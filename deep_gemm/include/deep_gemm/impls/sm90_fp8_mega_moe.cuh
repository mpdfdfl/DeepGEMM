#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/scheduler/sm90_mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// SM90 (Hopper) FP8 MegaMoE — N-split kernel.
// BLOCK_M=64, BLOCK_N=256: the two math warpgroups N-split each tile (one
// m64n128 WGMMA per WG) and share one A-tile load. The L1 epilogue
// cross-reduces the per-row amax over the two 64-col halves via SMEM so a
// single per-128 SF covers the full 128 post-SwiGLU columns; the L2 epilogue
// casts to BF16 and NVLink-scatters to remote combine buffers.
template <
    uint32_t kNumMaxTokensPerRank,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts, uint32_t kNumTopk,
    uint32_t kNumExpertsPerWave,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kNumMaxPoolTokens,
    uint32_t kNumPaddedSFPoolTokens,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
    uint32_t kNumEpilogueThreads,
    uint32_t kNumSMs, uint32_t kNumRanks,
    float kActivationClamp,
    bool kFastMath,
    bool kL2NMajorSchedule,
    uint32_t L1_SHAPE_N              = kIntermediateHidden * 2,
    uint32_t L1_SHAPE_K              = kHidden,
    uint32_t L2_SHAPE_N              = kHidden,
    uint32_t L2_SHAPE_K              = kIntermediateHidden,
    uint32_t kNumDispatchWarps       = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps       = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups  = kNumEpilogueWarps / 4,
    uint32_t kNumThreads             = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp       = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank      = kNumExperts / kNumRanks>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm90_fp8_mega_moe_impl(void* y,
                       int* cumulative_local_expert_recv_stats,
                       const uint32_t num_tokens,
                       const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
                       const float* __restrict__ l1_weights_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
                       const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
                       const float* __restrict__ l2_weights_sf) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000))
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 32 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads % 32 == 0, "Invalid number of GEMM TMA warps");
    DG_STATIC_ASSERT(kNumDispatchThreads + kNumNonEpilogueThreads == 128,
                     "Dispatch + TMA must form exactly 1 HW warpgroup (128 threads)");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of math/epilogue threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    DG_STATIC_ASSERT(BLOCK_M == 64, "BLOCK_M is fixed to 64 (one m64 WGMMA covers all rows)");
    DG_STATIC_ASSERT(BLOCK_N == 256, "BLOCK_N is fixed to 256 (N-split into two 128-col halves)");
    DG_STATIC_ASSERT(BLOCK_K == 128, "BLOCK_K is fixed to 128 (per-128 SF)");

    // Thread / warp identification
    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx   = ptx::get_lane_idx();

    // Prefetch all TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
    }

    // Workspaces and symmetric buffer slicing (mirror SM100 layout; both L1
    // and L2 activation SF are per-128 K float)
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    constexpr auto fp8_token_layout              = layout::Data(kHidden);
    constexpr auto bf16_token_layout             = layout::Data(kHidden * sizeof(nv_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    // Per-128 K float SF: 4 bytes per group => `kHidden / 32` bytes/token
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);
    // NOTES: must match the host buffer allocation in `get_symm_buffer_size_for_mega_moe`
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32);
    constexpr auto input_topk_idx_layout      = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout  = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout     = layout::Data(sizeof(float), false);

    // Registered input area
    const auto input_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank, workspace.get_end_ptr());
    const auto input_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank, input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer     = layout::Buffer(input_topk_idx_layout, 1, kNumMaxTokensPerRank, input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank, input_topk_idx_buffer.get_end_ptr());

    // L1 input area
    const auto l1_token_buffer        = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens, input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer           = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens, l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens, l1_sf_buffer.get_end_ptr());

    // L2 input area
    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens, l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer    = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens, l2_token_buffer.get_end_ptr());

    // Combine input area
    const auto combine_token_buffer = layout::Buffer(bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, l2_sf_buffer.get_end_ptr());

    // GEMM data types and shape constants
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;
    // N-split: each math WG owns WG_BLOCK_N = 128 columns (one m64n128 WGMMA, 64 accum-floats/thread)
    constexpr uint32_t WG_BLOCK_N = BLOCK_N / 2;                                          // 128
    using L1WGMMA                 = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type; // M=64, N=128, K=32
    using L2WGMMA                 = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
    static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
                  "Unexpected WGMMA shape");

    // Cluster=1, no multicast: A (shared by both WGs) and B (split across WGs) are loaded full-sized
    constexpr uint32_t LOAD_BLOCK_M   = BLOCK_M;                     // 64
    constexpr uint32_t LOAD_BLOCK_N   = BLOCK_N;                     // 256
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;                 // 128 post-SwiGLU (one per-128 SF group)
    constexpr uint32_t kSwizzleAMode  = BLOCK_K * sizeof(a_dtype_t); // 128
    constexpr uint32_t kSwizzleBMode  = BLOCK_K * sizeof(b_dtype_t); // 128
    constexpr uint32_t kGranK         = 128; // L1 acts SF, weights SF
    constexpr uint32_t kL2ActsSFGranK = 128; // L2 acts SF (per-128 K, matches DeepEP)

    // Shared memory layout
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    // Per-warp send buffers hold one pull chunk (not the full token)
    constexpr auto pull_layout = layout::Data(kNumBytesPerPull);
    DG_STATIC_ASSERT(kHidden % kNumBytesPerPull == 0, "kNumBytesPerPull must divide hidden");
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(pull_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    // SFA per-stage: one BLOCK_K=128 tile maps to exactly one per-128 SF group per row
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE =
        math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u);
    // Per-stage weight SF slots (2 floats), filled by the B loader right next to its
    // B-tile TMA and published through the SAME full/empty protocol (mbarrier arrive
    // has release semantics; the consumer's full-wait has acquire) — so the math
    // warps never touch global memory for weight SF. Slot meaning per phase:
    //   L1: slot0 = gate_sf[k], slot1 = up_sf[k]  (shared by both WGs)
    //   L2: slot g = SF of 128-col block n*2+g    (WG g reads slot g)
    constexpr uint32_t kNumSFBSlotsPerStage    = 2;
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = kNumSFBSlotsPerStage * sizeof(float);

    // Weight SF layout constants (block (128, 128) scales, MN-major).
    // L1 SF shape (E, 2*IH/128, H/128); the gate/up gran-8 interleave keeps a 256-wide
    // tile inside ONE gate + ONE up SF block shared by both WGs.
    // L2 SF shape (E, H/128, IH/128), one block per 128 output cols.
    constexpr uint32_t kL1SFKBlocks   = kHidden / 128;
    constexpr uint32_t kL2SFKBlocks   = kIntermediateHidden / 128;
    constexpr uint32_t kL1SFGateBlks  = kIntermediateHidden / 128;
    constexpr uint32_t kL1SFPerExpert = (kIntermediateHidden * 2 / 128) * kL1SFKBlocks;
    constexpr uint32_t kL2SFPerExpert = (kHidden / 128) * kL2SFKBlocks;

    // CD staging: max of L1 FP8 (BLOCK_M x L1_OUT_BLOCK_N) and L2 BF16 (BLOCK_M x BLOCK_N);
    // each WG writes its own 128-col half of a single full tile.
    // Double-buffered so a tile's TMA store drains while the next tile's MMA+epilogue
    // fills the other buffer (the host pipeline config accounts for the 2x CD region).
    constexpr uint32_t kNumCDStages     = 2;
    constexpr uint32_t SMEM_CD_L1_SIZE  = BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t SMEM_CD_L2_SIZE  = BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_BUF_SIZE = math::constexpr_align(
        SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_CD_SIZE = kNumCDStages * SMEM_CD_BUF_SIZE;

    // Cross-WG per-row amax exchange for the L1 epilogue (one BLOCK_M-float slot per WG);
    // must match the host `smem_amax` accounting
    constexpr uint32_t SMEM_AMAX_SIZE = math::constexpr_align<uint32_t>(
        2 * BLOCK_M * sizeof(float), kSharedMemoryAlignment);

    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + SMEM_AMAX_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

    // SMEM pointers
    auto smem_expert_count       = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        pull_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE);

    // CD staging is shared by L1 (FP8) and L2 (BF16); reinterpret-cast per phase
    auto smem_cd_l1 = utils::PatternVisitor([=](const uint32_t& buf) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(
            math::advance_ptr(smem_gemm_base, buf * SMEM_CD_BUF_SIZE));
    });
    auto smem_cd_l2 = utils::PatternVisitor([=](const uint32_t& buf) {
        return reinterpret_cast<nv_bfloat16*>(
            math::advance_ptr(smem_gemm_base, buf * SMEM_CD_BUF_SIZE));
    });

    // Cross-WG amax exchange region: `smem_amax[wg]` = that WG's BLOCK_M per-row amax slots
    auto smem_amax = utils::PatternVisitor([=](const uint32_t& wg) {
        return reinterpret_cast<float*>(
            math::advance_ptr(smem_gemm_base, SMEM_CD_SIZE + wg * BLOCK_M * sizeof(float)));
    });

    constexpr uint32_t SMEM_GEMM_OFFSET = SMEM_CD_SIZE + SMEM_AMAX_SIZE;
    auto smem_a                         = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_GEMM_OFFSET + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b                         = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_GEMM_OFFSET + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr                   = math::advance_ptr<uint8_t>(smem_gemm_base,
                                                                     SMEM_GEMM_OFFSET + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa                       = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    // Per-stage weight SF slots (see SMEM_SFB_SIZE_PER_STAGE above)
    auto smem_sfb                       = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<float*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE) +
               i * kNumSFBSlotsPerStage;
    });

    // Barrier layout: dispatch | full | empty | combine
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + i;
    });
    auto full_barriers     = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + i;
    });
    auto empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages + i;
    });
    auto combine_barriers  = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i;
    });

    // Initialization
    if (warp_idx == 0) {
// Clean expert-count shared memory
#pragma unroll
        for (uint32_t i = lane_idx; i < kNumExperts; i += 32)
            ptx::st_shared(smem_expert_count + i, 0u);
    } else if (warp_idx == 1) {
// Init dispatch m-barriers
#pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM full/empty and combine barriers
        if (cute::elect_one_sync()) {
#pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                // 2 producer warps (A+SFA loader, B+SFB loader)
                full_barriers[i]->init(2);
                // Released by lane 0 of every epilogue warp (both WGs consume every stage)
                empty_barriers[i]->init(kNumEpilogueWarps);
            }
#pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Scheduler (cluster=1)
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank, kNumExpertsPerWave,
        kNumSMs, kNumRanks, /*kClusterSize=*/1u, kL2NMajorSchedule>(workspace);

    // Pipeline state shared by TMA loaders and math warpgroups
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx  = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase     ^= stage_idx == 0;
    };

    // Intra-SM barrier indices (mirroring SM100)
    constexpr uint32_t kDispatchBarrierIdx             = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx         = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx      = 3;

    // Cross-rank NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag  = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Register reconfiguration: dispatch + TMA warps share HW WG0 (same dealloc);
    // budget = 128*48 + 256*224 = 63488 <= 64512, setmaxnreg must be a multiple of 8
    constexpr uint32_t kNumDispatchRegisters    = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 48;  // must match dispatch (same HW WG)
    constexpr uint32_t kNumEpilogueRegisters    = 224;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                             kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                             kNumEpilogueRegisters * kNumEpilogueThreads <=
                         64512,
                     "Too many registers");

    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // Dispatch warps: mirror SM100, but SF is per-128 channel float copied
    // linearly into the MN-major L1 SF buffer (no UTCCP 4x32 transpose)
    if (warp_idx < kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto read_topk_idx             = [&](const auto& process) {
#pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count tokens per expert
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

// Stake out per-expert SM offsets via global atomic
#pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i]      = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source token-topk indices to remote ranks
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
            const auto dst_ptr      = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });

        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() {
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            });

        if (sm_idx == 0) {
#pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx         = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status        = *workspace.get_expert_send_count_ptr(i);
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() {
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            },
            false, true);

        // Sync with epilogue warps before pulling tokens
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Token / SF pull loop
        uint32_t pull_mbarrier_phase = 0;
        const auto pull_buffer       = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier     = dispatch_barriers[warp_idx];

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane          = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx                       = -1;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};
        uint32_t expert_start_idx = 0, expert_end_idx = 0;
        uint32_t expert_pool_block_offset = 0;

        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx;; token_idx += kNumGlobalWarps) {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= kNumExpertsPerRank)
                    break;
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
                expert_start_idx          = expert_end_idx;
                expert_end_idx           += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
#pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    const uint32_t j     = i * 32 + lane_idx;
                    stored_rank_count[i] = j < kNumRanks ? static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // Round-robin rank selection (identical to SM100)
            uint32_t current_rank_in_expert_idx;
            uint32_t remaining[kNumRanksPerLane];
#pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];
            uint32_t offset              = 0;
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t slot_idx            = token_idx_in_expert;
            uint32_t token_idx_in_rank;
            while (true) {
                uint32_t num_actives_in_lane = 0;
                uint32_t min_in_lane         = 0xffffffff;
#pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                    num_actives_in_lane += remaining[i] > 0;
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);
                }
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                const uint32_t length           = __reduce_min_sync(0xffffffff, min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks          = 0;
                    current_rank_in_expert_idx       = 0;
#pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
                        const uint32_t mask             = __ballot_sync(0xffffffff, remaining[i] > 0);
                        const uint32_t num_active_lanes = __popc(mask);
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }
                slot_idx -= num_round_tokens;
                offset   += length;
#pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++i)
                    remaining[i] -= cute::min(remaining[i], length);
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            // Hidden bytes are divided into chunks (single-buffered pull chunk per warp)
            constexpr uint32_t kNumChunks = kHidden / kNumBytesPerPull;

            // TMA load token from remote rank and store into local, chunk by chunk
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            const auto src_base_ptr       = sym_buffer.map(
                input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                current_rank_in_expert_idx);
            const auto dst_base_ptr = l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr();
            const auto issue_and_wait_pull_store = [&](const uint32_t& i) {
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);
                ptx::tma_store_1d(
                    math::advance_ptr(dst_base_ptr, i * kNumBytesPerPull),
                    pull_buffer.get_base_ptr(), kNumBytesPerPull);
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            };
            if (cute::elect_one_sync()) {
#pragma unroll
                for (uint32_t i = 0; i < kNumChunks; ++i) {
                    ptx::tma_load_1d(
                        pull_buffer.get_base_ptr(),
                        math::advance_ptr(src_base_ptr, i * kNumBytesPerPull),
                        pull_mbarrier, kNumBytesPerPull);
                    ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
                    i != (kNumChunks - 1) ? issue_and_wait_pull_store(i) : void();
                }
            }
            __syncwarp();

            // Copy SF (overlaps with the last chunk's TMA load from remote):
            // per-128 K floats written linearly (no UTCCP transpose)
            constexpr uint32_t kNumSFFloats = kHidden / 128;
            DG_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                current_rank_in_expert_idx);
            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<float>();
#pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFFloats)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and metadata, then complete the last chunk's store
            if (cute::elect_one_sync()) {
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                issue_and_wait_pull_store(kNumChunks - 1);
                ptx::red_add_rel(
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        // Cleanup workspace, overlapping with combine
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
#pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j)  = 0;
                }
                __syncwarp();
            }
        }

        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() {
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
            },
            true, false);

        // GEMM TMA load warps: warp 0 loads A+SFA, warp 1 loads B+SFB, the rest idle
    } else if (warp_idx == kNumDispatchWarps) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // NOTES: manually inlined scheduler loop, avoids lambda outlining that
        // causes C7510 WGMMA serialization in the math warpgroup path
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == sched::BlockPhase::None)
                break;
            const auto num_k_blocks       = block_phase == sched::BlockPhase::Linear2
                                                ? L2_SHAPE_K / BLOCK_K
                                                : L1_SHAPE_K / BLOCK_K;
            const auto tensor_map_a_ptr   = block_phase == sched::BlockPhase::Linear2
                                                ? &tensor_map_l2_acts
                                                : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                                                ? &tensor_map_l2_acts_sf
                                                : &tensor_map_l1_acts_sf;

            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait for the pool to be ready
            if (block_phase == sched::BlockPhase::Linear1) {
                const auto ptr      = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();
                while (ptx::ld_acq(ptr) != expected)
                    ;
            } else {
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                // Each L1 N block sets one bit; total bits = L1_SHAPE_N / BLOCK_N.
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                const uint64_t expected          = (kNumL1BlockNs >= 64)
                                                       ? ~0ull
                                                       : ((1ull << kNumL1BlockNs) - 1ull);
                while (ptx::ld_acq_gpu(ptr) != expected)
                    ;
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t m_idx = pool_block_idx * BLOCK_M;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx],
                        k_idx, m_idx, 1);

                    // TMA load SFA: one BLOCK_K=128 tile = one per-128 SF group, box (BLOCK_M, 1)
                    tma::copy<BLOCK_M, 1, 0, float>(
                        tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx],
                        m_idx, k_block_idx, 1);
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        SMEM_A_SIZE_PER_STAGE + BLOCK_M * sizeof(float));
                }
                __syncwarp();
            }
        }

    } else if (warp_idx == kNumDispatchWarps + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // Manually inlined scheduler loop (matches A loader expansion)
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == sched::BlockPhase::None)
                break;
            const auto num_k_blocks = block_phase == sched::BlockPhase::Linear2
                                          ? L2_SHAPE_K / BLOCK_K
                                          : L1_SHAPE_K / BLOCK_K;

            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;

            const uint32_t shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            // Weight SF bases for this tile's two per-stage slots (see SMEM_SFB layout)
            const float* sfb_base_0;
            const float* sfb_base_1;
            if (block_phase == sched::BlockPhase::Linear1) {
                const float* base = l1_weights_sf + local_expert_idx * kL1SFPerExpert;
                sfb_base_0        = base + n_block_idx * kL1SFKBlocks;                    // gate
                sfb_base_1        = base + (kL1SFGateBlks + n_block_idx) * kL1SFKBlocks;  // up
            } else {
                // L2: n_block_idx is in 256-col units -> SF block pair (n*2, n*2+1)
                const float* base = l2_weights_sf + local_expert_idx * kL2SFPerExpert;
                sfb_base_0        = base + (n_block_idx * 2u) * kL2SFKBlocks;
                sfb_base_1        = base + (n_block_idx * 2u + 1) * kL2SFKBlocks;
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                if (cute::elect_one_sync()) {
                    const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load B
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx],
                        k_idx, n_idx, 1);

                    // Stage this k's weight SF next to the B tile (the ~200-cycle __ldg
                    // hides inside the B TMA flight); the SAME arrive below has release
                    // semantics and publishes these stores to the math warps' full-wait
                    ptx::st_shared(smem_sfb[stage_idx] + 0, __ldg(sfb_base_0 + k_block_idx));
                    ptx::st_shared(smem_sfb[stage_idx] + 1, __ldg(sfb_base_1 + k_block_idx));

                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }

    } else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Idle non-epilogue warps (dead with 2 dispatch + 2 TMA warps filling HW WG0);
        // must still join the warpgroup-collective setmaxnreg
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // Math warpgroups: WGMMA + epilogue + combine
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        const uint32_t epilogue_warp_idx   = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
        const uint32_t epilogue_wg_idx     = epilogue_warp_idx / 4;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        const uint32_t warp_idx_in_wg      = epilogue_warp_idx % 4;

        // WGMMA-output register layout helpers
        const uint32_t row_idx = lane_idx / 4;
        const uint32_t col_idx = lane_idx % 4;
        const uint32_t r_0     = warp_idx_in_wg * 16 + row_idx;
        const uint32_t r_1     = r_0 + 8;

        // N-split: WG g owns cols [g*WG_BLOCK_N, (g+1)*WG_BLOCK_N) of the same tile;
        // A + SFA are shared (no row offset), B is offset by this WG's column half
        DG_STATIC_ASSERT(WG_BLOCK_N == L1WGMMA::N, "WG_BLOCK_N must equal one WGMMA N (128)");
        DG_STATIC_ASSERT(kNumEpilogueWarpgroups == 2, "N-split requires exactly 2 math warpgroups");
        const uint32_t col_block_offset = epilogue_wg_idx * WG_BLOCK_N;
        const uint32_t b_col_off        = col_block_offset * BLOCK_K; // elems into smem_b
        constexpr uint32_t a_row_off    = 0;
        constexpr uint32_t sfa_row_off  = 0;

        // Sync with dispatch
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // NOTES: manually inlined scheduler loop, avoids lambda outlining that
        // causes ptxas C7510 (serialized WGMMA pipeline)
        scheduler.fetch_expert_recv_count();
        scheduler.set_expert_idx(0);
        uint32_t pos = 0;
        // Deferred L1-store retirement: a tile's arrival-mask bit (which tells L2 the
        // L1 output is in HBM) is published one tile late, after its TMA store drains
        bool l1_store_pending              = false;
        uint32_t l1_pending_pool_block_idx = 0;
        uint32_t l1_pending_n_block_idx    = 0;
        while (true) {
            CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == sched::BlockPhase::None)
                break;
            const auto num_k_blocks = block_phase == sched::BlockPhase::Linear2
                                          ? L2_SHAPE_K / BLOCK_K
                                          : L1_SHAPE_K / BLOCK_K;

            // Both WGs process every tile, each handling its own column half
            const uint32_t valid_m        = scheduler.template get_valid_m<false>();
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            const uint32_t m_idx          = pool_block_idx * BLOCK_M;
            // Alternate CD buffer per tile so the previous L1 store drains in the background
            const uint32_t cd_buf = pos % kNumCDStages;
            const uint32_t n_idx  = n_block_idx * BLOCK_N;

            // L1->L2 transition: flush the deferred L1 store so its mask bit is
            // published before any L2 loader spins on it
            if (block_phase == sched::BlockPhase::Linear2 and l1_store_pending) {
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(l1_pending_pool_block_idx),
                        1ull << l1_pending_n_block_idx);
                }
                l1_store_pending = false;
            }

            // GEMM (MMA region)
            using WGMMA                        = L1WGMMA;
            constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum; // 64 for M=64,N=128
            float final_accum[kAccumPerThread] = {};
            float accum[kAccumPerThread];

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);

                // Activation SF lives in a separate SMEM region, so its ld_shared is
                // issued after WGMMA issue and overlaps the async MMA
                float scale_a_0_lo, scale_a_1_lo;

                if (block_phase == sched::BlockPhase::Linear1) {
// Single per-128 K-block WGMMA group
#pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
#pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + a_row_off + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + b_col_off + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();

                    // Read act SF + this k's weight SF (per-stage slots the B loader
                    // filled, published by its full-barrier arrive) while WGMMA executes
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + sfa_row_off + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + sfa_row_off + r_1);
                    const float cur_gate_sf = ptx::ld_shared(smem_sfb[stage_idx] + 0);
                    const float cur_up_sf   = ptx::ld_shared(smem_sfb[stage_idx] + 1);

#pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

                    // Gate/up alternate at gran=8 along N; pre-multiply act-SF x weight-SF
                    // once per k-block so the inner element loop is pure FMA
                    const float sg0 = scale_a_0_lo * cur_gate_sf;
                    const float sg1 = scale_a_1_lo * cur_gate_sf;
                    const float su0 = scale_a_0_lo * cur_up_sf;
                    const float su1 = scale_a_1_lo * cur_up_sf;
#pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
                        const float s0          = (i & 1u) ? su0 : sg0;
                        const float s1          = (i & 1u) ? su1 : sg1;
                        // NOTES: FMA promote (1 rounding) is required for bit-exactness with
                        // the m_grouped reference; plain mul+add diverges by 1 fp32 ULP
                        final_accum[i * 4 + 0] = __fmaf_rn(s0, accum[i * 4 + 0], final_accum[i * 4 + 0]);
                        final_accum[i * 4 + 1] = __fmaf_rn(s0, accum[i * 4 + 1], final_accum[i * 4 + 1]);
                        final_accum[i * 4 + 2] = __fmaf_rn(s1, accum[i * 4 + 2], final_accum[i * 4 + 2]);
                        final_accum[i * 4 + 3] = __fmaf_rn(s1, accum[i * 4 + 3], final_accum[i * 4 + 3]);
                    }
                } else {
// L2: per-128 K — a single BLOCK_K=128 WGMMA group with one SFA (matching L1).
#pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_arrive();
#pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_a[stage_idx] + a_row_off + k * WGMMA::K, 1);
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_b[stage_idx] + b_col_off + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    ptx::warpgroup_commit_batch();

                    // Read act SF + this k's weight SF (per-stage slot the B loader
                    // filled, published by its full-barrier arrive) while WGMMA executes
                    scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + sfa_row_off + r_0);
                    scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + sfa_row_off + r_1);
                    const float cur_l2_sf = ptx::ld_shared(smem_sfb[stage_idx] + epilogue_wg_idx);

#pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread; ++i)
                        ptx::warpgroup_fence_operand(accum[i]);
                    ptx::warpgroup_wait<0>();

                    if (lane_idx == 0)
                        empty_barriers[stage_idx]->arrive();

                    // Single scalar SF broadcast across N; pre-multiply once, inner loop is pure FMA
                    const float s0 = scale_a_0_lo * cur_l2_sf;
                    const float s1 = scale_a_1_lo * cur_l2_sf;
#pragma unroll
                    for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
                        // FMA promote — see the L1 path
                        final_accum[i * 4 + 0] = __fmaf_rn(s0, accum[i * 4 + 0], final_accum[i * 4 + 0]);
                        final_accum[i * 4 + 1] = __fmaf_rn(s0, accum[i * 4 + 1], final_accum[i * 4 + 1]);
                        final_accum[i * 4 + 2] = __fmaf_rn(s1, accum[i * 4 + 2], final_accum[i * 4 + 2]);
                        final_accum[i * 4 + 3] = __fmaf_rn(s1, accum[i * 4 + 3], final_accum[i * 4 + 3]);
                    }
                }
            }

            if (block_phase == sched::BlockPhase::Linear1) {
                // L1 epilogue: SwiGLU + FP8 quantize + TMA store.
                // `final_accum` = 16 chunks of 8 N-cols (4 floats/thread: r0c0, r0c1, r1c0, r1c1);
                // gate chunks even, up chunks odd; pair p covers output cols p*8 + col_idx*2 + {0,1}
                constexpr uint32_t kNumPairs = kAccumPerThread / 8; // 8 (this WG's 64 output cols)
                float swiglu_r0[kNumPairs][2];
                float swiglu_r1[kNumPairs][2];

                // Per-row amax across all 8 pairs
                float amax_r0 = 0.0f, amax_r1 = 0.0f;

// Compute SwiGLU + per-pair amax
#pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++p) {
                    const uint32_t gate = 2 * p, up = 2 * p + 1;

                    // NOTES: round the L1 output to bf16 before clamp/SiLU to match sglang's
                    // grouped_gemm_nt_f8f8bf16 epilogue cast (bit-exactness)
                    auto bf16_round = [](float x) -> float {
                        return __bfloat162float(__float2bfloat16_rn(x));
                    };
                    // Clamp: gate upper-side only (SiLU(-inf) -> 0), up both sides (matches SM100)
                    auto clamp_gate = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(x, kActivationClamp);
                    };
                    auto clamp_up = [](float& x) {
                        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
                            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
                    };
                    float g_r0_c0 = bf16_round(final_accum[gate * 4 + 0]);
                    clamp_gate(g_r0_c0);
                    float g_r0_c1 = bf16_round(final_accum[gate * 4 + 1]);
                    clamp_gate(g_r0_c1);
                    float g_r1_c0 = bf16_round(final_accum[gate * 4 + 2]);
                    clamp_gate(g_r1_c0);
                    float g_r1_c1 = bf16_round(final_accum[gate * 4 + 3]);
                    clamp_gate(g_r1_c1);
                    float u_r0_c0 = bf16_round(final_accum[up * 4 + 0]);
                    clamp_up(u_r0_c0);
                    float u_r0_c1 = bf16_round(final_accum[up * 4 + 1]);
                    clamp_up(u_r0_c1);
                    float u_r1_c0 = bf16_round(final_accum[up * 4 + 2]);
                    clamp_up(u_r1_c0);
                    float u_r1_c1 = bf16_round(final_accum[up * 4 + 3]);
                    clamp_up(u_r1_c1);

                    // SiLU*up matching sglang's `sgl_kernel.silu_and_mul` (built with
                    // -use_fast_math): kFastMath reproduces its approx PTX via inline asm,
                    // and the fp32 result is rounded to bf16 (that op's output dtype)
                    auto silu_mul = [](float x, float u) -> float {
                        float o;
                        if constexpr (kFastMath) {
                            float t, e, d, s;
                            asm("mul.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(x));   // -x*log2(e)
                            asm("ex2.approx.ftz.f32 %0, %1;"      : "=f"(e) : "f"(t));   // 2^t
                            asm("add.ftz.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));   // 1+exp(-x)
                            asm("div.approx.ftz.f32 %0, %1, %2;"  : "=f"(s) : "f"(x), "f"(d)); // silu
                            asm("mul.ftz.f32 %0, %1, %2;"         : "=f"(o) : "f"(s), "f"(u)); // *up
                        } else {
                            const float s = x / (1.0f + expf(-x));
                            o = s * u;
                        }
                        return __bfloat162float(__float2bfloat16_rn(o));  // store → bf16
                    };

                    swiglu_r0[p][0] = silu_mul(g_r0_c0, u_r0_c0);
                    swiglu_r0[p][1] = silu_mul(g_r0_c1, u_r0_c1);
                    swiglu_r1[p][0] = silu_mul(g_r1_c0, u_r1_c0);
                    swiglu_r1[p][1] = silu_mul(g_r1_c1, u_r1_c1);

                    amax_r0 = cute::max(amax_r0, cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
                    amax_r1 = cute::max(amax_r1, cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
                }

                // NOTES: the topk weight is NOT applied here — sglang quantizes the
                // unweighted SwiGLU output and applies the weight only at combine

                // Reduce amax across the 4 col-lanes sharing a row: must be the intra-group
                // `warp_reduce<4, false>` (`<4, true>` would merge amax across different rows)
                amax_r0 = math::warp_reduce<4, false>(amax_r0, math::ReduceMax<float>());
                amax_r1 = math::warp_reduce<4, false>(amax_r1, math::ReduceMax<float>());

                // Cross-WG amax exchange via SMEM so both WGs derive the SAME per-128 SF
                // covering the two 64-col halves (rows are shared, index by row)
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                if (col_idx == 0) {
                    smem_amax[epilogue_wg_idx][r_0] = amax_r0;
                    smem_amax[epilogue_wg_idx][r_1] = amax_r1;
                }
                // 256-thread rendezvous: both WGs' halves are now in SMEM.
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                {
                    const uint32_t other_wg = epilogue_wg_idx ^ 1u;
                    amax_r0                 = cute::max(amax_r0, smem_amax[other_wg][r_0]);
                    amax_r1                 = cute::max(amax_r1, smem_amax[other_wg][r_1]);
                }

                // Quant matching sglang's `per_token_group_quant_8bit` JIT kernel (NOT
                // fast-math): stored y_s = amax/448 and code = clamp(val / y_s, ±448),
                // both PRECISE div.rn — reciprocal-mul forms diverge by 1 ULP
                constexpr float kE4M3Max = 448.0f;
                const float amax_c0   = fmaxf(amax_r0, 1e-10f);
                const float amax_c1   = fmaxf(amax_r1, 1e-10f);
                const float sf_r0     = amax_c0 / kE4M3Max;
                const float sf_r1     = amax_c1 / kE4M3Max;
                // Quant divides by the SAME stored y_s, not a separately-rounded reciprocal
                const float yscale_r0 = sf_r0;
                const float yscale_r1 = sf_r1;

                // Quantize and write to smem_cd_l1 (row-major, matches the swizzle-0 TMA
                // descriptor); each WG writes its own 64-col half of the 128-wide staging tile
                constexpr uint32_t WG_OUT_BLOCK_N = L1WGMMA::N / 2; // 64 output cols/WG
                const uint32_t wg_out_col_off     = epilogue_wg_idx * WG_OUT_BLOCK_N;
                auto* smem_cd_l1_buf              = smem_cd_l1[cd_buf];
#pragma unroll
                for (uint32_t p = 0; p < kNumPairs; ++p) {
                    // True div by the stored y_s (a separately-rounded reciprocal diverges
                    // by 1 ULP at e4m3 grid midpoints)
                    auto qmul = [](float v, float ys) -> float {
                        return fminf(fmaxf(v / ys, -448.0f), 448.0f);
                    };
                    const float v00 = qmul(swiglu_r0[p][0], yscale_r0);
                    const float v01 = qmul(swiglu_r0[p][1], yscale_r0);
                    const float v10 = qmul(swiglu_r1[p][0], yscale_r1);
                    const float v11 = qmul(swiglu_r1[p][1], yscale_r1);

                    const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
                    const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

                    const uint32_t col = wg_out_col_off + p * 8 + col_idx * 2;
                    auto* p0           = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_buf + r_0 * L1_OUT_BLOCK_N + col);
                    auto* p1 = reinterpret_cast<uint16_t*>(
                        smem_cd_l1_buf + r_1 * L1_OUT_BLOCK_N + col);
                    *p0 = r0_pair.__x;
                    *p1 = r1_pair.__x;
                }

                // Write the shared per-128 SF into the L2 acts SF buffer; both WGs derived
                // identical sf_r*, so only WG0's col_idx==0 lanes write
                if (epilogue_wg_idx == 0 and col_idx == 0) {
                    auto sf_base_ptr                                          = l2_sf_buffer.get_base_ptr<float>();
                    const uint32_t token_r0                                   = pool_block_idx * BLOCK_M + r_0;
                    const uint32_t token_r1                                   = pool_block_idx * BLOCK_M + r_1;
                    const uint32_t k_sf_idx                                   = n_block_idx; // one per-128 SF per L1 block
                    sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
                    sf_base_ptr[k_sf_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
                }

                // 256-thread rendezvous: both halves written before the full-tile store
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

                // Single full-tile TMA store, issued by one warp of WG0 only
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_l1_output,
                        smem_cd_l1_buf,
                        out_n_idx,
                        m_idx);
                    cute::tma_store_arrive();
                }
                __syncwarp();

                // Retire the PREVIOUS pending L1 store (tma_store_wait<1> leaves this tile's
                // store in flight) and publish its arrival-mask bit; this tile's own store
                // keeps draining while the next tile's MMA runs
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (l1_store_pending) {
                    ptx::tma_store_wait<1>();
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        ptx::red_or_rel_gpu(
                            workspace.get_l2_arrival_mask_ptr(l1_pending_pool_block_idx),
                            1ull << l1_pending_n_block_idx);
                    }
                }
                l1_store_pending          = true;
                l1_pending_pool_block_idx = pool_block_idx;
                l1_pending_n_block_idx    = n_block_idx;
            } else {
                // L2 epilogue: BF16 cast + NVLink scatter. Each WG writes/scatters its
                // 128-col half of the shared 256-wide tile
                constexpr uint32_t kNumRowsPerWarp = BLOCK_M / 8;

                // NOTES: scatter the RAW (unweighted) bf16 L2 output — sglang applies the
                // topk weight only at combine (see the two-layer combine below)
#pragma unroll
                for (uint32_t i = 0; i < kAccumPerThread / 8; ++i) {
                    // 8 floats per i = 2 chunks of (r0c0, r0c1, r1c0, r1c1)
                    const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;

                    // Pack each (row, col) pair into BF162
                    const uint32_t r0_lo = math::cast_into_bf16_and_pack(
                        final_accum[chunk_lo * 4 + 0], final_accum[chunk_lo * 4 + 1]);
                    const uint32_t r1_lo = math::cast_into_bf16_and_pack(
                        final_accum[chunk_lo * 4 + 2], final_accum[chunk_lo * 4 + 3]);
                    const uint32_t r0_hi = math::cast_into_bf16_and_pack(
                        final_accum[chunk_hi * 4 + 0], final_accum[chunk_hi * 4 + 1]);
                    const uint32_t r1_hi = math::cast_into_bf16_and_pack(
                        final_accum[chunk_hi * 4 + 2], final_accum[chunk_hi * 4 + 3]);

                    auto write_pair = [&](uint32_t row, uint32_t col, uint32_t packed) {
                        auto smem_ptr = smem_cd_l2[cd_buf] + row * BLOCK_N + col_block_offset + col;
                        *reinterpret_cast<uint32_t*>(smem_ptr) = packed;
                    };
                    write_pair(r_0, chunk_lo * 8 + col_idx * 2, r0_lo);
                    write_pair(r_0, chunk_hi * 8 + col_idx * 2, r0_hi);
                    write_pair(r_1, chunk_lo * 8 + col_idx * 2, r1_lo);
                    write_pair(r_1, chunk_hi * 8 + col_idx * 2, r1_hi);
                }

                ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                // NVLink scatter: 16 lanes per row, each lane 8 cols (1 uint4)
                const uint32_t row_in_warp_block = lane_idx / 16; // 0 or 1
                const uint32_t lane_in_row       = lane_idx % 16;
                const uint32_t cols_per_lane     = WG_BLOCK_N / 16; // 8 cols per lane
                static_assert(WG_BLOCK_N == 128, "Layout assumes WG_BLOCK_N=128");

#pragma unroll
                for (uint32_t j = 0; j < kNumRowsPerWarp; ++j) {
                    const uint32_t tile_row = warp_idx_in_wg * 16 + j * 2 + row_in_warp_block;
                    if (tile_row >= valid_m)
                        break;

                    const auto src_metadata      = *workspace.get_token_src_metadata_ptr(m_idx + tile_row);
                    const uint32_t dst_rank_idx  = src_metadata.rank_idx;
                    const uint32_t dst_token_idx = src_metadata.token_idx;
                    const uint32_t dst_topk_idx  = src_metadata.topk_idx;

                    // Read 8 BF16s (1 uint4) from SMEM
                    auto smem_ptr     = smem_cd_l2[cd_buf] + tile_row * BLOCK_N + col_block_offset + lane_in_row * cols_per_lane;
                    const auto packed = *reinterpret_cast<uint4*>(smem_ptr);

                    // Write to remote at the destination column offset
                    const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                    auto dst_ptr         = math::advance_ptr<uint4>(
                        dst_token.get_base_ptr(),
                        (n_idx + col_block_offset) * sizeof(nv_bfloat16) + lane_in_row * sizeof(uint4));
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                }

                // Tile-boundary 256-thread rendezvous: without it a WG can start writing
                // the NEXT tile's smem_cd while the other still scatters THIS tile (the
                // FP8/BF16 layouts alias in raw bytes). L1 runs the symmetric barrier
                // once per tile, so the arrival counts stay consistent
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
            ++pos;
        }

        // Flush any still-pending L1 store (e.g. a wave ending on L1 blocks)
        if (l1_store_pending) {
            ptx::tma_store_wait<0>();
            ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                ptx::red_or_rel_gpu(
                    workspace.get_l2_arrival_mask_ptr(l1_pending_pool_block_idx),
                    1ull << l1_pending_n_block_idx);
            }
            l1_store_pending = false;
        }

        // Combine. NVLink barrier first: signals remote ranks that this rank's
        // scatter targets are fully written
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() {
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            });

        // Sync with dispatch so it may now safely clean workspace state
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        constexpr uint32_t kNumHiddenBytes   = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

        constexpr uint32_t kNumChunkSlots            = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        constexpr uint32_t kNumChunks =
            (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and kHidden <= 32 * kNumMaxRegistersForBuffer) ? 1 : 2;
        constexpr uint32_t kNumChunkBytes   = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4   = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
                                                                                    reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        const auto combine_load_buffer  = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer = math::advance_ptr<uint4>(
            smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        uint32_t combine_phase  = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // Two-layer combine matching sglang's ep_gather + deep_ep combine (verified
            // bit-exact): group topk slots by sender rank (= expert_id / kNumExpertsPerRank),
            // fp32-accumulate `L2_bf16 * weight` per group, ONE bf16 round per rank, then
            // fp32-sum the per-rank results. Lane l < kNumTopk owns slot l.
            const int64_t raw_expert = lane_idx < kNumTopk
                ? __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)
                : static_cast<int64_t>(-1);
            const bool slot_active   = (lane_idx < kNumTopk) and (raw_expert >= 0);
            const uint32_t slot_rank = slot_active
                ? static_cast<uint32_t>(raw_expert) / kNumExpertsPerRank
                : kNumRanks; // sentinel: inactive slots sort to the end
            const float slot_weight  = slot_active
                ? __ldg(input_topk_weights_buffer.get_base_ptr<float>() + token_idx * kNumTopk + lane_idx)
                : 0.0f;

            // Stable-sort active slots by (rank, slot) — matches ep_gather's intra-rank +
            // deep_ep's cross-rank order; `my_pos` = this slot's index in the sorted sequence
            uint32_t my_pos = 0xffffffffu, base = 0;
#pragma unroll
            for (uint32_t r = 0; r < kNumRanks; ++r) {
                const uint32_t rmask = __ballot_sync(0xffffffffu, slot_active and slot_rank == r);
                if (slot_active and slot_rank == r)
                    my_pos = base + __popc(rmask & ((1u << lane_idx) - 1u));
                base += __popc(rmask);
            }
            const uint32_t num_active = base; // == __popc(total_mask)

            // Gather per-sorted-position metadata (replicated to all lanes)
            uint32_t seq_slot[kNumTopk];
            uint32_t seq_rank[kNumTopk];
            float    seq_weight[kNumTopk];
#pragma unroll
            for (uint32_t p = 0; p < kNumTopk; ++p) {
                const uint32_t hit = __ballot_sync(0xffffffffu, slot_active and my_pos == p);
                const bool valid   = (p < num_active);
                const uint32_t src = valid ? static_cast<uint32_t>(__ffs(hit) - 1) : 0u;
                seq_slot[p]   = __shfl_sync(0xffffffffu, lane_idx,    src);
                seq_rank[p]   = valid ? __shfl_sync(0xffffffffu, slot_rank,   src) : kNumRanks;
                seq_weight[p] = valid ? __shfl_sync(0xffffffffu, slot_weight, src) : 0.0f;
            }

            for (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // Stream the rank-sorted sequence; the TMA double-buffer flows across
                // rank-group boundaries (state is per-token, not per-group)
                const auto load_pos = [&](const uint32_t& stage, const uint32_t& p) {
                    if (cute::elect_one_sync()) {
                        const auto src_ptr = math::advance_ptr<uint8_t>(
                            combine_token_buffer.get_rank_buffer(seq_slot[p])
                                .get_data_buffer(token_idx)
                                .get_base_ptr(),
                            chunk_byte_offset);
                        ptx::tma_load_1d(combine_load_buffer[stage], src_ptr, combine_load_barriers[stage], kNumChunkBytes);
                        ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[stage], kNumChunkBytes);
                    }
                    __syncwarp();
                };

                constexpr uint32_t kW = kNumUint4PerLane * kNumElemsPerUint4;
                float2 cross[kW] = {};      // inter-rank fp32 fold of per-rank bf16
                if (num_active > 0) {
                    load_pos(load_stage_idx, 0);
                    uint32_t next_p = 1, p = 0;
                    while (p < num_active) {
                        // Intra-rank fp32 accumulator for the current rank group
                        float2 group_acc[kW] = {};
                        const uint32_t cur_rank = seq_rank[p];
                        while (p < num_active and seq_rank[p] == cur_rank) {
                            if (next_p < num_active) { load_pos(load_stage_idx ^ 1, next_p); ++next_p; }
                            combine_load_barriers[load_stage_idx]->wait(combine_phase);
                            const float wgt = seq_weight[p];
#pragma unroll
                            for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                                const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                                const auto bf16_values  = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
#pragma unroll
                                for (uint32_t l = 0; l < kNumElemsPerUint4; ++l) {
                                    const float2 f2 = __bfloat1622float2(bf16_values[l]);
                                    float2& g       = group_acc[j * kNumElemsPerUint4 + l];
                                    // NOTES: FMA matches ep_gather's Triton contraction to fma.rn.f32
                                    // (mul+add diverges by 1 ULP when a rank has >=2 experts)
                                    g.x = __fmaf_rn(f2.x, wgt, g.x);
                                    g.y = __fmaf_rn(f2.y, wgt, g.y);
                                }
                            }
                            combine_phase  ^= load_stage_idx;
                            load_stage_idx ^= 1;
                            ++p;
                        }
                        // Rank-group boundary: ONE bf16 round per rank, fold into the
                        // cross-rank fp32 accumulator (ascending-rank order)
#pragma unroll
                        for (uint32_t j = 0; j < kW; ++j)
                            ptx::accumulate(cross[j], __float22bfloat162_rn(group_acc[j]));
                    }
                }

#pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
#pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)
                        casted_bf16[l] = __float22bfloat162_rn(cross[j * kNumElemsPerUint4 + l]);

                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
