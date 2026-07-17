import torch
from typing import Optional

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load DeepGEMM comm kernels, please check your PyTorch version: {exception}')

from .. import _C


# Deterministic swap-AB token-tile size (BLOCK_M) for the fused RS kernel, as a function of
# the per-rank token count ONLY. Both the Python buffer padding and the C++ launcher must
# agree on this value (the launcher recomputes `min(128, m_pad/num_ranks)` which equals this),
# so tiles are single-owner without any cross-language handshake.
#   - per_rank <= 128 : BLOCK_M = round_up_16(per_rank)  -> exactly ONE tile/owner, no waste
#   - per_rank  > 128 : BLOCK_M = 128 (the 4-stage TMEM cap: 4*BLOCK_M <= 512)
# Each owner segment is then padded to a multiple of BLOCK_M so a full-BLOCK_M TMA store box
# never straddles two owners; the epilogue batches the whole tile into 2 TMA stores.
RS_STORE_BLOCK_M_MAX = 256

def rs_block_m(m_per_rank: int) -> int:
    bm = ((m_per_rank + 15) // 16) * 16          # round up to a multiple of 16
    return min(bm, RS_STORE_BLOCK_M_MAX)


class ReduceScatterBuffer:
    """Symmetric buffer for the fused BF16 GEMM + ReduceScatter kernel (swap-AB).

    Layout (single symmetric allocation, shared across ranks over NVLink):
        [ barrier region (32 B) ]
        [ FP32 output : m_per_rank_pad x n ]
        [ BF16 scratch: num_ranks x m_per_rank_pad x n ]   (num_ranks slots)

    Semantics: each rank writes its BF16 partial contribution into the owner rank's
    scratch SLOT `rank` with a plain remote store (no atomic); the owner then sums its
    `num_ranks` slots in FP32 into the output region. Owner-side FP32 accumulation gives
    better accuracy than a BF16 NCCL reduce-scatter.

    swap-AB padding: token(M) is the UMMA N-dim, tiled by BLOCK_M (=`rs_block_m`). Each
    owner's per-rank segment is padded up to a multiple of BLOCK_M (`m_per_rank_pad`) so a
    full-BLOCK_M TMA store box lands wholly inside one owner's slot (single-owner tiles).
    The epilogue then batches the whole tile into 2 TMA stores instead of 16. Padded rows
    carry zeros (their `a` rows are zero-padded) and are sliced off by `.output`.
    """

    # Must match `layout::Workspace::kNumBarrierSignalBytes`
    BARRIER_BYTES = 32

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        assert m % group.size() == 0, 'M must be divisible by world size'
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.m = m
        self.n = n
        self.m_per_rank = m // self.world_size                        # real rows per rank
        # Pad each owner segment up to a multiple of BLOCK_M (single-owner tiles)
        self.block_m = rs_block_m(self.m_per_rank)
        self.m_per_rank_pad = ((self.m_per_rank + self.block_m - 1) // self.block_m) * self.block_m
        self.m_pad = self.m_per_rank_pad * self.world_size

        shard_elems = self.m_per_rank_pad * n                         # padded slot size
        num_out_bytes = shard_elems * torch.float32.itemsize
        num_scratch_bytes = self.world_size * shard_elems * torch.bfloat16.itemsize
        num_bytes = self.BARRIER_BYTES + num_out_bytes + num_scratch_bytes
        self._out_bytes = num_out_bytes

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def buffer_ptrs(self):
        # Per-rank base pointers into the symmetric allocation
        return self.handle.buffer_ptrs

    @property
    def output(self) -> torch.Tensor:
        # FP32 [m_per_rank, n] view of the output region (padded tail rows sliced off)
        data = self.buffer[self.BARRIER_BYTES:self.BARRIER_BYTES + self._out_bytes]
        padded = data.view(torch.float32).view(self.m_per_rank_pad, self.n)
        return padded[:self.m_per_rank]

    def zero_(self):
        # Clear barrier + scratch (output is fully overwritten by the combine pass).
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def bf16_gemm_reduce_scatter(a: torch.Tensor,
                             b: torch.Tensor,
                             rs_buffer: ReduceScatterBuffer,
                             compiled_dims: str = 'nk') -> torch.Tensor:
    """Fused BF16 GEMM (`a @ b.T`) + ReduceScatter over single-node NVLink (single kernel).

    `a`: `[M, K]` bf16, `b`: `[N, K]` bf16 (NT layout). Every rank passes its own partial
    `a`/`b`; the output is reduced (summed) across ranks and scattered along M. Returns
    this rank's `[M // world_size, N]` FP32 shard. BF16 over NVLink + plain stores (no
    remote atomics) + owner-side FP32 combine. swap-AB tiles token(M) as the UMMA N-dim,
    so it stays efficient at small M (decode).
    """
    # Fresh accumulation target every call
    rs_buffer.zero_()
    m, k = a.shape
    n, _ = b.shape
    ws = rs_buffer.world_size
    m_per_rank = rs_buffer.m_per_rank
    m_per_rank_pad = rs_buffer.m_per_rank_pad

    # Repack `a` into the padded owner-segment layout [world_size * m_per_rank_pad, K]:
    # owner `o` occupies rows [o*m_per_rank_pad, o*m_per_rank_pad + m_per_rank); the padded
    # tail rows are zeros (contribute 0 to the GEMM, sliced off by `.output`).
    if m_per_rank_pad != m_per_rank:
        if not a.is_contiguous():
            a = a.contiguous()
        a = a.view(ws, m_per_rank, k)
        a = torch.nn.functional.pad(a, (0, 0, 0, m_per_rank_pad - m_per_rank))
        a = a.reshape(ws * m_per_rank_pad, k)
    m_pad = a.shape[0]

    # Local BF16 [m_pad, n] scratch (kept for signature compat; unused by the fused push)
    local_scratch = torch.empty((m_pad, n), dtype=torch.bfloat16, device=a.device)
    _C.bf16_gemm_reduce_scatter_nt(
        a, b, local_scratch, rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank, compiled_dims)
    # Make peers' pushes visible (the kernel's NVLink barrier already syncs, but
    # a host-side barrier keeps the Python-level contract simple)
    rs_buffer.group.barrier()
    return rs_buffer.output


class ReduceScatterBufferPull:
    """Symmetric buffer for the PULL-based split reduce-scatter (fastest at large M).

    Layout: [ barrier (32 B) ][ BF16 output m_per_rank*n ][ BF16 partial shape_m*n ]

    The GEMM writes this rank's full [m, n] BF16 partial into the `partial` region
    (symmetric), then the pull comm kernel reads all peers' partials for this rank's
    shard rows, sums in FP32, and writes the output region.
    """

    BARRIER_BYTES = 32

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        assert m % group.size() == 0, 'M must be divisible by world size'
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.m = m
        self.n = n
        self.m_per_rank = m // self.world_size

        shard_elems = self.m_per_rank * n
        self._out_bytes = shard_elems * torch.bfloat16.itemsize
        num_partial_bytes = m * n * torch.bfloat16.itemsize
        num_bytes = self.BARRIER_BYTES + self._out_bytes + num_partial_bytes

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def buffer_ptrs(self):
        return self.handle.buffer_ptrs

    @property
    def output(self) -> torch.Tensor:
        data = self.buffer[self.BARRIER_BYTES:self.BARRIER_BYTES + self._out_bytes]
        return data.view(torch.bfloat16).view(self.m_per_rank, self.n)

    @property
    def partial(self) -> torch.Tensor:
        # This rank's full [m, n] BF16 GEMM output region (symmetric)
        start = self.BARRIER_BYTES + self._out_bytes
        data = self.buffer[start:]
        return data.view(torch.bfloat16).view(self.m, self.n)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def bf16_gemm_reduce_scatter_pull(a: torch.Tensor,
                                  b: torch.Tensor,
                                  rs_buffer: ReduceScatterBufferPull,
                                  compiled_dims: str = 'nk') -> torch.Tensor:
    """PULL-based split RS: plain BF16 GEMM into the symmetric partial region, then a
    pull comm kernel (barrier + read-all-peers + fp32 reduce + write shard).

    Returns this rank's `[M//world_size, N]` BF16 shard (fp32 accumulation internally,
    so accuracy exceeds sglang's bf16 NCCL reduce-scatter). No local scratch round-trip.
    Fastest at large M (prefill).
    """
    from .. import bf16_gemm_nt
    m, _ = a.shape
    n, _ = b.shape
    bf16_gemm_nt(a, b, rs_buffer.partial)
    _C.reduce_scatter_comm_pull(rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank, m, n)
    rs_buffer.group.barrier()
    return rs_buffer.output


# Crossover between the two fused reduce-scatter kernels (per attention-TP-group M).
# Below this, the fused push+combine kernel wins; at/above, pull (read-all-peers) wins.
# Measured on SM100 (M504), n=4096 k=2048, 4-rank attn-TP group.
RS_PULL_CROSSOVER_M = 2048


class ReduceScatterBufferAuto:
    """Holds both a fused buffer (small M) and a pull buffer (large M); dispatches by M.

    Use when M varies at runtime (decode vs prefill). Allocates both symmetric buffers
    once (sized for the max M). For a fixed M, prefer the specific kernel directly.
    """

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        self.group = group
        self.m, self.n = m, n
        self._fused = ReduceScatterBuffer(group, m, n)
        self._pull = ReduceScatterBufferPull(group, m, n)

    def destroy(self):
        self._fused.destroy()
        self._pull.destroy()


def bf16_gemm_reduce_scatter_auto(a: torch.Tensor,
                                  b: torch.Tensor,
                                  rs_buffer: ReduceScatterBufferAuto,
                                  compiled_dims: str = 'nk') -> torch.Tensor:
    """Dispatch to the faster RS kernel by M: the fused push+combine kernel for small M
    (decode), pull for large M (prefill). Returns this rank's `[M//world_size, N]` shard
    (fp32 for the fused kernel, bf16 for pull)."""
    m = a.shape[0]
    if m < RS_PULL_CROSSOVER_M:
        return bf16_gemm_reduce_scatter(a, b, rs_buffer._fused, compiled_dims)
    return bf16_gemm_reduce_scatter_pull(a, b, rs_buffer._pull, compiled_dims)


def bf16_gemm_reduce_scatter_split(a: torch.Tensor,
                                   b: torch.Tensor,
                                   rs_buffer: ReduceScatterBuffer,
                                   compiled_dims: str = 'nk') -> torch.Tensor:
    """Split-kernel RS: plain BF16 GEMM -> bf16 scratch, then a standalone lightweight
    comm kernel (push + barrier + local FP32 combine) that runs at full occupancy.

    Uses the same buffer layout as `ReduceScatterBuffer`. Returns this rank's
    `[M//world_size, N]` FP32 shard.
    """
    from .. import bf16_gemm_nt
    rs_buffer.zero_()
    m, _ = a.shape
    n, _ = b.shape
    local_scratch = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    bf16_gemm_nt(a, b, local_scratch)
    _C.reduce_scatter_comm(local_scratch, rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank)
    rs_buffer.group.barrier()
    return rs_buffer.output
class ReduceScatterRingBuffer:
    """Symmetric buffer for the standalone PUSH-RING reduce-scatter comm kernel
    (`_C.reduce_scatter_ring`).

    Layout (single symmetric allocation, shared across ranks over NVLink):
        [ barrier region (32 B) ]
        [ output   : m_per_rank x n  bf16 ]
        [ ring recv: num_ranks x m_per_rank x n  bf16 ]   (one owner slot each)
        [ flags    : num_ranks x int32 ]                  (per-owner ready flag)

    Ring reduce-scatter: each rank pushes its running partial-sum downstream (rank->rank-1),
    adding the upstream partial at each hop; after num_ranks-1 hops the sum lands at the
    owner's output. Reduction folded into transfer (no separate combine pass). Output is
    BF16 (fp32 hadd internally per hop -> actually bf16 hadd; matches NCCL precision).
    """

    BARRIER_BYTES = 32

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        assert m % group.size() == 0, 'M must be divisible by world size'
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.m, self.n = m, n
        self.m_per_rank = m // self.world_size

        shard_elems = self.m_per_rank * n
        self._out_bytes = shard_elems * torch.bfloat16.itemsize
        ring_bytes = self.world_size * shard_elems * torch.bfloat16.itemsize
        flag_bytes = self.world_size * 4                       # int32 per owner
        num_bytes = self.BARRIER_BYTES + self._out_bytes + ring_bytes + flag_bytes

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def buffer_ptrs(self):
        return self.handle.buffer_ptrs

    @property
    def output(self) -> torch.Tensor:
        data = self.buffer[self.BARRIER_BYTES:self.BARRIER_BYTES + self._out_bytes]
        return data.view(torch.bfloat16).view(self.m_per_rank, self.n)

    def zero_(self):
        # Ring + flags must be cleared every call (flags are consumed, ring accumulated).
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def bf16_gemm_reduce_scatter_ring(a: torch.Tensor,
                                  b: torch.Tensor,
                                  rs_buffer: ReduceScatterRingBuffer) -> torch.Tensor:
    """Split-kernel RING reduce-scatter: plain BF16 GEMM -> bf16 scratch, then the standalone
    push-ring comm kernel (bidirectional NVLink, reduction folded into transfer).
    Returns this rank's [M//world_size, N] BF16 shard. Validation vehicle for the ring logic
    before fusing it into the GEMM epilogue."""
    from .. import bf16_gemm_nt
    rs_buffer.zero_()
    m, _ = a.shape
    n, _ = b.shape
    local_scratch = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    bf16_gemm_nt(a, b, local_scratch)
    _C.reduce_scatter_ring(local_scratch, rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank)
    rs_buffer.group.barrier()
    return rs_buffer.output


class ReduceScatterRingFusedBuffer:
    """Symmetric buffer for the FULLY FUSED BF16 GEMM + PUSH-RING reduce-scatter kernel
    (`_C.bf16_gemm_reduce_scatter_ring_nt`).

    Layout (single symmetric allocation, shared across ranks over NVLink):
        [ barrier region (32 B) ]
        [ output   : m_per_rank_pad x n  bf16 ]                     (END writes here)
        [ ring recv: num_ranks x m_per_rank_pad x n  bf16 ]         (upstream running-sum, per owner)
        [ mask     : num_ranks x ceil(seg_tiles/64) x uint64_t ]    (per-tile ready bitmask)

    swap-AB padding mirrors `ReduceScatterBuffer`: token(M) is the UMMA N-dim tiled by
    BLOCK_M=`rs_block_m(m_per_rank)`; each owner segment is padded to a multiple of BLOCK_M so
    every tile is single-owner. `num_m_blocks = m_pad / BLOCK_M`, `num_n_blocks = n / 128`.
    `seg_tiles = num_m_blocks / num_ranks * num_n_blocks` — each tile owns one bit in its
    segment's uint64_t bitmask; upstream (WG2 on rank i+1) sets bits per-tile via
    `red.release.sys.or`, downstream (rank i, w3) polls with `ld.acquire.sys`.
    """

    BARRIER_BYTES = 32
    BLOCK_N = 128
    BLOCK_M = 128     # LOCKED: the fused-ring kernel does whole-[128,128]-tile stores

    def __init__(self, group: 'dist.ProcessGroup', m: int, n: int):
        assert m % group.size() == 0, 'M must be divisible by world size'
        assert n % self.BLOCK_N == 0, 'N must be a multiple of 128'
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.m, self.n = m, n
        self.m_per_rank = m // self.world_size
        self.block_m = self.BLOCK_M
        self.m_per_rank_pad = ((self.m_per_rank + self.block_m - 1) // self.block_m) * self.block_m
        self.m_pad = self.m_per_rank_pad * self.world_size
        self.num_m_blocks = self.m_pad // self.block_m
        self.num_n_blocks = n // self.BLOCK_N

        shard_elems = self.m_per_rank_pad * n
        self._out_bytes = shard_elems * torch.bfloat16.itemsize
        ring_bytes = self.world_size * shard_elems * torch.bfloat16.itemsize
        # Per-TILE bitmask protocol: one bit per tile in the owner segment, uint64_t words.
        # Kernel does `red.release.sys.or` per-tile; downstream polls `ld.acquire.sys`.
        bpo = self.m_per_rank_pad // self.block_m                       # blocks per owner
        seg_tiles = bpo * self.num_n_blocks                             # tiles per owner segment
        words_per_seg = (seg_tiles + 63) // 64
        mask_bytes = self.world_size * words_per_seg * 8                # uint64_t per word
        num_bytes = self.BARRIER_BYTES + self._out_bytes + ring_bytes + mask_bytes

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def buffer_ptrs(self):
        return self.handle.buffer_ptrs

    @property
    def output(self) -> torch.Tensor:
        data = self.buffer[self.BARRIER_BYTES:self.BARRIER_BYTES + self._out_bytes]
        padded = data.view(torch.bfloat16).view(self.m_per_rank_pad, self.n)
        return padded[:self.m_per_rank]

    def zero_(self):
        # Ring + mask must be cleared every call (the kernel also self-resets the mask at entry,
        # but a full zero keeps the ring recv region clean for repeated launches).
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def bf16_gemm_reduce_scatter_ring_fused(a: torch.Tensor,
                                        b: torch.Tensor,
                                        rs_buffer: ReduceScatterRingFusedBuffer,
                                        compiled_dims: str = 'nk') -> torch.Tensor:
    """FULLY FUSED BF16 GEMM (`a @ b.T`) + PUSH-RING reduce-scatter (single kernel).

    The ring reduction is folded into the GEMM epilogue's cross-rank transfer (bidirectional
    NVLink, no local-scratch round-trip, no separate combine pass). Returns this rank's
    `[M//world_size, N]` BF16 shard. swap-AB pads token(M) exactly like `ReduceScatterBuffer`.
    """
    rs_buffer.zero_()
    m, k = a.shape
    n, _ = b.shape
    ws = rs_buffer.world_size
    m_per_rank = rs_buffer.m_per_rank
    m_per_rank_pad = rs_buffer.m_per_rank_pad

    # Repack `a` into padded owner-segment layout [ws * m_per_rank_pad, K] (tail rows zeroed).
    if m_per_rank_pad != m_per_rank:
        if not a.is_contiguous():
            a = a.contiguous()
        a = a.view(ws, m_per_rank, k)
        a = torch.nn.functional.pad(a, (0, 0, 0, m_per_rank_pad - m_per_rank))
        a = a.reshape(ws * m_per_rank_pad, k)

    _C.bf16_gemm_reduce_scatter_ring_nt(
        a, b, rs_buffer.buffer, rs_buffer.buffer_ptrs, rs_buffer.rank, compiled_dims)
    rs_buffer.group.barrier()
    return rs_buffer.output
