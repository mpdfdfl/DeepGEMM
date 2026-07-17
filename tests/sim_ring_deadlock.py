"""Faithful deadlock simulator for the FUSED push-ring GEMM+RS kernel.

The fused kernel has NO grid_sync between ring steps (unlike the standalone comm kernel).
Instead every SM runs the GEMM persistent tile loop; WG2 (ring warp) processes each tile
IN ORDER, and for a MIDDLE/END tile it WAITS on the upstream rank's flag for that SAME
physical tile. Deadlock-freedom is therefore NOT automatic; it holds iff the combined
dependency graph is acyclic:

  edge A (per-SM in-order): WG2(rank i, queue pos p) depends on WG2(rank i, pos p-1)
  edge B (cross-rank ring): WG2(rank i, tile X) depends on WG2(up=i+1, tile X) if X is MIDDLE/END

A cycle in "depends-on" == a set of WG2 consumers each blocked on the next == hang.

We replicate `get_swizzled_block_idx` (kIsMulticastOnA=false / grouping on M, kNumMulticast=1,
arch>=1000 so the SM90 multicast fix is skipped) and `get_next_block` EXACTLY, then apply the
stagger owner' = (owner_raw + rank + 1) % R and detect cycles.
"""
import argparse
import sys


def num_1d_blocks_per_group(block_m, block_n, num_sms):
    # kIsMulticastOnA == false -> grouping on M
    best, min_usage = 0, 1 << 62
    for cand in (8, 16):
        usage = cand * block_m + ((num_sms + cand - 1) // cand) * block_n
        if usage < min_usage:
            min_usage, best = usage, cand
    return best


def swizzled_block_idx(block_idx, num_m_blocks, num_n_blocks, k1d):
    # kIsMulticastOnA == false: primary=M, secondary=N
    primary = num_m_blocks
    secondary = num_n_blocks
    num_blocks_per_group = secondary * k1d
    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * k1d
    in_group_idx = block_idx % num_blocks_per_group
    num_blocks_in_group = min(k1d, primary - first_block_idx)
    # arch>=1000: no multicast fix
    m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
    n_block_idx = in_group_idx // num_blocks_in_group
    return m_block_idx, n_block_idx


def build_queues(rank, R, num_sms, num_m_blocks, num_n_blocks, bpo, k1d, stagger):
    """Return per-SM ordered list of physical tiles (c, off, nb) for this rank."""
    num_blocks = num_m_blocks * num_n_blocks
    queues = [[] for _ in range(num_sms)]
    for s in range(num_sms):
        it = 0
        while True:
            gbi = it * num_sms + s
            if gbi >= num_blocks:
                break
            it += 1
            raw_m, nb = swizzled_block_idx(gbi, num_m_blocks, num_n_blocks, k1d)
            owner_raw = raw_m // bpo
            off = raw_m % bpo
            if stagger:
                c = (owner_raw + rank + 1) % R
            else:
                c = owner_raw
            queues[s].append((c, off, nb))
    return queues


def build_queues_segordered(rank, R, num_sms, bpo, num_n_blocks, stagger):
    """SEGMENT-ORDERED enumeration: process owner segments in staggered order c=(rank+1+t)%R,
    t=0..R-1; within a segment distribute its (off,nb) tiles grid-strided across SMs; a whole
    segment is enumerated before the next. This is the order needed for a per-segment flag to be
    deadlock-free (all of a segment completes as one contiguous phase, like the standalone RS)."""
    queues = [[] for _ in range(num_sms)]
    seg_tiles = bpo * num_n_blocks
    for t in range(R):
        c = (rank + 1 + t) % R if stagger else t
        # enumerate this segment's tiles (off, nb) and hand them to SMs grid-strided
        idx = 0
        for off in range(bpo):
            for nb in range(num_n_blocks):
                sm = idx % num_sms
                queues[sm].append((c, off, nb))
                idx += 1
    return queues


def ring_state(rank, c, R):
    d = (rank - c + R) % R
    if d == R - 1:
        return "START"   # no upstream wait
    if d == 0:
        return "END"     # i==c, waits upstream, writes output
    return "MIDDLE"      # waits upstream, forwards


def detect_deadlock(R, num_sms, M, N, block_n=128, stagger=True, verbose=False,
                    seg_flag=False, seg_order=False, drain_k=1, no_boundary_flush=False):
    m_per_rank = M // R
    block_m = min(128, m_per_rank) if (seg_flag or seg_order) else min(256, m_per_rank)  # fused ring locks 128
    assert m_per_rank % block_m == 0, (m_per_rank, block_m)
    bpo = m_per_rank // block_m
    num_m_blocks = R * bpo
    num_n_blocks = (N + block_n - 1) // block_n
    k1d = num_1d_blocks_per_group(block_m, block_n, num_sms)

    # Build queues for every rank.
    if seg_order:
        all_q = [build_queues_segordered(i, R, num_sms, bpo, num_n_blocks, stagger)
                 for i in range(R)]
    else:
        all_q = [build_queues(i, R, num_sms, num_m_blocks, num_n_blocks, bpo, k1d, stagger)
                 for i in range(R)]

    # Node id = (rank, c, off, nb). Map each node -> (sm, pos) so we can add the in-order edge.
    # Build adjacency of "depends-on" edges.
    # For cycle detection we use iterative DFS with colors.
    # Represent node as tuple.
    pos_of = {}          # (rank, tile) -> (sm, pos)
    for i in range(R):
        for s in range(num_sms):
            for p, tile in enumerate(all_q[i][s]):
                pos_of[(i, tile)] = (s, p)

    # For segment-flag mode: per (rank, owner c) the LAST tile that rank processes for owner c
    # (max global "position" across its SMs). Downstream's owner-c tiles depend on upstream
    # having stored ALL of owner-c => depend on that upstream last-owner-c tile.
    # Global processing order key = pos (iteration index within SM); use (pos, sm) as a proxy for
    # "when" a tile completes. We take the upstream tile with the max pos for owner c.
    seg_last = {}   # (rank, c) -> tile with max pos among that rank's owner-c tiles
    if seg_flag:
        for i in range(R):
            for s in range(num_sms):
                for p, tile in enumerate(all_q[i][s]):
                    c = tile[0]
                    key = (i, c)
                    if key not in seg_last or p > seg_last[key][0]:
                        seg_last[key] = (p, tile)

    # For drain-ring mode (drain_k > 1): the OR for tile X fires when the SAME upstream SM has
    # issued K-1 further stores within X's segment (drain-when-full), or at the owner-boundary
    # flush after the last tile of X's segment in that SM's queue, whichever comes first.
    # Precompute, per (rank, sm, owner), the last queue position of that owner's tiles.
    seg_last_pos = {}   # (rank, sm, owner) -> last pos in that SM's queue
    if drain_k > 1:
        for i in range(R):
            for s in range(num_sms):
                for p, tile in enumerate(all_q[i][s]):
                    seg_last_pos[(i, s, tile[0])] = p

    def deps(node):
        i, tile = node
        c, off, nb = tile
        out = []
        s, p = pos_of[(i, tile)]
        # edge A: in-order within SM (depends on previous tile in same SM queue)
        if p > 0:
            prev = all_q[i][s][p - 1]
            out.append((i, prev))
        # edge B: cross-rank upstream dependency (MIDDLE/END wait upstream)
        st = ring_state(i, c, R)
        if st in ("MIDDLE", "END"):
            up = (i + 1) % R
            if seg_flag:
                # SEGMENT flag: this tile can't start until upstream finished ALL of owner c,
                # i.e. depends on upstream's LAST owner-c tile (which itself depends, via edge A,
                # on every earlier upstream owner-c tile).
                out.append((up, seg_last[(up, c)][1]))
            elif drain_k > 1:
                # DRAIN-RING: OR for tile X fires after upstream's SM processes the tile at
                # min(pos_X + K-1, DRAIN_HORIZON). With the owner-boundary flush (default),
                # DRAIN_HORIZON = last pos of X's segment in that SM's queue. With
                # --no-boundary-flush, DRAIN_HORIZON = end of the whole queue (ORs for the
                # last K-1 tiles of a segment only fire in the trailing drain after the main
                # loop — the scheme that deadlocked on hardware at M=1024/2048/4096, R=4).
                us, up_pos = pos_of[(up, tile)]
                horizon = len(all_q[up][us]) - 1 if no_boundary_flush else seg_last_pos[(up, us, c)]
                drain_pos = min(up_pos + drain_k - 1, horizon)
                out.append((up, all_q[up][us][drain_pos]))
            else:
                out.append((up, tile))   # per-tile flag
        return out

    # Iterative DFS cycle detection over all nodes.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    nodes = list(pos_of.keys())
    for n in nodes:
        color[n] = WHITE

    cycle = None
    for start in nodes:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(deps(start)))]
        color[start] = GRAY
        path = [start]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nb_node in it:
                if nb_node not in color:
                    # tile not present in upstream's queue -> would wait forever!
                    cycle = ("MISSING_UPSTREAM", node, nb_node)
                    return False_result(cycle, block_m, bpo, num_m_blocks, num_n_blocks, k1d)
                if color[nb_node] == WHITE:
                    color[nb_node] = GRAY
                    stack.append((nb_node, iter(deps(nb_node))))
                    path.append(nb_node)
                    advanced = True
                    break
                elif color[nb_node] == GRAY:
                    # found a back-edge -> cycle
                    idx = path.index(nb_node)
                    cycle = ("CYCLE", path[idx:] + [nb_node])
                    return False_result(cycle, block_m, bpo, num_m_blocks, num_n_blocks, k1d)
            if not advanced:
                color[node] = BLACK
                stack.pop()
                path.pop()
    return {"deadlock": False, "block_m": block_m, "bpo": bpo,
            "num_m_blocks": num_m_blocks, "num_n_blocks": num_n_blocks, "k1d": k1d,
            "nodes": len(nodes)}


def False_result(cycle, block_m, bpo, nmb, nnb, k1d):
    return {"deadlock": True, "reason": cycle[0], "detail": cycle[1:],
            "block_m": block_m, "bpo": bpo, "num_m_blocks": nmb,
            "num_n_blocks": nnb, "k1d": k1d}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--sms", type=int, default=148)
    p.add_argument("--tokens", type=str, default="512,1024,2048,4096,8192")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--seg-flag", action="store_true", help="model per-owner-segment flag deps (BLOCK_M=128)")
    p.add_argument("--seg-order", action="store_true", help="segment-ordered processing (contiguous per owner)")
    p.add_argument("--drain-k", type=int, default=1,
                   help="drain-ring depth K: OR for tile X fires K-1 tiles later in-segment, "
                        "or at the owner-boundary flush (models the WG2 drain-ring; K=1 = fire in-iteration)")
    p.add_argument("--no-boundary-flush", action="store_true",
                   help="naive drain-ring: ORs for a segment's last K-1 tiles only fire in the "
                        "trailing drain after the main loop (expected to deadlock — sanity check)")
    args = p.parse_args()

    any_dead = False
    for stagger in (True, False):
        print(f"\n===== stagger={stagger} seg_flag={args.seg_flag} seg_order={args.seg_order} drain_k={args.drain_k} no_boundary_flush={args.no_boundary_flush} | R={args.ranks} SMs={args.sms} n={args.n} =====")
        for m in (int(x) for x in args.tokens.split(",")):
            if (m // args.ranks) == 0:
                continue
            r = detect_deadlock(args.ranks, args.sms, m, args.n, stagger=stagger,
                                seg_flag=args.seg_flag, seg_order=args.seg_order,
                                drain_k=args.drain_k, no_boundary_flush=args.no_boundary_flush)
            tag = "DEADLOCK" if r["deadlock"] else "ok"
            extra = ""
            if r["deadlock"]:
                any_dead = True
                extra = f"  reason={r['reason']} detail={str(r['detail'])[:200]}"
            print(f"  M={m:>6} block_m={r['block_m']:>3} bpo={r['bpo']} "
                  f"m_blocks={r['num_m_blocks']} n_blocks={r['num_n_blocks']} "
                  f"k1d={r['k1d']} nodes={r.get('nodes','?'):>6}  -> {tag}{extra}")
    sys.exit(1 if any_dead else 0)
