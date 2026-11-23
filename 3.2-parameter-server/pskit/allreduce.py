"""Ring all-reduce (simulated ranks, exact byte counts) and an alpha-beta cost
model to compare it with the parameter server."""
import numpy as np


def ring_allreduce(arrays):
    """Sum `arrays` (one per rank) with the ring algorithm: reduce-scatter then
    all-gather, n-1 steps each. Returns (results per rank, bytes sent per rank)."""
    n = len(arrays)
    bufs = [np.array_split(a.astype(np.float32).copy(), n) for a in arrays]
    sent = [0] * n
    for step in range(n - 1):  # reduce-scatter: after it, rank r owns the full sum of chunk (r+1) % n
        msgs = [(r, (r - step) % n, bufs[r][(r - step) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            bufs[(r + 1) % n][c] += data
            sent[r] += data.nbytes
    for step in range(n - 1):  # all-gather: pass the finished chunks around
        msgs = [(r, (r + 1 - step) % n, bufs[r][(r + 1 - step) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            bufs[(r + 1) % n][c] = data
            sent[r] += data.nbytes
    return [np.concatenate(b) for b in bufs], sent


def ring_time(params, n, bw, alpha):
    """Seconds for one dense all-reduce of `params` float32 values over n ranks."""
    return 2 * (n - 1) * alpha + 2 * (n - 1) / n * params * 4 / bw


def ps_time(keys_per_worker, n, shards, bw, alpha, dim=1):
    """Seconds for one pull + push when each worker touches keys_per_worker keys.
    Worker link: 24 bytes/key for dim 1 (8 key + 4 value pulled, 8 key + 4 gradient pushed).
    Server link: each shard carries 1/shards of all workers' traffic."""
    per_worker = keys_per_worker * (2 * 8 + 2 * 4 * dim)
    per_shard = n * per_worker / shards
    return 2 * 2 * alpha + max(per_worker, per_shard) / bw


def compare(n=8, shards=8, bw=12.5e9, alpha=20e-6):
    """Per-step sync time on n workers for four workloads (100 Gb/s links, 20 us latency).
    Dense models are stored on the PS as blocks of 1,024 values per key, so key overhead is tiny."""
    rows = []
    for name, params, rows_touched, dim in [("dense MLP, 25 M params", 25_000_000, 25_000_000 // 1024, 1024),
                                            ("dense model, 1 B params", 1_000_000_000, 1_000_000_000 // 1024, 1024),
                                            ("CTR, 100 M keys, 40 k touched", 100_000_000, 40_000, 1),
                                            ("CTR, 10 B keys, 400 k touched", 10_000_000_000, 400_000, 1)]:
        r = ring_time(params, n, bw, alpha)
        p = ps_time(rows_touched, n, shards, bw, alpha, dim)
        rows.append(dict(name=name, params=params, ring_s=r, ps_s=p, ring_bytes=2 * (n - 1) / n * params * 4,
                         ps_bytes=rows_touched * (16 + 8 * dim), winner="ring" if r < p else "PS"))
    return rows
