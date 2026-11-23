"""Measurements: all-reduce bytes and time, training scaling, and communication overlap."""
import time

import numpy as np

from .allreduce import ring_allreduce, ring_bytes_per_rank
from .launch import launch
from .trainer import Config, train_worker
from .transport import RingLink


def _allreduce_worker(rank, addrs, sock, sizes, reps):
    link = RingLink(rank, len(addrs), addrs, 30.0, sock)
    world, out = len(addrs), []
    try:
        for n in sizes:
            buf, times = np.empty(n), []
            for rep in range(reps + 1):                       # rep 0 is a warm-up
                buf[:] = rank + 1.0
                sent0 = link.bytes_sent
                t0 = time.perf_counter()
                ring_allreduce(buf, link, tag=rep * 2 * world)
                if rep:
                    times.append(time.perf_counter() - t0)
            if not np.all(buf == world * (world + 1) / 2):       # 1 + 2 + ... + N on every element
                raise AssertionError("wrong all-reduce result")
            out.append(dict(n=n, seconds=float(np.median(times)), sent=link.bytes_sent - sent0))
        return out
    finally:
        link.close()


def allreduce_bench(world, sizes, reps=5):
    """Per message size: slowest rank's median time, bytes sent by one rank in one call, theory."""
    per_rank = launch(_allreduce_worker, world, (sizes, reps))
    rows = []
    for i, n in enumerate(sizes):
        s = n * 8
        t = max(r[i]["seconds"] for r in per_rank)
        rows.append(dict(world=world, bytes=s, seconds=t, sent=per_rank[0][i]["sent"],
                         theory=ring_bytes_per_rank(s, world), busbw=ring_bytes_per_rank(s, world) / t))
    return rows


def scaling_bench(worlds=(1, 2, 4), **kw):
    """Same global batch and steps for every worker count (strong scaling)."""
    rows = []
    for w in worlds:
        cfg = Config(world=w, **kw)
        res = launch(train_worker, w, (cfg,))
        wall = max(r["total_s"] for r in res)
        rows.append(dict(world=w, wall=wall, samples_s=res[0]["steps"] * cfg.global_batch / wall,
                         compute=np.mean([r["compute_s"] for r in res]), wait=np.mean([r["wait_s"] for r in res]),
                         sent_mb=res[0]["bytes_sent"] / 1e6, loss=res[0]["losses"][-1]))
    return rows


def overlap_bench(world=2, settings=((0, False), (0, True), (64, True), (256, True), (1024, True)), **kw):
    """(bucket KiB, overlap) pairs; 0 KiB = one bucket holding every gradient."""
    rows = []
    for kb, ov in settings:
        cfg = Config(world=world, bucket_kb=kb, overlap=ov, **kw)
        res = launch(train_worker, world, (cfg,))
        steps = res[0]["steps"]
        rows.append(dict(bucket_kb=kb, overlap=ov, buckets=res[0]["n_buckets"],
                         step_ms=1e3 * max(r["total_s"] for r in res) / steps,
                         compute_ms=1e3 * np.mean([r["compute_s"] for r in res]) / steps,
                         wait_ms=1e3 * np.mean([r["wait_s"] for r in res]) / steps,
                         comm_ms=1e3 * np.mean([r["comm_s"] for r in res]) / steps))
    return rows
