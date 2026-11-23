"""Measurements: training stall (sync vs async), write throughput, recovery time."""
import shutil
import tempfile
import time

import numpy as np

from .checkpointer import Checkpointer
from .loader import load, load_latest
from .state import make_state, nbytes, shard_state
from .storage import LocalFS, ObjectStore
from .train import Trainer

MB = 1e6


def stall(storage_factory, n_params, world, saves=4):
    """Median stall per checkpoint and write throughput, sync vs async, for one backend."""
    out = {}
    state = make_state(n_params)
    for mode in ("sync", "async"):
        store = storage_factory()
        ck = Checkpointer(store, world, keep_last=2)
        recs = []
        for i in range(saves):
            r = ck.save(i + 1, shard_state(state, world), blocking=(mode == "sync"))
            ck.wait()                                   # let it finish so saves do not queue
            recs.append(r)
        out[mode] = {"stall_s": float(np.median([r["stall_s"] for r in recs])),
                     "snapshot_s": float(np.median([r["snapshot_s"] for r in recs])),
                     "write_s": float(np.median([r["write_s"] for r in recs])),
                     "mb": recs[0]["bytes"] / MB}
        out[mode]["write_mb_s"] = out[mode]["mb"] / out[mode]["write_s"]
    return out


def training(storage_factory, n_params, world, steps, every, repeats=3):
    """Wall-clock time of a training run: no checkpoints, sync, async (median of `repeats`, interleaved)."""
    runs = {m: [] for m in ("none", "sync", "async")}
    for _ in range(repeats):
        for mode in runs:
            tr, store = Trainer(n_params=n_params, layers=4, batch=128), storage_factory()
            ck = Checkpointer(store, world, keep_last=2)
            stalls, t0 = [], time.perf_counter()
            for _ in range(steps):
                tr.train_step()
                if mode != "none" and tr.step % every == 0:
                    stalls.append(ck.save(tr.step, shard_state(tr.state, world), blocking=(mode == "sync"))["stall_s"])
            ck.wait()
            runs[mode].append((time.perf_counter() - t0, sum(stalls), len(stalls)))
    res = {}
    for mode, rs in runs.items():
        total, stall, n = sorted(rs)[len(rs) // 2]
        res[mode] = {"total_s": total, "stall_s": stall, "saves": n}
    return res


def recovery(store, worlds, verify=True):
    """Time to find the newest checkpoint and load it onto each world size."""
    out = {}
    for w in worlds:
        t = time.perf_counter()
        load_latest(store, w, verify=verify)
        out[w] = time.perf_counter() - t
    return out


def run(n_params=4_000_000, world=4, quick=False, root=None):
    root = root or tempfile.mkdtemp(prefix="ackpt-bench-")
    lines = []
    p = lines.append
    st = make_state(n_params)
    p(f"state: {n_params:,} params x (weight + Adam m + v) = {nbytes(st) / MB:.1f} MB, {world} ranks")
    backends = {
        "local disk (fsync)": lambda: LocalFS(tempfile.mkdtemp(dir=root)),
        "object store sim (200 MB/s, 20 ms)": lambda: ObjectStore(latency_s=0.02, bandwidth_bps=200 * MB),
    }
    p("")
    p(f"{'backend':38s} {'mode':6s} {'stall ms':>9s} {'snapshot ms':>12s} {'write ms':>9s} {'write MB/s':>11s}")
    for name, fac in backends.items():
        r = stall(fac, n_params, world, saves=3 if quick else 5)
        for mode in ("sync", "async"):
            x = r[mode]
            p(f"{name:38s} {mode:6s} {x['stall_s'] * 1e3:9.1f} {x['snapshot_s'] * 1e3:12.1f} {x['write_s'] * 1e3:9.1f} {x['write_mb_s']:11.0f}")
    # copy bandwidth = the speed limit of phase 1
    a = np.ones(int(nbytes(st) / 4), np.float32); b = np.empty_like(a)
    t = time.perf_counter(); np.copyto(b, a); dt = time.perf_counter() - t
    p(f"\nsnapshot copy bandwidth (np.copyto, {a.nbytes / MB:.0f} MB): {a.nbytes / dt / 1e9:.2f} GB/s")

    p("\ntraining run: 40 steps, checkpoint every 10, object store sim (200 MB/s), median of 3")
    tr = training(backends["object store sim (200 MB/s, 20 ms)"], n_params, world, 20 if quick else 40, 10, 1 if quick else 3)
    for mode in ("none", "sync", "async"):
        x = tr[mode]
        extra = x["total_s"] - tr["none"]["total_s"]
        p(f"  {mode:6s} total {x['total_s']:6.2f} s   stalled {x['stall_s']:6.2f} s over {x['saves']} saves   "
          f"extra vs none {extra:+5.2f} s")

    p("\nrecovery (find newest + read + verify CRC + reshard), saved with 4 ranks")
    fs_store = LocalFS(tempfile.mkdtemp(dir=root))
    os_store = ObjectStore(latency_s=0.02, bandwidth_bps=200 * MB)
    for s in (fs_store, os_store):
        ck = Checkpointer(s, world)
        ck.save(1, shard_state(st, world), blocking=True)
    for name, s in (("local disk", fs_store), ("object store sim", os_store)):
        r = recovery(s, (4, 2, 8))
        p(f"  {name:18s} " + "   ".join(f"-> {w} ranks {t * 1e3:7.1f} ms" for w, t in r.items()))
    stats = {}
    load(os_store, 1, 2, verify=False, stats=stats)
    p(f"  ranged reads without CRC check, 4 -> 2 ranks: {stats['bytes_read'] / MB:.1f} MB read")
    shutil.rmtree(root, ignore_errors=True)
    return "\n".join(lines)
