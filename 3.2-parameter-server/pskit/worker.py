"""One training worker: pull -> gradient -> push, with a clock for SSP."""
import os
import signal
import time

import numpy as np

from pskit.client import PSClient
from pskit.model import batch_grad

MODES = {"sync": 0, "async": -1}  # "ssp" uses the given staleness


def staleness_of(mode, s=0):
    return MODES[mode] if mode in MODES else int(s)


def run_worker(addrs, wid, n_workers, keys, y, steps, batch=256, staleness=0, seed=0, die_at=None, barrier=None):
    """Train for `steps` iterations on this worker's slice of the data.
    Clock c = iterations finished; the pull at clock c may see weights that
    miss at most `staleness` clocks of the slowest worker.
    die_at: SIGKILL this process at that step (failure drills)."""
    cl = PSClient(addrs, worker=wid)
    mine = np.arange(wid, len(y), n_workers)
    rng = np.random.default_rng(seed + wid)
    losses, t = [], {"pull": 0.0, "compute": 0.0, "push": 0.0}
    if barrier is not None:
        barrier.wait()
    t0 = time.perf_counter()
    for c in range(steps):
        if die_at is not None and c == die_at:
            os.kill(os.getpid(), signal.SIGKILL)
        idx = rng.choice(mine, batch, replace=False)

        def pull(uniq):
            a = time.perf_counter()
            w = cl.pull(uniq, clock=c, staleness=staleness)
            t["pull"] += time.perf_counter() - a
            return w

        a = time.perf_counter()
        uniq, g, loss = batch_grad(keys[idx], y[idx], pull)
        b = time.perf_counter()
        cl.push(uniq, g, clock=c + 1)
        t["push"] += time.perf_counter() - b
        t["compute"] += b - a
        losses.append(loss)
    wall = time.perf_counter() - t0
    t["compute"] -= t["pull"]
    cl.leave()
    out = dict(worker=wid, steps=steps, wall=wall, losses=losses, bytes_sent=cl.bytes_sent,
               bytes_recv=cl.bytes_recv, retries=cl.retries, **{"t_" + k: v for k, v in t.items()})
    cl.close()
    return out
