"""One training process (one 'GPU'). Started by an agent with its rank and generation in the environment.

Exit codes: 0 training finished · 3 fenced (generation no longer active) · 4 coordinator unreachable.
Anything else (a crash, kill -9 = -9) is a failure that the agent reports.
"""
import os
import sys
import time

import numpy as np

from . import model as M
from .checkpoint import CheckpointStore, StaleWriter
from .client import Client, Stale, pack, unpack
from .events import EventLog

EXIT_FENCED, EXIT_UNREACHABLE = 3, 4


def env(name, default=None, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


def retry(fn, *a, patience=30.0, **kw):
    """Retry through short network trouble. Stale (fenced) is never retried."""
    end = time.monotonic() + patience
    while True:
        try:
            return fn(*a, **kw)
        except ConnectionError:
            if time.monotonic() > end:
                raise
            time.sleep(0.05)


def slow_delay(faults, node):
    if not faults:
        return 0.0
    try:
        with open(os.path.join(faults, f"slow-{node}")) as f:
            return float(f.read().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0.0


def train():
    node, rank, world = env("ELASTIC_NODE"), env("RANK", cast=int), env("WORLD_SIZE", cast=int)
    gen, total = env("ELASTIC_GEN", cast=int), env("ELASTIC_STEPS", 300, int)
    every, nap = env("ELASTIC_CKPT_EVERY", 20, int), env("ELASTIC_STEP_SLEEP", 0.01, float)
    seed, faults = env("ELASTIC_SEED", 0, int), env("ELASTIC_FAULTS")
    log = EventLog(env("ELASTIC_EVENTS"), f"worker{rank}@{node}")
    c = Client(env("ELASTIC_ENDPOINT"), node)
    store = CheckpointStore(env("ELASTIC_CKPT_DIR"))
    log.emit("worker_start", pid=os.getpid(), rank=rank, world=world, gen=gen)

    # 1. agree on where to resume: rank 0 claims the fence, reads LATEST and publishes the step
    like = M.init_params(seed)
    theta, opt = M.flatten(like), M.SGDMomentum(M.flatten(like).size)
    if rank == 0:
        store.claim(gen)                              # zombies of older generations are now refused
        ck = store.load()
        start = ck[0] if ck else 0
        retry(c.kv_set, gen, {"resume": str(start)})
    else:
        start = int(retry(c.kv_get, gen, ["resume"])["resume"])
        ck = store.load(start) if start else None
    if ck:
        theta, opt.v = ck[1]["theta"], ck[1]["momentum"]
    if rank == 0:
        log.emit("resume", step=start, gen=gen, world=world)

    # 2. train: each rank computes its chunks of the global batch, all ranks add all chunks in order
    x, y = M.make_data(seed)
    mine = M.my_chunks(rank, world)
    for step in range(start, total):
        t0 = time.perf_counter()
        time.sleep(nap + slow_delay(faults, node))    # stands in for forward/backward on an accelerator
        params = M.unflatten(theta, like)
        idx = M.batch_indices(step, seed)
        items = {}
        for ch in mine:
            sel = idx[ch * M.CHUNK:(ch + 1) * M.CHUNK]
            g, loss = M.chunk_grad(params, x[sel], y[sel])
            items[f"s{step}/c{ch}"] = pack(np.append(g, loss))
        old = [f"s{step - 2}/c{ch}" for ch in mine] if step >= 2 else []
        retry(c.kv_set, gen, items, rank=rank, step=step, delete=old)
        vals = retry(c.kv_get, gen, [f"s{step}/c{ch}" for ch in range(M.N_CHUNKS)])
        chunks = {}
        for ch in range(M.N_CHUNKS):
            v = unpack(vals[f"s{step}/c{ch}"])
            chunks[ch] = (v[:-1], v[-1])
        grad, loss = M.reduce_chunks(chunks)
        theta = opt.step(theta, grad)
        if rank == 0:
            log.emit("step", step=step, gen=gen, world=world, loss=round(loss, 5), ms=round((time.perf_counter() - t0) * 1000, 2))
            if (step + 1) % every == 0 or step + 1 == total:
                name, secs, size = store.save(step + 1, {"theta": theta, "momentum": opt.v}, {"gen": gen, "world": world}, token=gen)
                log.emit("checkpoint", step=step + 1, gen=gen, ms=round(secs * 1000, 2), bytes=size)
    if rank == 0:
        log.emit("train_done", gen=gen, digest=M.digest(theta), acc=round(M.accuracy(theta, like, x, y), 4))
    return 0


def main():
    try:
        return train()
    except (Stale, StaleWriter) as e:
        EventLog(env("ELASTIC_EVENTS"), f"worker{env('RANK')}@{env('ELASTIC_NODE')}").emit("fenced", gen=env("ELASTIC_GEN", cast=int), why=str(e)[:120])
        return EXIT_FENCED
    except ConnectionError:
        return EXIT_UNREACHABLE


if __name__ == "__main__":
    sys.exit(main())
