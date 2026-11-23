"""Streaming loader: decode workers fill bounded prefetch queues; the main process takes batches round-robin.

State for exact resume is three numbers: {"epoch", "step", "fp"}. Each worker recomputes its position from
the integer-only plan, refills its shuffle buffer with random-access reads, and continues streaming.
"""
import bisect
import json
import multiprocessing as mp
import queue
import time
import traceback
import zlib
from multiprocessing import resource_tracker, shared_memory

import numpy as np

from .codec import augment, decode_image, sample_rng
from .plan import EpochPlan, batch_sizes, buffered_shuffle
from .reader import ShardReader
from .storage import make_store


class WorkerError(RuntimeError):
    """A decode worker raised, died, or ended before its planned number of batches."""


def _plan(cfg, epoch):
    return EpochPlan(cfg["shards"], cfg["seed"], epoch, cfg["batch_size"], cfg["num_workers"],
                     cfg["world_size"], cfg["rank"], cfg["shuffle"], cfg["drop_last"])


def _collate(cfg, epoch, items):
    keys, xs, ys = [], [], []
    for (name, idx), payload in items:
        if payload is None:                       # corrupt record skipped by the reader
            continue
        keys.append((name, idx))
        if cfg["decode"] == "image":
            img, label = decode_image(payload)
            xs.append(augment(img, cfg["crop"], sample_rng(cfg["seed"], epoch, name, idx), cfg["dtype"]))
            ys.append(label)
        else:
            xs.append(payload)
    if cfg["decode"] == "image":
        x = np.stack(xs) if xs else np.zeros((0,), cfg["dtype"])
        return {"x": x, "y": np.asarray(ys, dtype=np.int64), "keys": keys}
    return {"x": xs, "keys": keys}


def worker_batches(cfg, epoch, worker, start_batch=0):
    """All batches of one worker for one epoch, beginning at batch number start_batch."""
    plan = _plan(cfg, epoch)
    shards = plan.worker_shards(worker)
    starts = [0]
    for _, c in shards:
        starts.append(starts[-1] + c)
    n = starts[-1]
    sizes = batch_sizes(n, cfg["batch_size"], cfg["drop_last"])
    store, readers = make_store(cfg["store"]), {}

    def reader(k):
        if k not in readers:
            name = shards[k][0]
            readers[k] = ShardReader(store, name, cfg["block_bytes"], cfg["on_corrupt"], cfg["sizes"].get(name))
        return readers[k]

    def locate(p):                              # stream position -> (shard slot, record index)
        k = bisect.bisect_right(starts, p) - 1
        return k, p - starts[k]

    def stream(pos):                            # sequential read from stream position pos onward
        for k in range(len(shards)):
            if pos < starts[k + 1]:
                for i, payload in reader(k).records(max(0, pos - starts[k])):
                    yield starts[k] + i, payload

    order = buffered_shuffle(n, cfg["shuffle_buffer"] if cfg["shuffle"] else 1, plan.rng(worker))
    emitted, n_read = set(), 0
    for _ in range(sum(sizes[:start_batch])):   # fast-forward on integers only
        p, n_read = next(order)
        emitted.add(p)
    pending = {}
    for p in range(n_read):                     # refill the shuffle buffer by random access
        if p not in emitted:
            k, i = locate(p)
            pending[p] = reader(k).get(i)
    src = stream(n_read)
    for b in range(start_batch, len(sizes)):
        items = []
        for _ in range(sizes[b]):
            p, _ = next(order)
            while p not in pending:
                q, payload = next(src)
                pending[q] = payload
            k, i = locate(p)
            items.append(((shards[k][0], i), pending.pop(p)))
        yield _collate(cfg, epoch, items)


def _to_shm(batch):
    """Put the big array in shared memory; only its name, shape and dtype go through the pipe."""
    x = batch["x"]
    shm = shared_memory.SharedMemory(create=True, size=max(1, x.nbytes))
    resource_tracker.unregister(shm._name, "shared_memory")   # the consumer owns and unlinks it
    np.ndarray(x.shape, x.dtype, buffer=shm.buf)[...] = x
    meta = (shm.name, x.shape, x.dtype.str)
    shm.close()
    return ("shm", {**batch, "x": meta})


def _from_shm(batch):
    name, shape, dtype = batch["x"]
    shm = shared_memory.SharedMemory(name=name)
    try:
        x = np.ndarray(shape, np.dtype(dtype), buffer=shm.buf).copy()
    finally:
        shm.close()
        shm.unlink()
    return {**batch, "x": x}


def _worker_main(cfg, epoch, worker, start_batch, q, transport):
    try:
        for batch in worker_batches(cfg, epoch, worker, start_batch):
            if transport == "shm" and isinstance(batch["x"], np.ndarray):
                q.put(_to_shm(batch))
            else:
                q.put(("batch", batch))
        q.put(("done", None))
    except BaseException:
        q.put(("error", f"worker {worker}:\n{traceback.format_exc()}"))


class StreamingLoader:
    def __init__(self, store, manifest, batch_size=32, num_workers=0, prefetch=2, shuffle_buffer=256, seed=0,
                 rank=0, world_size=1, shuffle=True, drop_last=False, decode="image", crop=24, dtype="uint8",
                 block_bytes=1 << 20, on_corrupt="raise", timeout=120.0, processes=True, transport="shm",
                 mp_context="forkserver"):
        spec = store if isinstance(store, dict) else store.spec()
        self.cfg = dict(store=spec, shards=[(s["name"], s["records"]) for s in manifest["shards"]],
                        sizes={s["name"]: s["bytes"] for s in manifest["shards"]},
                        batch_size=batch_size, num_workers=max(1, num_workers), shuffle_buffer=shuffle_buffer,
                        seed=seed, rank=rank, world_size=world_size, shuffle=shuffle, drop_last=drop_last,
                        decode=decode, crop=crop, dtype=dtype, block_bytes=block_bytes, on_corrupt=on_corrupt)
        self.use_procs = processes and num_workers > 0
        self.prefetch, self.timeout, self.mp_context, self.transport = prefetch, timeout, mp_context, transport
        self.epoch, self.step = 0, 0
        self.wait_s = 0.0

    # ---- deterministic plan and resume state
    def plan(self):
        return _plan(self.cfg, self.epoch)

    def __len__(self):
        return self.plan().steps()

    def set_epoch(self, epoch):
        self.epoch, self.step = epoch, 0

    def fingerprint(self):
        keys = ("shards", "batch_size", "num_workers", "shuffle_buffer", "seed", "rank", "world_size",
                "shuffle", "drop_last", "decode", "crop", "dtype")
        blob = json.dumps({k: self.cfg[k] for k in keys}, sort_keys=True).encode()
        return f"{zlib.crc32(blob):08x}"

    def state_dict(self):
        return {"epoch": self.epoch, "step": self.step, "fp": self.fingerprint()}

    def load_state_dict(self, state):
        if state["fp"] != self.fingerprint():
            raise ValueError("loader state was saved with a different dataset or loader configuration")
        self.epoch, self.step = state["epoch"], state["step"]

    # ---- iteration
    def __iter__(self):
        plan = self.plan()
        schedule, starts = plan.schedule(), plan.start_batches(self.step)
        todo = schedule[self.step:]
        source = self._processes(starts) if self.use_procs else self._inline(starts)
        try:
            for w in todo:
                batch = source(w)
                self.step += 1
                yield batch
        finally:
            self.close()
        self.epoch, self.step = self.epoch + 1, 0     # a finished epoch rolls the state forward

    def _inline(self, starts):
        gens = [worker_batches(self.cfg, self.epoch, w, starts[w]) for w in range(self.cfg["num_workers"])]
        return lambda w: next(gens[w])

    def _processes(self, starts):
        ctx = mp.get_context(self.mp_context)
        if self.mp_context == "forkserver":
            ctx.set_forkserver_preload(["streamload.loader"])
        n = self.cfg["num_workers"]
        qs = [ctx.Queue(maxsize=self.prefetch) for _ in range(n)]
        ps = [ctx.Process(target=_worker_main, args=(self.cfg, self.epoch, w, starts[w], qs[w], self.transport),
                          daemon=True)
              for w in range(n)]
        for p in ps:
            p.start()
        self._live = (ps, qs)

        def take(w):
            t0 = time.perf_counter()
            while True:
                try:
                    kind, val = qs[w].get(timeout=0.2)
                    break
                except queue.Empty:
                    if not ps[w].is_alive() and qs[w].empty():
                        raise WorkerError(f"worker {w} died (exit code {ps[w].exitcode})")
                    if time.perf_counter() - t0 > self.timeout:
                        raise TimeoutError(f"worker {w} produced nothing for {self.timeout}s")
            self.wait_s += time.perf_counter() - t0
            if kind == "error":
                raise WorkerError(val)
            if kind == "done":
                raise WorkerError(f"worker {w} ended before its planned batches")
            return _from_shm(val) if kind == "shm" else val
        return take

    def close(self):
        ps, qs = getattr(self, "_live", ((), ()))
        for p in ps:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        for q in qs:                    # free shared memory of batches nobody will consume
            try:
                while True:
                    kind, val = q.get_nowait()
                    if kind == "shm":
                        _from_shm(val)
            except Exception:          # empty, or a message cut short by terminate()
                pass
            q.cancel_join_thread()
            q.close()
        self._live = ((), ())

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
