"""Two-phase checkpoint save.

Phase 1 (training waits): copy every rank's shards into staging buffers (on a GPU: device -> pinned host memory).
Phase 2 (background):     each rank writes its file in parallel; then the coordinator writes the manifest (commit);
                          then old checkpoints are garbage-collected.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .manifest import MANIFEST, build_manifest, encode_rank, step_dir
from .retention import gc
from .state import Shard


class Checkpointer:
    def __init__(self, storage, world_size, keep_last=3, fault=None):
        self.storage, self.world, self.keep_last = storage, world_size, keep_last
        self.fault = fault              # tests: fault(point, step=..., rank=...) may crash or raise
        self._thread, self._error = None, None
        self._staging = None            # reused copy buffers, like pinned host memory
        self.history = []               # one record per save

    # ---------------------------------------------------------------- phase 1
    def _snapshot(self, rank_states):
        same = self._staging is not None and all(
            set(a) == set(b) and all(a[k].data.shape == b[k].data.shape and a[k].data.dtype == b[k].data.dtype for k in a)
            for a, b in zip(self._staging, rank_states))
        if not same:                    # first save (or shapes changed): allocate once
            self._staging = [{k: Shard(np.empty_like(sh.data), sh.global_shape, sh.row_offset) for k, sh in rs.items()}
                             for rs in rank_states]
        for dst, src in zip(self._staging, rank_states):
            for k, sh in src.items():
                np.copyto(dst[k].data, sh.data)     # the only work the training loop waits for
        return self._staging

    # ---------------------------------------------------------------- phase 2
    def _write_rank(self, step, rank, shards):
        fname, blob, chunks = encode_rank(rank, shards)
        if self.fault:
            self.fault("rank_write", step=step, rank=rank, key=f"{step_dir(step)}/{fname}", data=blob)
        self.storage.put(f"{step_dir(step)}/{fname}", blob)
        return chunks, len(blob)

    def _persist(self, step, snap, rec, extra):
        try:
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.world) as ex:          # ranks write in parallel
                results = list(ex.map(lambda r: self._write_rank(step, r, snap[r]), range(self.world)))
            if self.fault:
                self.fault("before_manifest", step=step, rank=0)
            man = build_manifest(step, self.world, [c for c, _ in results], extra)
            self.storage.put(f"{step_dir(step)}/{MANIFEST}", man)           # COMMIT: last, atomic
            rec["write_s"] = time.perf_counter() - t
            rec["bytes"] = sum(n for _, n in results) + len(man)
            rec["deleted"] = gc(self.storage, self.keep_last, protect=step) if self.keep_last else []
        except BaseException as e:      # keep the error for the training thread (wait() re-raises it)
            self._error = e
            rec["error"] = repr(e)

    # ---------------------------------------------------------------- public API
    def save(self, step, rank_states, blocking=False, extra=None):
        """Returns a record with stall_s = how long training was paused."""
        t0 = time.perf_counter()
        self.wait()                     # back-pressure: at most one save in flight
        waited = time.perf_counter() - t0
        snap = self._snapshot(rank_states)
        rec = {"step": step, "mode": "sync" if blocking else "async", "waited_s": waited}
        rec["snapshot_s"] = time.perf_counter() - t0 - waited
        if blocking:
            self._persist(step, snap, rec, extra)
            self.wait()
        else:
            self._thread = threading.Thread(target=self._persist, args=(step, snap, rec, extra), daemon=True)
            self._thread.start()
        rec["stall_s"] = time.perf_counter() - t0
        self.history.append(rec)
        return rec

    def wait(self):
        """Block until the in-flight save is committed; re-raise its error if it failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            e, self._error = self._error, None
            raise RuntimeError(f"background checkpoint failed: {e!r}") from e
