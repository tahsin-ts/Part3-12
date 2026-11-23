"""Data-parallel wrapper: gradient buckets, all-reduced on a communication thread while backward runs."""
import queue
import threading
import time

import numpy as np

from .allreduce import ring_allreduce


class Bucket:
    def __init__(self, index, param_ids, size):
        self.index, self.param_ids = index, param_ids
        self.flat = np.zeros(size)                         # the gradients live INSIDE this buffer
        self.pending = 0
        self.done = threading.Event()


class DistributedDataParallel:
    def __init__(self, model, link, bucket_bytes=256 * 1024, overlap=True):
        self.model, self.link, self.overlap = model, link, overlap
        ready_order = list(reversed(range(len(model.params))))   # backward produces the last layer first
        groups, cur, cur_bytes = [], [], 0
        for i in ready_order:
            cur.append(i)
            cur_bytes += model.params[i].nbytes
            if bucket_bytes and cur_bytes >= bucket_bytes:
                groups.append(cur)
                cur, cur_bytes = [], 0
        if cur:
            groups.append(cur)
        self.buckets, self.bucket_of = [], {}
        for bi, ids in enumerate(groups):
            b = Bucket(bi, ids, sum(model.params[i].size for i in ids))
            off = 0
            for i in ids:                                   # gradient-as-bucket-view: no copy into the bucket
                p = model.params[i]
                model.grads[i] = b.flat[off:off + p.size].reshape(p.shape)
                self.bucket_of[i] = b
                off += p.size
            self.buckets.append(b)
        self.error, self.calls = None, 0
        self.compute_s = self.wait_s = self.comm_s = 0.0
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._comm_loop, daemon=True)
        self._thread.start()

    # ---- communication thread: one bucket at a time, in launch order (the same on every rank)
    def _comm_loop(self):
        while True:
            b = self._q.get()
            if b is None:
                return
            if self.error is None:
                try:
                    t0 = time.perf_counter()
                    ring_allreduce(b.flat, self.link, tag=(self.calls * 2 * self.link.world) % 2**30)
                    b.flat /= self.link.world                # sum -> mean
                    self.comm_s += time.perf_counter() - t0
                except Exception as e:                       # remember it; the training thread raises it
                    self.error = e
            self.calls += 1
            b.done.set()

    def _on_grad(self, i):
        b = self.bucket_of[i]
        b.pending -= 1
        if b.pending == 0 and self.overlap:
            self._q.put(b)                                   # start all-reducing while backward continues

    def backward(self, x, y):
        """Forward + backward + gradient all-reduce. Returns the local loss; grads are averaged."""
        for b in self.buckets:
            b.pending = len(b.param_ids)
            b.done.clear()
        t0 = time.perf_counter()
        loss = self.model.loss_and_backward(x, y, self._on_grad)
        t1 = time.perf_counter()
        if not self.overlap:
            for b in self.buckets:
                self._q.put(b)
        for b in self.buckets:
            b.done.wait()
        t2 = time.perf_counter()
        self.compute_s += t1 - t0
        self.wait_s += t2 - t1                               # communication NOT hidden behind compute
        if self.error:
            raise self.error
        return loss

    def allreduce_mean(self, values):
        """Average a few numbers (e.g. the loss) over all ranks, outside backward."""
        buf = np.array(values, dtype=np.float64)
        ring_allreduce(buf, self.link, tag=2**30)
        return buf / self.link.world

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=1.0)
