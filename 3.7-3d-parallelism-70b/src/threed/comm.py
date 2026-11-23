"""Message passing between worker processes: point-to-point send/recv and ring collectives.

Every rank owns one inbox (a multiprocessing.Queue). send() never blocks, so a pipeline
schedule cannot deadlock on a full buffer. recv() waits for a message with a given
(source, tag) and parks any other message it sees in a local mailbox. Every byte sent is
counted per traffic kind ("tp", "pp", "dp") so tests can compare it with the formulas.
"""
import queue as _queue
from collections import defaultdict

import numpy as np


class CommTimeout(RuntimeError):
    """A peer did not answer in time: treated as a failed rank by the launcher."""


class Comm:
    def __init__(self, rank, world, inboxes, timeout=30.0):
        self.rank, self.world, self.inboxes, self.timeout = rank, world, inboxes, timeout
        self.mailbox = defaultdict(list)          # (src, tag) -> messages that arrived early
        self.bytes_sent = defaultdict(int)        # kind -> bytes this rank sent
        self.messages = defaultdict(int)          # kind -> messages this rank sent
        self._seq = defaultdict(int)              # per-group collective counter (keeps tags unique)

    # ------------------------------------------------------------ point to point
    def send(self, dst, tag, arr, kind="pp"):
        arr = np.ascontiguousarray(arr)
        self.bytes_sent[kind] += arr.nbytes
        self.messages[kind] += 1
        self.inboxes[dst].put((self.rank, tag, arr))

    def recv(self, src, tag):
        key = (src, tag)
        if self.mailbox[key]:
            return self.mailbox[key].pop(0)
        while True:
            try:
                s, t, arr = self.inboxes[self.rank].get(timeout=self.timeout)
            except _queue.Empty:
                raise CommTimeout(f"rank {self.rank}: nothing from rank {src} (tag {tag}) in {self.timeout}s")
            if (s, t) == key:
                return arr
            self.mailbox[(s, t)].append(arr)

    # ------------------------------------------------------------ collectives (ring algorithms)
    def _tag(self, group, name):
        g = tuple(group)
        self._seq[(g, name)] += 1
        return f"{name}{g}#{self._seq[(g, name)]}"

    def all_reduce(self, x, group, op="sum", kind="tp"):
        """Ring all-reduce = reduce-scatter then all-gather. Each rank sends 2(n-1)/n of the data."""
        group = list(group)
        n = len(group)
        if n == 1:
            return x.copy()
        tag = self._tag(group, "ar")
        i = group.index(self.rank)
        right, left = group[(i + 1) % n], group[(i - 1) % n]
        flat = np.array(x, dtype=x.dtype).reshape(-1)
        chunks = np.array_split(flat, n)
        red = np.add if op == "sum" else np.maximum
        for step in range(n - 1):                 # reduce-scatter: after it, rank i owns chunk (i+1) % n
            c_send, c_recv = (i - step) % n, (i - step - 1) % n
            self.send(right, (tag, "rs", step), chunks[c_send], kind)
            chunks[c_recv] = red(chunks[c_recv], self.recv(left, (tag, "rs", step)))
        for step in range(n - 1):                 # all-gather: pass the finished chunks around the ring
            c_send, c_recv = (i + 1 - step) % n, (i - step) % n
            self.send(right, (tag, "ag", step), chunks[c_send], kind)
            chunks[c_recv] = self.recv(left, (tag, "ag", step))
        return np.concatenate(chunks).reshape(x.shape)

    def all_gather(self, x, group, kind="tp"):
        """Returns the list of every rank's x, in group order (ring, n-1 steps)."""
        group = list(group)
        n = len(group)
        parts = [None] * n
        i = group.index(self.rank)
        parts[i] = x
        if n == 1:
            return parts
        tag = self._tag(group, "gather")
        right, left = group[(i + 1) % n], group[(i - 1) % n]
        for step in range(n - 1):
            send_idx, recv_idx = (i - step) % n, (i - step - 1) % n
            self.send(right, (tag, step), parts[send_idx], kind)
            parts[recv_idx] = self.recv(left, (tag, step))
        return parts

    def barrier(self, group, kind="ctl"):
        self.all_reduce(np.zeros(1), group, kind=kind)
