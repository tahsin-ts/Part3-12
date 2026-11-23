"""Deterministic epoch plan: shard shuffle, rank/worker assignment, buffered shuffle, round-robin schedule.

Everything here is a pure function of (manifest, seed, epoch, world size, workers, batch size), computed on
integers only. That is what makes a resume state tiny: given the step number, every worker can recompute
exactly where it was without reading any data.
"""
import random
from collections import Counter


def shard_order(n_shards, seed, epoch, shuffle=True):
    order = list(range(n_shards))
    if shuffle:
        random.Random(f"shards:{seed}:{epoch}").shuffle(order)
    return order


def assign_shards(order, world_size, rank, num_workers, worker):
    """Global worker g = rank * num_workers + worker takes every (world_size * num_workers)-th shard.
    Shards are disjoint across all (rank, worker) pairs, so no sample is seen twice per epoch."""
    g, total = rank * num_workers + worker, world_size * num_workers
    return order[g::total]


def buffered_shuffle(n, buffer_size, rng):
    """Shuffle a stream of n items using a buffer of buffer_size. Yields (position, items_read_so_far).
    Memory is O(buffer_size); position p is only emitted after it has been read (p < items_read)."""
    buf = []
    for p in range(n):
        buf.append(p)
        if len(buf) >= buffer_size:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop(), p + 1
    while buf:
        j = rng.randrange(len(buf))
        buf[j], buf[-1] = buf[-1], buf[j]
        yield buf.pop(), n


def batch_sizes(n, batch_size, drop_last):
    full, rest = divmod(n, batch_size)
    return [batch_size] * full + ([rest] if rest and not drop_last else [])


def rr_schedule(batches_per_worker):
    """Round-robin over workers, skipping exhausted ones: the order in which the main process takes batches."""
    left, out, w = list(batches_per_worker), [], 0
    while any(left):
        if left[w]:
            out.append(w)
            left[w] -= 1
        w = (w + 1) % len(left)
    return out


class EpochPlan:
    """The plan for one rank in one epoch. shards: list of (name, record_count) from the manifest."""

    def __init__(self, shards, seed, epoch, batch_size, num_workers=1, world_size=1, rank=0,
                 shuffle=True, drop_last=False):
        self.shards, self.seed, self.epoch = shards, seed, epoch
        self.batch_size, self.num_workers, self.world_size, self.rank = batch_size, num_workers, world_size, rank
        self.shuffle, self.drop_last = shuffle, drop_last
        self.order = shard_order(len(shards), seed, epoch, shuffle)

    def worker_shards(self, worker, rank=None):
        r = self.rank if rank is None else rank
        return [self.shards[i] for i in assign_shards(self.order, self.world_size, r, self.num_workers, worker)]

    def worker_samples(self, worker, rank=None):
        return sum(c for _, c in self.worker_shards(worker, rank))

    def worker_batches(self, rank=None):
        return [len(batch_sizes(self.worker_samples(w, rank), self.batch_size, self.drop_last))
                for w in range(self.num_workers)]

    def steps(self):
        """Steps per epoch = the minimum over ranks, so every rank calls all-reduce the same number of times."""
        return min(sum(self.worker_batches(r)) for r in range(self.world_size))

    def schedule(self):
        return rr_schedule(self.worker_batches())[:self.steps()]

    def start_batches(self, step):
        """How many batches each worker had already delivered when `step` batches were consumed."""
        c = Counter(self.schedule()[:step])
        return [c.get(w, 0) for w in range(self.num_workers)]

    def rng(self, worker):
        return random.Random(f"buffer:{self.seed}:{self.epoch}:{self.rank}:{worker}")
