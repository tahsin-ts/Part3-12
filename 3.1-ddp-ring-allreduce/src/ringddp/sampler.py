"""Each rank gets a disjoint slice of a shuffled index list; the shuffle changes every epoch."""
import numpy as np


class DistributedSampler:
    def __init__(self, n, world, rank, shuffle=True, seed=0, drop_last=False):
        self.n, self.world, self.rank = n, world, rank
        self.shuffle, self.seed, self.drop_last, self.epoch = shuffle, seed, drop_last, 0
        if drop_last:
            self.per_rank = n // world                       # throw away the tail
        else:
            self.per_rank = -(-n // world)                   # ceil: pad by repeating the head

    def set_epoch(self, epoch):
        self.epoch = epoch                                   # same seed + epoch on every rank -> same permutation

    def indices(self):
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(self.n)
        else:
            order = np.arange(self.n)
        total = self.per_rank * self.world
        order = np.resize(order, total)                      # cut the tail, or repeat the head as padding
        return order[self.rank::self.world]                  # interleaved: rank r takes r, r+N, r+2N, ...

    def __len__(self):
        return self.per_rank
