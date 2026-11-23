import numpy as np

from ringddp import DistributedSampler


def test_shards_are_disjoint_and_cover_the_dataset():
    shards = [DistributedSampler(100, 4, r, seed=3).indices() for r in range(4)]
    allidx = np.concatenate(shards)
    assert len(allidx) == 100 and set(allidx) == set(range(100))


def test_new_shuffle_every_epoch_same_on_every_rank():
    a, b = DistributedSampler(64, 2, 0, seed=1), DistributedSampler(64, 2, 1, seed=1)
    e0 = a.indices()
    a.set_epoch(1)
    b.set_epoch(1)
    assert not np.array_equal(e0, a.indices())
    assert set(a.indices()).isdisjoint(b.indices())


def test_padding_and_drop_last():
    pad = [DistributedSampler(10, 4, r, shuffle=False).indices() for r in range(4)]
    assert all(len(p) == 3 for p in pad)                     # 12 slots: 2 samples repeated
    drop = [DistributedSampler(10, 4, r, shuffle=False, drop_last=True).indices() for r in range(4)]
    assert all(len(d) == 2 for d in drop)                    # 8 slots: 2 samples skipped


def test_global_batch_is_the_same_set_as_single_process():
    n, world, gb = 64, 4, 16
    single = DistributedSampler(n, 1, 0, seed=5).indices()
    shards = [DistributedSampler(n, world, r, seed=5).indices() for r in range(world)]
    b = gb // world
    for step in range(n // gb):
        dist = np.concatenate([s[step * b:(step + 1) * b] for s in shards])
        assert set(dist) == set(single[step * gb:(step + 1) * gb])
