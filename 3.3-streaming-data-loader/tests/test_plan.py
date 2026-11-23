import random
from collections import Counter

from streamload.plan import EpochPlan, batch_sizes, buffered_shuffle, rr_schedule

SHARDS = [(f"s{i}", 10 + i) for i in range(12)]


def test_buffered_shuffle_is_a_permutation_and_bounded():
    out = list(buffered_shuffle(1000, 64, random.Random(1)))
    assert sorted(p for p, _ in out) == list(range(1000))
    assert all(p < read for p, read in out)                 # only emits what was already read
    assert all(read - i <= 64 for i, (_, read) in enumerate(out))   # at most 64 items held
    assert [p for p, _ in out] != list(range(1000))


def test_buffered_shuffle_deterministic_and_seeded():
    a = [p for p, _ in buffered_shuffle(500, 32, random.Random("x"))]
    b = [p for p, _ in buffered_shuffle(500, 32, random.Random("x"))]
    c = [p for p, _ in buffered_shuffle(500, 32, random.Random("y"))]
    assert a == b and a != c
    assert [p for p, _ in buffered_shuffle(50, 1, random.Random(0))] == list(range(50))


def test_no_duplicates_across_ranks_and_workers():
    for world, workers in ((1, 1), (2, 2), (3, 2), (4, 3)):
        seen = []
        for r in range(world):
            p = EpochPlan(SHARDS, seed=7, epoch=3, batch_size=4, num_workers=workers, world_size=world, rank=r)
            for w in range(workers):
                seen += [n for n, _ in p.worker_shards(w)]
        assert sorted(seen) == sorted(n for n, _ in SHARDS)    # every shard exactly once


def test_shard_order_changes_by_epoch_not_by_rank():
    p0 = EpochPlan(SHARDS, 1, 0, 4, world_size=2, rank=0)
    p1 = EpochPlan(SHARDS, 1, 0, 4, world_size=2, rank=1)
    q0 = EpochPlan(SHARDS, 1, 1, 4, world_size=2, rank=0)
    assert p0.order == p1.order and p0.order != q0.order


def test_equal_steps_on_every_rank():
    for world in (2, 3, 5):
        steps = {EpochPlan(SHARDS, 0, 0, 8, 2, world, r).steps() for r in range(world)}
        assert len(steps) == 1


def test_round_robin_and_start_batches():
    assert rr_schedule([2, 3, 1]) == [0, 1, 2, 0, 1, 1]
    p = EpochPlan(SHARDS, 0, 0, batch_size=5, num_workers=3)
    sched = p.schedule()
    assert len(sched) == sum(p.worker_batches()) == p.steps()
    for k in range(len(sched) + 1):
        assert sum(p.start_batches(k)) == k
    assert p.start_batches(len(sched)) == p.worker_batches()


def test_batch_sizes():
    assert batch_sizes(10, 4, False) == [4, 4, 2]
    assert batch_sizes(10, 4, True) == [4, 4]
    assert Counter(batch_sizes(0, 4, False)) == Counter()
