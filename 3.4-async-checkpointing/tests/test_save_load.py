import numpy as np
import pytest

from ackpt import Checkpointer, ObjectStore, load, load_latest, shard_state, unshard
from ackpt.manifest import MANIFEST
from ackpt.state import row_ranges


def same(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)


def test_row_ranges_cover_every_row_once():
    for n in (1, 7, 8, 100, 1001):
        for w in (1, 2, 3, 4, 8):
            rr = row_ranges(n, w)
            assert rr[0][0] == 0 and rr[-1][1] == n
            assert all(rr[i][1] == rr[i + 1][0] for i in range(w - 1))
            assert max(b - a for a, b in rr) - min(b - a for a, b in rr) <= 1


def test_shard_unshard_roundtrip(state):
    for w in (1, 2, 4, 5):
        assert same(unshard(shard_state(state, w)), state)


@pytest.mark.parametrize("blocking", [True, False])
def test_save_then_load_same_world(store, state, blocking):
    ck = Checkpointer(store, 4)
    ck.save(5, shard_state(state, 4), blocking=blocking)
    ck.wait()
    step, ranks, skipped = load_latest(store, 4)
    assert step == 5 and skipped == [] and same(unshard(ranks), state)


@pytest.mark.parametrize("w_save,w_load", [(4, 2), (4, 8), (4, 3), (3, 4), (2, 1), (1, 6)])
def test_reshard_on_load(store, state, w_save, w_load):
    ck = Checkpointer(store, w_save)
    ck.save(1, shard_state(state, w_save), blocking=True)
    ranks = load(store, 1, w_load)
    assert len(ranks) == w_load
    assert same(unshard(ranks), state)
    expect = shard_state(state, w_load)                    # each rank gets exactly its new rows
    for r in range(w_load):
        assert set(ranks[r]) == set(expect[r])
        for k in ranks[r]:
            assert ranks[r][k].row_offset == expect[r][k].row_offset
            assert np.array_equal(ranks[r][k].data, expect[r][k].data)


def test_async_snapshot_is_isolated_from_later_updates(state):
    store = ObjectStore(latency_s=0.05)                     # slow writes: training runs on meanwhile
    ck = Checkpointer(store, 4)
    before = {k: v.copy() for k, v in state.items()}
    ck.save(1, shard_state(state, 4))
    for v in state.values():
        v += 1                                              # the "next training step" changes everything in place
    ck.wait()
    assert same(unshard(load(store, 1, 4)), before)


def test_manifest_is_written_last(state):
    order = []

    class Recording(ObjectStore):
        def put(self, key, data, crash_after=None):
            order.append(key)
            super().put(key, data, crash_after)
    ck = Checkpointer(Recording(), 4)
    ck.save(3, shard_state(state, 4))
    ck.wait()
    assert order[-1].endswith(MANIFEST) and len(order) == 5


def test_staging_buffers_are_reused(state):
    ck = Checkpointer(ObjectStore(), 2)
    ck.save(1, shard_state(state, 2))
    ids = [id(sh.data) for rs in ck._staging for sh in rs.values()]
    ck.save(2, shard_state(state, 2))
    assert [id(sh.data) for rs in ck._staging for sh in rs.values()] == ids
    ck.wait()


def test_only_one_save_in_flight(state):
    store = ObjectStore(latency_s=0.03)
    ck = Checkpointer(store, 4)
    ck.save(1, shard_state(state, 4))
    r2 = ck.save(2, shard_state(state, 4))                  # must wait for save 1 (back-pressure)
    assert r2["waited_s"] > 0.02
    ck.wait()


def test_background_failure_is_reported_and_not_committed(state):
    store = ObjectStore(fail_every_n=2)                     # 2nd request fails like a 503
    ck = Checkpointer(store, 4)
    ck.save(1, shard_state(state, 4))
    with pytest.raises(RuntimeError, match="503"):
        ck.wait()
    assert load_latest(store, 4)[0] is None


def test_stall_is_recorded(state):
    ck = Checkpointer(ObjectStore(), 2)
    r = ck.save(1, shard_state(state, 2), blocking=True)
    assert r["stall_s"] >= r["snapshot_s"] > 0 and r["bytes"] > 0 and ck.history == [r]
