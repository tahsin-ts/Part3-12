"""Integration tests: real Go shards (subprocesses) and NumPy worker processes."""
import time

import numpy as np
import pytest

from pskit.client import PSClient, PSError
from pskit.cluster import Shards, summarize_stats, train


@pytest.fixture(scope="module")
def shards(psserver):
    with Shards(2, 4, base=7600, worker_timeout="1s") as sh:
        yield sh


@pytest.mark.parametrize("staleness", [0, 2, -1])
def test_training_converges_and_respects_the_bound(shards, staleness):
    shards.reset(4)
    r = train(shards.addrs, workers=4, steps=80, staleness=staleness)
    st = summarize_stats(shards.stats())
    assert r["train_loss_curve"][-1] < 0.5 < r["train_loss_curve"][0]
    assert r["test_loss"] < 0.5
    if staleness >= 0:
        assert st["max_clock_gap"] <= staleness  # the SSP guarantee, checked on every pull
    else:
        assert st["waits"] == 0  # async never blocks


def test_wrong_ring_is_rejected(shards):
    shards.reset(1)
    cl = PSClient(shards.addrs[::-1], retry_for=1)  # shard order swapped: every key goes to the wrong shard
    keys = np.arange(100, dtype=np.uint64)
    with pytest.raises(PSError, match="409"):
        cl.pull(keys)
    cl.close()
    assert sum(s["Misrouted"] for s in shards.stats()) >= 1


def test_shard_crash_and_restart_from_snapshot(psserver, tmp_path):
    with Shards(2, 2, base=7650, snapdir=str(tmp_path), snapshot_every="100ms") as sh:
        cl = PSClient(sh.addrs, worker=0)
        keys = np.arange(1000, dtype=np.uint64)
        cl.push(keys, np.ones((1000, 1), np.float32), clock=1)
        time.sleep(0.4)  # at least one snapshot
        before = cl.pull(keys, clock=1)
        sh.kill(1)
        sh.start(1)  # restarted process loads shard-1.snap
        after = cl.pull(keys, clock=1)
        np.testing.assert_array_equal(before, after)
        assert sh.stats()[1]["RestoredKeys"] > 0
        cl.close()


def test_training_survives_a_shard_restart(psserver, tmp_path):
    import threading
    with Shards(2, 2, base=7700, snapdir=str(tmp_path), snapshot_every="100ms") as sh:
        def chaos(t0):
            def go():
                time.sleep(0.05)  # early, so even a fast machine is still training
                sh.kill(0)
                time.sleep(0.3)
                sh.start(0)
            threading.Thread(target=go, daemon=True).start()
        r = train(sh.addrs, workers=2, steps=400, staleness=2, on_step=chaos)
    assert r["retries"] > 0  # workers did hit the dead shard
    assert r["test_loss"] < 0.5


def test_dead_worker_is_evicted_and_sync_training_finishes(shards):
    shards.reset(4)
    t = time.perf_counter()
    r = train(shards.addrs, workers=4, steps=60, staleness=0, die={3: 10})
    st = summarize_stats(shards.stats())
    assert st["evicted"] == [3]
    assert r["test_loss"] < 0.6 and time.perf_counter() - t < 30
