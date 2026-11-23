import time

import pytest

from ringddp import CommTimeout, RingLink, WorkerFailure, launch
from ringddp.trainer import Config, train_worker

CFG = dict(epochs=2, samples=2048, global_batch=128, hidden=64)


def test_crash_is_reported_quickly_by_every_rank():
    t0 = time.monotonic()
    with pytest.raises(WorkerFailure) as e:
        launch(train_worker, 4, (Config(world=4, crash_rank=1, crash_step=3, timeout=10, **CFG),), grace=5)
    assert time.monotonic() - t0 < 5                      # EOF travels round the ring, no need to wait for timeouts
    st = e.value.status
    assert "exit code 17" in st[1]
    assert all("PeerClosed" in st[r] for r in (0, 2, 3))


def test_hung_worker_times_out_and_is_killed():
    t0 = time.monotonic()
    with pytest.raises(WorkerFailure) as e:
        launch(train_worker, 4, (Config(world=4, hang_rank=1, crash_step=2, timeout=1.0, **CFG),), grace=1.5)
    assert time.monotonic() - t0 < 6
    st = e.value.status
    assert "killed" in st[1]
    assert any("CommTimeout" in v for v in st.values())


def test_bad_batch_is_a_clean_error():
    with pytest.raises(WorkerFailure, match="not divisible"):
        launch(train_worker, 3, (Config(world=3, **CFG),))


def test_missing_peer_times_out_at_startup():
    with pytest.raises(CommTimeout):
        RingLink(0, 2, [("127.0.0.1", 0), ("127.0.0.1", 9)], timeout=0.3)
