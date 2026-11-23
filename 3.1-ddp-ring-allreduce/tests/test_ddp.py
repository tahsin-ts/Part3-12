import numpy as np
import pytest

from ringddp import MLP
from ringddp.ddp import DistributedDataParallel
from ringddp.launch import launch
from ringddp.trainer import Config, train_reference, train_worker

CFG = dict(epochs=2, samples=2048, global_batch=128, hidden=64)


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    m = MLP([5, 7, 3], seed=1)
    x, y = rng.normal(size=(4, 5)), np.array([0, 2, 1, 2])
    m.loss_and_backward(x, y)
    for i in range(len(m.params)):
        p, g = m.params[i], m.grads[i].copy()
        j = np.unravel_index(1 % p.size, p.shape)
        old = p[j]
        p[j] = old + 1e-6
        up = m.loss_and_backward(x, y)
        p[j] = old - 1e-6
        dn = m.loss_and_backward(x, y)
        p[j] = old
        assert abs((up - dn) / 2e-6 - g[j]) < 1e-7


def test_buckets_follow_backward_order_and_hold_the_grads():
    class OneRank:
        world, rank = 1, 0
    m = MLP([8, 256, 256, 4], seed=0)
    ddp = DistributedDataParallel(m, OneRank(), bucket_bytes=64 * 1024)
    assert ddp.buckets[0].param_ids[0] == len(m.params) - 1          # last layer's bias first
    for i, g in enumerate(m.grads):
        assert np.shares_memory(g, ddp.bucket_of[i].flat)             # gradient-as-bucket-view
    ddp.close()


@pytest.mark.parametrize("world", [2, 4])
def test_n_workers_equal_one_process(world):
    ref = train_reference(Config(**CFG))
    res = launch(train_worker, world, (Config(world=world, **CFG),))
    assert np.abs(res[0]["params"] - ref["params"]).max() <= 1e-12
    assert len({r["digest"] for r in res}) == 1
    assert np.allclose(res[0]["losses"], ref["losses"], rtol=0, atol=1e-12)


def test_bucket_size_and_overlap_do_not_change_the_result():
    ref = train_reference(Config(**CFG))
    for kb, ov in [(0, False), (0, True), (16, True), (1024, False)]:
        res = launch(train_worker, 2, (Config(world=2, bucket_kb=kb, overlap=ov, **CFG),))
        assert np.abs(res[0]["params"] - ref["params"]).max() <= 1e-12


def test_bytes_per_worker_match_theory():
    cfg = Config(world=4, **CFG)
    res = launch(train_worker, 4, (cfg,))
    n = res[0]["n_params"]
    grad_bytes = 2 * 3 / 4 * n * 8 * res[0]["steps"]              # gradient all-reduces
    total = sum(r["bytes_sent"] for r in res) / 4
    assert abs(total - grad_bytes) <= 64 * cfg.epochs                # plus one tiny loss all-reduce per epoch
