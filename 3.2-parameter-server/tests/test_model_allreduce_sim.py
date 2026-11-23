import numpy as np
import pytest

from pskit import sim
from pskit.allreduce import compare, ring_allreduce
from pskit.data import make_dataset
from pskit.model import batch_grad, logloss


def test_gradient_matches_finite_differences():
    keys, y = make_dataset(n=64, vocab=50, k=5, seed=1)
    w = {int(k): v for k, v in zip(range(51), np.random.default_rng(0).normal(0, 0.3, 51))}
    look = lambda u: np.array([[w[int(k)]] for k in u], np.float32)
    uniq, g, loss = batch_grad(keys, y, look)
    k0 = int(uniq[3])
    eps = 1e-3
    w[k0] += eps
    up = logloss(keys, y, lambda u: look(u)[:, 0])
    w[k0] -= 2 * eps
    down = logloss(keys, y, lambda u: look(u)[:, 0])
    assert g[3, 0] == pytest.approx((up - down) / (2 * eps), rel=1e-2, abs=1e-5)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_ring_allreduce_is_correct_and_sends_2_n_minus_1_over_n(n):
    arrs = [np.random.default_rng(r).normal(size=1000).astype(np.float32) for r in range(n)]
    res, sent = ring_allreduce(arrs)
    for r in res:
        np.testing.assert_allclose(r, sum(arrs), atol=1e-4)
    assert sum(sent) == pytest.approx(2 * (n - 1) * 1000 * 4, abs=8 * n)


def test_ring_wins_dense_ps_wins_sparse():
    rows = compare()
    assert [r["winner"] for r in rows] == ["ring", "ring", "PS", "PS"]


def test_large_staleness_hurts_sgd():
    data = make_dataset(n=30_000, vocab=20_000, k=16, seed=0)
    fresh = sim.run(0, lr=2.0, opt="sgd", updates=600, data=data, every=600)["test_loss"]
    stale = sim.run(32, lr=2.0, opt="sgd", updates=600, data=data, every=600)["test_loss"]
    assert stale > fresh * 1.5
