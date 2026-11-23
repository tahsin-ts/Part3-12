import numpy as np
import pytest

from threed import engine as E
from threed import tensor_parallel as T
from threed.model import Config
from threed.topology import Grid


@pytest.mark.parametrize("t", [2, 4])
def test_parallel_mlp_equals_dense(t):
    y0, dx0, g0 = T.dense_reference(16, 32, 6, 0)
    res = E.launch(t, T.check_worker, (t, 16, 32, 6, 0))
    for r in res:
        assert np.allclose(r["y"], y0, atol=1e-12) and np.allclose(r["dx"], dx0, atol=1e-12)
    assert np.allclose(np.concatenate([r["g"]["w1"] for r in res], 1), g0["w1"], atol=1e-12)
    assert np.allclose(np.concatenate([r["g"]["w2"] for r in res], 0), g0["w2"], atol=1e-12)
    assert all(np.allclose(r["g"]["b2"], g0["b2"]) for r in res)     # replicated bias: same grad everywhere


def test_one_forward_and_one_backward_all_reduce():
    res = E.launch(2, T.check_worker, (2, 16, 32, 6, 0))
    assert res[0]["bytes"]["tp"] == 2 * E.ring_bytes(6 * 16, 2, 0)


@pytest.mark.parametrize("t", [2, 4])
def test_tensor_parallel_transformer_equals_dense(t):
    cfg = Config(layers=2)
    ref_l, ref = E.train_reference(cfg, 2, 8)
    res = E.train(cfg, Grid(t, 1, 1), 2, 8, 8, return_params=True)
    assert max(abs(a - b) for a, b in zip(res[0]["losses"], ref_l)) < 1e-12
    assert E.max_diff(ref, E.gather_full(cfg, Grid(t, 1, 1), res)) < 1e-12
