import numpy as np

from elastic import model as M


def test_chunk_grad_matches_finite_differences():
    x, y = M.make_data(0)
    p = M.init_params(0)
    g, _ = M.chunk_grad(p, x[:8], y[:8])
    theta = M.flatten(p)
    eps = 1e-6
    for i in (0, 100, 520, theta.size - 1):
        d = np.zeros_like(theta); d[i] = eps
        lp = M.chunk_grad(M.unflatten(theta + d, p), x[:8], y[:8])[1]
        lm = M.chunk_grad(M.unflatten(theta - d, p), x[:8], y[:8])[1]
        assert abs((lp - lm) / (2 * eps) - g[i]) < 1e-5


def test_every_chunk_owned_exactly_once():
    for w in range(1, 9):
        owned = sorted(c for r in range(w) for c in M.my_chunks(r, w))
        assert owned == list(range(M.N_CHUNKS))


def test_reduction_is_bitwise_identical_for_any_world_size():
    x, y = M.make_data(0)
    p = M.init_params(0)
    idx = M.batch_indices(5)
    results = set()
    for w in (1, 2, 3, 4, 8):
        chunks = {}
        for r in range(w):
            for c in M.my_chunks(r, w):
                sel = idx[c * M.CHUNK:(c + 1) * M.CHUNK]
                chunks[c] = M.chunk_grad(p, x[sel], y[sel])
        g, _ = M.reduce_chunks(chunks)
        results.add(g.tobytes())
    assert len(results) == 1


def test_data_order_is_a_pure_function_of_step():
    assert (M.batch_indices(70) == M.batch_indices(70)).all()
    epoch0 = np.concatenate([M.batch_indices(s) for s in range(M.N_SAMPLES // M.GLOBAL_BATCH)])
    assert sorted(epoch0.tolist()) == list(range(M.N_SAMPLES))      # one epoch visits every sample once
