import numpy as np

from threed import model as M

CFG = M.Config(vocab=16, hidden=8, heads=2, ffn=16, layers=1, seq=4)


def _loss(full, x, y):
    tp = M.TP()
    h, _ = M.emb_fwd(CFG, full["emb"], x, tp)
    h, _ = M.block_fwd(CFG, full["blocks"][0], h, tp)
    return M.head_fwd(CFG, full["head"], h, y, tp, 1.0)[0]


def _grads(full, x, y):
    tp = M.TP()
    h, ce = M.emb_fwd(CFG, full["emb"], x, tp)
    h, cb = M.block_fwd(CFG, full["blocks"][0], h, tp)
    _, ch = M.head_fwd(CFG, full["head"], h, y, tp, 1.0)
    dh, gh = M.head_bwd(CFG, full["head"], ch, tp)
    dh, gb = M.block_bwd(CFG, full["blocks"][0], dh, cb, tp)
    return M.emb_bwd(CFG, full["emb"], dh, ce), gb, gh


def test_backward_matches_finite_differences():
    full = M.init_full(CFG, 3)
    for blk in full["blocks"]:                    # larger weights so every path matters
        for k in ("wqkv", "wo", "w1", "w2"):
            blk[k] *= 20
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 16, (2, 4)), rng.integers(0, 16, (2, 4))
    ge, gb, gh = _grads(full, x, y)
    checks = [(full["blocks"][0], gb, k) for k in gb] + [(full["head"], gh, k) for k in gh] + [(full["emb"], ge, "emb")]
    for tree, g, k in checks:
        a = tree[k].reshape(-1)
        for idx in rng.choice(a.size, size=min(4, a.size), replace=False):
            old = a[idx]
            a[idx] = old + 1e-6; lp = _loss(full, x, y)
            a[idx] = old - 1e-6; lm = _loss(full, x, y)
            a[idx] = old
            num = (lp - lm) / 2e-6
            assert abs(num - g[k].reshape(-1)[idx]) < 1e-6 + 1e-4 * abs(num), k


def test_shard_unshard_roundtrip():
    full = M.init_full(CFG, 0)["blocks"][0]
    for t in (1, 2):
        back = M.unshard_block(CFG, [M.shard_block(CFG, full, r, t) for r in range(t)])
        assert all(np.array_equal(back[k], full[k]) for k in full)


def test_shard_shapes():
    s = M.shard_block(M.Config(), M.init_full(M.Config(), 0)["blocks"][0], 1, 4)
    assert s["wqkv"].shape == (32, 24) and s["wo"].shape == (8, 32)
    assert s["w1"].shape == (32, 16) and s["w2"].shape == (16, 32)
