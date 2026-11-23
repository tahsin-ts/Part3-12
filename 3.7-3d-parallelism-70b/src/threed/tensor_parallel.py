"""Column- and row-parallel linear layers, the two building blocks of Megatron-style tensor parallelism.

    Y = X W  with W split by COLUMNS over t ranks:  Y_r = X W_r      (no communication; output is split)
    Y = X W  with W split by ROWS    over t ranks:  Y = sum_r X_r W_r (one all-reduce; input must be split)

Chaining them (column then row) gives an MLP with ONE all-reduce in the forward pass and ONE in the
backward pass. The two conjugate operators make this explicit:
    f: forward = identity,   backward = all-reduce   (placed before the column-parallel layer)
    g: forward = all-reduce, backward = identity     (placed after the row-parallel layer)
"""
import numpy as np

from .model import TP, gelu_bwd, gelu_fwd


def column_fwd(x, w_r, b_r):
    return x @ w_r + b_r


def row_fwd(x_r, w_r, b, tp):
    return tp.all_reduce(x_r @ w_r) + b                  # g operator


def mlp_fwd(x, p, tp):
    u = column_fwd(x, p["w1"], p["b1"])
    a, ge = gelu_fwd(u)
    return row_fwd(a, p["w2"], p["b2"], tp), (x, a, ge)


def mlp_bwd(dy, p, cache, tp):
    x, a, ge = cache
    fl = lambda t: t.reshape(-1, t.shape[-1])
    g = {"w2": fl(a).T @ fl(dy), "b2": fl(dy).sum(0)}
    du = gelu_bwd(dy @ p["w2"].T, ge)
    g["w1"], g["b1"] = fl(x).T @ fl(du), fl(du).sum(0)
    dx = tp.all_reduce(du @ p["w1"].T)                    # f operator (backward)
    return dx, g


def split_mlp(p, r, t):
    f = p["w1"].shape[1] // t
    return {"w1": p["w1"][:, r * f:(r + 1) * f], "b1": p["b1"][r * f:(r + 1) * f],
            "w2": p["w2"][r * f:(r + 1) * f], "b2": p["b2"]}


def make_mlp(h, f, seed=0):
    rng = np.random.default_rng(seed)
    return {"w1": rng.normal(0, 0.1, (h, f)), "b1": rng.normal(0, 0.1, f),
            "w2": rng.normal(0, 0.1, (f, h)), "b2": rng.normal(0, 0.1, h)}


def check_worker(comm, rank, t, h, f, batch, seed):
    """Every rank runs its shard of the MLP; returns output, input gradient and its weight-gradient shards."""
    full = make_mlp(h, f, seed)
    p = split_mlp(full, rank, t)
    tp = TP(comm, list(range(t)), rank)
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(batch, h))
    dy = rng.normal(size=(batch, h))
    y, cache = mlp_fwd(x, p, tp)
    dx, g = mlp_bwd(dy, p, cache, tp)
    return {"y": y, "dx": dx, "g": g, "bytes": dict(comm.bytes_sent)}


def dense_reference(h, f, batch, seed):
    full = make_mlp(h, f, seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(batch, h))
    dy = rng.normal(size=(batch, h))
    y, cache = mlp_fwd(x, full, TP())
    dx, g = mlp_bwd(dy, full, cache, TP())
    return y, dx, g
