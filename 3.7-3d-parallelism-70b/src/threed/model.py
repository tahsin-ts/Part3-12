"""A small GPT-style transformer in NumPy with hand-written backward passes, written so that every
matrix can be split across tensor-parallel ranks (Megatron-LM style).

Pieces and how they are split over t tensor-parallel ranks:
  embedding      vocab-parallel: rank r holds rows [r V/t, (r+1) V/t); outputs are summed (all-reduce)
  attention      column-parallel QKV (rank r owns heads [r a/t, (r+1) a/t)), row-parallel output + all-reduce
  MLP            column-parallel W1 (h -> f/t), row-parallel W2 (f/t -> h) + all-reduce
  head + loss    column-parallel over the vocabulary, cross-entropy computed without gathering logits
  LayerNorms and the biases added after an all-reduce are replicated on every tensor-parallel rank.
With t = 1 every all-reduce is the identity and this is the ordinary dense model.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    vocab: int = 64
    hidden: int = 32
    heads: int = 4
    ffn: int = 64
    layers: int = 4
    seq: int = 8

    @property
    def head_dim(self):
        return self.hidden // self.heads


class TP:
    """Tensor-parallel context: the communicator and the ranks of this rank's tensor-parallel group."""

    def __init__(self, comm=None, group=(0,), rank_in_group=0):
        self.comm, self.group, self.r = comm, list(group), rank_in_group

    @property
    def size(self):
        return len(self.group)

    def all_reduce(self, x, op="sum"):
        if self.size == 1:
            return x
        return self.comm.all_reduce(x, self.group, op=op, kind="tp")


# ---------------------------------------------------------------------------------------- init / shard
def init_full(cfg, seed=0):
    """Full (unsplit) parameters: one dict for the embedding, one per block, one for the head."""
    rng = np.random.default_rng(seed)
    h, f, V = cfg.hidden, cfg.ffn, cfg.vocab
    n = lambda *s: rng.normal(0.0, 0.02, s)
    emb = {"emb": n(V, h)}
    blocks = []
    for _ in range(cfg.layers):
        blocks.append({"ln1_g": 1 + n(h), "ln1_b": n(h), "wqkv": n(h, 3 * h), "bqkv": n(3 * h),
                       "wo": n(h, h), "bo": n(h), "ln2_g": 1 + n(h), "ln2_b": n(h),
                       "w1": n(h, f), "b1": n(f), "w2": n(f, h), "b2": n(h)})
    head = {"lnf_g": 1 + n(h), "lnf_b": n(h), "wout": n(h, V)}
    return {"emb": emb, "blocks": blocks, "head": head}


def _qkv_cols(cfg, r, t):
    h, d, hl = cfg.hidden, cfg.head_dim, cfg.heads // t
    idx = []
    for part in range(3):                                   # Q, K, V sections of the fused matrix
        start = part * h + r * hl * d
        idx.extend(range(start, start + hl * d))
    return np.array(idx)


def shard_block(cfg, p, r, t):
    """Tensor-parallel shard r of t of one block's parameters."""
    if cfg.heads % t or cfg.ffn % t:
        raise ValueError("heads and ffn must be divisible by the tensor-parallel size")
    d, hl, fl = cfg.head_dim, cfg.heads // t, cfg.ffn // t
    cols = _qkv_cols(cfg, r, t)
    rows = slice(r * hl * d, (r + 1) * hl * d)
    return {"ln1_g": p["ln1_g"].copy(), "ln1_b": p["ln1_b"].copy(),
            "wqkv": p["wqkv"][:, cols].copy(), "bqkv": p["bqkv"][cols].copy(),
            "wo": p["wo"][rows].copy(), "bo": p["bo"].copy(),
            "ln2_g": p["ln2_g"].copy(), "ln2_b": p["ln2_b"].copy(),
            "w1": p["w1"][:, r * fl:(r + 1) * fl].copy(), "b1": p["b1"][r * fl:(r + 1) * fl].copy(),
            "w2": p["w2"][r * fl:(r + 1) * fl].copy(), "b2": p["b2"].copy()}


def unshard_block(cfg, shards):
    """Inverse of shard_block: rebuild the full block from the t shards (used to check results)."""
    t = len(shards)
    full = {k: shards[0][k].copy() for k in ("ln1_g", "ln1_b", "bo", "ln2_g", "ln2_b", "b2")}
    h = cfg.hidden
    wqkv, bqkv = np.zeros((h, 3 * h)), np.zeros(3 * h)
    for r, s in enumerate(shards):
        cols = _qkv_cols(cfg, r, t)
        wqkv[:, cols], bqkv[cols] = s["wqkv"], s["bqkv"]
    full.update(wqkv=wqkv, bqkv=bqkv, wo=np.concatenate([s["wo"] for s in shards]),
                w1=np.concatenate([s["w1"] for s in shards], 1), b1=np.concatenate([s["b1"] for s in shards]),
                w2=np.concatenate([s["w2"] for s in shards]))
    return full


def shard_emb(cfg, p, r, t):
    vl = cfg.vocab // t
    return {"emb": p["emb"][r * vl:(r + 1) * vl].copy()}


def shard_head(cfg, p, r, t):
    vl = cfg.vocab // t
    return {"lnf_g": p["lnf_g"].copy(), "lnf_b": p["lnf_b"].copy(), "wout": p["wout"][:, r * vl:(r + 1) * vl].copy()}


# ---------------------------------------------------------------------------------------- basic layers
def ln_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv, g)


def ln_bwd(dy, cache):
    xh, inv, g = cache
    dg = (dy * xh).reshape(-1, xh.shape[-1]).sum(0)
    db = dy.reshape(-1, xh.shape[-1]).sum(0)
    dxh = dy * g
    n = xh.shape[-1]
    dx = inv / n * (n * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
    return dx, dg, db


_C = np.sqrt(2.0 / np.pi)


def gelu_fwd(u):
    th = np.tanh(_C * (u + 0.044715 * u ** 3))
    return 0.5 * u * (1 + th), (u, th)


def gelu_bwd(dg, cache):
    u, th = cache
    dth = (1 - th ** 2) * _C * (1 + 3 * 0.044715 * u ** 2)
    return dg * (0.5 * (1 + th) + 0.5 * u * dth)


def _flat(x):
    return x.reshape(-1, x.shape[-1])


# ---------------------------------------------------------------------------------------- transformer block
def block_fwd(cfg, p, x, tp):
    B, S, h = x.shape
    d = cfg.head_dim
    hl = p["wqkv"].shape[1] // (3 * d)                    # heads held by this rank
    a, ln1 = ln_fwd(x, p["ln1_g"], p["ln1_b"])
    qkv = a @ p["wqkv"] + p["bqkv"]                       # column-parallel: no communication
    q, k, v = (qkv[..., i * hl * d:(i + 1) * hl * d].reshape(B, S, hl, d).transpose(0, 2, 1, 3) for i in range(3))
    sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)
    sc = sc + np.triu(np.full((S, S), -1e9), 1)           # causal mask
    sc = sc - sc.max(-1, keepdims=True)
    P = np.exp(sc)
    P /= P.sum(-1, keepdims=True)
    ctx = (P @ v).transpose(0, 2, 1, 3).reshape(B, S, hl * d)
    o = tp.all_reduce(ctx @ p["wo"]) + p["bo"]           # row-parallel: partial sums -> all-reduce
    x1 = x + o
    c, ln2 = ln_fwd(x1, p["ln2_g"], p["ln2_b"])
    u = c @ p["w1"] + p["b1"]                             # column-parallel
    g, ge = gelu_fwd(u)
    m = tp.all_reduce(g @ p["w2"]) + p["b2"]             # row-parallel + all-reduce
    y = x1 + m
    return y, (a, ln1, q, k, v, P, ctx, c, ln2, u, g, ge, hl)


def block_bwd(cfg, p, dy, cache, tp):
    a, ln1, q, k, v, P, ctx, c, ln2, u, g, ge, hl = cache
    B, S, h = dy.shape
    d = cfg.head_dim
    gr = {}
    # MLP
    gr["b2"] = _flat(dy).sum(0)
    gr["w2"] = _flat(g).T @ _flat(dy)
    du = gelu_bwd(dy @ p["w2"].T, ge)
    gr["w1"] = _flat(c).T @ _flat(du)
    gr["b1"] = _flat(du).sum(0)
    dc = tp.all_reduce(du @ p["w1"].T)                     # backward of the column-parallel input: all-reduce
    dx1, gr["ln2_g"], gr["ln2_b"] = ln_bwd(dc, ln2)
    dx1 = dx1 + dy
    # attention
    gr["bo"] = _flat(dx1).sum(0)
    gr["wo"] = _flat(ctx).T @ _flat(dx1)
    dctx = (dx1 @ p["wo"].T).reshape(B, S, hl, d).transpose(0, 2, 1, 3)
    dP = dctx @ v.transpose(0, 1, 3, 2)
    dv = P.transpose(0, 1, 3, 2) @ dctx
    dS = P * (dP - (dP * P).sum(-1, keepdims=True)) / np.sqrt(d)
    dq, dk = dS @ k, dS.transpose(0, 1, 3, 2) @ q
    dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, S, hl * d) for t in (dq, dk, dv)], -1)
    gr["wqkv"] = _flat(a).T @ _flat(dqkv)
    gr["bqkv"] = _flat(dqkv).sum(0)
    da = tp.all_reduce(dqkv @ p["wqkv"].T)
    dx, gr["ln1_g"], gr["ln1_b"] = ln_bwd(da, ln1)
    return dx + dx1, gr


# ---------------------------------------------------------------------------------------- embedding and loss
def emb_fwd(cfg, p, ids, tp):
    vl = p["emb"].shape[0]
    local = ids - tp.r * vl
    mask = (local >= 0) & (local < vl)
    out = p["emb"][np.where(mask, local, 0)] * mask[..., None]
    return tp.all_reduce(out), (local, mask)


def emb_bwd(cfg, p, dout, cache):
    local, mask = cache
    g = np.zeros_like(p["emb"])
    np.add.at(g, local[mask], dout[mask])
    return {"emb": g}


def head_fwd(cfg, p, x, targets, tp, scale):
    """Final LayerNorm, column-parallel logits and vocab-parallel cross-entropy.
    Returns (loss contribution, cache). The full logits are never gathered on one rank."""
    z, lnf = ln_fwd(x, p["lnf_g"], p["lnf_b"])
    logits = z @ p["wout"]                                     # (B, S, V/t)
    vl = logits.shape[-1]
    mx = tp.all_reduce(logits.max(-1), op="max")               # global max for a stable softmax
    e = np.exp(logits - mx[..., None])
    se = tp.all_reduce(e.sum(-1))                              # global sum of exp
    local_t = targets - tp.r * vl
    mask = (local_t >= 0) & (local_t < vl)
    tl = np.where(mask, np.take_along_axis(logits, np.where(mask, local_t, 0)[..., None], -1)[..., 0], 0.0)
    tl = tp.all_reduce(tl)                                     # the target's logit, from whichever rank owns it
    per_tok = np.log(se) + mx - tl
    loss = per_tok.mean() * scale
    return loss, (z, lnf, e, se, local_t, mask, scale, per_tok.size)


def head_bwd(cfg, p, cache, tp):
    z, lnf, e, se, local_t, mask, scale, n = cache
    dlog = e / se[..., None]
    idx = np.nonzero(mask)
    dlog[idx + (local_t[idx],)] -= 1.0
    dlog *= scale / n
    gr = {"wout": _flat(z).T @ _flat(dlog)}
    dz = tp.all_reduce(dlog @ p["wout"].T)
    dx, gr["lnf_g"], gr["lnf_b"] = ln_bwd(dz, lnf)
    return dx, gr
