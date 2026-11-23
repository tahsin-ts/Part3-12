"""Model + optimizer state, and how it is split (sharded) across ranks.

A rank owns a contiguous block of rows (dimension 0) of every large tensor.
Small tensors (fewer rows than ranks, or scalars) are owned whole by one rank.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Shard:
    """The part of one global tensor that one rank holds."""
    data: np.ndarray          # the local rows
    global_shape: tuple       # shape of the whole tensor
    row_offset: int           # first global row held here


def row_ranges(n_rows, world):
    """Split n_rows into `world` contiguous ranges; the first ones get one extra row."""
    base, extra = divmod(n_rows, world)
    out, start = [], 0
    for r in range(world):
        n = base + (1 if r < extra else 0)
        out.append((start, start + n))
        start += n
    return out


def owner_of_small(name, world):
    """Deterministic owner for a tensor too small to split (same on every rank)."""
    return sum(name.encode()) % world


def is_split(shape, world):
    return len(shape) >= 1 and shape[0] >= world


def shard_state(state, world):
    """Global {name: array} -> list of per-rank {name: Shard}."""
    ranks = [dict() for _ in range(world)]
    for name, arr in state.items():
        arr = np.asarray(arr)
        if is_split(arr.shape, world):
            for r, (a, b) in enumerate(row_ranges(arr.shape[0], world)):
                ranks[r][name] = Shard(arr[a:b], arr.shape, a)
        else:
            ranks[owner_of_small(name, world)][name] = Shard(arr, arr.shape, 0)
    return ranks


def unshard(rank_states):
    """Inverse of shard_state: per-rank shards -> global {name: array}."""
    full = {}
    for rs in rank_states:
        for name, sh in rs.items():
            if name not in full:
                full[name] = np.empty(sh.global_shape, dtype=sh.data.dtype)
            if sh.data.ndim == 0:
                full[name][...] = sh.data
            else:
                full[name][sh.row_offset:sh.row_offset + sh.data.shape[0]] = sh.data
    return full


def make_state(n_params=1_000_000, seed=0, layers=4):
    """A small MLP's weights + Adam state (m, v) + step counter, all float32.
    Bytes ~= 12 x n_params (weights, first moment, second moment)."""
    rng = np.random.default_rng(seed)
    width = max(8, int((n_params / layers) ** 0.5))
    state = {}
    for i in range(layers):
        w = rng.standard_normal((width, width), dtype=np.float32) * 0.02
        state[f"model.layer{i}.weight"] = w
        state[f"model.layer{i}.bias"] = np.zeros(width, np.float32)
        state[f"optim.layer{i}.weight.m"] = np.zeros_like(w)
        state[f"optim.layer{i}.weight.v"] = np.zeros_like(w)
    state["optim.step"] = np.array(0, dtype=np.int64)
    return state


def nbytes(state):
    return sum(np.asarray(a).nbytes for a in state.values())
