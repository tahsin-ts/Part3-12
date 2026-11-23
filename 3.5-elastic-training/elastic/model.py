"""A small NumPy MLP trained data-parallel with a world-size-invariant reduction.

The global batch is cut into a fixed number of chunks. Each rank computes the gradient
sum of the chunks it owns; everyone adds the chunks in chunk order. The result is
bit-for-bit the same for any world size, so a job that shrinks from 3 to 2 workers
after a failure ends with exactly the same weights as a job that never failed.
"""
import hashlib

import numpy as np

N_SAMPLES, N_FEATURES, HIDDEN = 4096, 16, 32
GLOBAL_BATCH, N_CHUNKS = 64, 8          # 8 chunks of 8 samples; any world size 1..8 works
CHUNK = GLOBAL_BATCH // N_CHUNKS


def make_data(seed=0):
    """Synthetic binary classification: a hidden two-layer teacher labels random points."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES))
    w = rng.standard_normal((N_FEATURES, 8))
    y = (np.tanh(x @ w).sum(axis=1) > 0).astype(np.float64)
    return x, y


def init_params(seed=0):
    rng = np.random.default_rng(seed + 1)
    return {"w1": rng.standard_normal((N_FEATURES, HIDDEN)) * 0.3, "b1": np.zeros(HIDDEN),
            "w2": rng.standard_normal(HIDDEN) * 0.3, "b2": np.zeros(1)}


NAMES = ("w1", "b1", "w2", "b2")


def flatten(p):
    return np.concatenate([p[k].ravel() for k in NAMES])


def unflatten(v, like):
    out, i = {}, 0
    for k in NAMES:
        n = like[k].size
        out[k] = v[i:i + n].reshape(like[k].shape)
        i += n
    return out


def batch_indices(step, seed=0):
    """Sample ids of global step `step`: a pure function of the step (no hidden RNG state).

    Resuming at step s therefore needs only s, whatever the world size was."""
    per_epoch = N_SAMPLES // GLOBAL_BATCH
    epoch, k = divmod(step, per_epoch)
    perm = np.random.default_rng(seed * 1000 + epoch).permutation(N_SAMPLES)
    return perm[k * GLOBAL_BATCH:(k + 1) * GLOBAL_BATCH]


def my_chunks(rank, world):
    """Round-robin chunk ownership: chunk c belongs to rank c % world."""
    return [c for c in range(N_CHUNKS) if c % world == rank]


def chunk_grad(p, x, y):
    """Sum (not mean) of per-sample gradients of the logistic loss, plus the loss sum."""
    h = np.tanh(x @ p["w1"] + p["b1"])
    z = h @ p["w2"] + p["b2"][0]
    prob = 1.0 / (1.0 + np.exp(-z))
    loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
    dz = prob - y                                   # dL/dz per sample
    g = {"w2": h.T @ dz, "b2": np.array([dz.sum()])}
    dh = np.outer(dz, p["w2"]) * (1.0 - h * h)
    g["w1"], g["b1"] = x.T @ dh, dh.sum(axis=0)
    return flatten(g), loss


def reduce_chunks(chunks):
    """chunks: {chunk_id: (grad_sum_vector, loss_sum)}. Fixed order -> world-size invariant."""
    total = np.zeros_like(chunks[0][0])
    loss = 0.0
    for c in range(N_CHUNKS):
        total = total + chunks[c][0]
        loss += chunks[c][1]
    return total / GLOBAL_BATCH, loss / GLOBAL_BATCH


class SGDMomentum:
    def __init__(self, n, lr=0.1, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, np.zeros(n)

    def step(self, theta, grad):
        self.v = self.beta * self.v + grad
        return theta - self.lr * self.v


def digest(theta):
    return hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest()[:16]


def accuracy(theta, like, x, y):
    p = unflatten(theta, like)
    z = np.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"][0]
    return float(np.mean((z > 0) == (y > 0.5)))
