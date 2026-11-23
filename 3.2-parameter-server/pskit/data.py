"""Synthetic click-through-style data: sparse binary features with a power-law
(a few very common keys, a long tail of rare ones), labels from a hidden logistic model."""
import numpy as np


def make_dataset(n=200_000, vocab=100_000, k=16, alpha=1.1, seed=0):
    """Returns (keys [n, k] uint64, labels [n] float32). Key `vocab` is a bias feature in every row."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** alpha
    p /= p.sum()
    perm = rng.permutation(vocab)  # popular keys are not simply the small numbers
    feats = perm[rng.choice(vocab, size=(n, k - 1), p=p)]
    keys = np.concatenate([feats, np.full((n, 1), vocab)], axis=1).astype(np.uint64)
    w_true = rng.normal(0, 1.0, vocab + 1) * (rng.random(vocab + 1) < 0.3)  # 30% of features matter
    w_true[vocab] = -0.5
    logits = w_true[keys.astype(np.int64)].sum(1)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    return keys, y


def dense_dataset(n=50_000, vocab=4_096, k=16, seed=0):
    """'Dense' workload: a small, uniformly used vocabulary, so a batch touches almost every key."""
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, vocab, size=(n, k)).astype(np.uint64)
    w_true = rng.normal(0, 0.5, vocab)
    y = (rng.random(n) < 1 / (1 + np.exp(-w_true[keys.astype(np.int64)].sum(1)))).astype(np.float32)
    return keys, y
