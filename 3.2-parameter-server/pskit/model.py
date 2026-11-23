"""Sparse logistic regression: score = sum of the weights of the active keys."""
import numpy as np


def batch_grad(keys, y, pull):
    """keys [B, k], y [B]; pull(uniq) returns weights [len(uniq), 1].
    Returns (uniq keys, gradient per unique key, mean log-loss before the update)."""
    uniq, inv = np.unique(keys.ravel(), return_inverse=True)
    w = pull(uniq)[:, 0].astype(np.float64)
    s = np.clip(w[inv].reshape(keys.shape).sum(1), -30, 30)
    p = 1 / (1 + np.exp(-s))
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    g_ex = (p - y) / len(y)  # d loss / d score, averaged over the batch
    g = np.bincount(inv, weights=np.repeat(g_ex, keys.shape[1]), minlength=len(uniq))
    return uniq, g.astype(np.float32)[:, None], loss


def logloss(keys, y, weights_of):
    """Mean log-loss with weights_of(uniq) -> [len(uniq)] array."""
    uniq, inv = np.unique(keys.ravel(), return_inverse=True)
    s = np.asarray(weights_of(uniq), np.float64)[inv].reshape(keys.shape).sum(1)
    p = np.clip(1 / (1 + np.exp(-s)), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
