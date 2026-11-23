"""Deterministic delayed-gradient simulator: every gradient is computed on the
weights as they were `delay` updates ago. Isolates the effect of staleness from
machine noise (the real cluster cannot choose its staleness)."""
from collections import deque

import numpy as np

from pskit.data import make_dataset
from pskit.model import logloss


def run(delay, lr=0.1, updates=1500, batch=128, opt="adagrad", seed=0, data=None, every=250):
    keys, y = data if data is not None else make_dataset(n=60_000, vocab=50_000, k=16, seed=seed)
    cut = int(len(y) * 0.85)
    ktr, ytr, kte, yte = keys[:cut], y[:cut], keys[cut:], y[cut:]
    V = int(keys.max()) + 1
    w, acc = np.zeros(V), np.zeros(V)
    hist = deque([w.copy()], maxlen=delay + 1)  # hist[0] = weights `delay` updates ago
    rng = np.random.default_rng(seed)
    curve = []
    for t in range(updates):
        idx = rng.integers(0, len(ytr), batch)
        kb, yb = ktr[idx].astype(np.int64), ytr[idx]
        stale = hist[0]
        s = np.clip(stale[kb].sum(1), -30, 30)
        g_ex = (1 / (1 + np.exp(-s)) - yb) / batch
        g = np.bincount(kb.ravel(), weights=np.repeat(g_ex, kb.shape[1]), minlength=V)
        if opt == "adagrad":
            acc += g * g
            w -= lr * g / np.sqrt(acc + 1e-8)
        else:
            w -= lr * g
        hist.append(w.copy())
        if (t + 1) % every == 0:
            curve.append(logloss(kte, yte, lambda u: w[u.astype(np.int64)]))
    return dict(delay=delay, lr=lr, test_loss=curve[-1], curve=curve)


SETTINGS = (("sgd", 0.5), ("sgd", 2.0), ("adagrad", 0.1))


def sweep(delays=(0, 1, 2, 4, 8, 16, 32, 64), settings=SETTINGS, **kw):
    """Final test loss for every (optimizer, learning rate) and delay."""
    data = make_dataset(n=60_000, vocab=50_000, k=16, seed=kw.get("seed", 0))
    return [dict(run(d, lr=lr, opt=opt, data=data, **kw), opt=opt) for opt, lr in settings for d in delays]
