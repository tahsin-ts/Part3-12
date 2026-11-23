"""A tiny deterministic NumPy training loop (MLP + Adam) to checkpoint and resume.

Batches come from a generator seeded by the step number, so a run resumed from a
checkpoint must produce bit-identical weights to a run that never stopped.
"""
import numpy as np

from .state import make_state


class Trainer:
    def __init__(self, n_params=1_000_000, layers=4, batch=64, lr=1e-3, seed=0):
        self.state = make_state(n_params, seed, layers)
        self.layers, self.batch, self.lr = layers, batch, lr
        self.width = self.state["model.layer0.weight"].shape[0]
        self.teacher = np.random.default_rng(seed + 1).standard_normal((self.width, self.width), dtype=np.float32) / self.width ** 0.5

    @property
    def step(self):
        return int(self.state["optim.step"])

    def train_step(self):
        st, L = self.state, self.layers
        rng = np.random.default_rng(1000 + self.step)            # data depends only on the step
        x = rng.standard_normal((self.batch, self.width), dtype=np.float32)
        y = np.tanh(x @ self.teacher)
        hs = [x]
        for i in range(L):                                        # forward
            z = hs[-1] @ st[f"model.layer{i}.weight"] + st[f"model.layer{i}.bias"]
            hs.append(np.tanh(z) if i < L - 1 else z)
        g = 2 * (hs[-1] - y) / y.size                             # d(MSE)/d(out)
        loss = float(np.mean((hs[-1] - y) ** 2))
        t = self.step + 1
        for i in reversed(range(L)):                              # backward + Adam
            W = st[f"model.layer{i}.weight"]
            gW, gb = hs[i].T @ g, g.sum(0)
            if i > 0:
                g = (g @ W.T) * (1 - hs[i] ** 2)
            m, v = st[f"optim.layer{i}.weight.m"], st[f"optim.layer{i}.weight.v"]
            m *= 0.9; m += 0.1 * gW
            v *= 0.999; v += 0.001 * gW * gW
            W -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            st[f"model.layer{i}.bias"] -= self.lr * gb
        st["optim.step"][...] = t
        return loss
