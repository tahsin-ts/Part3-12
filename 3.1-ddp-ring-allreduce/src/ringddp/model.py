"""A NumPy MLP (float64) whose backward pass reports each gradient as soon as it is ready."""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.params = []                                   # [W1, b1, W2, b2, ...]
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.params.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.params.append(np.zeros(fan_out))
        self.grads = [np.zeros_like(p) for p in self.params]

    def n_params(self):
        return sum(p.size for p in self.params)

    def forward(self, x):
        self.acts = [x]                                    # saved for the backward pass
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            z = self.acts[-1] @ self.params[2 * i] + self.params[2 * i + 1]
            self.acts.append(np.tanh(z) if i < n_layers - 1 else z)
        return self.acts[-1]

    def loss_and_backward(self, x, y, on_grad=None):
        """Mean softmax cross-entropy. Fills self.grads from the LAST layer to the first and
        calls on_grad(i) right after grads[i] is written (this is where DDP starts communication)."""
        logits = self.forward(x)
        z = logits - logits.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        loss = -logp[np.arange(len(y)), y].mean()
        d = np.exp(logp)
        d[np.arange(len(y)), y] -= 1.0
        d /= len(y)                                        # gradient of the MEAN loss
        for i in reversed(range(len(self.params) // 2)):
            a = self.acts[i]
            self.grads[2 * i + 1][...] = d.sum(axis=0)
            if on_grad:
                on_grad(2 * i + 1)
            self.grads[2 * i][...] = a.T @ d
            if on_grad:
                on_grad(2 * i)
            if i:
                d = (d @ self.params[2 * i].T) * (1.0 - a * a)   # tanh' = 1 - tanh^2
        return float(loss)


class SGD:
    """SGD with momentum, same update rule as torch.optim.SGD: v = mu v + g;  p -= lr v."""

    def __init__(self, params, lr=0.1, momentum=0.9):
        self.params, self.lr, self.mu = params, lr, momentum
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.v):
            v *= self.mu
            v += g
            p -= self.lr * v
