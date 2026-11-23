"""One training worker: shard the data, compute local gradients, all-reduce, update."""
import hashlib
import os
import time
from dataclasses import dataclass, replace

import numpy as np

from .ddp import DistributedDataParallel
from .model import MLP, SGD
from .sampler import DistributedSampler
from .transport import CommError, RingLink


@dataclass
class Config:
    world: int = 2
    epochs: int = 3
    global_batch: int = 256
    samples: int = 8192
    features: int = 32
    hidden: int = 256
    layers: int = 2               # hidden layers
    classes: int = 10
    lr: float = 0.05
    momentum: float = 0.9
    seed: int = 0
    bucket_kb: int = 256          # 0 = one bucket for all gradients
    overlap: bool = True
    timeout: float = 30.0
    crash_rank: int = -1          # failure injection: this rank exits ...
    crash_step: int = -1          # ... at this global step
    hang_rank: int = -1           # failure injection: this rank stops responding at crash_step


def make_data(cfg):
    """Gaussian blobs: 10 classes in 32 dimensions; identical on every rank (same seed)."""
    rng = np.random.default_rng(cfg.seed + 1000)
    centers = rng.normal(0, 1.0, (cfg.classes, cfg.features))
    y = rng.integers(0, cfg.classes, cfg.samples)
    x = centers[y] + rng.normal(0, 1.5, (cfg.samples, cfg.features))
    return x, y


def train_worker(rank, addrs, listen_sock, cfg):
    if cfg.global_batch % cfg.world:
        raise ValueError(f"global batch {cfg.global_batch} is not divisible by {cfg.world} workers")
    x, y = make_data(cfg)
    link = RingLink(rank, cfg.world, addrs, cfg.timeout, listen_sock)
    model = MLP([cfg.features] + [cfg.hidden] * cfg.layers + [cfg.classes], seed=cfg.seed)   # same init everywhere
    ddp = DistributedDataParallel(model, link, cfg.bucket_kb * 1024, cfg.overlap)
    opt = SGD(model.params, cfg.lr, cfg.momentum)
    sampler = DistributedSampler(cfg.samples, cfg.world, rank, shuffle=True, seed=cfg.seed)
    local_b = cfg.global_batch // cfg.world
    losses, step, t_start = [], 0, time.perf_counter()
    try:
        for epoch in range(cfg.epochs):
            sampler.set_epoch(epoch)
            idx = sampler.indices()
            total = 0.0
            for s in range(len(idx) // local_b):
                if step == cfg.crash_step and rank == cfg.crash_rank:
                    os._exit(17)                                 # simulate a hard crash (no cleanup)
                if step == cfg.crash_step and rank == cfg.hang_rank:
                    time.sleep(3600)                             # simulate a hung worker
                batch = idx[s * local_b:(s + 1) * local_b]
                total += ddp.backward(x[batch], y[batch])
                opt.step(model.grads)
                step += 1
            losses.append(float(ddp.allreduce_mean([total / (len(idx) // local_b)])[0]))
        flat = np.concatenate([p.ravel() for p in model.params])
        return dict(rank=rank, world=cfg.world, losses=losses, steps=step,
                    digest=hashlib.sha256(flat.tobytes()).hexdigest()[:16],
                    params=flat if rank == 0 else None, n_params=flat.size, n_buckets=len(ddp.buckets),
                    bytes_sent=link.bytes_sent, bytes_recv=link.bytes_recv, wire_sent=link.wire_sent,
                    compute_s=ddp.compute_s, wait_s=ddp.wait_s, comm_s=ddp.comm_s,
                    total_s=time.perf_counter() - t_start)
    except CommError as e:
        raise type(e)(f"{e} (global step {step})") from None    # say WHERE it broke
    finally:
        ddp.close()
        link.close()


def train_reference(cfg):
    """The same training in ONE process with the full global batch: the ground truth."""
    return train_worker(0, None, None, replace(cfg, world=1))
