"""Optional: the same training with torch DistributedDataParallel on gloo (CPU). Skipped without torch."""
import numpy as np
import pytest

from ringddp import MLP, DistributedSampler, launch
from ringddp.trainer import Config, make_data, train_reference

torch = pytest.importorskip("torch")
CFG = Config(world=2, epochs=2, samples=2048, global_batch=128, hidden=64)


def _torch_worker(rank, addrs, sock, cfg):
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    if sock is not None:
        sock.close()                                      # free the port; torch's rendezvous takes rank 0's
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://{addrs[0][0]}:{addrs[0][1]}", rank=rank, world_size=cfg.world)
    try:
        x, y = make_data(cfg)
        ours = MLP([cfg.features] + [cfg.hidden] * cfg.layers + [cfg.classes], seed=cfg.seed)
        mods = []
        for i in range(len(ours.params) // 2):
            w, b = ours.params[2 * i], ours.params[2 * i + 1]
            lin = torch.nn.Linear(*w.shape).double()
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(w.T.copy()))
                lin.bias.copy_(torch.from_numpy(b))
            mods += [lin, torch.nn.Tanh()]
        model = DDP(torch.nn.Sequential(*mods[:-1]))
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
        sampler = DistributedSampler(cfg.samples, cfg.world, rank, seed=cfg.seed)
        lb = cfg.global_batch // cfg.world
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        for epoch in range(cfg.epochs):
            sampler.set_epoch(epoch)
            idx = torch.from_numpy(sampler.indices())
            for s in range(len(idx) // lb):
                batch = idx[s * lb:(s + 1) * lb]
                opt.zero_grad()
                F.cross_entropy(model(xt[batch]), yt[batch]).backward()
                opt.step()
        out = []
        for m in model.module:
            if isinstance(m, torch.nn.Linear):
                out += [m.weight.detach().numpy().T.ravel(), m.bias.detach().numpy().ravel()]
        return np.concatenate(out)
    finally:
        dist.destroy_process_group()


def test_matches_torch_ddp_gloo():
    ref = train_reference(CFG)["params"]
    res = launch(_torch_worker, 2, (CFG,), timeout=300)
    assert np.abs(res[0] - ref).max() < 1e-9
    assert np.array_equal(res[0], res[1])
