"""The same recipe with PyTorch's own elastic launcher (torchrun) and the gloo backend on CPU.

    torchrun --nnodes=1:2 --nproc-per-node=2 --max-restarts=2 --rdzv-backend=c10d \
             --rdzv-endpoint=localhost:29400 --rdzv-id=demo examples/torch_elastic.py --ckpt /tmp/tckpt

On the first attempt rank 1 crashes at step CRASH_AT; torchrun restarts the group and training resumes
from the last checkpoint. On GPUs use backend nccl and move model and data to cuda:LOCAL_RANK.
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/tckpt")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--crash-at", type=int, default=int(os.environ.get("CRASH_AT", 35)))
    a = ap.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0))
    torch.manual_seed(0)
    model = DDP(torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    path, start = os.path.join(a.ckpt, "latest.pt"), 0
    if os.path.exists(path):
        ck = torch.load(path, weights_only=True)
        model.module.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    print(f"[attempt {attempt}] rank {rank}/{world} resumes at step {start}", flush=True)
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(4096, 16, generator=g), torch.randint(0, 2, (4096, 1), generator=g).float()
    for step in range(start, a.steps):
        if attempt == 0 and rank == 1 and step == a.crash_at:
            os._exit(1)                                     # simulated hardware fault on the first attempt
        idx = torch.arange(step * 64, step * 64 + 64) % 4096
        mine = idx[rank::world]                             # re-shard the same global batch for any world size
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[mine]), y[mine])
        opt.zero_grad(); loss.backward(); opt.step()
        if rank == 0 and (step + 1) % a.every == 0:
            os.makedirs(a.ckpt, exist_ok=True)
            torch.save({"model": model.module.state_dict(), "opt": opt.state_dict(), "step": step + 1}, path + ".tmp")
            os.replace(path + ".tmp", path)                 # atomic: a crash never leaves half a checkpoint
    if rank == 0:
        print(f"[attempt {attempt}] done at step {a.steps}, loss {loss.item():.4f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
