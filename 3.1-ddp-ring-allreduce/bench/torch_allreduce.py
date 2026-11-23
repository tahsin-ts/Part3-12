"""Measure all-reduce time with torch.distributed (gloo on CPU, nccl on GPUs).

  torchrun --nproc_per_node 2 bench/torch_allreduce.py --backend gloo
  torchrun --nproc_per_node 8 bench/torch_allreduce.py --backend nccl --sizes-mb 100,1000,13480
"""
import argparse
import time

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    p.add_argument("--sizes-mb", default="1,10,100")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    dist.init_process_group(a.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", rank % torch.cuda.device_count()) if a.backend == "nccl" else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    dtype = getattr(torch, a.dtype)
    for mb in [float(s) for s in a.sizes_mb.split(",")]:
        n = int(mb * 1e6) // dtype.itemsize
        x = torch.ones(n, dtype=dtype, device=dev)
        dist.all_reduce(x)                                    # warm-up (and NCCL communicator setup)
        times = []
        for _ in range(a.reps):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        t = sorted(times)[len(times) // 2]
        busbw = 2 * (world - 1) / world * n * dtype.itemsize / t / 1e9
        if rank == 0:
            print(f"{a.backend} N={world} {mb:>9.1f} MB  {t * 1e3:9.3f} ms  busbw {busbw:7.2f} GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
