"""Is the GPU starved? Samples/s a GPU needs vs samples/s the input pipeline delivers.

needed  = peak_flops x mfu / train_flops_per_sample
supply  = per_core_rate x cores_per_gpu  (decode is CPU work and scales with cores, until I/O caps it)
GPU busy <= min(1, supply / needed)
"""
# dense BF16/FP16 Tensor Core peaks from the datasheets (no sparsity)
GPUS = {"A100": 312e12, "H100": 989e12, "L4": 121e12, "T4": 65e12}

# training FLOPs per sample = 3 x forward (forward + backward ~ 2x forward); MACs count as 2 FLOPs
MODELS = {
    "resnet50": {"unit": "image", "flops": 3 * 2 * 4.1e9, "mfu": 0.25, "bytes": 110_000},
    "vit-b16": {"unit": "image", "flops": 3 * 2 * 17.6e9, "mfu": 0.40, "bytes": 110_000},
    "llama-7b": {"unit": "token", "flops": 6 * 6.74e9, "mfu": 0.40, "bytes": 2},
    "gpt-1.3b": {"unit": "token", "flops": 6 * 1.3e9, "mfu": 0.40, "bytes": 2},
}


def needed_rate(gpu, model, mfu=None):
    m = MODELS[model]
    return GPUS[gpu] * (m["mfu"] if mfu is None else mfu) / m["flops"]


def starve(gpu, model, per_core, cores_per_gpu, mfu=None):
    need = needed_rate(gpu, model, mfu)
    supply = per_core * cores_per_gpu
    return {"gpu": gpu, "model": model, "unit": MODELS[model]["unit"], "needed_per_s": need,
            "supplied_per_s": supply, "gpu_busy_max": min(1.0, supply / need),
            "cores_needed": need / per_core, "read_MBps": need * MODELS[model]["bytes"] / 1e6}


def report(rows):
    out = [f"{'gpu':5} {'model':9} {'needed/s':>10} {'supplied/s':>10} {'GPU busy':>8} {'cores needed':>12} {'read MB/s':>9}"]
    for r in rows:
        out.append(f"{r['gpu']:5} {r['model']:9} {r['needed_per_s']:10,.0f} {r['supplied_per_s']:10,.0f} "
                   f"{r['gpu_busy_max']:8.0%} {r['cores_needed']:12.1f} {r['read_MBps']:9.1f}")
    return "\n".join(out)
