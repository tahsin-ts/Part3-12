"""Design study for a Llama-2-70B-like model: memory per GPU, communication per step, step time, MFU,
days to train, and checkpoint policy, for a choice of tensor (t), pipeline (p) and data (d) parallelism.

Everything here is an analytical MODEL with stated assumptions (see HARDWARE and ASSUME). It projects;
it does not measure. Sources of the formulas:
  parameters: exact count of the Llama-2-70B architecture (h 8192, 80 layers, 64 heads, 8 KV heads, ffn 28672)
  FLOPs per token: 6 N + 12 L h s (Chowdhery et al. 2022, PaLM, appendix B)
  activations: 34 s b h / t bytes per layer with sequence parallelism and selective recomputation
               (Korthikanti et al. 2022), x p micro-batches in flight on stage 0 under 1F1B
  bubble: (p - 1) / (v m) of the compute time for 1F1B with v interleaved chunks (Narayanan et al. 2021)
  checkpoint interval: Young/Daly, sqrt(2 x checkpoint stall x MTBF)
"""
import math
from dataclasses import dataclass, replace

GiB = 1024 ** 3


@dataclass(frozen=True)
class ModelSpec:
    name: str = "llama2-70b"
    hidden: int = 8192
    layers: int = 80
    heads: int = 64
    kv_heads: int = 8
    ffn: int = 28672
    vocab: int = 32000
    seq: int = 4096

    @property
    def layer_params(self):
        h, kv = self.hidden, self.hidden * self.kv_heads // self.heads
        return 2 * h * h + 2 * h * kv + 3 * h * self.ffn + 2 * h      # q,o + k,v + SwiGLU + 2 RMSNorm

    @property
    def embed_params(self):
        return self.vocab * self.hidden

    @property
    def params(self):
        return self.layers * self.layer_params + 2 * self.embed_params + self.hidden   # untied in/out + final norm

    def flops_per_token(self):
        return 6 * self.params + 12 * self.layers * self.hidden * self.seq


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_tflops: float      # dense BF16 Tensor Core peak (datasheet)
    hbm_gib: float
    nvlink_gbs: float       # per GPU, one direction
    net_gbs: float          # per GPU, one direction (InfiniBand / RoCE NIC per GPU)
    gpus_per_node: int = 8


HARDWARE = {
    "h100": Hardware("H100 SXM 80GB", 989.4, 80, 450, 50),     # NVLink 4: 900 GB/s bidirectional; 1 x 400 Gb/s NIC per GPU
    "a100": Hardware("A100 SXM 80GB", 312.0, 80, 300, 25),     # NVLink 3: 600 GB/s bidirectional; 1 x 200 Gb/s NIC per GPU
}

ASSUME = {
    "kernel_eff": 0.50,     # fraction of peak the kernels reach; calibrated so A100 x 2048 lands near Llama 2's published GPU-hours
    "nccl_eff": 0.75,       # achieved bus bandwidth / link bandwidth for large collectives
    "dp_overlap": 0.8,      # share of data-parallel communication hidden behind the backward pass
    "act_coef": 34,         # bytes per s*b*h per layer (Korthikanti; Llama SwiGLU+GQA count is similar)
    "reserve_gib": 6.0,     # CUDA context, NCCL buffers, fragmentation, temporaries
    "usable": 0.95,         # plan to at most 95% of HBM
    "global_batch": 1024,   # sequences of 4096 tokens = 4.19 M tokens per step (Llama 2 setting)
    "micro_batch": 1,
}


@dataclass(frozen=True)
class Plan:
    t: int
    p: int
    d: int
    v: int = 1                  # interleaved virtual stages per GPU
    recompute: str = "selective"  # selective (flash attention, nothing extra) or full

    @property
    def n(self):
        return self.t * self.p * self.d


def memory(ms, hw, pl, a=ASSUME):
    """Bytes per GPU on the most loaded stage (stage 0: embedding + most activations)."""
    layers_per_stage = math.ceil(ms.layers / pl.p)
    params = (layers_per_stage * ms.layer_params + ms.embed_params) / pl.t
    weights = 2 * params                                  # bf16
    grads = 4 * params                                    # fp32 main gradients
    optim = 12 * params / pl.d                            # fp32 master + Adam m, v; distributed optimizer (ZeRO-1)
    sbh = ms.seq * a["micro_batch"] * ms.hidden
    per_layer = (a["act_coef"] * sbh / pl.t) if pl.recompute == "selective" else (2 * sbh / pl.t)
    in_flight = pl.p if pl.p > 1 else 1                   # 1F1B: stage 0 keeps p micro-batches
    acts = per_layer * layers_per_stage * in_flight
    if pl.v > 1:
        acts *= 1 + (pl.p - 1) / (pl.p * pl.v)            # interleaving keeps a little more on stage 0
    total = weights + grads + optim + acts + a["reserve_gib"] * GiB
    return {"params_per_gpu": params, "weights": weights, "grads": grads, "optimizer": optim,
            "activations": acts, "reserve": a["reserve_gib"] * GiB, "total": total, "fits": total <= a["usable"] * hw.hbm_gib * GiB}


def ring(size, n):
    return 0.0 if n <= 1 else 2 * (n - 1) / n * size


def step(ms, hw, pl, a=ASSUME):
    """Per-step time model and the bytes each GPU sends, by kind."""
    B, b = a["global_batch"], a["micro_batch"]
    m = B // (pl.d * b)                                   # micro-batches per pipeline per step
    tokens = B * ms.seq
    model_flops = ms.flops_per_token() * tokens
    extra = 4 / 3 if pl.recompute == "full" else 1.0      # full recompute repeats the forward pass
    t_comp = model_flops * extra / (pl.n * hw.peak_tflops * 1e12 * a["kernel_eff"])
    bubble = (pl.p - 1) / (pl.v * m) if pl.p > 1 else 0.0
    layers_per_stage = ms.layers / pl.p
    act_bytes = ms.seq * b * ms.hidden * 2                # one bf16 activation tensor
    ars = (6 if pl.recompute == "full" else 4) * layers_per_stage * m
    tp_bytes = ars * ring(act_bytes, pl.t)
    tp_bw = (hw.nvlink_gbs if pl.t <= hw.gpus_per_node else hw.net_gbs) * 1e9 * a["nccl_eff"]
    t_tp = tp_bytes / tp_bw
    pp_bytes = 2 * m * pl.v * act_bytes / pl.t if pl.p > 1 else 0.0   # send fwd + bwd; scatter/gather over t
    t_pp = 2 * (m + pl.p - 1) * pl.v * (act_bytes / pl.t) / (hw.net_gbs * 1e9 * a["nccl_eff"]) if pl.p > 1 else 0.0
    params = (math.ceil(ms.layers / pl.p) * ms.layer_params + ms.embed_params) / pl.t
    dp_bytes = ring(2 * params, pl.d)                     # bf16 grad reduce-scatter + param all-gather
    dp_in_node = pl.t * pl.d <= hw.gpus_per_node
    t_dp = dp_bytes / ((hw.nvlink_gbs if dp_in_node else hw.net_gbs) * 1e9 * a["nccl_eff"])
    total = t_comp * (1 + bubble) + t_tp + t_pp + (1 - a["dp_overlap"]) * t_dp
    mfu = model_flops / (pl.n * hw.peak_tflops * 1e12 * total)
    return {"m": m, "t_compute": t_comp, "bubble": bubble, "t_bubble": t_comp * bubble, "t_tp": t_tp, "t_pp": t_pp,
            "t_dp": t_dp, "t_dp_exposed": (1 - a["dp_overlap"]) * t_dp, "step_s": total, "mfu": mfu,
            "tp_bytes": tp_bytes, "pp_bytes": pp_bytes, "dp_bytes": dp_bytes, "tokens": tokens}


def days(ms, hw, pl, tokens_total, a=ASSUME):
    s = step(ms, hw, pl, a)
    seconds = tokens_total / s["tokens"] * s["step_s"]
    return seconds / 86400, seconds * pl.n / 3600


def candidates(ms, hw, n, a=ASSUME):
    """Every (t, p, d, v, recompute) that divides n, the layers and the global batch."""
    out = []
    for t in (1, 2, 4, 8, 16):
        for p in (1, 2, 4, 5, 8, 10, 16, 20):
            if n % (t * p) or ms.layers % p:
                continue
            d = n // (t * p)
            if a["global_batch"] % (d * a["micro_batch"]) or a["global_batch"] // d < p:
                continue
            for v in ((1, 2, 4) if p > 1 else (1,)):
                if (ms.layers // p) % v:
                    continue
                for rc in ("selective", "full"):
                    out.append(Plan(t, p, d, v, rc))
    return out


def best(ms, hw, n, a=ASSUME):
    """Fastest plan that fits in memory; also returns why the others lost."""
    scored, rejected = [], []
    for pl in candidates(ms, hw, n, a):
        mem = memory(ms, hw, pl, a)
        if not mem["fits"]:
            rejected.append((pl, f"needs {mem['total'] / GiB:.0f} GiB"))
            continue
        scored.append((step(ms, hw, pl, a)["step_s"], pl))
    scored.sort(key=lambda x: (x[0], x[1].t, x[1].p))
    return scored, rejected


def checkpoint_policy(ms, n, write_gbs=100.0, failures_per_gpu_day=419 / (54 * 16384), restart_min=10.0):
    """Checkpoint size, Young/Daly interval and the expected share of time lost to failures.
    Default failure rate: 419 unexpected interruptions in 54 days on 16,384 H100s (Llama 3 report, 2024)."""
    size = ms.params * (2 + 12)                           # bf16 weights + fp32 master, m, v
    stall = size / (write_gbs * 1e9)                      # seconds the job is paused (synchronous write)
    mtbf = 86400 / (failures_per_gpu_day * n)             # seconds between failures for the whole job
    tau = math.sqrt(2 * stall * mtbf)
    lost = stall / tau + tau / (2 * mtbf) + restart_min * 60 / mtbf
    return {"size_bytes": size, "stall_s": stall, "mtbf_h": mtbf / 3600, "interval_min": tau / 60,
            "lost_fraction": lost, "failures_per_day": failures_per_gpu_day * n}


def gib(x):
    return x / GiB
