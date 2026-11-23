"""Step-time model: compute at an efficiency factor + TP all-reduces + PP bubble and
point-to-point + DP gradient sync (alpha-beta ring) + optimizer step.

Rank order is TP fastest, then DP, then PP (Megatron default), so a group of size g whose
ranks are `stride` apart fits in one node when g * stride <= gpus_per_node.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Cluster, Knobs, Model, Plan
from .memory import OPT_B, stage_params


def forward_flops(m: Model, tokens: float) -> dict:
    """FLOPs of one forward pass over `tokens` tokens, split by kind."""
    l, h, s, V = m.layers, m.hidden, m.seq, m.vocab
    return {"gemm": 24 * tokens * l * h * h,          # QKV, projection, two MLP matmuls
            "attn": 4 * tokens * s * l * h,           # QK^T and scores x V
            "logits": 2 * tokens * h * V}


def model_flops(m: Model, tokens: float) -> float:
    """FLOPs the model needs per step (no recomputation): 3 x forward."""
    return 3 * sum(forward_flops(m, tokens).values())


def hardware_flops(m: Model, tokens: float, recompute: str) -> float:
    """FLOPs the GPUs actually execute, including recomputation."""
    f = forward_flops(m, tokens)
    extra = {"none": 0.0, "selective": f["attn"], "full": f["gemm"] + f["attn"]}[recompute]
    return model_flops(m, tokens) + extra


def ring_allreduce_s(nbytes: float, n: int, gbs: float, alpha_us: float) -> float:
    """alpha-beta cost of a ring all-reduce: 2(n-1) steps, each moving nbytes/n."""
    if n <= 1:
        return 0.0
    return 2 * (n - 1) / n * nbytes / (gbs * 1e9) + 2 * (n - 1) * alpha_us * 1e-6


def p2p_s(nbytes: float, gbs: float, alpha_us: float) -> float:
    return nbytes / (gbs * 1e9) + alpha_us * 1e-6


def link(c: Cluster, group: int, stride: int) -> tuple[float, float]:
    """(bandwidth, alpha) for a group of `group` ranks spaced `stride` apart."""
    if group * stride <= c.gpus_per_node:
        return c.intra_gbs, c.intra_alpha_us
    return c.inter_gbs, c.inter_alpha_us


@dataclass
class StepReport:
    compute: float       # pure compute on the busiest stage, all micro-batches
    tp_comm: float       # exposed tensor-parallel all-reduces
    pp_comm: float       # pipeline sends/receives
    bubble: float        # pipeline idle time
    dp_comm: float       # exposed data-parallel gradient sync
    optimizer: float
    total: float
    micro_batches: int
    bubble_fraction: float
    tokens_per_s_per_gpu: float
    mfu: float           # model FLOPs utilization
    hfu: float           # hardware FLOPs utilization (counts recomputation)
    tflops_per_gpu: float  # hardware TFLOP/s per GPU (Megatron convention)


def step_time(m: Model, c: Cluster, plan: Plan, global_batch: int, knobs: Knobs = Knobs()) -> StepReport:
    n_gpu = plan.dp * plan.tp * plan.pp
    mb = global_batch // (plan.dp * plan.mbs)              # micro-batches per pipeline per step
    tok_mb = plan.mbs * m.seq
    width = m.hidden / plan.tp                             # narrow side of the per-GPU GEMMs
    peak = c.peak_tflops * 1e12 * knobs.efficiency * width / (width + knobs.gemm_knee)
    # compute of one micro-batch on one stage: tensor-core FLOPs + memory-bound elementwise work
    t_mb = hardware_flops(m, tok_mb, plan.recompute) / (plan.tp * plan.pp) / peak
    passes = 3 + (1 if plan.recompute == "full" else 0)   # backward moves ~2x the forward's bytes
    elem = (knobs.elem_bytes * tok_mb * m.hidden + knobs.score_bytes * plan.mbs * m.heads * m.seq ** 2) / plan.tp
    t_mb += passes * elem * (m.layers / plan.pp) / (c.hbm_gbs * 1e9)
    # tensor parallel: 2 all-reduces forward, 2 backward, +2 when the forward is recomputed (per layer)
    bw, al = link(c, plan.tp, 1)
    act = tok_mb * m.hidden * 2                            # bf16 activation of one micro-batch
    n_ar = 4 + (2 if plan.recompute == "full" else 0)
    tp_mb = n_ar * (m.layers / plan.pp) * ring_allreduce_s(act, plan.tp, bw, al)
    # pipeline: one activation forward and one gradient backward per micro-batch, split over tp ranks
    pp_mb = 0.0
    if plan.pp > 1:
        bw, al = link(c, plan.pp, plan.tp * plan.dp)
        pp_mb = 2 * p2p_s(act / plan.tp, bw, al)
    per_mb = t_mb + tp_mb + pp_mb
    pipe = (mb + plan.pp - 1) * per_mb
    bubble = (plan.pp - 1) * per_mb
    # data parallel gradient sync over the stage's parameters (reduce-scatter + all-gather for ZeRO)
    params = stage_params(m, plan, first=True)
    bw, al = link(c, plan.dp, plan.tp)
    ar = ring_allreduce_s(params * knobs.grad_bytes, plan.dp, bw, al)   # one full all-reduce
    if plan.zero <= 1:
        dp = ar                                    # gradients accumulate locally, sync once per step
    elif plan.zero == 2:
        dp = mb * ar / 2 + ar / 2                  # reduce-scatter every micro-batch + all-gather weights once
    else:
        dp = mb * ar * 1.5                         # all-gather (fwd) + all-gather (bwd) + reduce-scatter per micro-batch
    dp_exposed = dp * (1 - knobs.dp_overlap)
    opt_params = params / plan.dp if plan.zero >= 1 else params
    opt = opt_params * (OPT_B + 2 + 2) * 2 / (c.hbm_gbs * 1e9)   # read + write all state once
    total = pipe + dp_exposed + opt
    tokens = global_batch * m.seq
    return StepReport(compute=mb * t_mb, tp_comm=mb * tp_mb, pp_comm=mb * pp_mb, bubble=bubble, dp_comm=dp_exposed,
                      optimizer=opt, total=total, micro_batches=mb,
                      bubble_fraction=(plan.pp - 1) / (mb + plan.pp - 1),
                      tokens_per_s_per_gpu=tokens / total / n_gpu,
                      mfu=model_flops(m, tokens) / (total * n_gpu * c.peak_tflops * 1e12),
                      hfu=hardware_flops(m, tokens, plan.recompute) / (total * n_gpu * c.peak_tflops * 1e12),
                      tflops_per_gpu=hardware_flops(m, tokens, plan.recompute) / (total * n_gpu) / 1e12)
