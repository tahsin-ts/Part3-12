"""Per-GPU memory model for Megatron-style DP x TP x PP training with ZeRO.

Model state: mixed precision + Adam = 2 (bf16 weights) + 2 (grads) + 12 (fp32 master,
m, v) bytes per parameter (ZeRO paper, K = 12). Activations: Korthikanti et al. 2022.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import GIB, Knobs, Model, Plan

WEIGHT_B, GRAD_B, OPT_B = 2, 2, 12


def layer_params(m: Model) -> float:
    return 12 * m.hidden ** 2 + 13 * m.hidden


def stage_params(m: Model, plan: Plan, first: bool = True) -> float:
    """Parameters held by one GPU of the first (or last) pipeline stage."""
    n = layer_params(m) * (m.layers / plan.pp) / plan.tp
    emb = m.vocab * m.hidden / plan.tp           # word embedding (first) / output layer (last)
    if first:
        emb += m.seq * m.hidden                  # position embedding, not split
    return n + emb


def model_state_bytes(params: float, zero: int, dp: int) -> dict:
    """Bytes of weights, gradients and optimizer state per GPU after ZeRO sharding."""
    w, g, o = WEIGHT_B * params, GRAD_B * params, OPT_B * params
    if zero >= 1:
        o /= dp
    if zero >= 2:
        g /= dp
    if zero >= 3:
        w /= dp
    return {"weights": w, "grads": g, "optimizer": o}


def layer_activation_bytes(m: Model, mbs: int, tp: int, recompute: str, seq_parallel: bool = True) -> float:
    """Activation bytes stored for ONE layer and ONE micro-batch (Korthikanti et al. Table 2)."""
    s, b, h, a = m.seq, mbs, m.hidden, m.heads
    sbh = s * b * h
    attn = 5 * a * s / h                          # softmax scores, dropout mask, ...
    if recompute == "full":
        return 2 * sbh / (tp if seq_parallel else 1)     # only the layer input is kept
    if recompute == "selective":
        return sbh * 34 / tp if seq_parallel else sbh * (10 + 24 / tp)
    if seq_parallel or tp == 1:
        return sbh * (34 + attn) / tp
    return sbh * (10 + 24 / tp + attn / tp)


@dataclass
class MemoryReport:
    weights: float
    grads: float
    optimizer: float
    activations: float
    logits: float
    overhead: float

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.optimizer + self.activations + self.logits + self.overhead

    def gib(self) -> dict:
        d = {k: v / GIB for k, v in self.__dict__.items()}
        d["total"] = self.total / GIB
        return d


def in_flight(plan: Plan, micro_batches: int, stage: int = 0) -> int:
    """Micro-batches whose activations stage `stage` holds at the peak of 1F1B."""
    return min(plan.pp - stage, micro_batches)


def estimate(m: Model, plan: Plan, micro_batches: int, knobs: Knobs = Knobs()) -> MemoryReport:
    """Peak memory of the busiest GPU: the max of the first and the last pipeline stage."""
    sp = knobs.seq_parallel and plan.tp > 1
    per_layer = layer_activation_bytes(m, plan.mbs, plan.tp, plan.recompute, sp)
    layers = m.layers / plan.pp
    extra = 0.0
    if plan.recompute == "full":   # during backward one layer is rebuilt in full
        extra = layer_activation_bytes(m, plan.mbs, plan.tp, "none", sp)
    reports = []
    for first in (True, False):
        if plan.pp == 1 and not first:
            break
        stage = 0 if first else plan.pp - 1
        p = stage_params(m, plan, first=first)
        st = model_state_bytes(p, plan.zero, plan.dp)
        act = per_layer * layers * in_flight(plan, micro_batches, stage) + extra
        last = (not first) or plan.pp == 1
        logits = plan.mbs * m.seq * m.vocab / plan.tp * (2 + 4) if last else 0.0   # bf16 logits + fp32 for the loss
        reports.append(MemoryReport(st["weights"], st["grads"], st["optimizer"], act, logits, knobs.overhead_gib * GIB))
    return max(reports, key=lambda r: r.total)
