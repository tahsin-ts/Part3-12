"""Check the models against published numbers.

1. Throughput: Narayanan et al. 2021 (Megatron-LM, SC'21) Table 1, 10 GPT models on
   DGX A100 80 GB (Selene), seq 2048, vocab 51,200, full activation recomputation.
2. Model-state memory: Rajbhandari et al. 2020 (ZeRO) Figure 1, 7.5B parameters, 64 GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import GIB, Knobs, Plan, get_cluster, get_model
from . import memory, timing

# name, tp, pp, gpus, global batch, published TFLOP/s per GPU
MEGATRON_TABLE1 = [
    ("gpt-1.7b", 1, 1, 32, 512, 137), ("gpt-3.6b", 2, 1, 64, 512, 138), ("gpt-7.5b", 4, 1, 128, 512, 142),
    ("gpt-18b", 8, 1, 256, 1024, 135), ("gpt-39b", 8, 2, 512, 1536, 138), ("gpt-76b", 8, 4, 1024, 1792, 140),
    ("gpt-146b", 8, 8, 1536, 2304, 148), ("gpt-310b", 8, 16, 1920, 2160, 155), ("gpt-530b", 8, 35, 2520, 2520, 163),
    ("gpt-1t", 8, 64, 3072, 3072, 163),
]

# ZeRO paper, Figure 1: psi = 7.5e9, K = 12, Nd = 64 -> GB (1e9 bytes) per GPU
ZERO_FIG1 = {0: 120.0, 1: 31.4, 2: 16.6, 3: 1.9}


@dataclass
class Row:
    name: str
    plan: Plan
    published: float
    predicted: float
    bubble: float
    mem_gib: float

    @property
    def error_pct(self) -> float:
        return 100 * (self.predicted / self.published - 1)


def megatron_rows(knobs: Knobs = Knobs(), rows=MEGATRON_TABLE1) -> list[Row]:
    out = []
    for name, tp, pp, gpus, batch, pub in rows:
        m, c = get_model(name), get_cluster("dgx-a100", gpus)
        plan = Plan(gpus // (tp * pp), tp, pp, 1, 1, "full")   # mbs 1 (not stated in the paper), distributed optimizer
        r = timing.step_time(m, c, plan, batch, knobs)
        mem = memory.estimate(m, plan, r.micro_batches, knobs)
        out.append(Row(name, plan, pub, r.tflops_per_gpu, r.bubble_fraction, mem.total / GIB))
    return out


def mape(rows: list[Row]) -> float:
    return float(np.mean([abs(r.error_pct) for r in rows]))


def fit_efficiency(rows=MEGATRON_TABLE1, knobs: Knobs = Knobs(), grid=np.arange(0.40, 0.801, 0.005)) -> float:
    """One-parameter least squares on log error."""
    def loss(e):
        rs = megatron_rows(replace(knobs, efficiency=float(e)), rows)
        return sum(np.log(r.predicted / r.published) ** 2 for r in rs)
    return float(min(grid, key=loss))


def leave_one_out(knobs: Knobs = Knobs()) -> list[tuple[str, float, float]]:
    """For every row: fit the efficiency on the other 9, predict this one. -> (name, fitted eff, error %)."""
    out = []
    for i, row in enumerate(MEGATRON_TABLE1):
        rest = MEGATRON_TABLE1[:i] + MEGATRON_TABLE1[i + 1:]
        e = fit_efficiency(rest, knobs)
        r = megatron_rows(replace(knobs, efficiency=e), [row])[0]
        out.append((row[0], e, r.error_pct))
    return out


def zero_check() -> list[tuple[int, float, float]]:
    """(stage, published GB, model GB) for the ZeRO paper's Figure 1."""
    out = []
    for stage, pub in ZERO_FIG1.items():
        st = memory.model_state_bytes(7.5e9, stage, 64)
        out.append((stage, pub, sum(st.values()) / 1e9))
    return out
