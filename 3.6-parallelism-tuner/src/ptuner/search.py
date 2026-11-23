"""Enumerate DP x TP x PP x ZeRO x micro-batch x recomputation, prune, rank."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .config import GIB, Cluster, Knobs, Model, Plan
from . import memory, timing

RECOMPUTE = ("none", "selective", "full")


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass
class Candidate:
    plan: Plan
    mem: memory.MemoryReport
    step: timing.StepReport

    @property
    def score(self) -> float:
        return self.step.tokens_per_s_per_gpu


@dataclass
class SearchResult:
    ranked: list[Candidate]
    pruned: Counter = field(default_factory=Counter)
    examined: int = 0

    @property
    def best(self) -> Candidate | None:
        return self.ranked[0] if self.ranked else None


def check(m: Model, c: Cluster, plan: Plan, global_batch: int) -> str | None:
    """Return why a plan is invalid (None if structurally valid)."""
    if m.heads % plan.tp:
        return "tp does not divide heads"
    if plan.tp > c.gpus_per_node:
        return "tp spans nodes"
    if m.layers % plan.pp:
        return "pp does not divide layers"
    if global_batch % (plan.dp * plan.mbs):
        return "batch not divisible by dp x mbs"
    if plan.zero >= 2 and plan.pp > 1:
        return "zero>=2 with pp>1 unsupported"
    if plan.zero >= 1 and plan.dp == 1:
        return "zero needs dp>1"
    return None


def search(m: Model, c: Cluster, global_batch: int, knobs: Knobs = Knobs(),
           mbs_options=(1, 2, 4, 8), zero_options=(0, 1, 2, 3), recompute_options=RECOMPUTE,
           headroom: float = 1.0) -> SearchResult:
    res = SearchResult([])
    cap = c.mem_gib * GIB * headroom
    for tp in divisors(c.gpus):
        for pp in divisors(c.gpus // tp):
            dp = c.gpus // (tp * pp)
            for zero in zero_options:
                for mbs in mbs_options:
                    for rc in recompute_options:
                        plan = Plan(dp, tp, pp, zero, mbs, rc)
                        res.examined += 1
                        why = check(m, c, plan, global_batch)
                        if why:
                            res.pruned[why] += 1
                            continue
                        mb = global_batch // (dp * mbs)
                        mem = memory.estimate(m, plan, mb, knobs)
                        if mem.total > cap:
                            res.pruned["out of memory"] += 1
                            continue
                        res.ranked.append(Candidate(plan, mem, timing.step_time(m, c, plan, global_batch, knobs)))
    # best throughput first; ties -> less memory
    res.ranked.sort(key=lambda x: (-round(x.score, 6), x.mem.total))
    return res
