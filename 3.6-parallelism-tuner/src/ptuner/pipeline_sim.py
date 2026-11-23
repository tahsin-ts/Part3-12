"""Discrete-event simulation of GPipe and 1F1B pipeline schedules (NumPy).

Each stage runs its operations in a fixed order; an operation starts when the stage is
free AND its dependency is done: F(i, j) needs F(i-1, j); B(i, j) needs B(i+1, j)
(the last stage's B needs its own F). The result checks the bubble formula
(p - 1) / (m + p - 1) and counts how many micro-batches each stage keeps in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def order_1f1b(stage: int, p: int, m: int) -> list[tuple[str, int]]:
    warm = min(p - stage - 1, m)
    ops = [("F", j) for j in range(warm)]
    f, b = warm, 0
    while b < m:
        if f < m:
            ops.append(("F", f)); f += 1
        ops.append(("B", b)); b += 1
    return ops


def order_gpipe(stage: int, p: int, m: int) -> list[tuple[str, int]]:
    return [("F", j) for j in range(m)] + [("B", j) for j in range(m)]


@dataclass
class SimResult:
    makespan: float
    busy: np.ndarray          # busy time per stage
    bubble_fraction: float    # idle / (p x makespan)
    peak_in_flight: np.ndarray  # max micro-batches holding activations, per stage
    start: dict               # (kind, stage, mb) -> start time


def simulate(p: int, m: int, tf=1.0, tb=2.0, schedule: str = "1f1b", comm: float = 0.0) -> SimResult:
    """tf, tb: scalars or arrays of shape (p, m) with per-op durations. comm: send time between stages."""
    tf = np.broadcast_to(np.asarray(tf, dtype=float), (p, m))
    tb = np.broadcast_to(np.asarray(tb, dtype=float), (p, m))
    orders = [(order_1f1b if schedule == "1f1b" else order_gpipe)(i, p, m) for i in range(p)]
    end_f = np.full((p, m), np.nan)
    end_b = np.full((p, m), np.nan)
    free = np.zeros(p)
    nxt = [0] * p
    start, live, peak = {}, np.zeros(p, int), np.zeros(p, int)
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for i in range(p):
            if nxt[i] == len(orders[i]):
                continue
            kind, j = orders[i][nxt[i]]
            if kind == "F":
                dep = 0.0 if i == 0 else end_f[i - 1, j] + comm
            else:
                dep = end_f[i, j] if i == p - 1 else end_b[i + 1, j] + comm
            if np.isnan(dep):
                continue                       # dependency not scheduled yet
            t0 = max(free[i], dep)
            dur = tf[i, j] if kind == "F" else tb[i, j]
            (end_f if kind == "F" else end_b)[i, j] = t0 + dur
            free[i] = t0 + dur
            start[(kind, i, j)] = t0
            live[i] += 1 if kind == "F" else -1
            peak[i] = max(peak[i], live[i])
            nxt[i] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("deadlock: schedule order violates dependencies")
    makespan = float(np.nanmax(end_b))
    busy = tf.sum(axis=1) + tb.sum(axis=1)
    return SimResult(makespan, busy, 1 - busy.sum() / (p * makespan), peak, start)


def formula(p: int, m: int) -> float:
    return (p - 1) / (m + p - 1)
