"""Turn a search result into tables and plain-English reasons."""
from __future__ import annotations

from .config import GIB, Cluster, Model
from .search import Candidate, SearchResult


def plan_table(cands: list[Candidate], cap_gib: float, top: int = 10) -> str:
    head = f"{'#':>2}  {'dp':>3} {'tp':>3} {'pp':>3} {'zero':>4} {'mbs':>3} {'recompute':<9} {'mem GiB':>8} {'step s':>7} {'bubble':>6} {'tok/s/GPU':>9} {'MFU':>5}"
    lines = [head, "-" * len(head)]
    for i, c in enumerate(cands[:top], 1):
        p, s = c.plan, c.step
        lines.append(f"{i:>2}  {p.dp:>3} {p.tp:>3} {p.pp:>3} {p.zero:>4} {p.mbs:>3} {p.recompute:<9} "
                     f"{c.mem.total / GIB:>8.1f} {s.total:>7.2f} {100 * s.bubble_fraction:>5.1f}% {s.tokens_per_s_per_gpu:>9,.0f} {100 * s.mfu:>4.1f}%")
    return "\n".join(lines)


def best_per_shape(res: SearchResult) -> list[Candidate]:
    """The best candidate for every (tp, pp) layout, ranked."""
    seen, out = set(), []
    for c in res.ranked:
        k = (c.plan.tp, c.plan.pp)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def breakdown(c: Candidate) -> list[tuple[str, float]]:
    s = c.step
    return [("compute", s.compute), ("tensor-parallel comm", s.tp_comm), ("pipeline send/recv", s.pp_comm),
            ("pipeline bubble", s.bubble), ("data-parallel sync (exposed)", s.dp_comm), ("optimizer step", s.optimizer)]


def reasons(res: SearchResult, m: Model, c: Cluster, tie_pct: float = 4.0) -> list[str]:
    b = res.best
    if b is None:
        return ["No configuration fits: add GPUs, use a GPU with more memory, or shorten the sequence."]
    p, s, out = b.plan, b.step, []
    state_gib = (b.mem.weights + b.mem.grads + b.mem.optimizer) / GIB
    full_state = m.params() * 16 / GIB
    out.append(f"Model state is {full_state:,.0f} GiB unsharded (16 B/param x {m.params() / 1e9:.1f} B params); "
               f"this plan keeps {state_gib:.1f} GiB per GPU (tp{p.tp} x pp{p.pp}, ZeRO-{p.zero} over dp{p.dp}).")
    out.append(f"Peak memory {b.mem.total / GIB:.1f} of {c.mem_gib:g} GiB: activations {b.mem.activations / GIB:.1f} GiB "
               f"with recompute={p.recompute}, micro-batch {p.mbs}.")
    tot = s.total
    parts = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in breakdown(b) if v / tot >= 0.005)
    out.append(f"Step {tot:.2f} s = {parts}.")
    shapes = best_per_shape(res)
    for alt in shapes[1:3]:
        loss = 100 * (1 - alt.score / b.score)
        why = []
        if alt.step.tp_comm > s.tp_comm * 1.2 + 1e-9:
            why.append("more tensor-parallel traffic")
        if alt.step.bubble > s.bubble * 1.2 + 1e-9:
            why.append("a bigger pipeline bubble")
        if alt.step.dp_comm > s.dp_comm * 1.2 + 1e-9:
            why.append("more gradient traffic")
        if alt.plan.recompute != p.recompute and alt.plan.recompute != "none":
            why.append(f"needs {alt.plan.recompute} recompute to fit")
        out.append(f"Next layout tp{alt.plan.tp} x pp{alt.plan.pp}: {loss:.1f}% slower"
                   + (f" ({', '.join(why)})." if why else "."))
    close = sum(1 for x in res.ranked[1:] if x.score >= b.score * (1 - tie_pct / 100))
    if close:
        out.append(f"{close} other plans are within {tie_pct:g}% (the model's validation error): "
                   "benchmark the top few for 20 steps before committing.")
    top = res.pruned.most_common(3)
    out.append(f"Examined {res.examined} combinations: {len(res.ranked)} feasible; pruned "
               + ", ".join(f"{n} ({k})" for k, n in top) + ".")
    return out
