"""Pipeline schedules (GPipe and 1F1B), an event simulator for them, and the bubble formula.

A schedule is, for every stage, the ordered list of operations it runs: ("F", i) is the forward
pass of micro-batch i and ("B", i) its backward pass. The executor in engine.py runs exactly
these lists on real processes; the simulator here computes the ideal timeline.
"""


def gpipe(p, m):
    """All forwards, then all backwards (in reverse order). Every stage holds m activations."""
    return [[("F", i) for i in range(m)] + [("B", i) for i in reversed(range(m))] for _ in range(p)]


def one_f_one_b(p, m):
    """PipeDream-Flush / Megatron 1F1B: warm up with p-s-1 forwards, then alternate F and B, then drain.
    Stage s holds at most p-s activations at once instead of m."""
    sched = []
    for s in range(p):
        warm = min(p - s - 1, m)
        ops = [("F", i) for i in range(warm)]
        f, b = warm, 0
        while f < m:
            ops.append(("F", f)); f += 1
            ops.append(("B", b)); b += 1
        while b < m:
            ops.append(("B", b)); b += 1
        sched.append(ops)
    return sched


SCHEDULES = {"gpipe": gpipe, "1f1b": one_f_one_b}


def check_schedule(sched, m):
    """Every stage runs each micro-batch's F and B exactly once, F before B."""
    for ops in sched:
        assert sorted(ops) == sorted([("F", i) for i in range(m)] + [("B", i) for i in range(m)])
        seen = set()
        for kind, i in ops:
            if kind == "B":
                assert i in seen, "backward before forward"
            seen.add(i)
    return True


def peak_in_flight(ops):
    """Largest number of micro-batches whose activations a stage holds at the same time."""
    live = peak = 0
    for kind, _ in ops:
        live += 1 if kind == "F" else -1
        peak = max(peak, live)
    return peak


def simulate(sched, tf=1.0, tb=2.0, comm=0.0):
    """Ideal timeline. F(s, i) needs F(s-1, i); B(s, i) needs B(s+1, i) (and, on the last stage, F(s, i)).
    Returns (makespan, bubble fraction, per-stage list of (op, start, end))."""
    p = len(sched)
    done = {}
    pos = [0] * p
    free = [0.0] * p
    timeline = [[] for _ in range(p)]
    remaining = sum(len(o) for o in sched)
    while remaining:
        progressed = False
        for s in range(p):
            if pos[s] == len(sched[s]):
                continue
            kind, i = sched[s][pos[s]]
            if kind == "F":
                dep = done.get(("F", s - 1, i)) if s > 0 else 0.0
                dep = None if dep is None else dep + (comm if s > 0 else 0.0)
            else:
                dep = done.get(("B", s + 1, i)) if s < p - 1 else done.get(("F", s, i))
                dep = None if dep is None else dep + (comm if s < p - 1 else 0.0)
            if dep is None:
                continue
            start = max(free[s], dep)
            end = start + (tf if kind == "F" else tb)
            done[(kind, s, i)] = end
            free[s] = end
            timeline[s].append(((kind, i), start, end))
            pos[s] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("schedule deadlocks")
    makespan = max(free)
    m = len(sched[0]) // 2
    busy = m * (tf + tb)
    return makespan, 1 - busy / makespan, timeline


def bubble_formula(p, m):
    """Idle fraction of a pipeline flush with equal stages: (p - 1) / (m + p - 1)."""
    return (p - 1) / (m + p - 1)


def split_layers(layers, p):
    """Contiguous, as even as possible: stage s gets layers [start, end)."""
    base, extra = divmod(layers, p)
    out, start = [], 0
    for s in range(p):
        n = base + (1 if s < extra else 0)
        out.append((start, start + n))
        start += n
    return out


def render(timeline, unit=1.0, width=None):
    """ASCII Gantt chart: digits are forward micro-batches, letters a, b, c... their backwards, '.' is idle."""
    end = max(t[-1][2] for t in timeline if t)
    cols = int(round(end / unit))
    rows = []
    for s, ops in enumerate(timeline):
        line = ["."] * cols
        for (kind, i), a, b in ops:
            ch = str(i % 10) if kind == "F" else chr(ord("a") + i % 26)
            for c in range(int(round(a / unit)), int(round(b / unit))):
                line[c] = ch
        rows.append(f"stage {s} |" + "".join(line) + "|")
    return "\n".join(rows)
