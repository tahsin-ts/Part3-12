"""Turn an event log into the numbers that matter: detection, recovery, lost work, goodput, fencing."""

DETECT = {"kill_worker": ("worker_failed",), "kill_node": ("node_lost",), "partition": ("node_lost", "self_fenced"),
          "pause": ("node_lost",), "slow": ("straggler_evicted",), "scale_up": ("node_joined",)}


def analyze(events):
    steps = [e for e in events if e["ev"] == "step"]
    start = min(e["t"] for e in events if e["ev"] == "round_closed")
    done = [e for e in events if e["ev"] == "train_done"]
    end = done[-1]["t"] if done else max(e["t"] for e in events)
    useful = len({e["step"] for e in steps})
    rep = {"wall_s": round(end - start, 3), "steps_run": len(steps), "unique_steps": useful,
           "redone_steps": len(steps) - useful, "generations": max([e["gen"] for e in events if e["ev"] == "round_closed"] or [0]),
           "worlds": [e["world"] for e in events if e["ev"] == "round_closed"],
           "fenced_exits": sum(e["ev"] == "fenced" for e in events),
           "fenced_requests": sum(e["ev"] == "fenced_request" for e in events), "digest": done[-1]["digest"] if done else None,
           "acc": done[-1]["acc"] if done else None, "faults": []}
    ck = [e for e in events if e["ev"] == "checkpoint"]
    rep["ckpt_ms_mean"] = round(sum(e["ms"] for e in ck) / len(ck), 2) if ck else None
    rep["ckpt_bytes"] = ck[0]["bytes"] if ck else None
    for f in (e for e in events if e["ev"] == "fault"):
        before = [e for e in steps if e["t"] <= f["t"]]
        g = before[-1]["gen"] if before else 1
        last_old = max([e["step"] for e in steps if e["gen"] == g] or [0])
        det = next((e for e in events if e["t"] >= f["t"] and e["ev"] in DETECT[f["kind"]]), None)
        closed = next((e for e in events if e["t"] >= f["t"] and e["ev"] == "round_closed" and e["gen"] > g), None)
        first = next((e for e in steps if e["gen"] > g), None)
        resume = next((e for e in events if e["ev"] == "resume" and e["gen"] > g), None)
        r = {"kind": f["kind"], "node": f["node"], "at_step": before[-1]["step"] if before else 0, "old_gen": g}
        if det:
            r["detect_s"] = round(det["t"] - f["t"], 3)
        if closed:
            r["new_gen"], r["new_world"], r["rendezvous_s"] = closed["gen"], closed["world"], round(closed["t"] - f["t"], 3)
        if first:
            r["recover_s"] = round(first["t"] - f["t"], 3)
        if resume:
            r["resume_step"], r["lost_steps"] = resume["step"], last_old + 1 - resume["step"]
        rep["faults"].append(r)
    # world size timeline: (gen, world, first step, last step)
    tl = {}
    for e in steps:
        a = tl.setdefault(e["gen"], [e["world"], e["step"], e["step"], 0.0])
        a[2] = e["step"]
        a[3] += e["ms"]
    rep["timeline"] = [(gen, w, a, b, round(ms / max(1, b - a + 1), 2)) for gen, (w, a, b, ms) in sorted(tl.items())]
    return rep


def fmt(rep, title=""):
    out = [f"== {title}" if title else "=="]
    out.append(f"wall {rep['wall_s']:.2f} s | steps run {rep['steps_run']} (unique {rep['unique_steps']}, redone {rep['redone_steps']}) | "
               f"generations {rep['generations']} worlds {rep['worlds']}")
    for gen, w, a, b, ms in rep["timeline"]:
        out.append(f"  gen {gen}: world {w}  steps {a:>3}-{b:<3}  {ms:6.2f} ms/step")
    for f in rep["faults"]:
        parts = [f"  fault {f['kind']} on {f['node']} at step {f['at_step']}:"]
        for k, lab in (("detect_s", "detected"), ("rendezvous_s", "new round"), ("recover_s", "training again")):
            if k in f:
                parts.append(f"{lab} +{f[k]:.2f}s")
        if "new_world" in f:
            parts.append(f"world {f['new_world']}")
        if "lost_steps" in f:
            parts.append(f"resumed at {f['resume_step']}, lost {f['lost_steps']} steps")
        out.append(" ".join(parts))
    out.append(f"fenced requests {rep['fenced_requests']}, fenced worker exits {rep['fenced_exits']} | checkpoint {rep['ckpt_ms_mean']} ms ({rep['ckpt_bytes']} B) | "
               f"final digest {rep['digest']} acc {rep['acc']}")
    return "\n".join(out)
