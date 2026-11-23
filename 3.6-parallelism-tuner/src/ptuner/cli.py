"""ptuner: pick DP x TP x PP x ZeRO x micro-batch x recomputation for a transformer."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

from .config import CLUSTERS, GIB, MODELS, Knobs, Model, get_cluster, get_model, load_cluster
from . import explain, pipeline_sim, search, validate


def _model(a) -> Model:
    m = get_model(a.model) if a.model else Model("custom", a.layers, a.hidden, a.heads)
    return replace(m, **{k: v for k, v in (("seq", a.seq), ("vocab", a.vocab)) if v})


def _knobs(a) -> Knobs:
    return replace(Knobs(), efficiency=a.efficiency, dp_overlap=a.overlap)


def cmd_search(a) -> int:
    m = _model(a)
    c = load_cluster(a.cluster_file) if a.cluster_file else get_cluster(a.cluster, a.gpus)
    res = search.search(m, c, a.batch, _knobs(a), headroom=a.headroom)
    if a.json:
        json.dump([{"plan": asdict(x.plan), "mem_gib": x.mem.total / GIB, "step_s": x.step.total,
                    "tokens_per_s_per_gpu": x.step.tokens_per_s_per_gpu, "mfu": x.step.mfu} for x in res.ranked[:a.top]],
                  sys.stdout, indent=1)
        print()
        return 0 if res.best else 1
    print(f"model {m.name}: {m.params() / 1e9:.2f} B params, L={m.layers} h={m.hidden} a={m.heads} s={m.seq} V={m.vocab}")
    print(f"cluster {c.name}: {c.gpus} GPUs ({c.gpus_per_node}/node), {c.mem_gib:g} GiB, {c.peak_tflops:g} TFLOP/s, "
          f"intra {c.intra_gbs:g} GB/s, inter {c.inter_gbs:g} GB/s   [PROJECTED: model, not measured]")
    print(f"global batch {a.batch} sequences = {a.batch * m.seq:,} tokens per step\n")
    if not res.best:
        print("\n".join(explain.reasons(res, m, c)))
        return 1
    print(explain.plan_table(res.ranked, c.mem_gib, a.top))
    print("\nwhy #1:")
    for r in explain.reasons(res, m, c):
        print(" -", r)
    return 0


def cmd_validate(a) -> int:
    k = _knobs(a)
    rows = validate.megatron_rows(k)
    print(f"Megatron-LM (Narayanan et al. 2021) Table 1, DGX A100; model efficiency={k.efficiency:.3f}")
    print(f"{'model':<10} {'tp':>3} {'pp':>3} {'dp':>4} {'bubble':>6} {'mem GiB':>7} {'published':>9} {'model':>7} {'error':>7}")
    for r in rows:
        print(f"{r.name:<10} {r.plan.tp:>3} {r.plan.pp:>3} {r.plan.dp:>4} {100 * r.bubble:>5.1f}% {r.mem_gib:>7.1f} "
              f"{r.published:>9.0f} {r.predicted:>7.1f} {r.error_pct:>+6.1f}%")
    print(f"mean absolute error: {validate.mape(rows):.1f}%")
    if a.loo:
        loo = validate.leave_one_out(k)
        print("leave-one-out (fit efficiency on 9 rows, predict the 10th):")
        print("  " + "  ".join(f"{n} {e:+.1f}%" for n, _, e in loo))
        print(f"  held-out mean absolute error: {sum(abs(e) for _, _, e in loo) / len(loo):.1f}%; "
              f"fitted efficiency {min(f for _, f, _ in loo):.3f}-{max(f for _, f, _ in loo):.3f}")
    print("ZeRO (Rajbhandari et al. 2020) Figure 1, 7.5B params, 64 GPUs, model state GB per GPU:")
    for st, pub, mod in validate.zero_check():
        print(f"  stage {st}: published {pub:6.1f}  model {mod:6.2f}")
    return 0


def cmd_bubble(a) -> int:
    print(f"{'p':>3} {'m':>4} {'formula':>8} {'1F1B sim':>9} {'GPipe sim':>9} {'1F1B peak mb':>12} {'GPipe peak mb':>13}")
    for p in a.p:
        for m in a.m:
            f = pipeline_sim.simulate(p, m, a.tf, 2 * a.tf, "1f1b")
            g = pipeline_sim.simulate(p, m, a.tf, 2 * a.tf, "gpipe")
            print(f"{p:>3} {m:>4} {pipeline_sim.formula(p, m):>8.4f} {f.bubble_fraction:>9.4f} {g.bubble_fraction:>9.4f} "
                  f"{f.peak_in_flight[0]:>12} {g.peak_in_flight[0]:>13}")
    return 0


def cmd_list(a) -> int:
    print("models:")
    for m in MODELS.values():
        print(f"  {m.name:<10} {m.params() / 1e9:8.1f} B  L={m.layers:<4} h={m.hidden:<6} a={m.heads}")
    print("clusters (effective bandwidths are assumptions):")
    for c in CLUSTERS.values():
        print(f"  {c.name:<10} {c.gpus} x {c.mem_gib:g} GiB, {c.peak_tflops:g} TFLOP/s, intra {c.intra_gbs:g} GB/s, inter {c.inter_gbs:g} GB/s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ptuner", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--efficiency", type=float, default=Knobs().efficiency, help="fraction of peak for GEMMs")
    common.add_argument("--overlap", type=float, default=Knobs().dp_overlap, help="fraction of DP sync hidden")
    s = sub.add_parser("search", parents=[common], help="rank parallel layouts")
    s.add_argument("--model", choices=sorted(MODELS))
    s.add_argument("--layers", type=int); s.add_argument("--hidden", type=int); s.add_argument("--heads", type=int)
    s.add_argument("--seq", type=int); s.add_argument("--vocab", type=int)
    s.add_argument("--cluster", default="dgx-a100", choices=sorted(CLUSTERS)); s.add_argument("--cluster-file")
    s.add_argument("--gpus", type=int); s.add_argument("--batch", type=int, default=512)
    s.add_argument("--headroom", type=float, default=0.95, help="usable fraction of GPU memory")
    s.add_argument("--top", type=int, default=8); s.add_argument("--json", action="store_true")
    v = sub.add_parser("validate", parents=[common], help="compare with published results")
    v.add_argument("--loo", action="store_true", help="leave-one-out calibration of the efficiency")
    b = sub.add_parser("bubble", help="simulate 1F1B and GPipe, compare with (p-1)/(m+p-1)")
    b.add_argument("--p", type=int, nargs="+", default=[2, 4, 8]); b.add_argument("--m", type=int, nargs="+", default=[4, 8, 32])
    b.add_argument("--tf", type=float, default=1.0)
    sub.add_parser("list", help="presets")
    a = ap.parse_args(argv)
    if a.cmd == "search" and not a.model and not (a.layers and a.hidden and a.heads):
        ap.error("give --model or --layers/--hidden/--heads")
    return {"search": cmd_search, "validate": cmd_validate, "bubble": cmd_bubble, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
