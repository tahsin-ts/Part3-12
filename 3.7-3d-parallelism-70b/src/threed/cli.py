"""Command line: threed layout | tp-check | pipeline | train | failover | plan | demo"""
import argparse
import os
import shutil
import tempfile
import time

import numpy as np

from . import engine as E
from . import pipeline as PL
from . import planner as PN
from . import tensor_parallel as T
from .model import Config
from .topology import parse_grid


def cmd_layout(a):
    g = parse_grid(a.grid)
    print(f"grid tp={g.tp} pp={g.pp} dp={g.dp}  world={g.world}  ({a.gpus_per_node} GPUs per node)")
    print("rank  node  (pp, dp, tp)  tp-group          dp-group")
    for r in range(min(g.world, a.show)):
        print(f"{r:4d}  {g.node_of(r, a.gpus_per_node):4d}  {str(g.coords(r)):12s}  {str(g.tp_group(r)):16s}  {g.dp_group(r)}")
    if g.world > a.show:
        print(f"... ({g.world - a.show} more ranks)")
    for kind, info in g.describe(a.gpus_per_node).items():
        print(f"{kind}: {info['groups']} groups of {info['size']}, {info['inside_one_node']} inside one node")


def cmd_tp_check(a):
    h, f, b = 64, 256, 32
    y0, dx0, g0 = T.dense_reference(h, f, b, 0)
    print(f"MLP h={h} f={f} batch={b}, float64; column-parallel W1, row-parallel W2")
    print("  t   max|y - y_dense|   max|dx - dx_dense|   max|dW - dW_dense|   bytes sent per rank")
    for t in a.tp:
        res = E.launch(t, T.check_worker, (t, h, f, b, 0))
        ey = max(np.abs(r["y"] - y0).max() for r in res)
        ex = max(np.abs(r["dx"] - dx0).max() for r in res)
        w1 = np.concatenate([r["g"]["w1"] for r in res], 1)
        w2 = np.concatenate([r["g"]["w2"] for r in res], 0)
        ew = max(np.abs(w1 - g0["w1"]).max(), np.abs(w2 - g0["w2"]).max())
        print(f"{t:3d}   {ey:17.2e}   {ex:18.2e}   {ew:18.2e}   {res[0]['bytes'].get('tp', 0):,}")


def cmd_pipeline(a):
    p, m = a.stages, a.micro
    for name in ("gpipe", "1f1b"):
        sched = PL.SCHEDULES[name](p, m)
        span, bub, tl = PL.simulate(sched, 1, 2)
        print(f"{name}: p={p} m={m} simulated bubble {bub:.3f}  formula (p-1)/(m+p-1) = {PL.bubble_formula(p, m):.3f}  "
              f"peak activations held per stage {[PL.peak_in_flight(o) for o in sched]}")
        if a.chart:
            print(PL.render(tl))
        for lat in a.latency:
            print(f"   with p2p latency {lat:g} x forward time: simulated bubble {PL.simulate(sched, 1, 2, lat)[1]:.3f}")
    if a.measure:
        cfg = Config(layers=p)
        print(f"\nmeasured on {p} processes (each forward padded to {a.tf * 1000:.0f} ms, backward {2 * a.tf * 1000:.0f} ms):")
        print("schedule   m   makespan_s   measured_bubble   formula   peak_live(stage0)")
        for name in ("gpipe", "1f1b"):
            for mm in a.sweep:
                res = E.train(cfg, parse_grid(f"1x{p}x1"), 1, mm, 1, schedule=name, pad=(a.tf, 2 * a.tf), timeout=60)
                t0 = min(r["timing"][0]["t0"] for r in res)
                t1 = max(r["timing"][0]["t1"] for r in res)
                busy = sum(r["timing"][0]["busy"] for r in res) / p
                print(f"{name:8s} {mm:3d}   {t1 - t0:10.3f}   {1 - busy / (t1 - t0):15.3f}   {PL.bubble_formula(p, mm):7.3f}   "
                      f"{res[0]['timing'][0]['peak_live']:3d}")


def cmd_train(a):
    cfg = Config(layers=a.layers)
    g = parse_grid(a.grid)
    t = time.time()
    ref_l, ref = E.train_reference(cfg, a.steps, a.batch, lr=a.lr)
    print(f"reference (1 process, whole batch): {time.time() - t:.2f}s")
    t = time.time()
    res = E.train(cfg, g, a.steps, a.batch, a.micro, lr=a.lr, schedule=a.schedule, return_params=True)
    el = time.time() - t
    full = E.gather_full(cfg, g, [r for r in res if r["coords"][1] == 0])
    print(f"3D run tp={g.tp} pp={g.pp} dp={g.dp} ({g.world} processes, {a.schedule}, m={res[0]['m']} micro-batches): {el:.2f}s")
    for i in range(0, a.steps, max(1, a.steps // 6)):
        print(f"  step {i:3d}  loss 3D {res[0]['losses'][i]:.6f}   reference {ref_l[i]:.6f}")
    print(f"  final loss 3D {res[0]['losses'][-1]:.6f}   reference {ref_l[-1]:.6f}")
    print(f"max |loss diff| {max(abs(x - y) for x, y in zip(res[0]['losses'], ref_l)):.1e}   "
          f"max |param diff| {E.max_diff(ref, full):.1e}")
    print("bytes sent per step (rank: measured tp/pp/dp == formula):")
    for r in res:
        e = E.expected_bytes(cfg, g, r["rank"], a.micro, r["m"])
        got = {k: r["bytes"].get(k, 0) // a.steps for k in ("tp", "pp", "dp")}
        ok = all(got[k] == e[k] for k in got)
        print(f"  rank {r['rank']} {r['coords']}: tp {got['tp']:>9,} pp {got['pp']:>7,} dp {got['dp']:>7,}  {'match' if ok else 'MISMATCH'}")


def cmd_failover(a):
    cfg = Config()
    g = parse_grid(a.grid)
    d = tempfile.mkdtemp(prefix="ckpt-")
    try:
        ref_l, ref = E.train_reference(cfg, a.steps, 16)
        marker = os.path.join(d, "failed-once")
        res, restarts, log = E.train_with_restarts(cfg, g, a.steps, 16, 2, d, fail_at=(a.kill_rank, a.kill_step, marker),
                                                   return_params=True, timeout=5)
        for line in log:
            print(line)
        full = E.gather_full(cfg, g, [r for r in res if r["coords"][1] == 0])
        print(f"restarts: {restarts}; checkpoints on disk: {sorted(x for x in os.listdir(d) if x.startswith('step'))[-3:]}")
        print(f"max |param diff| vs uninterrupted reference: {E.max_diff(ref, full):.1e}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cmd_plan(a):
    ms, hw = PN.ModelSpec(), PN.HARDWARE[a.hw]
    print(f"{ms.name}: {ms.params:,} parameters, {ms.flops_per_token() / 1e9:.0f} GFLOP per token (train), "
          f"seq {ms.seq}, global batch {PN.ASSUME['global_batch']} sequences; {hw.name}; PROJECTED")
    print("GPUs   t  p   d  v  recompute  mem/GPU GiB  step s  bubble  MFU    days  GPU-hours   tp GB  pp GB  dp GB")
    for n in a.gpus:
        sc, _ = PN.best(ms, hw, n)
        if not sc:
            print(f"{n:5d}  nothing fits")
            continue
        _, pl = sc[0]
        s, m = PN.step(ms, hw, pl), PN.memory(ms, hw, pl)
        dd, gh = PN.days(ms, hw, pl, a.tokens)
        print(f"{n:5d} {pl.t:2d} {pl.p:2d} {pl.d:3d} {pl.v:2d}  {pl.recompute:9s}  {PN.gib(m['total']):11.1f}  {s['step_s']:6.2f}  "
              f"{s['bubble']:6.3f}  {s['mfu']:.3f}  {dd:6.1f}  {gh / 1e6:6.2f} M  {s['tp_bytes'] / 1e9:6.1f} {s['pp_bytes'] / 1e9:6.2f} {s['dp_bytes'] / 1e9:6.2f}")
    if a.explain:
        n = a.explain
        sc, rej = PN.best(ms, hw, n)
        print(f"\n{n} GPUs: top 5 of {len(sc)} plans that fit ({len(rej)} rejected for memory)")
        for t_, pl in sc[:5]:
            print(f"  t={pl.t} p={pl.p} d={pl.d} v={pl.v} {pl.recompute:9s} step {t_:.2f}s  mem {PN.gib(PN.memory(ms, hw, pl)['total']):.1f} GiB")
        for pl in (PN.Plan(8, 1, n // 8), PN.Plan(1, 1, n), PN.Plan(16, 1, n // 16)):
            if n % pl.n == 0 and pl.d >= 1:
                mem, s = PN.memory(ms, hw, pl), PN.step(ms, hw, pl)
                print(f"  for contrast t={pl.t} p={pl.p} d={pl.d}: {PN.gib(mem['total']):.0f} GiB/GPU ({'fits' if mem['fits'] else 'does NOT fit'}), "
                      f"step {s['step_s']:.2f}s")
        ck = PN.checkpoint_policy(ms, n)
        print(f"checkpoint {ck['size_bytes'] / 1e9:.0f} GB, stall {ck['stall_s']:.1f}s at 100 GB/s; job MTBF {ck['mtbf_h']:.1f} h "
              f"({ck['failures_per_day']:.2f} failures/day); Young/Daly interval {ck['interval_min']:.0f} min; "
              f"expected time lost {100 * ck['lost_fraction']:.1f}%")


def cmd_demo(a):
    cmd_layout(argparse.Namespace(grid="8x4x2", gpus_per_node=8, show=10))
    print()
    cmd_tp_check(argparse.Namespace(tp=[1, 2, 4]))
    print()
    cmd_train(argparse.Namespace(grid="2x2x2", steps=12, batch=16, micro=2, lr=0.5, schedule="1f1b", layers=4))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="threed", description="3D parallelism (tensor x pipeline x data) on CPU processes + 70B planner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("layout", help="rank -> (pp, dp, tp) and the process groups")
    s.add_argument("--grid", default="8x4x2", help="TPxPPxDP, e.g. 8x4x2")
    s.add_argument("--gpus-per-node", type=int, default=8)
    s.add_argument("--show", type=int, default=16)
    s.set_defaults(fn=cmd_layout)
    s = sub.add_parser("tp-check", help="tensor-parallel MLP vs the dense MLP")
    s.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8])
    s.set_defaults(fn=cmd_tp_check)
    s = sub.add_parser("pipeline", help="GPipe vs 1F1B: simulated and measured bubble")
    s.add_argument("--stages", type=int, default=4)
    s.add_argument("--micro", type=int, default=8)
    s.add_argument("--chart", action="store_true")
    s.add_argument("--measure", action="store_true")
    s.add_argument("--sweep", type=int, nargs="+", default=[1, 4, 8, 16])
    s.add_argument("--latency", type=float, nargs="*", default=[], help="also simulate a send latency (fraction of tf)")
    s.add_argument("--tf", type=float, default=0.02, help="forward time per op in seconds (backward = 2x)")
    s.set_defaults(fn=cmd_pipeline)
    s = sub.add_parser("train", help="3D-parallel training checked against a single process")
    s.add_argument("--grid", default="2x2x2")
    s.add_argument("--steps", type=int, default=12)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--micro", type=int, default=2)
    s.add_argument("--lr", type=float, default=0.5)
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--schedule", default="1f1b", choices=list(PL.SCHEDULES))
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("failover", help="kill a rank mid-run, restart from the checkpoint")
    s.add_argument("--grid", default="2x2x2")
    s.add_argument("--steps", type=int, default=6)
    s.add_argument("--kill-rank", type=int, default=5)
    s.add_argument("--kill-step", type=int, default=3)
    s.set_defaults(fn=cmd_failover)
    s = sub.add_parser("plan", help="70B design study (projected)")
    s.add_argument("--hw", default="h100", choices=list(PN.HARDWARE))
    s.add_argument("--gpus", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    s.add_argument("--tokens", type=float, default=2e12)
    s.add_argument("--explain", type=int, default=0, help="show the alternatives for this GPU count")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("demo", help="layout + tensor-parallel check + 3D training")
    s.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
