"""Benchmarks: every fault scenario, detection vs lease TTL, lost work vs checkpoint interval."""
import json
import os
import statistics

from .analysis import fmt
from .cli import SCENARIOS, run_scenario


def run_bench(workdir, out):
    res = {"scenarios": {}, "lease": [], "interval": []}
    print("# 1. fault scenarios (3 nodes x 1 worker, min 2, max 3, lease 1.5 s, checkpoint every 20 steps, 300 steps)")
    for name in SCENARIOS:
        rep = run_scenario(name, os.path.join(workdir, name), quiet=True)
        res["scenarios"][name] = rep
        print(fmt(rep, name))
    print("\n# 2. detection and recovery vs lease TTL (kill -9 a whole node at step 120; heartbeat = TTL / 6)")
    print(f"{'lease_s':>8} {'detect_s':>9} {'new_round_s':>12} {'training_again_s':>17}")
    for ttl in (0.75, 1.5, 3.0):
        rep = run_scenario("kill-node", os.path.join(workdir, f"lease{ttl}"), quiet=True, lease_ttl=ttl, hb_interval=ttl / 6)
        f = rep["faults"][0]
        res["lease"].append({"lease_s": ttl, **f})
        print(f"{ttl:>8} {f['detect_s']:>9.2f} {f['rendezvous_s']:>12.2f} {f['recover_s']:>17.2f}")
    print("\n# 3. lost work vs checkpoint interval (kill -9 one worker at steps 137 and 163; 200 steps)")
    print(f"{'every':>6} {'lost@137':>9} {'lost@163':>9} {'mean_lost':>10} {'ckpt_ms':>8} {'wall_s':>7}")
    for every in (1, 5, 10, 25, 50):
        lost, ck, wall = [], [], []
        for at in (137, 163):
            rep = run_scenario("kill-worker", os.path.join(workdir, f"k{every}-{at}"), quiet=True, at_step=at, steps=200, ckpt_every=every)
            lost.append(rep["faults"][0]["lost_steps"])
            ck.append(rep["ckpt_ms_mean"])
            wall.append(rep["wall_s"])
        row = {"every": every, "lost": lost, "mean_lost": statistics.mean(lost), "ckpt_ms": round(statistics.mean(ck), 2),
               "wall_s": round(statistics.mean(wall), 2)}
        res["interval"].append(row)
        print(f"{every:>6} {lost[0]:>9} {lost[1]:>9} {row['mean_lost']:>10.1f} {row['ckpt_ms']:>8.2f} {row['wall_s']:>7.2f}")
    digests = {r["digest"] for r in res["scenarios"].values()}
    print(f"\nfinal weights identical in all {len(res['scenarios'])} scenarios: {len(digests) == 1} ({digests.pop()})")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res
