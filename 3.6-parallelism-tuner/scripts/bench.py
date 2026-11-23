"""Measure how fast the tuner searches (CPU wall time) and how fast the pipeline simulator runs."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ptuner import pipeline_sim, search  # noqa: E402
from ptuner.config import get_cluster, get_model  # noqa: E402


def best_of(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), out


print(f"{'case':<34} {'examined':>8} {'feasible':>8} {'time ms':>8} {'plans/s':>9}")
for model, gpus, batch in [("gpt3-13b", 8, 512), ("gpt3-175b", 64, 1536), ("gpt3-175b", 512, 1536), ("gpt-1t", 3072, 3072)]:
    m, c = get_model(model), get_cluster("dgx-a100", gpus)
    t, r = best_of(lambda: search.search(m, c, batch), 3)
    print(f"{model + ' on ' + str(gpus) + ' GPUs':<34} {r.examined:>8} {len(r.ranked):>8} {1000 * t:>8.1f} {r.examined / t:>9,.0f}")
print()
print(f"{'pipeline simulation':<34} {'ops':>8} {'time ms':>8}")
for p, m in [(4, 32), (8, 64), (16, 256)]:
    t, _ = best_of(lambda: pipeline_sim.simulate(p, m), 3)
    print(f"{'1F1B p=' + str(p) + ' m=' + str(m):<34} {2 * p * m:>8} {1000 * t:>8.1f}")
