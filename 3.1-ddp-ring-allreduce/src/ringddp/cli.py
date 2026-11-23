"""ringddp train | check | bench | worker | project"""
import argparse
import json
import sys
import time

import numpy as np

from . import bench, costmodel
from .launch import WorkerFailure, launch
from .trainer import Config, train_reference, train_worker


def _cfg(a, world):
    return Config(world=world, epochs=a.epochs, global_batch=a.batch, samples=a.samples, hidden=a.hidden,
                  layers=a.layers, lr=a.lr, bucket_kb=a.bucket_kb, overlap=not a.no_overlap, timeout=a.timeout,
                  crash_rank=a.crash_rank, crash_step=a.crash_step, hang_rank=a.hang_rank)


def cmd_train(a):
    cfg = _cfg(a, a.workers)
    t0 = time.perf_counter()
    try:
        res = launch(train_worker, a.workers, (cfg,), grace=a.timeout + 2)
    except WorkerFailure as e:
        print(f"training failed after {time.perf_counter() - t0:.2f} s: {e}")
        return 3
    r0 = res[0]
    print(f"workers={a.workers}  params={r0['n_params']:,}  buckets={r0['n_buckets']}  steps={r0['steps']}  "
          f"local batch={a.batch // a.workers}")
    for e, loss in enumerate(r0["losses"]):
        print(f"  epoch {e}: mean loss {loss:.6f}")
    print("  replicas identical:", len({r['digest'] for r in res}) == 1, f"(sha256 {r0['digest']})")
    for r in res:
        print(f"  rank {r['rank']}: sent {r['bytes_sent'] / 1e6:8.2f} MB  compute {r['compute_s']:.2f} s  "
              f"exposed comm {r['wait_s']:.2f} s  total {r['total_s']:.2f} s")
    return 0


def cmd_check(a):
    ref = train_reference(_cfg(a, 1))
    ok = True
    print(f"reference: 1 process, global batch {a.batch}, final loss {ref['losses'][-1]:.12f}")
    for w in a.workers_list:
        res = launch(train_worker, w, (_cfg(a, w),))
        diff = float(np.abs(res[0]["params"] - ref["params"]).max())
        same = len({r["digest"] for r in res}) == 1
        good = diff <= a.tol and same
        ok &= good
        print(f"workers={w}: max |param - reference| = {diff:.3e}  replicas identical={same}  "
              f"{'PASS' if good else 'FAIL'}")
    return 0 if ok else 1


def cmd_worker(a):
    """One rank of a multi-host job (e.g. a Kubernetes Indexed Job): --rank R --peers h0:p0,h1:p1,..."""
    addrs = [(h, int(port)) for h, port in (x.rsplit(":", 1) for x in a.peers.split(","))]
    r = train_worker(a.rank, addrs, None, _cfg(a, len(addrs)))
    print(f"rank {a.rank}/{len(addrs)}: losses {['%.6f' % v for v in r['losses']]}  sha256 {r['digest']}  "
          f"sent {r['bytes_sent'] / 1e6:.2f} MB  total {r['total_s']:.2f} s")
    return 0


def _table(rows):
    keys = list(rows[0])
    print("  ".join(f"{k:>10}" for k in keys))
    for r in rows:
        print("  ".join(f"{v:>10.4g}" if isinstance(v, float) else f"{str(v):>10}" for v in r.values()))


def _best(rows, key, time_field):
    """Best of the repeats for each configuration: on a shared machine, the minimum is the least disturbed run."""
    best = {}
    for r in rows:
        k = tuple(r[f] for f in key)
        if k not in best or r[time_field] < best[k][time_field]:
            best[k] = r
    return list(best.values())


def cmd_bench(a):
    rows = []
    for _ in range(a.repeat):
        if a.what == "allreduce":
            for w in a.workers_list:
                rows += bench.allreduce_bench(w, [int(s) for s in a.sizes.split(",")], a.reps)
        elif a.what == "scaling":
            rows += bench.scaling_bench(a.workers_list, epochs=a.epochs, global_batch=a.batch, samples=a.samples,
                                        hidden=a.hidden, layers=a.layers)
        else:
            rows += bench.overlap_bench(a.workers, epochs=a.epochs, global_batch=a.batch, samples=a.samples,
                                        hidden=a.hidden, layers=a.layers)
    key, tf = {"allreduce": (("world", "bytes"), "seconds"), "scaling": (("world",), "wall"),
               "overlap": (("bucket_kb", "overlap"), "step_ms")}[a.what]
    rows = _best([{k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()} for r in rows], key, tf)
    _table(rows)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


def cmd_project(a):
    print("PROJECTED (alpha-beta model, not measured). Assumptions:")
    for name, c in costmodel.CLUSTERS.items():
        print(f"  {name}: N={c['world']}, alpha={c['alpha'] * 1e6:g} us/step, bandwidth={c['bw'] / 1e9:g} GB/s per direction")
    names = list(costmodel.CLUSTERS)
    print(f"{'model':<12} {'grad bytes':>11}  " + "  ".join(f"{n:>24}" for n in names))
    for name, nbytes, times in costmodel.projection():
        print(f"{name:<12} {nbytes / 1e9:>9.2f} GB  " + "  ".join(f"{t * 1e3:>21.1f} ms" for t in times))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="ringddp", description="Data-parallel training with a ring all-reduce")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "check", "bench", "worker"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--workers-list", type=lambda v: [int(x) for x in v.split(",")], default=[1, 2, 4])
        s.add_argument("--epochs", type=int, default=3)
        s.add_argument("--batch", type=int, default=256, help="global batch (split across workers)")
        s.add_argument("--samples", type=int, default=8192)
        s.add_argument("--hidden", type=int, default=256)
        s.add_argument("--layers", type=int, default=2)
        s.add_argument("--lr", type=float, default=0.05)
        s.add_argument("--bucket-kb", type=int, default=256)
        s.add_argument("--no-overlap", action="store_true")
        s.add_argument("--timeout", type=float, default=30.0, help="seconds without data before a peer counts as hung")
        s.add_argument("--crash-rank", type=int, default=-1)
        s.add_argument("--hang-rank", type=int, default=-1)
        s.add_argument("--crash-step", type=int, default=-1)
        s.add_argument("--tol", type=float, default=1e-12)
        if name == "worker":
            s.add_argument("--rank", type=int, required=True)
            s.add_argument("--peers", required=True, help="host:port of every rank, in rank order")
        if name == "bench":
            s.add_argument("what", choices=["allreduce", "scaling", "overlap"])
            s.add_argument("--sizes", default="1000,10000,100000,1000000,4000000", help="float64 elements")
            s.add_argument("--reps", type=int, default=5)
            s.add_argument("--repeat", type=int, default=1)
            s.add_argument("--json")
    sub.add_parser("project")
    a = p.parse_args(argv)
    return {"train": cmd_train, "check": cmd_check, "worker": cmd_worker, "bench": cmd_bench, "project": cmd_project}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
