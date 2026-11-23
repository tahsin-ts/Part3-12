"""Command line: python -m pskit {train,worker,bench,sim,ring,compare,failover}."""
import argparse
import json
import os
import tempfile
import threading
import time

import numpy as np


def _mode(a):
    from pskit.worker import staleness_of
    return staleness_of(a.mode, a.staleness)


def _label(s):
    return {0: "sync (BSP)", -1: "async"}.get(s, f"SSP s={s}")


def cmd_train(a):
    from pskit.cluster import Shards, summarize_stats, train
    s = _mode(a)
    with Shards(a.shards, a.workers, base=a.port) as sh:
        r = train(sh.addrs, workers=a.workers, steps=a.steps, batch=a.batch, staleness=s, kind=a.kind)
        st = summarize_stats(sh.stats())
    print(f"mode={_label(s)} shards={a.shards} workers={a.workers} steps/worker={a.steps} batch={a.batch} data={a.kind}")
    print(f"wall {r['wall']:.2f} s   throughput {r['examples_per_s']:,.0f} examples/s   ({r['steps_per_s']:.0f} steps/s)")
    print(f"train loss {r['train_loss_curve'][0]:.4f} -> {np.mean(r['train_loss_curve'][-20:]):.4f}   test loss {r['test_loss']:.4f}")
    print(f"payload bytes/step {r['bytes_per_step']:,.0f}   keys on shards {st['keys_per_shard']}")
    print(f"clock gap max {st['max_clock_gap']}   blocked pulls {st['waits']} ({st['wait_ms']} ms)   "
          f"mean staleness {st['mean_staleness']:.2f} updates")
    print(f"time per step: pull {1e3 * r['t_pull'] / a.steps:.2f} ms  compute {1e3 * r['t_compute'] / a.steps:.2f} ms  "
          f"push {1e3 * r['t_push'] / a.steps:.2f} ms")
    return r, st


def cmd_worker(a):
    """One worker against running servers (used by the Kubernetes Job)."""
    from pskit.cluster import dataset
    from pskit.worker import run_worker
    wid = a.id if a.id is not None else int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    ktr, ytr, _, _ = dataset(a.kind)
    r = run_worker(a.servers.split(","), wid, a.workers, ktr, ytr, a.steps, a.batch, _mode(a))
    print(json.dumps({k: v for k, v in r.items() if k != "losses"} | {"final_loss": float(np.mean(r["losses"][-20:]))}))


def cmd_bench(a):
    from pskit.cluster import Shards, summarize_stats, train
    out = {"modes": [], "workers": [], "shards": [], "dense_vs_sparse": []}
    steps = a.steps
    with Shards(2, 4, base=a.port) as sh:
        print(f"== 1. consistency model (2 shards, 4 workers, {steps} steps each, batch 256)")
        print(f"{'mode':<12}{'wall s':>8}{'ex/s':>9}{'test loss':>11}{'max gap':>9}{'blocked':>9}{'wait ms':>9}{'staleness':>11}")
        for s in (0, 1, 3, -1):
            sh.reset(4)
            r = train(sh.addrs, workers=4, steps=steps, staleness=s)
            st = summarize_stats(sh.stats())
            out["modes"].append(r | st)
            print(f"{_label(s):<12}{r['wall']:>8.2f}{r['examples_per_s']:>9,.0f}{r['test_loss']:>11.4f}{st['max_clock_gap']:>9}"
                  f"{st['waits']:>9}{st['wait_ms']:>9}{st['mean_staleness']:>11.2f}")
        print("\n== 2. dense vs sparse traffic (SSP s=3, 4 workers, 2 shards)")
        for kind in ("sparse", "dense"):
            sh.reset(4)
            r = train(sh.addrs, workers=4, steps=steps // 2, staleness=3, kind=kind)
            st = summarize_stats(sh.stats())
            keys_step = sum(x["KeysPulled"] for x in sh.stats()) / (4 * (steps // 2))
            out["dense_vs_sparse"].append(dict(kind=kind, bytes_per_step=r["bytes_per_step"], keys_per_step=keys_step,
                                               model_keys=st["keys"], examples_per_s=r["examples_per_s"]))
            print(f"{kind:<7} model keys {st['keys']:>7,}  keys pulled/step {keys_step:>7,.0f} ({100 * keys_step / st['keys']:.1f}% of model)"
                  f"  payload bytes/step {r['bytes_per_step']:>9,.0f}  bytes/key {r['bytes_per_step'] / keys_step:.1f}")
    print("\n== 3. throughput vs workers (SSP s=3, 2 shards; this machine has %d CPU cores)" % os.cpu_count())
    with Shards(2, 1, base=a.port + 10) as sh:
        for w in (1, 2, 4, 8):
            sh.reset(w)
            r = train(sh.addrs, workers=w, steps=steps // 2, staleness=3)
            out["workers"].append(r)
            print(f"workers {w}: {r['examples_per_s']:>9,.0f} examples/s   test loss {r['test_loss']:.4f}   "
                  f"pull {1e3 * r['t_pull'] / (steps // 2):.2f} ms  compute {1e3 * r['t_compute'] / (steps // 2):.2f} ms  "
                  f"push {1e3 * r['t_push'] / (steps // 2):.2f} ms per step")
    print("\n== 4. throughput vs shards (SSP s=3, 4 workers)")
    for n in (1, 2, 4):
        with Shards(n, 4, base=a.port + 20) as sh:
            r = train(sh.addrs, workers=4, steps=steps // 2, staleness=3)
            out["shards"].append(r | {"shards": n})
            print(f"shards {n}: {r['examples_per_s']:>9,.0f} examples/s   push+pull {1e3 * (r['t_pull'] + r['t_push']) / (steps // 2):.2f} ms per step")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"\nwrote {a.out}")


def cmd_sim(a):
    from pskit import sim
    rows = sim.sweep(delays=tuple(int(x) for x in a.delays.split(",")))
    delays = sorted({r["delay"] for r in rows})
    print("final test loss after 1,500 updates (batch 128); gradient computed on weights `delay` updates old")
    print(f"{'optimizer':<16}" + "".join(f"{'d=' + str(d):>8}" for d in delays))
    for opt, lr in sim.SETTINGS:
        rs = {r["delay"]: r["test_loss"] for r in rows if r["opt"] == opt and r["lr"] == lr}
        print(f"{opt + ' lr=' + str(lr):<16}" + "".join(f"{rs[d]:>8.4f}" for d in delays))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


def cmd_ring(a):
    from pskit.hashring import Ring
    keys = np.arange(a.keys, dtype=np.uint64)
    print(f"{a.keys:,} keys, {a.shards} -> {a.shards + 1} shards")
    old, new = Ring(a.shards).owners(keys), Ring(a.shards + 1).owners(keys)
    mo, mn = keys % np.uint64(a.shards), keys % np.uint64(a.shards + 1)
    print(f"consistent hashing (64 vnodes): {100 * np.mean(old != new):5.1f}% of keys move (ideal {100 / (a.shards + 1):.1f}%)")
    print(f"modulo hashing  (key % n)     : {100 * np.mean(mo != mn):5.1f}% of keys move")
    print("balance (largest shard / mean) with 8 shards:")
    for v in (1, 4, 16, 64, 256):
        c = np.bincount(Ring(8, v).owners(keys), minlength=8)
        print(f"  vnodes {v:>3}: {c.max() / c.mean():.3f}")


def cmd_compare(a):
    from pskit.allreduce import compare, ring_allreduce
    arrs = [np.random.default_rng(r).normal(size=1_000_000).astype(np.float32) for r in range(a.workers)]
    res, sent = ring_allreduce(arrs)
    ok = all(np.allclose(x, sum(arrs), atol=1e-3) for x in res)
    print(f"ring all-reduce check: {a.workers} ranks x 1,000,000 floats, correct={ok}, bytes sent per rank {sent[0]:,} "
          f"(formula 2(n-1)/n x 4 MB = {2 * (a.workers - 1) / a.workers * 4e6:,.0f})")
    print(f"\nmodel: {a.workers} workers, {a.workers} PS shards, 100 Gb/s links, 20 us per message")
    print(f"{'workload':<34}{'ring bytes':>12}{'PS bytes':>12}{'ring ms':>10}{'PS ms':>10}  winner")
    for r in compare(n=a.workers, shards=a.workers):
        print(f"{r['name']:<34}{r['ring_bytes'] / 1e6:>10,.1f}MB{r['ps_bytes'] / 1e6:>10,.2f}MB{1e3 * r['ring_s']:>10.2f}{1e3 * r['ps_s']:>10.3f}  {r['winner']}")


def cmd_failover(a):
    from pskit.cluster import Shards, summarize_stats, train
    d = tempfile.mkdtemp(prefix="ps-snap-")
    print(f"== shard crash: 2 shards (snapshot every {a.every}), 4 workers, SSP s=2, {a.steps} steps")
    with Shards(2, 4, base=a.port, snapdir=d, snapshot_every=a.every, worker_timeout="1s") as sh:
        base = train(sh.addrs, workers=4, steps=a.steps, staleness=2)
        sh.reset(4)
        log = {}

        def chaos(t0):
            def run():
                last = None
                while time.perf_counter() - t0 < a.kill_at:
                    try:
                        last = sh.stats()[1]
                    except OSError:
                        pass
                    time.sleep(0.05)
                sh.kill(1)
                log["killed_at"] = time.perf_counter() - t0
                log["version_before"] = last["Version"] if last else 0
                time.sleep(a.down)
                sh.start(1)
                s1 = sh.stats()[1]
                log["restarted_at"] = time.perf_counter() - t0
                log["restored_keys"], log["restored_version"] = s1["RestoredKeys"], s1["Version"]
            threading.Thread(target=run, daemon=True).start()

        r = train(sh.addrs, workers=4, steps=a.steps, staleness=2, on_step=chaos)
    print(f"no failure : wall {base['wall']:.2f} s  test loss {base['test_loss']:.4f}")
    print(f"shard crash: wall {r['wall']:.2f} s  test loss {r['test_loss']:.4f}  client retries {r['retries']}")
    print(f"  killed shard 1 at {log['killed_at']:.2f} s (version ~{log['version_before']}), restarted at {log['restarted_at']:.2f} s")
    print(f"  restored {log['restored_keys']:,} keys at version {log['restored_version']}: "
          f"~{log['version_before'] - log['restored_version']} pushes lost (between last snapshot and crash)")
    print(f"\n== worker death: worker 3 is SIGKILLed at step {a.steps // 4}; eviction timeout 1 s; sync (BSP)")
    with Shards(2, 4, base=a.port + 10, worker_timeout="1s") as sh:
        b2 = train(sh.addrs, workers=4, steps=a.steps, staleness=0)
        sh.reset(4)
        r2 = train(sh.addrs, workers=4, steps=a.steps, staleness=0, die={3: a.steps // 4})
        st = summarize_stats(sh.stats())
    print(f"no failure  : wall {b2['wall']:.2f} s  test loss {b2['test_loss']:.4f}")
    print(f"worker death: wall {r2['wall']:.2f} s  test loss {r2['test_loss']:.4f}  evicted workers {st['evicted']}  "
          f"blocked pulls {st['waits']} ({st['wait_ms']} ms total)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pskit", description="Sharded parameter server: workers, benchmarks, drills")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, steps=200):
        q.add_argument("--mode", choices=["sync", "ssp", "async"], default="ssp")
        q.add_argument("--staleness", type=int, default=2, help="SSP bound s (clocks)")
        q.add_argument("--steps", type=int, default=steps)
        q.add_argument("--batch", type=int, default=256)
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--kind", choices=["sparse", "dense"], default="sparse")

    t = sub.add_parser("train", help="start local shards and train once")
    common(t)
    t.add_argument("--shards", type=int, default=2)
    t.add_argument("--port", type=int, default=7100)
    w = sub.add_parser("worker", help="run one worker against running shards")
    common(w)
    w.add_argument("--servers", required=True, help="host:port,host:port (shard order)")
    w.add_argument("--id", type=int)
    b = sub.add_parser("bench", help="consistency models, traffic, scaling")
    b.add_argument("--steps", type=int, default=300)
    b.add_argument("--port", type=int, default=7200)
    b.add_argument("--out")
    s = sub.add_parser("sim", help="convergence vs staleness (deterministic simulator)")
    s.add_argument("--delays", default="0,1,2,4,8,16,32,64")
    s.add_argument("--out")
    r = sub.add_parser("ring", help="consistent hashing: movement and balance")
    r.add_argument("--keys", type=int, default=1_000_000)
    r.add_argument("--shards", type=int, default=4)
    c = sub.add_parser("compare", help="parameter server vs ring all-reduce")
    c.add_argument("--workers", type=int, default=8)
    f = sub.add_parser("failover", help="shard crash + restart from snapshot; worker death")
    f.add_argument("--steps", type=int, default=400)
    f.add_argument("--every", default="300ms")
    f.add_argument("--kill-at", type=float, default=1.0)
    f.add_argument("--down", type=float, default=0.5)
    f.add_argument("--port", type=int, default=7300)
    a = p.parse_args(argv)
    {"train": cmd_train, "worker": cmd_worker, "bench": cmd_bench, "sim": cmd_sim, "ring": cmd_ring,
     "compare": cmd_compare, "failover": cmd_failover}[a.cmd](a)


if __name__ == "__main__":
    main()
