"""ackpt command line: demo, bench, inspect, verify, gc, plan, crash-test."""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

from . import planning as pl
from .checkpointer import Checkpointer
from .loader import committed_steps, load, load_latest, read_manifest
from .retention import gc
from .state import shard_state, unshard
from .storage import LocalFS
from .train import Trainer


def demo(a):
    root = a.dir or tempfile.mkdtemp(prefix="ackpt-demo-")
    fs, world = LocalFS(root), 4
    tr = Trainer(n_params=a.params)
    ck = Checkpointer(fs, world, keep_last=2)
    print(f"training on {world} ranks, async checkpoint every 10 steps -> {root}")
    for _ in range(40):
        loss = tr.train_step()
        if tr.step % 10 == 0:
            r = ck.save(tr.step, shard_state(tr.state, world))
            print(f"  step {tr.step:3d}  loss {loss:.5f}  checkpoint stall {r['stall_s'] * 1e3:6.2f} ms")
    ck.wait()
    print(f"committed checkpoints: {committed_steps(fs)}   (keep_last=2 removed the rest)")
    print("CRASH: all in-memory state lost; restarting on 2 ranks instead of 4")
    step, states, _ = load_latest(fs, 2)
    resumed = Trainer(n_params=a.params)
    resumed.state = unshard(states)
    print(f"resumed at step {step} from {len(states)} ranks")
    while resumed.step < 60:
        resumed.train_step()
    ref = Trainer(n_params=a.params)
    while ref.step < 60:
        ref.train_step()
    same = all(np.array_equal(ref.state[k], resumed.state[k]) for k in ref.state)
    print(f"step 60: resumed run == uninterrupted run, bit for bit: {same}")
    return 0 if same else 1


def bench(a):
    from .bench import run
    print(run(n_params=a.params, quick=a.quick))


def inspect(a):
    fs = LocalFS(a.dir)
    for s in committed_steps(fs):
        m = read_manifest(fs, s)
        size = sum(c["nbytes"] for c in m["chunks"])
        print(f"step {s:6d}  world {m['world_size']}  tensors {len({c['tensor'] for c in m['chunks']}):3d}  "
              f"chunks {len(m['chunks']):3d}  {size / 1e6:8.2f} MB")


def verify(a):
    fs = LocalFS(a.dir)
    bad = 0
    for s in committed_steps(fs):
        try:
            load(fs, s, read_manifest(fs, s)["world_size"], verify=True)
            print(f"step {s}: OK")
        except Exception as e:
            bad += 1
            print(f"step {s}: CORRUPT ({e})")
    return 1 if bad else 0


def gc_cmd(a):
    print("deleted steps:", gc(LocalFS(a.dir), a.keep))


def plan(a):
    total = pl.checkpoint_bytes(a.params)
    print(f"{a.params / 1e9:.0f}B params x 14 B (bf16 weights + fp32 master + Adam m, v) = {total / 1e9:,.0f} GB")
    for name, bw in (("NVMe", a.nvme_gbps), ("S3", a.s3_gbps)):
        print(f"  write to {name:4s}: {pl.write_seconds(total, a.nodes, bw * 1e9):7.1f} s  ({a.nodes} nodes x {bw} GB/s)")
    snap = total / (a.nodes * a.gpus_per_node) / (a.d2h_gbps * 1e9)
    mtbf = pl.job_mtbf(a.gpu_mtbf_h * 3600, a.nodes * a.gpus_per_node)
    print(f"  async stall (GPU -> host copy at {a.d2h_gbps} GB/s per GPU): {snap:.2f} s;  job MTBF {mtbf / 3600:.1f} h")
    write = pl.write_seconds(total, a.nodes, a.s3_gbps * 1e9)
    for name, c in (("sync to S3", write), ("async", snap)):
        t = max(pl.young_daly_interval(c, mtbf), write)      # async cannot start a save before the last one is written
        print(f"  {name:10s}: Young/Daly interval {t / 60:6.1f} min, waste {pl.waste_fraction(t, c, mtbf) * 100:5.2f}%")


def crash_test(a):
    ok = True
    for point in ("mid_shard", "before_manifest", "mid_manifest"):
        root = tempfile.mkdtemp(prefix="ackpt-crash-")
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        p = subprocess.run([sys.executable, "-m", "ackpt.crashsim", root, point], capture_output=True, env=env)
        step, states, skipped = load_latest(LocalFS(root), 4)
        debris = [k for k in LocalFS(root).list_all() if k.startswith("step_000000020")]
        print(f"{point:16s} exit {p.returncode:4d} (SIGKILL = -9)  latest loadable step: {step}  step-20 debris files: {len(debris)}")
        ok &= (p.returncode == -9 and step == 10)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ackpt", description="Asynchronous distributed checkpointing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--dir"); d.add_argument("--params", type=int, default=400_000); d.set_defaults(f=demo)
    b = sub.add_parser("bench"); b.add_argument("--params", type=int, default=4_000_000); b.add_argument("--quick", action="store_true"); b.set_defaults(f=bench)
    for name, f in (("inspect", inspect), ("verify", verify)):
        x = sub.add_parser(name); x.add_argument("dir"); x.set_defaults(f=f)
    g = sub.add_parser("gc"); g.add_argument("dir"); g.add_argument("--keep", type=int, default=3); g.set_defaults(f=gc_cmd)
    p = sub.add_parser("plan")
    p.add_argument("--params", type=float, default=7e9); p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=8); p.add_argument("--nvme-gbps", type=float, default=3.0)
    p.add_argument("--s3-gbps", type=float, default=2.5); p.add_argument("--d2h-gbps", type=float, default=25.0)
    p.add_argument("--gpu-mtbf-h", type=float, default=50_000.0); p.set_defaults(f=plan)
    c = sub.add_parser("crash-test"); c.set_defaults(f=crash_test)
    a = ap.parse_args(argv)
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
