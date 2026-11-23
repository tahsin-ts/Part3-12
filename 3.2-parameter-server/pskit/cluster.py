"""Start Go shards, run NumPy workers as processes, collect metrics."""
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
import urllib.request

import numpy as np

from pskit.client import PSClient
from pskit.data import dense_dataset, make_dataset
from pskit.model import logloss
from pskit.worker import run_worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def server_binary():
    b = os.environ.get("PSSERVER") or os.path.join(ROOT, "bin", "psserver")
    if not os.path.exists(b):
        b = shutil.which("psserver") or b
    if not os.path.exists(b):
        raise FileNotFoundError(f"{b} not found: run `make build` (go build -o bin/psserver ./cmd/psserver)")
    return b


def http(addr, path, method="GET"):
    req = urllib.request.Request(f"http://{addr}{path}", method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


class Shards:
    """N psserver processes on localhost ports base..base+N-1."""

    def __init__(self, n, workers, base=7100, lr=0.1, opt="adagrad", dim=1, snapdir=None,
                 snapshot_every=None, worker_timeout="5s"):
        self.n, self.base, self.procs = n, base, [None] * n
        self.args = dict(workers=workers, lr=lr, opt=opt, dim=dim, snapdir=snapdir,
                         snapshot_every=snapshot_every, worker_timeout=worker_timeout)
        self.addrs = [f"127.0.0.1:{base + i}" for i in range(n)]
        for i in range(n):
            self.start(i)

    def start(self, i):
        a = self.args
        cmd = [server_binary(), f"-addr=127.0.0.1:{self.base + i}", f"-id={i}", f"-shards={self.n}",
               f"-workers={a['workers']}", f"-lr={a['lr']}", f"-opt={a['opt']}", f"-dim={a['dim']}",
               f"-worker-timeout={a['worker_timeout']}"]
        if a["snapdir"]:
            cmd.append(f"-snapshot={os.path.join(a['snapdir'], f'shard-{i}.snap')}")
            if a["snapshot_every"]:
                cmd.append(f"-snapshot-every={a['snapshot_every']}")
        self.procs[i] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                http(self.addrs[i], "/healthz")
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError(f"shard {i} did not start")

    def kill(self, i):
        self.procs[i].kill()  # SIGKILL: no final snapshot, like a machine crash
        self.procs[i].wait()

    def reset(self, workers):
        for a in self.addrs:
            http(a, f"/reset?workers={workers}", "POST")

    def stats(self):
        return [json.loads(http(a, "/stats")) for a in self.addrs]

    def close(self):
        for p in self.procs:
            if p and p.poll() is None:
                p.terminate()
                p.wait(5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


DATA = {}


def dataset(kind="sparse", seed=0):
    if (kind, seed) not in DATA:
        if kind == "sparse":
            k, y = make_dataset(n=120_000, vocab=200_000, k=16, seed=seed)
        else:
            k, y = dense_dataset(n=60_000, vocab=4_096, k=16, seed=seed)
        cut = int(len(y) * 0.85)
        DATA[(kind, seed)] = (k[:cut], y[:cut], k[cut:], y[cut:])
    return DATA[(kind, seed)]


def _child(q, *args, **kw):
    q.put(run_worker(*args, **kw))


def train(addrs, workers=4, steps=200, batch=256, staleness=0, kind="sparse", seed=0, die=None, on_step=None):
    """Run `workers` worker processes to completion. die = {worker: step} kills workers.
    Returns a summary with throughput, bytes and final test loss."""
    ktr, ytr, kte, yte = dataset(kind, seed)
    ctx = mp.get_context("fork")
    q, bar = ctx.Queue(), ctx.Barrier(workers + 1)
    ps = [ctx.Process(target=_child, args=(q, addrs, w, workers, ktr, ytr, steps),
                      kwargs=dict(batch=batch, staleness=staleness, seed=seed, barrier=bar,
                                  die_at=(die or {}).get(w))) for w in range(workers)]
    for p in ps:
        p.start()
    bar.wait()
    t0 = time.perf_counter()
    if on_step:
        on_step(t0)
    alive = [w for w in range(workers) if w not in (die or {})]
    res = [q.get(timeout=600) for _ in alive]
    wall = time.perf_counter() - t0
    for p in ps:
        p.join()
    cl = PSClient(addrs)
    test = logloss(kte, yte, lambda u: cl.pull(u)[:, 0])
    cl.close()
    done = sum(r["steps"] for r in res)
    curve = np.mean([r["losses"] for r in res], axis=0)
    return dict(workers=workers, steps=steps, batch=batch, staleness=staleness, wall=wall,
                examples_per_s=done * batch / wall, steps_per_s=done / wall, test_loss=test,
                train_loss_curve=[float(x) for x in curve],
                client_bytes=sum(r["bytes_sent"] + r["bytes_recv"] for r in res),
                bytes_per_step=sum(r["bytes_sent"] + r["bytes_recv"] for r in res) / done,
                retries=sum(r["retries"] for r in res),
                t_pull=np.mean([r["t_pull"] for r in res]), t_compute=np.mean([r["t_compute"] for r in res]),
                t_push=np.mean([r["t_push"] for r in res]))


def summarize_stats(stats):
    hist = {}
    for s in stats:
        for k, v in s["StalenessHist"].items():
            hist[k] = hist.get(k, 0) + v
    n = sum(hist.values()) or 1
    return dict(keys=sum(s["Keys"] for s in stats), max_clock_gap=max(s["MaxClockGap"] for s in stats),
                waits=sum(s["Waits"] for s in stats), wait_ms=sum(s["WaitMs"] for s in stats),
                mean_staleness=sum(s["MeanStaleness"] * sum(s["StalenessHist"].values()) for s in stats) / n,
                staleness_hist=hist, server_bytes=sum(s["BytesIn"] + s["BytesOut"] for s in stats),
                keys_per_shard=[s["Keys"] for s in stats], evicted=sorted({w for s in stats for w in (s["Evicted"] or [])}))
