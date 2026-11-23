"""3D-parallel training on CPU processes: tensor x pipeline x data parallelism, checkpoints and restarts.

launch() starts one process per rank and returns their results. train_worker() is what every rank runs:
it builds its shard of the model (its pipeline stage's layers, split over its tensor-parallel group),
executes a GPipe or 1F1B schedule over micro-batches, all-reduces gradients over its data-parallel group,
and takes an SGD step. train_reference() is the plain single-process version used to check it.
"""
import multiprocessing as mp
import os
import time

import numpy as np

from . import model as M
from .comm import Comm, CommTimeout
from .pipeline import SCHEDULES, peak_in_flight, split_layers
from .topology import Grid


class WorkerFailed(RuntimeError):
    def __init__(self, msg, ranks):
        super().__init__(msg)
        self.ranks = ranks


# ------------------------------------------------------------------------------------------- launcher
def _entry(fn, rank, world, inboxes, results, timeout, args):
    comm = Comm(rank, world, inboxes, timeout=timeout)
    try:
        results.put((rank, "ok", fn(comm, rank, *args)))
    except CommTimeout as e:
        results.put((rank, "timeout", str(e)))
    except Exception as e:  # report, never hang the launcher
        results.put((rank, "error", repr(e)))


def launch(world, fn, args=(), timeout=30.0):
    """Run fn(comm, rank, *args) on `world` processes. Raises WorkerFailed if any rank dies or times out."""
    ctx = mp.get_context("fork")
    inboxes = [ctx.Queue() for _ in range(world)]
    results = ctx.Queue()
    procs = [ctx.Process(target=_entry, args=(fn, r, world, inboxes, results, timeout, args), daemon=True) for r in range(world)]
    for p in procs:
        p.start()
    out, bad = {}, {}
    deadline = time.monotonic() + timeout * 4
    while len(out) + len(bad) < world:
        try:
            rank, status, val = results.get(timeout=0.2)
            (out if status == "ok" else bad)[rank] = val
        except Exception:
            dead = [r for r, p in enumerate(procs) if p.exitcode not in (None, 0) and r not in out and r not in bad]
            for r in dead:
                bad[r] = f"exit code {procs[r].exitcode}"
            if bad or time.monotonic() > deadline:
                break
    if bad or len(out) < world:
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(1)
        missing = sorted(set(range(world)) - set(out))
        raise WorkerFailed(f"ranks failed: {bad or missing}", sorted(bad) or missing)
    for p in procs:
        p.join(5)
    return [out[r] for r in range(world)]


# ------------------------------------------------------------------------------------------- data
def batch(cfg, step, global_batch, seed=1):
    """Learnable synthetic text: each sequence counts upward by a random stride (mod vocab), with 10% noise."""
    rng = np.random.default_rng(seed * 100003 + step)
    start = rng.integers(0, cfg.vocab, (global_batch, 1))
    stride = rng.integers(1, 4, (global_batch, 1))
    ids = (start + stride * np.arange(cfg.seq + 1)) % cfg.vocab
    noise = rng.random(ids.shape) < 0.1
    ids = np.where(noise, rng.integers(0, cfg.vocab, ids.shape), ids)
    return ids[:, :-1], ids[:, 1:]


# ------------------------------------------------------------------------------------------- stage
class Stage:
    """This rank's parameters: its layers [lo, hi), plus the embedding (first stage) and head (last stage)."""

    def __init__(self, cfg, grid, rank, seed):
        self.cfg, self.grid = cfg, grid
        self.pp, self.dp, self.tpr = grid.coords(rank)
        self.first, self.last = self.pp == 0, self.pp == grid.pp - 1
        self.lo, self.hi = split_layers(cfg.layers, grid.pp)[self.pp]
        full = M.init_full(cfg, seed)
        t, r = grid.tp, self.tpr
        self.params = {}
        if self.first:
            self.params["emb"] = M.shard_emb(cfg, full["emb"], r, t)
        for li in range(self.lo, self.hi):
            self.params[f"block{li}"] = M.shard_block(cfg, full["blocks"][li], r, t)
        if self.last:
            self.params["head"] = M.shard_head(cfg, full["head"], r, t)

    def keys(self):
        return [(g, k) for g in sorted(self.params) for k in sorted(self.params[g])]

    def flat(self, tree):
        return np.concatenate([tree[g][k].ravel() for g, k in self.keys()])

    def unflat(self, vec):
        out, i = {}, 0
        for g, k in self.keys():
            a = self.params[g][k]
            out.setdefault(g, {})[k] = vec[i:i + a.size].reshape(a.shape)
            i += a.size
        return out


def _ckpt_path(d, step, pp, tp):
    return os.path.join(d, f"step_{step:06d}", f"pp{pp}_tp{tp}.npz")


def _save(stage, d, step):
    path = _ckpt_path(d, step, stage.pp, stage.tpr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{f"{g}.{k}": stage.params[g][k] for g, k in stage.keys()})
    os.replace(tmp, path)                       # a shard is either complete or absent


def latest_step(d):
    try:
        return int(open(os.path.join(d, "latest")).read())
    except (FileNotFoundError, ValueError):
        return 0


def _load(stage, d, step):
    z = np.load(_ckpt_path(d, step, stage.pp, stage.tpr))
    for g, k in stage.keys():
        stage.params[g][k] = z[f"{g}.{k}"].copy()


# ------------------------------------------------------------------------------------------- worker
def train_worker(comm, rank, cfg, grid, steps, global_batch, micro_batch, lr=0.1, schedule="1f1b", seed=0,
                 data_seed=1, ckpt_dir=None, fail_at=None, pad=None, return_params=False):
    st = Stage(cfg, grid, rank, seed)
    tp = M.TP(comm, grid.tp_group(rank), st.tpr)
    dpg, ppg = grid.dp_group(rank), grid.pp_group(rank)
    prev_rank = ppg[st.pp - 1] if not st.first else None
    next_rank = ppg[st.pp + 1] if not st.last else None
    per_dp = global_batch // grid.dp
    m = per_dp // micro_batch
    if per_dp % micro_batch or global_batch % grid.dp:
        raise ValueError("global batch must split evenly into data-parallel ranks and micro-batches")
    ops = SCHEDULES[schedule](grid.pp, m)[st.pp]
    scale = 1.0 / (m * grid.dp)                 # loss = mean over the whole global batch
    start = latest_step(ckpt_dir) if ckpt_dir else 0
    if start:
        _load(st, ckpt_dir, start)
    losses, timing = [], []
    for step in range(start, steps):
        if fail_at and fail_at[0] == rank and fail_at[1] == step and not os.path.exists(fail_at[2]):
            open(fail_at[2], "w").close()       # fail once: the marker file stops it failing again after restart
            os._exit(17)                        # abrupt death, like a GPU falling off the bus
        x_all, y_all = batch(cfg, step, global_batch, data_seed)
        mine = slice(st.dp * per_dp, (st.dp + 1) * per_dp)
        xs, ys = x_all[mine], y_all[mine]
        grads = {g: {k: np.zeros_like(v) for k, v in d.items()} for g, d in st.params.items()}
        caches, loss_sum, live, peak = {}, 0.0, 0, 0
        comm.barrier(list(range(grid.world)))
        t0 = time.monotonic()
        busy = 0.0
        for kind, i in ops:
            o0 = time.monotonic()
            mb = slice(i * micro_batch, (i + 1) * micro_batch)
            if kind == "F":
                cache = []
                if st.first:
                    h, c = M.emb_fwd(cfg, st.params["emb"], xs[mb], tp)
                    cache.append(c)
                else:
                    h = comm.recv(prev_rank, ("act", step, i))
                    o0 = time.monotonic()           # waiting for the previous stage is bubble, not work
                for li in range(st.lo, st.hi):
                    h, c = M.block_fwd(cfg, st.params[f"block{li}"], h, tp)
                    cache.append(c)
                _pad(pad, 0, o0)
                if st.last:
                    loss, c = M.head_fwd(cfg, st.params["head"], h, ys[mb], tp, scale)
                    loss_sum += loss
                    cache.append(c)
                else:
                    comm.send(next_rank, ("act", step, i), h, "pp")
                caches[i] = cache
                live += 1
                peak = max(peak, live)
            else:
                cache = caches.pop(i)
                if st.last:
                    dh, gh = M.head_bwd(cfg, st.params["head"], cache.pop(), tp)
                    _acc(grads["head"], gh)
                else:
                    dh = comm.recv(next_rank, ("grad", step, i))
                    o0 = time.monotonic()
                for li in reversed(range(st.lo, st.hi)):
                    dh, gb = M.block_bwd(cfg, st.params[f"block{li}"], dh, cache.pop(), tp)
                    _acc(grads[f"block{li}"], gb)
                _pad(pad, 1, o0)
                if st.first:
                    _acc(grads["emb"], M.emb_bwd(cfg, st.params["emb"], dh, cache.pop()))
                else:
                    comm.send(prev_rank, ("grad", step, i), dh, "pp")
                live -= 1
            busy += time.monotonic() - o0
        t1 = time.monotonic()
        # data parallelism: one bucket, one all-reduce over the data-parallel group
        g = comm.all_reduce(st.flat(grads), dpg, kind="dp") if grid.dp > 1 else st.flat(grads)
        new = st.unflat(st.flat(st.params) - lr * g)
        st.params = {gk: {k: v.copy() for k, v in d.items()} for gk, d in new.items()}
        total = comm.all_reduce(np.array([loss_sum]), list(range(grid.world)), kind="ctl")[0] / grid.tp
        losses.append(float(total))
        timing.append({"t0": t0, "t1": t1, "busy": busy, "peak_live": peak})
        if ckpt_dir:
            if st.dp == 0:
                _save(st, ckpt_dir, step + 1)
            comm.barrier(list(range(grid.world)))
            if rank == 0:
                tmp = os.path.join(ckpt_dir, "latest.tmp")
                open(tmp, "w").write(str(step + 1))
                os.replace(tmp, os.path.join(ckpt_dir, "latest"))
            comm.barrier(list(range(grid.world)))
    res = {"rank": rank, "coords": (st.pp, st.dp, st.tpr), "losses": losses, "timing": timing,
           "bytes": dict(comm.bytes_sent), "messages": dict(comm.messages), "peak_live": peak_in_flight(ops),
           "m": m, "start_step": start}
    if return_params:
        res["params"] = st.params
    return res


def _pad(pad, which, o0):
    """Bubble lab only: stretch an op to a fixed duration BEFORE its result is sent, like a slow GPU kernel."""
    if pad:
        left = pad[which] - (time.monotonic() - o0)
        if left > 0:
            time.sleep(left)


def _acc(dst, src):
    for k, v in src.items():
        dst[k] += v


def train_reference(cfg, steps, global_batch, lr=0.1, seed=0, data_seed=1):
    """Single process, whole batch at once, no communication: the ground truth for every 3D run."""
    full = M.init_full(cfg, seed)
    tp = M.TP()
    losses = []
    for step in range(steps):
        x, y = batch(cfg, step, global_batch, data_seed)
        h, ce = M.emb_fwd(cfg, full["emb"], x, tp)
        caches = []
        for p in full["blocks"]:
            h, c = M.block_fwd(cfg, p, h, tp)
            caches.append(c)
        loss, ch = M.head_fwd(cfg, full["head"], h, y, tp, 1.0)
        losses.append(float(loss))
        dh, gh = M.head_bwd(cfg, full["head"], ch, tp)
        gblocks = []
        for p, c in zip(reversed(full["blocks"]), reversed(caches)):
            dh, gb = M.block_bwd(cfg, p, dh, c, tp)
            gblocks.append(gb)
        ge = M.emb_bwd(cfg, full["emb"], dh, ce)
        for k in full["head"]:
            full["head"][k] = full["head"][k] - lr * gh[k]
        for p, gb in zip(full["blocks"], reversed(gblocks)):
            for k in p:
                p[k] = p[k] - lr * gb[k]
        full["emb"]["emb"] = full["emb"]["emb"] - lr * ge["emb"]
    return losses, full


def gather_full(cfg, grid, results):
    """Rebuild full parameters from the shards returned by the data-parallel-rank-0 workers."""
    t = grid.tp
    by = {r["coords"]: r["params"] for r in results}
    full = {"blocks": [None] * cfg.layers}
    for pp, (lo, hi) in enumerate(split_layers(cfg.layers, grid.pp)):
        shards = [by[(pp, 0, r)] for r in range(t)]
        for li in range(lo, hi):
            full["blocks"][li] = M.unshard_block(cfg, [s[f"block{li}"] for s in shards])
        if pp == 0:
            full["emb"] = {"emb": np.concatenate([s["emb"]["emb"] for s in shards])}
        if pp == grid.pp - 1:
            full["head"] = {"lnf_g": shards[0]["head"]["lnf_g"], "lnf_b": shards[0]["head"]["lnf_b"],
                            "wout": np.concatenate([s["head"]["wout"] for s in shards], 1)}
    return full


def max_diff(a, b):
    """Largest absolute difference between two parameter trees."""
    if isinstance(a, dict):
        return max(max_diff(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return max(max_diff(x, y) for x, y in zip(a, b))
    return float(np.max(np.abs(a - b)))


def train(cfg, grid, steps=3, global_batch=16, micro_batch=2, timeout=30.0, **kw):
    """Run train_worker on grid.world processes; extra keyword arguments go to every worker."""
    return launch(grid.world, _kw_worker, (cfg, grid, steps, global_batch, micro_batch, kw), timeout)


def _kw_worker(comm, rank, cfg, grid, steps, global_batch, micro_batch, kw):
    return train_worker(comm, rank, cfg, grid, steps, global_batch, micro_batch, **kw)


def train_with_restarts(cfg, grid, steps, global_batch, micro_batch, ckpt_dir, max_restarts=3, timeout=10.0, **kw):
    """What torchrun/a job controller does: on any rank failure, kill all ranks and restart from the last checkpoint."""
    restarts, log = 0, []
    while True:
        try:
            res = train(cfg, grid, steps, global_batch, micro_batch, timeout=timeout, ckpt_dir=ckpt_dir, **kw)
            log.append(f"run {restarts}: finished steps {res[0]['start_step']}..{steps - 1}")
            return res, restarts, log
        except WorkerFailed as e:
            log.append(f"run {restarts}: rank(s) {e.ranks} failed -> restart from checkpoint step {latest_step(ckpt_dir)}")
            restarts += 1
            if restarts > max_restarts:
                raise


# ------------------------------------------------------------------------------------------- expected traffic
def ring_bytes(numel, n, i, itemsize=8):
    """Exact bytes rank i (index in its group) sends in a ring all-reduce of numel elements over n ranks."""
    if n == 1:
        return 0
    sizes = [len(c) for c in np.array_split(np.arange(numel), n)]
    rs = sum(sizes) - sizes[(i + 1) % n]
    ag = sum(sizes) - sizes[(i + 2) % n]
    return (rs + ag) * itemsize


def expected_bytes(cfg, grid, rank, micro_batch, m):
    """Per-step bytes this rank should send, by kind, from first principles (tested against the counters)."""
    pp, dp, tpr = grid.coords(rank)
    st = Stage(cfg, grid, rank, 0)
    act = micro_batch * cfg.seq * cfg.hidden
    tok = micro_batch * cfg.seq
    t = grid.tp
    n_ar = []                                           # sizes of tensor-parallel all-reduces per micro-batch
    if st.first:
        n_ar += [act]                                   # embedding sum
    n_ar += [act] * 4 * (st.hi - st.lo)                 # per block: 2 forward + 2 backward
    if st.last:
        n_ar += [tok, tok, tok, act]                    # max, sum-exp, target logit, dz
    tp_b = m * sum(ring_bytes(n, t, tpr) for n in n_ar)
    pp_b = (0 if st.last else m * act * 8) + (0 if st.first else m * act * 8)
    nparam = st.flat(st.params).size
    dp_b = ring_bytes(nparam, grid.dp, dp)
    return {"tp": tp_b, "pp": pp_b, "dp": dp_b, "params_local": nparam}
