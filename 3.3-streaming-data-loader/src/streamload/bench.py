"""Throughput benchmarks: workers, remote block size, prefetch depth under a simulated GPU, record size."""
import os
import tempfile
import time

from .codec import make_samples
from .loader import StreamingLoader
from .shards import write_dataset
from .storage import HttpStore, LocalStore, ObjectStoreServer


def _epoch(loader, step_s=0.0):
    """One epoch. step_s > 0 sleeps after each batch, like a GPU step that the loader must hide."""
    t0, n, wait, first, wait0 = time.perf_counter(), 0, 0.0, None, 0.0
    it = iter(loader)
    while True:
        t1 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        wait += time.perf_counter() - t1
        if first is None:
            first, wait0 = time.perf_counter() - t0, wait
        n += len(b["keys"])
        if step_s:
            time.sleep(step_s)
    dt = time.perf_counter() - t0
    return {"samples": n, "seconds": round(dt, 3), "first_batch_s": round(first or 0, 3),
            "samples_per_s": round(n / dt, 1), "wait_frac": round(wait / dt, 3),
            "steady_wait_frac": round((wait - wait0) / max(1e-9, dt - first), 3)}


def dataset(root, n, side, codec, per_shard, width=None):
    os.makedirs(root, exist_ok=True)
    return write_dataset(make_samples(n, side, seed=1, codec=codec, width=width), root, per_shard)


def bench_workers(root, quick=False):
    """JPEG decode + crop, local disk: how throughput grows with decode processes."""
    m = dataset(os.path.join(root, "jpeg"), 192 if quick else 384, 375, "jpeg", 24, width=500)
    rows = []
    for w, transport, dtype in ((0, "-", "uint8"), (1, "shm", "uint8"), (2, "shm", "uint8"), (3, "shm", "uint8"),
                                (2, "pickle", "uint8"), (2, "pickle", "float32")):
        L = StreamingLoader(LocalStore(os.path.join(root, "jpeg")), m, batch_size=32, num_workers=w,
                            prefetch=2, crop=224, transport=transport, dtype=dtype)
        r = _epoch(L)
        rows.append({"workers": w, "transport": transport, "dtype": dtype, **r})
    return rows


def bench_remote(root, quick=False, latency_ms=5.0):
    """Raw records over the HTTP object store: coalesced range reads and parallel streams hide latency."""
    m = dataset(os.path.join(root, "small"), 512 if quick else 1024, 32, "raw", 128)   # ~3 kB records
    rows = []
    with ObjectStoreServer(os.path.join(root, "small"), latency_ms=latency_ms) as srv:
        for block, w in ((0, 1), (0, 4), (1 << 16, 1), (1 << 20, 1), (1 << 20, 2)):
            before = srv.requests
            L = StreamingLoader(HttpStore(srv.url), m, batch_size=64, num_workers=w, decode="raw", block_bytes=block)
            r = _epoch(L)
            rows.append({"block_kB": block // 1024, "workers": w, **r, "requests": srv.requests - before})
    return rows


def bench_prefetch(root, quick=False, step_ms=60.0, latency_ms=5.0, jitter_ms=30.0):
    """A simulated GPU step after each batch; remote shards with a slow tail; how deep must the queue be?"""
    m = dataset(os.path.join(root, "pf"), 384 if quick else 768, 64, "zlib", 32)
    rows = []
    with ObjectStoreServer(os.path.join(root, "pf"), latency_ms=latency_ms, jitter_ms=jitter_ms) as srv:
        for depth in (1, 2, 4, 8):
            L = StreamingLoader(HttpStore(srv.url), m, batch_size=16, num_workers=2, prefetch=depth,
                                block_bytes=1 << 16, crop=56)
            rows.append({"prefetch": depth, **_epoch(L, step_ms / 1000)})
    return rows


def bench_record_size(root, quick=False):
    """Bytes per record vs records/s and MB/s (local disk, no decode)."""
    rows = []
    total = (8 if quick else 24) << 20
    for side in (16, 64, 256, 512):
        rec = side * side * 3
        n = max(32, total // rec)
        sub = os.path.join(root, f"rs{side}")
        m = dataset(sub, n, side, "raw", max(8, n // 8))
        L = StreamingLoader(LocalStore(sub), m, batch_size=32, num_workers=2, decode="raw")
        r = _epoch(L)
        rows.append({"record_kB": round(rec / 1024, 1), **r, "MB_per_s": round(r["samples_per_s"] * rec / 1e6, 1)})
    return rows


def bench_decode_cpu(root, n=96):
    """CPU-seconds per sample for JPEG decode + crop in one process: a rate that other load cannot distort."""
    from .codec import augment, decode_image, sample_rng
    samples = list(make_samples(n, 375, seed=2, codec="jpeg", width=500))
    t0 = time.process_time()
    for i, p in enumerate(samples):
        img, _ = decode_image(p)
        augment(img, 224, sample_rng(0, 0, "x", i))
    cpu = time.process_time() - t0
    return [{"samples": n, "cpu_s": round(cpu, 3), "per_core_per_s": round(n / cpu, 1),
             "record_kB": round(sum(map(len, samples)) / n / 1e3, 1)}]


def warm_up(root):
    """Start the forkserver once, so its boot time is not charged to the first benchmark."""
    m = dataset(os.path.join(root, "warm"), 8, 8, "raw", 4)
    list(StreamingLoader(LocalStore(os.path.join(root, "warm")), m, batch_size=4, num_workers=1, decode="raw"))


def run_all(quick=False):
    with tempfile.TemporaryDirectory() as root:
        warm_up(root)
        load = os.getloadavg()[0]
        return {"machine": [{"cpus": os.cpu_count(), "load_avg_1m": round(load, 2)}],
                "decode_cpu": bench_decode_cpu(root), "workers": bench_workers(root, quick), "remote": bench_remote(root, quick),
                "prefetch": bench_prefetch(root, quick), "record_size": bench_record_size(root, quick)}


def fmt(res):
    out = []
    for name, rows in res.items():
        out.append(f"== {name}")
        keys = list(rows[0].keys())
        out.append("  ".join(f"{k:>13}" for k in keys))
        for r in rows:
            out.append("  ".join(f"{r[k]:>13}" for k in keys))
    return "\n".join(out)
