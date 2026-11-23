"""Command line: make-data, serve, verify, read, resume-check, bench, starve."""
import argparse
import hashlib
import json
import os
import sys
import time

from . import bench, starve
from .codec import make_samples
from .loader import StreamingLoader
from .shards import MANIFEST, write_dataset, verify_dataset
from .storage import HttpStore, LocalStore, ObjectStoreServer


def open_source(src):
    """A directory or an http:// URL that serves manifest.json and the shards."""
    store = HttpStore(src) if src.startswith("http") else LocalStore(src)
    data = store.read(MANIFEST, 0, store.size(MANIFEST))
    return store, json.loads(data)


def loader_from(a, store, manifest):
    return StreamingLoader(store, manifest, batch_size=a.batch, num_workers=a.workers, prefetch=a.prefetch,
                           shuffle_buffer=a.buffer, seed=a.seed, rank=a.rank, world_size=a.world, crop=a.crop)


def keyhash(keys):
    return hashlib.sha256(repr(keys).encode()).hexdigest()[:12]


def cmd_make(a):
    m = write_dataset(make_samples(a.records, a.side, a.seed, a.codec, width=a.width), a.out, a.per_shard)
    print(f"wrote {m['records']} records in {len(m['shards'])} shards, {m['bytes'] / 1e6:.1f} MB -> {a.out}")


def cmd_serve(a):
    srv = ObjectStoreServer(a.dir, port=a.port, latency_ms=a.latency_ms, mbps=a.mbps, fail_every=a.fail_every,
                            host=a.host).start()
    print(f"serving {a.dir} at {srv.url} (latency {a.latency_ms} ms, fail every {a.fail_every or 'never'})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()


def cmd_verify(a):
    store, m = open_source(a.src)
    bad = verify_dataset(store, m)
    print(f"{len(m['shards'])} shards, {m['records']} records: " + ("OK" if not bad else f"CORRUPT {bad}"))
    return 1 if bad else 0


def cmd_read(a):
    store, m = open_source(a.src)
    L = loader_from(a, store, m)
    if a.state_in:
        L.load_state_dict(json.load(open(a.state_in)))
    t0, n, keys, steps = time.perf_counter(), 0, [], 0
    for b in L:
        n += len(b["keys"])
        keys += b["keys"]
        steps += 1
        if a.max_steps and steps >= a.max_steps:
            break
    dt = time.perf_counter() - t0
    print(f"rank {a.rank}/{a.world}: {steps} steps, {n} samples in {dt:.2f} s = {n / dt:,.0f} samples/s, "
          f"waited {L.wait_s:.2f} s, order {keyhash(keys)}")
    print("state:", json.dumps(L.state_dict()))
    if a.state_out:
        json.dump(L.state_dict(), open(a.state_out, "w"))


def cmd_resume(a):
    """Run one epoch straight through, then again stopping at --stop-at and resuming from the saved state."""
    store, m = open_source(a.src)
    full = [k for b in loader_from(a, store, m) for k in b["keys"]]
    L = loader_from(a, store, m)
    first = []
    for i, b in enumerate(L, 1):
        first += b["keys"]
        if i == a.stop_at:
            break
    state = json.dumps(L.state_dict())
    L.close()
    L2 = loader_from(a, store, m)
    L2.load_state_dict(json.loads(state))
    rest = [k for b in L2 for k in b["keys"]]
    same = first + rest == full
    print(f"uninterrupted epoch : {len(full)} samples, order {keyhash(full)}")
    print(f"stopped at step {a.stop_at}: saved state {state} ({len(state)} bytes)")
    print(f"resumed epoch       : {len(first)} + {len(rest)} samples, order {keyhash(first + rest)}")
    print("exact resume:", "YES" if same else "NO", "| duplicates:", len(first + rest) - len(set(first + rest)))
    return 0 if same else 1


def cmd_bench(a):
    res = bench.run_all(a.quick)
    print(bench.fmt(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def cmd_starve(a):
    rows = [starve.starve(g, mo, a.per_core, a.cores_per_gpu) for g in a.gpus.split(",") for mo in a.models.split(",")]
    print(f"per-core loader rate {a.per_core:,.0f} samples/s, {a.cores_per_gpu} cores per GPU")
    print(starve.report(rows))


def main(argv=None):
    p = argparse.ArgumentParser(prog="streamload", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make-data", help="write a synthetic sharded image dataset")
    s.add_argument("out"); s.add_argument("--records", type=int, default=1000); s.add_argument("--side", type=int, default=64)
    s.add_argument("--width", type=int); s.add_argument("--codec", default="zlib", choices=["raw", "zlib", "jpeg"])
    s.add_argument("--per-shard", type=int, default=100); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_make)
    s = sub.add_parser("serve", help="HTTP object-store simulator with Range support")
    s.add_argument("dir"); s.add_argument("--port", type=int, default=8000); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--latency-ms", type=float, default=0); s.add_argument("--mbps", type=float, default=0)
    s.add_argument("--fail-every", type=int, default=0)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("verify", help="check every shard's sha256 against the manifest")
    s.add_argument("src"); s.set_defaults(fn=cmd_verify)
    for name, fn in (("read", cmd_read), ("resume-check", cmd_resume)):
        s = sub.add_parser(name)
        s.add_argument("src"); s.add_argument("--batch", type=int, default=32); s.add_argument("--workers", type=int, default=2)
        s.add_argument("--prefetch", type=int, default=2); s.add_argument("--buffer", type=int, default=256)
        s.add_argument("--seed", type=int, default=0); s.add_argument("--rank", type=int, default=0)
        s.add_argument("--world", type=int, default=1); s.add_argument("--crop", type=int, default=24)
        if name == "read":
            s.add_argument("--max-steps", type=int, default=0); s.add_argument("--state-in"); s.add_argument("--state-out")
        else:
            s.add_argument("--stop-at", type=int, default=5)
        s.set_defaults(fn=fn)
    s = sub.add_parser("bench", help="throughput sweeps"); s.add_argument("--quick", action="store_true")
    s.add_argument("--out"); s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("starve", help="needed vs delivered samples/s per GPU")
    s.add_argument("--per-core", type=float, required=True); s.add_argument("--cores-per-gpu", type=float, default=12)
    s.add_argument("--gpus", default="A100,H100"); s.add_argument("--models", default="resnet50,vit-b16,llama-7b")
    s.set_defaults(fn=cmd_starve)
    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
