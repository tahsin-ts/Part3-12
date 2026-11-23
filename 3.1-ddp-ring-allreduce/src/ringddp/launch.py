"""Start N worker processes on this machine, collect their results, and fail cleanly if any dies."""
import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait

from .transport import listen


class WorkerFailure(RuntimeError):
    def __init__(self, status):
        self.status = status
        lines = [f"  rank {r}: {msg}" for r, msg in sorted(status.items())]
        super().__init__(f"{len(status)} rank(s) failed\n" + "\n".join(lines))


def _entry(rank, fn, addrs, socks, conn, args):
    for i, s in enumerate(socks):                 # keep only our own listening socket
        if i != rank:
            s.close()
    try:
        res = fn(rank, addrs, socks[rank] if socks else None, *args)
        conn.send(("ok", res))
        code = 0
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"))
        code = 1
    conn.close()
    os._exit(code)


def launch(fn, world, args=(), timeout=120.0, grace=5.0):
    """Run fn(rank, addrs, listen_sock, *args) in `world` processes. Returns results by rank.
    After the first failure, the others get `grace` seconds to notice; then they are killed."""
    ctx = mp.get_context("fork")
    socks = [listen() for _ in range(world)] if world > 1 else []   # bound BEFORE fork: no port races
    addrs = [s.getsockname() for s in socks]
    procs, conns = [], {}
    for r in range(world):
        rx, tx = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_entry, args=(r, fn, addrs, socks, tx, args), daemon=True)
        p.start()
        tx.close()
        procs.append(p)
        conns[rx] = r
    for s in socks:
        s.close()
    results, errors = {}, {}
    t0, first_fail = time.monotonic(), None
    while conns:
        for c in wait(list(conns), timeout=0.05):
            r = conns.pop(c)
            try:
                kind, val = c.recv()
            except EOFError:                          # the process died without saying anything
                procs[r].join(1.0)
                kind, val = "error", f"died with exit code {procs[r].exitcode} (no result)"
            (results if kind == "ok" else errors)[r] = val
        now = time.monotonic()
        if errors and first_fail is None:
            first_fail = now
        expired = (first_fail is not None and now - first_fail > grace) or now - t0 > timeout
        if expired:
            for c, r in list(conns.items()):
                procs[r].kill()
                errors[r] = "killed by the launcher (no progress)"
                conns.pop(c)
    for p in procs:
        p.join(2.0)
    if errors:
        raise WorkerFailure(errors)
    return [results[r] for r in range(world)]
