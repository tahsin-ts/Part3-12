"""Append-only JSON-lines event log shared by every process of a run (server, agents, workers, harness).

One small write() per event on a file opened with O_APPEND: lines from different processes never interleave.
Every timing in the benchmarks is computed from these timestamps (time.time(), same clock for all processes).
"""
import json
import os
import time


class EventLog:
    def __init__(self, path, src):
        self.path, self.src = path, src
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if path else None

    def emit(self, ev, **kw):
        if self.fd is None:
            return
        rec = {"t": round(time.time(), 4), "src": self.src, "ev": ev}
        rec.update(kw)
        os.write(self.fd, (json.dumps(rec) + "\n").encode())


def read_events(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:       # a line being written right now
                    pass
    return out
