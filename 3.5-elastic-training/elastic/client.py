"""Client of the rendezvous server, used by agents and workers.

Keeps one HTTP/1.1 connection open. Fault injection lives here too: if the file
<faults_dir>/partition-<node> exists, every call fails as if the network were cut,
which simulates a partition between this node and the coordinator.
"""
import base64
import http.client
import json
import os
import time

import numpy as np


class Stale(Exception):
    """409: our generation was fenced off. Stop and re-rendezvous."""


class Gone(Exception):
    """410: the server no longer knows this node (lease expired or evicted)."""


class Partitioned(ConnectionError):
    """Simulated network partition."""


class Client:
    def __init__(self, endpoint, node="?", faults_dir=None, timeout=10.0):
        host, port = endpoint.rsplit(":", 1)
        self.host, self.port, self.node, self.timeout = host, int(port), node, timeout
        self.faults_dir = faults_dir if faults_dir is not None else os.environ.get("ELASTIC_FAULTS")
        self.conn = None

    def partitioned(self):
        return bool(self.faults_dir) and os.path.exists(os.path.join(self.faults_dir, f"partition-{self.node}"))

    def call(self, path, obj=None, timeout=None):
        if self.partitioned():
            raise Partitioned(f"{self.node}: simulated partition")
        body = json.dumps(obj or {}).encode()
        for attempt in (0, 1):                 # one reconnect: keep-alive connections can go stale
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
                self.conn.timeout = timeout or self.timeout
                if self.conn.sock is not None:
                    self.conn.sock.settimeout(self.conn.timeout)
                self.conn.request("POST" if obj is not None else "GET", path, body if obj is not None else None,
                                  {"Content-Type": "application/json"})
                resp = self.conn.getresponse()
                data = json.loads(resp.read() or b"{}")
                break
            except (OSError, http.client.HTTPException):
                self.close()
                if attempt:
                    raise
        if self.partitioned():                 # the reply arrived after the cut: drop it
            raise Partitioned(f"{self.node}: simulated partition")
        if resp.status == 409:
            raise Stale(data.get("error"))
        if resp.status == 410:
            raise Gone(data.get("error"))
        return resp.status, data

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ------------------------------------------------------------ agent API
    def join(self, nproc):
        return self.call("/join", {"node": self.node, "nproc": nproc})[1]

    def wait(self, timeout=2.0):
        return self.call("/wait", {"node": self.node, "timeout": timeout}, timeout=timeout + 5)[1]

    def heartbeat(self, gen):
        return self.call("/heartbeat", {"node": self.node, "gen": gen}, timeout=2.0)[1]

    def fail(self, gen, reason):
        self.call("/fail", {"node": self.node, "gen": gen, "reason": reason})

    def complete(self, gen):
        self.call("/complete", {"node": self.node, "gen": gen})

    def status(self):
        return self.call("/status")[1]

    # ------------------------------------------------------------ worker API (fenced by generation)
    def kv_set(self, gen, items, rank=None, step=None, delete=()):
        self.call("/kv/set", {"gen": gen, "items": items, "rank": rank, "step": step, "delete": list(delete)})

    def kv_get(self, gen, keys, deadline_s=60.0):
        """Block until every key exists. Raises Stale as soon as the generation is fenced."""
        end = time.monotonic() + deadline_s
        while True:
            code, data = self.call("/kv/get", {"gen": gen, "keys": keys, "timeout": 2.0}, timeout=7.0)
            if code == 200:
                return data["values"]
            if time.monotonic() > end:
                raise TimeoutError(f"keys not ready after {deadline_s}s: {keys[:3]}...")


def pack(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).decode()


def unpack(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float64)
