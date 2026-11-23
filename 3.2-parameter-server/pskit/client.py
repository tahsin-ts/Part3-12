"""Worker-side client: routes keys to shards, pulls and pushes in parallel, retries."""
import http.client
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pskit.codec import decode_rows, encode_keys, encode_push
from pskit.hashring import Ring


class PSError(RuntimeError):
    pass


class PSClient:
    def __init__(self, addrs, worker=-1, dim=1, vnodes=64, retry_for=20.0):
        self.addrs, self.worker, self.dim, self.retry_for = list(addrs), worker, dim, retry_for
        self.ring = Ring(len(self.addrs), vnodes)
        self.conns = [None] * len(self.addrs)
        self.pool = ThreadPoolExecutor(max(1, len(self.addrs)))
        self.versions = np.zeros(len(self.addrs), dtype=np.int64)  # shard version seen at the last pull
        self.bytes_sent = self.bytes_recv = self.retries = 0

    # -------------------------------------------------------------- transport
    def _request(self, shard, path, body=b""):
        """POST with keep-alive; reconnect and retry until retry_for seconds pass (shard restarts)."""
        deadline, delay = time.monotonic() + self.retry_for, 0.05
        while True:
            try:
                if self.conns[shard] is None:
                    host, port = self.addrs[shard].rsplit(":", 1)
                    self.conns[shard] = http.client.HTTPConnection(host, int(port), timeout=60)
                c = self.conns[shard]
                c.request("POST", path, body, {"Content-Type": "application/octet-stream"})
                r = c.getresponse()
                data = r.read()
                if r.status == 200:
                    self.bytes_sent += len(body)
                    self.bytes_recv += len(data)
                    return r, data
                if r.status not in (503, 504):  # 4xx: a bug, do not retry
                    raise PSError(f"shard {shard} {path}: {r.status} {data[:200]!r}")
            except (ConnectionError, http.client.HTTPException, OSError):
                if self.conns[shard] is not None:
                    self.conns[shard].close()
                self.conns[shard] = None
            if time.monotonic() > deadline:
                raise PSError(f"shard {shard} ({self.addrs[shard]}) unreachable for {self.retry_for}s")
            self.retries += 1
            time.sleep(delay)
            delay = min(delay * 2, 1.0)

    def _split(self, keys):
        own = self.ring.owners(keys)
        return [np.nonzero(own == s)[0] for s in range(len(self.addrs))]

    # -------------------------------------------------------------- API
    def pull(self, keys, clock=0, staleness=-1):
        """Rows for `keys` (shape len(keys) x dim). Blocks on the server while the
        slowest worker is more than `staleness` clocks behind (`staleness` < 0: never)."""
        keys = np.asarray(keys, np.uint64)
        parts, out = self._split(keys), np.empty((len(keys), self.dim), np.float32)

        def one(s):  # every shard is asked, even with no keys: the request also reports our clock
            r, data = self._request(s, f"/pull?w={self.worker}&c={clock}&s={staleness}", encode_keys(keys[parts[s]]))
            self.versions[s] = int(r.getheader("X-Version", "0"))
            out[parts[s]] = decode_rows(data)

        list(self.pool.map(one, range(len(self.addrs))))
        return out

    def push(self, keys, grads, clock):
        """Send gradients; `clock` = iterations this worker has now finished."""
        keys = np.asarray(keys, np.uint64)
        grads = np.asarray(grads, np.float32).reshape(len(keys), self.dim)
        parts = self._split(keys)

        def one(s):
            self._request(s, f"/push?w={self.worker}&c={clock}&v={self.versions[s]}", encode_push(keys[parts[s]], grads[parts[s]]))

        list(self.pool.map(one, range(len(self.addrs))))

    def leave(self):
        for s in range(len(self.addrs)):
            try:
                self._request(s, f"/leave?w={self.worker}")
            except PSError:
                pass

    def close(self):
        for c in self.conns:
            if c is not None:
                c.close()
        self.pool.shutdown()
