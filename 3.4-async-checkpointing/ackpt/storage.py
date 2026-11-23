"""Storage backends with one small interface: put, get (with byte ranges), list, delete.

LocalFS      - a POSIX directory. put = write temp file, fsync, rename, fsync directory.
ObjectStore  - a simulator of S3-like storage: whole-object puts that appear atomically,
               no rename, per-request latency, limited bandwidth and injectable faults.
"""
import os
import threading
import time


class TransientError(IOError):
    """A retryable failure (like HTTP 503 SlowDown from an object store)."""


def _crash():
    raise SystemExit("injected crash during put")


class LocalFS:
    def __init__(self, root, fsync=True):
        self.root, self.fsync = root, fsync
        self.on_crash = _crash                # tests replace this with a real SIGKILL
        os.makedirs(root, exist_ok=True)

    def _p(self, key):
        return os.path.join(self.root, key)

    def put(self, key, data, crash_after=None):
        """Atomic write. crash_after (tests only): stop after that many bytes, as a killed process would."""
        path = self._p(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "wb") as f:
            if crash_after is not None:
                f.write(memoryview(data)[:crash_after])
                f.flush()
                self.on_crash()
            f.write(data)
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())          # data is on disk before it gets its real name
        os.replace(tmp, path)                 # atomic: readers see old or new, never half
        if self.fsync:
            dfd = os.open(os.path.dirname(path), os.O_RDONLY)
            try:
                os.fsync(dfd)                 # the rename itself survives a power cut
            finally:
                os.close(dfd)

    def get(self, key, start=0, length=None):
        with open(self._p(key), "rb") as f:
            f.seek(start)
            return f.read() if length is None else f.read(length)

    def exists(self, key):
        return os.path.exists(self._p(key))

    def list(self, prefix=""):
        out = []
        for dirpath, _, files in os.walk(self.root):
            for fn in files:
                key = os.path.relpath(os.path.join(dirpath, fn), self.root).replace(os.sep, "/")
                if key.startswith(prefix) and ".tmp." not in key:
                    out.append(key)
        return sorted(out)

    def list_all(self, prefix=""):
        """Like list, but also shows temp files left by crashed writers (for garbage collection)."""
        out = []
        for dirpath, _, files in os.walk(self.root):
            for fn in files:
                key = os.path.relpath(os.path.join(dirpath, fn), self.root).replace(os.sep, "/")
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)

    def delete(self, key):
        try:
            os.remove(self._p(key))
        except FileNotFoundError:
            pass
        d = os.path.dirname(self._p(key))
        while d != os.path.abspath(self.root) and d.startswith(os.path.abspath(self.root)):
            try:
                os.rmdir(d)                   # remove empty step directories
            except OSError:
                break
            d = os.path.dirname(d)


class ObjectStore:
    """In-memory S3 simulator. latency_s per request, bandwidth_bps shared by all requests
    (a token bucket), fail_every_n: every n-th request raises TransientError."""

    def __init__(self, latency_s=0.0, bandwidth_bps=None, fail_every_n=0):
        self.objects, self.lock = {}, threading.Lock()
        self.latency_s, self.bw, self.fail_every_n = latency_s, bandwidth_bps, fail_every_n
        self.requests, self.bytes_in, self.bytes_out = 0, 0, 0
        self._next_free = 0.0                # when the shared link is free again

    def _request(self, nbytes):
        with self.lock:
            self.requests += 1
            if self.fail_every_n and self.requests % self.fail_every_n == 0:
                raise TransientError("503 SlowDown (injected)")
            now = time.perf_counter()
            start = max(now, self._next_free)
            dur = nbytes / self.bw if self.bw else 0.0
            self._next_free = start + dur     # requests queue for the shared bandwidth
            wait = start + dur - now
        time.sleep(self.latency_s + wait)    # sleeping releases the GIL, like real network I/O

    def put(self, key, data, crash_after=None):
        data = bytes(data)
        if crash_after is not None:
            raise SystemExit("injected crash during put")   # a dropped upload leaves no object
        self._request(len(data))
        with self.lock:
            self.objects[key] = data          # the whole object appears at once
            self.bytes_in += len(data)

    def get(self, key, start=0, length=None):
        with self.lock:
            if key not in self.objects:
                raise FileNotFoundError(key)
            blob = self.objects[key]
        part = blob[start:] if length is None else blob[start:start + length]
        self._request(len(part))
        with self.lock:
            self.bytes_out += len(part)
        return part

    def exists(self, key):
        with self.lock:
            return key in self.objects

    def list(self, prefix=""):
        self._request(0)
        with self.lock:
            return sorted(k for k in self.objects if k.startswith(prefix))

    list_all = list

    def delete(self, key):
        self._request(0)
        with self.lock:
            self.objects.pop(key, None)

    def corrupt(self, key, byte_index):
        """Flip one bit (tests): what a bad disk or a buggy copy tool does."""
        with self.lock:
            b = bytearray(self.objects[key])
            b[byte_index] ^= 0x01
            self.objects[key] = bytes(b)
