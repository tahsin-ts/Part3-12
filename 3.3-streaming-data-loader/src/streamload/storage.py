"""Where shards live: a local directory, or an HTTP object store read with Range requests.

ObjectStoreServer is a small S3-like simulator (stdlib http.server) with injectable latency,
bandwidth limit and failures, so remote-storage behaviour can be tested on one machine.
"""
import http.client
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


class LocalStore:
    def __init__(self, root):
        self.root = root
        self.requests = self.bytes = self.retries = 0

    def spec(self):
        return {"kind": "local", "root": self.root}

    def size(self, name):
        return os.path.getsize(os.path.join(self.root, name))

    def read(self, name, offset, length):
        self.requests += 1
        with open(os.path.join(self.root, name), "rb") as f:
            f.seek(offset)
            data = f.read(length)
        self.bytes += len(data)
        return data


class HttpStore:
    """Range reads over one keep-alive connection, with retries and exponential back-off."""

    def __init__(self, base_url, retries=4, backoff=0.02, timeout=10.0):
        u = urlparse(base_url)
        self.base_url, self.host, self.port = base_url, u.hostname, u.port or 80
        self.prefix = u.path.rstrip("/")
        self.max_retries, self.backoff, self.timeout = retries, backoff, timeout
        self.conn = None
        self.requests = self.bytes = self.retries = 0

    def spec(self):
        return {"kind": "http", "url": self.base_url, "retries": self.max_retries}

    def _call(self, method, name, headers):
        for attempt in range(self.max_retries + 1):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.request(method, f"{self.prefix}/{name}", headers=headers)
                r = self.conn.getresponse()
                body = r.read()
                self.requests += 1
                if r.status in (500, 502, 503, 504):
                    raise IOError(f"HTTP {r.status}")
                return r, body
            except (OSError, http.client.HTTPException) as e:
                if self.conn is not None:
                    self.conn.close()
                self.conn = None
                if attempt == self.max_retries:
                    raise IOError(f"{method} {name} failed after {attempt + 1} attempts: {e}") from e
                self.retries += 1
                time.sleep(self.backoff * (2 ** attempt))

    def size(self, name):
        r, _ = self._call("HEAD", name, {})
        if r.status != 200:
            raise FileNotFoundError(name)
        return int(r.getheader("Content-Length"))

    def read(self, name, offset, length):
        r, body = self._call("GET", name, {"Range": f"bytes={offset}-{offset + length - 1}"})
        if r.status != 206 or len(body) != length:
            raise IOError(f"bad range response {r.status} ({len(body)} of {length} bytes) for {name}")
        self.bytes += length
        return body


def make_store(spec):
    if spec["kind"] == "local":
        return LocalStore(spec["root"])
    if spec["kind"] == "http":
        return HttpStore(spec["url"], retries=spec.get("retries", 4))
    raise ValueError(f"unknown store kind {spec['kind']}")


_RANGE = re.compile(r"bytes=(\d+)-(\d*)$")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: one TCP connection per reader
    disable_nagle_algorithm = True  # headers and body are two writes: Nagle + delayed ACK would add ~40 ms

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _common(self):
        srv = self.server
        with srv.lock:
            srv.requests += 1
            n = srv.requests
        delay = srv.latency + (srv.rng.expovariate(1.0 / srv.jitter) if srv.jitter else 0.0)
        if delay:
            time.sleep(delay)
        if srv.fail_every and n % srv.fail_every == 0:
            self._send(503, b"injected failure")
            return None
        name = unquote(self.path.lstrip("/"))
        path = os.path.normpath(os.path.join(srv.root, name))
        if not path.startswith(os.path.abspath(srv.root) + os.sep) or not os.path.isfile(path):
            self._send(404, b"not found")
            return None
        return path

    def do_HEAD(self):
        path = self._common()
        if path:
            self.send_response(200)
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

    def do_GET(self):
        path = self._common()
        if not path:
            return
        size = os.path.getsize(path)
        m = _RANGE.match(self.headers.get("Range", ""))
        start, end = (int(m.group(1)), int(m.group(2) or size - 1)) if m else (0, size - 1)
        if start >= size or end < start:
            self._send(416, b"", [("Content-Range", f"bytes */{size}")])
            return
        end = min(end, size - 1)
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        if self.server.bandwidth:
            time.sleep(len(body) / self.server.bandwidth)
        hdr = [("Accept-Ranges", "bytes")]
        if m:
            hdr.append(("Content-Range", f"bytes {start}-{end}/{size}"))
        self._send(206 if m else 200, body, hdr)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):  # clients that hang up are normal
            super().handle_error(request, client_address)


class ObjectStoreServer:
    """S3-like range server. latency_ms (+ exponential jitter_ms tail) per request, mbps bandwidth per
    stream, every fail_every-th request answered with 503."""

    def __init__(self, root, port=0, latency_ms=0.0, mbps=0.0, fail_every=0, host="127.0.0.1", jitter_ms=0.0, seed=0):
        self.httpd = _Server((host, port), _Handler)
        self.httpd.root = os.path.abspath(root)
        self.httpd.latency = latency_ms / 1000.0
        self.httpd.jitter, self.httpd.rng = jitter_ms / 1000.0, random.Random(seed)
        self.httpd.bandwidth = mbps * 1e6 / 8
        self.httpd.fail_every = fail_every
        self.httpd.requests = 0
        self.httpd.lock = threading.Lock()
        self.url = f"http://{host}:{self.httpd.server_address[1]}"
        self.thread = None

    @property
    def requests(self):
        return self.httpd.requests

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
