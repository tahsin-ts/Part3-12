"""HTTP front end of the rendezvous: JSON over HTTP/1.1 keep-alive, long polling, one lock.

POST /join /wait /heartbeat /fail /complete /leave /kv/set /kv/get     GET /status /healthz /metrics
Status codes: 200 ok, 408 long poll timed out, 409 stale generation (fenced), 410 unknown node.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .events import EventLog
from .rendezvous import DONE, Rendezvous, StaleGeneration, UnknownNode


class Coordinator:
    """Rendezvous + a condition variable: every change wakes the long-polling waiters."""

    def __init__(self, rdzv):
        self.r = rdzv
        self.cond = threading.Condition()
        self.stop = threading.Event()
        threading.Thread(target=self._reaper, daemon=True).start()

    def _reaper(self):                      # leases expire even when nobody calls us
        while not self.stop.wait(0.05):
            with self.cond:
                before = (self.r.state, self.r.gen)
                self.r.expire()
                self.r.maybe_close()
                if (self.r.state, self.r.gen) != before:
                    self.cond.notify_all()

    def call(self, fn, *a, **kw):
        with self.cond:
            out = fn(*a, **kw)
            self.cond.notify_all()
            return out

    def wait_for(self, check, timeout):
        """Re-run check() under the lock until it returns something, or time out (None)."""
        end = time.monotonic() + timeout
        with self.cond:
            while True:
                out = check()
                if out is not None:
                    return out
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cond.wait(min(left, 0.05))


def make_handler(co):
    r = co.r

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True      # headers and body go out in two writes: avoid the 40 ms delayed-ACK stall

        def log_message(self, *a):
            pass

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                return self.reply(200, {"ok": True})
            if self.path == "/status":
                return self.reply(200, co.call(r.status))
            if self.path == "/metrics":
                st = co.call(r.status)
                text = (f"elastic_generation {st['gen']}\nelastic_world_size {st['world']}\n"
                        f"elastic_alive_nodes {len(st['alive'])}\nelastic_waiting_nodes {len(st['waiting'])}\n"
                        f"elastic_fenced_writes_total {st['rejected_writes']}\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                return self.wfile.write(text)
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self.reply(400, {"error": "bad json"})
            try:
                return self.route(req)
            except StaleGeneration as e:
                return self.reply(409, {"error": str(e), "gen": r.gen})
            except UnknownNode as e:
                return self.reply(410, {"error": f"unknown node {e}; rejoin"})
            except (KeyError, TypeError) as e:
                return self.reply(400, {"error": f"bad request: {e}"})

        def route(self, q):
            p = self.path
            if p == "/join":
                co.call(r.join, q["node"], int(q.get("nproc", 1)))
                return self.reply(200, co.call(r.assignment, q["node"]))
            if p == "/wait":                       # long poll until this node is in a running round
                co.call(r.touch, q["node"])          # waiting renews the lease (hot spares stay alive)
                def ready():
                    a = r.assignment(q["node"])
                    return a if a["state"] == DONE or "world" in a else None
                out = co.wait_for(ready, float(q.get("timeout", 5)))
                return self.reply(200, out or co.call(r.assignment, q["node"]))
            if p == "/heartbeat":
                return self.reply(200, co.call(r.heartbeat, q["node"], q.get("gen")))
            if p == "/fail":
                co.call(r.fail, q["node"], q["gen"], q.get("reason", "worker failed"))
                return self.reply(200, {"ok": True})
            if p == "/complete":
                co.call(r.complete, q["node"], q["gen"])
                return self.reply(200, {"ok": True})
            if p == "/leave":
                co.call(r.leave, q["node"])
                return self.reply(200, {"ok": True})
            if p == "/kv/set":
                co.call(r.kv_set, q["gen"], q["items"], q.get("rank"), q.get("step"), q.get("delete", ()))
                return self.reply(200, {"ok": True})
            if p == "/kv/get":
                out = co.wait_for(lambda: r.kv_get(q["gen"], q["keys"]), float(q.get("timeout", 5)))
                return self.reply(200, {"values": out}) if out is not None else self.reply(408, {"error": "timeout"})
            return self.reply(404, {"error": "not found"})

    return H


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass                                   # a killed worker resets its connection: expected, not news


def serve(host="127.0.0.1", port=0, events=None, **cfg):
    """Start the coordinator in a background thread; returns (http_server, coordinator)."""
    log = EventLog(events, "server")
    co = Coordinator(Rendezvous(log=log.emit, **cfg))
    httpd = QuietServer((host, port), make_handler(co))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log.emit("server_up", port=httpd.server_address[1], **{k: v for k, v in cfg.items() if v is not None})
    return httpd, co
