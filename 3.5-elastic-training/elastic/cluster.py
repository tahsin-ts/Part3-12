"""A local 'cluster' for experiments: one coordinator process and N agent processes on this machine.

Each agent runs in its own process group (its own 'machine'), so a whole node can be killed,
frozen or cut off. Faults are injected the way they happen in real clusters:
  kill_worker  kill -9 one training process            (GPU fell off the bus, OOM killer)
  kill_node    kill -9 the agent and its workers        (machine lost power)
  partition    the node can no longer reach the coordinator for a while
  pause        SIGSTOP / SIGCONT the whole node         (VM freeze, long GC pause: the zombie case)
  slow         one node takes extra time every step     (thermal throttling, bad link)
"""
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from .events import EventLog, read_events


class LocalCluster:
    def __init__(self, workdir, nodes=3, nproc=1, min_nodes=2, max_nodes=3, lease_ttl=1.5, hb_interval=0.25,
                 last_call=0.5, steps=300, ckpt_every=20, step_sleep=0.01, seed=0, evict_lag=None, max_restarts=3):
        self.wd = os.path.abspath(workdir)
        shutil.rmtree(self.wd, ignore_errors=True)
        os.makedirs(os.path.join(self.wd, "faults"))
        self.events_path = os.path.join(self.wd, "events.jsonl")
        self.faults = os.path.join(self.wd, "faults")
        self.n0, self.nproc, self.min, self.max = nodes, nproc, min_nodes, max_nodes
        self.lease, self.hb, self.last_call = lease_ttl, hb_interval, last_call
        self.steps, self.every, self.nap, self.seed = steps, ckpt_every, step_sleep, seed
        self.evict_lag, self.max_restarts = evict_lag, max_restarts
        self.log = EventLog(self.events_path, "harness")
        self.server, self.agents, self.endpoint = None, {}, None
        self.timers = []

    def _py(self, *args):
        return [sys.executable, "-m", "elastic", *args]

    def start(self):
        portfile = os.path.join(self.wd, "port")
        cmd = self._py("server", "--port", "0", "--port-file", portfile, "--min-nodes", str(self.min), "--max-nodes", str(self.max),
                       "--lease-ttl", str(self.lease), "--last-call", str(self.last_call), "--events", self.events_path)
        if self.evict_lag is not None:
            cmd += ["--evict-lag", str(self.evict_lag)]
        self.server = subprocess.Popen(cmd, start_new_session=True)
        for _ in range(200):
            if os.path.exists(portfile) and open(portfile).read().strip():
                break
            time.sleep(0.05)
        self.endpoint = f"127.0.0.1:{open(portfile).read().strip()}"
        for i in range(1, self.n0 + 1):
            self.add_node(f"n{i}")
        return self

    def add_node(self, name):
        cmd = self._py("agent", "--endpoint", self.endpoint, "--node", name, "--nproc", str(self.nproc),
                       "--ckpt-dir", os.path.join(self.wd, "ckpt"), "--events", self.events_path, "--faults", self.faults,
                       "--steps", str(self.steps), "--ckpt-every", str(self.every), "--step-sleep", str(self.nap),
                       "--seed", str(self.seed), "--hb-interval", str(self.hb), "--lease-ttl", str(self.lease),
                       "--max-restarts", str(self.max_restarts))
        self.agents[name] = subprocess.Popen(cmd, start_new_session=True)

    # ---------------------------------------------------------------- faults
    def worker_pid(self, node, local_rank=0):
        starts = [e for e in read_events(self.events_path) if e["ev"] == "workers_started" and e["src"] == f"agent@{node}"]
        return starts[-1]["pids"][local_rank]

    def kill_worker(self, node, local_rank=0):
        pid = self.worker_pid(node, local_rank)
        self.log.emit("fault", kind="kill_worker", node=node, pid=pid)
        os.kill(pid, signal.SIGKILL)

    def kill_node(self, node):
        self.log.emit("fault", kind="kill_node", node=node)
        os.killpg(self.agents[node].pid, signal.SIGKILL)

    def _later(self, seconds, fn):
        t = threading.Timer(seconds, fn)
        t.daemon = True
        t.start()
        self.timers.append(t)

    def partition(self, node, seconds):
        flag = os.path.join(self.faults, f"partition-{node}")
        open(flag, "w").close()
        self.log.emit("fault", kind="partition", node=node, seconds=seconds)

        def heal():
            os.unlink(flag)
            self.log.emit("healed", node=node)
        self._later(seconds, heal)

    def pause(self, node, seconds):
        pg = self.agents[node].pid
        os.killpg(pg, signal.SIGSTOP)
        self.log.emit("fault", kind="pause", node=node, seconds=seconds)

        def resume():
            os.killpg(pg, signal.SIGCONT)
            self.log.emit("healed", node=node)
        self._later(seconds, resume)

    def slow(self, node, extra_s):
        with open(os.path.join(self.faults, f"slow-{node}"), "w") as f:
            f.write(str(extra_s))
        self.log.emit("fault", kind="slow", node=node, extra_s=extra_s)

    # ---------------------------------------------------------------- waiting
    def events(self):
        return read_events(self.events_path)

    def wait_step(self, step, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if any(e["ev"] == "step" and e["step"] >= step for e in self.events()):
                return True
            time.sleep(0.02)
        raise TimeoutError(f"step {step} not reached")

    def wait_done(self, timeout=120):
        end = time.time() + timeout
        for name, p in self.agents.items():
            try:
                p.wait(timeout=max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                pass
        return {n: p.poll() for n, p in self.agents.items()}

    def stop(self):
        for t in self.timers:
            t.cancel()
        for p in list(self.agents.values()) + [self.server]:
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGCONT)
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()
