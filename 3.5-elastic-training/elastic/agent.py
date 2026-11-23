"""Node agent: joins the rendezvous, launches the local workers, watches them, restarts the group.

One agent per machine ("node"), like torchrun's elastic agent. Loop:
  rendezvous -> spawn nproc workers -> monitor (poll workers + heartbeat) -> on any change, stop and rendezvous again.
Self-fencing: if the agent cannot renew its lease for lease_ttl seconds it kills its own workers,
because the coordinator has already given their ranks to someone else.
"""
import os
import signal
import subprocess
import sys
import time

from .client import Client, Gone
from .events import EventLog

DONE, RESTART, FAILED = "done", "restart", "failed"


class Agent:
    def __init__(self, node, endpoint, nproc=1, ckpt_dir="ckpt", events=None, faults=None, steps=300,
                 ckpt_every=20, step_sleep=0.01, seed=0, hb_interval=0.25, lease_ttl=2.0,
                 max_restarts=3, join_timeout=60.0):
        self.node, self.endpoint, self.nproc = node, endpoint, nproc
        self.c = Client(endpoint, node, faults_dir=faults)
        self.log = EventLog(events, f"agent@{node}")
        self.hb_interval, self.lease_ttl = hb_interval, lease_ttl
        self.max_restarts, self.join_timeout = max_restarts, join_timeout
        self.base_env = {"ELASTIC_ENDPOINT": endpoint, "ELASTIC_NODE": node, "ELASTIC_CKPT_DIR": ckpt_dir,
                         "ELASTIC_EVENTS": events or "", "ELASTIC_FAULTS": faults or "", "ELASTIC_STEPS": str(steps),
                         "ELASTIC_CKPT_EVERY": str(ckpt_every), "ELASTIC_STEP_SLEEP": str(step_sleep), "ELASTIC_SEED": str(seed)}
        self.procs = []

    # ---------------------------------------------------------------- rendezvous
    def rendezvous(self):
        end = time.monotonic() + self.join_timeout
        while time.monotonic() < end:
            try:
                a = self.c.join(self.nproc)
                while "world" not in a and a["state"] != "done" and time.monotonic() < end:
                    a = self.c.wait(timeout=1.0)
                    if not a.get("waiting") and "world" not in a and a["state"] != "done":
                        break                                  # not registered for this round (e.g. banned): join again
                if "world" in a or a["state"] == "done":
                    return a
            except (ConnectionError, Gone):
                pass                                           # partitioned or forgotten: keep trying
            time.sleep(0.2)
        raise TimeoutError(f"{self.node}: no rendezvous within {self.join_timeout}s")

    # ---------------------------------------------------------------- workers
    def spawn(self, a):
        self.procs = []
        for local in range(self.nproc):
            env = dict(os.environ, **self.base_env, RANK=str(a["rank_offset"] + local), LOCAL_RANK=str(local),
                       WORLD_SIZE=str(a["world"]), ELASTIC_GEN=str(a["gen"]))
            self.procs.append(subprocess.Popen([sys.executable, "-m", "elastic.worker"], env=env))
        self.log.emit("workers_started", gen=a["gen"], world=a["world"], ranks=[a["rank_offset"] + i for i in range(self.nproc)],
                      pids=[p.pid for p in self.procs])

    def stop_workers(self, grace=1.0):
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        end = time.monotonic() + grace
        for p in self.procs:
            try:
                p.wait(timeout=max(0.01, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def monitor(self, a):
        gen, last_ok, last_hb = a["gen"], time.monotonic(), 0.0
        while True:
            codes = [p.poll() for p in self.procs]
            if all(cd == 0 for cd in codes):
                return DONE
            bad = [(i, cd) for i, cd in enumerate(codes) if cd not in (None, 0)]
            if bad:
                i, cd = bad[0]
                if cd == 3:                                     # fenced: the generation moved on
                    self.log.emit("worker_fenced", gen=gen, local_rank=i)
                    self.stop_workers()
                    return RESTART
                self.log.emit("worker_failed", gen=gen, local_rank=i, rank=a["rank_offset"] + i, exitcode=cd)
                try:
                    self.c.fail(gen, f"local rank {i} exited with {cd}")
                except (ConnectionError, Gone):
                    pass
                self.stop_workers()
                return FAILED
            now = time.monotonic()
            if now - last_hb >= self.hb_interval:
                last_hb = now
                try:
                    hb = self.c.heartbeat(gen)
                    last_ok = now
                    if hb["state"] == "done":
                        for p in self.procs:
                            try:
                                p.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                p.kill()
                        return DONE
                    if hb["gen"] != gen or hb["state"] != "running" or not hb["member"]:
                        self.log.emit("rerendezvous", old_gen=gen, new_gen=hb["gen"], state=hb["state"])
                        self.stop_workers()
                        return RESTART
                except Gone:
                    self.log.emit("lease_lost", gen=gen)
                    self.stop_workers()
                    return RESTART
                except ConnectionError:
                    if now - last_ok > self.lease_ttl:
                        self.log.emit("self_fenced", gen=gen, silent_s=round(now - last_ok, 3))
                        self.stop_workers(grace=0.0)
                        return RESTART
            time.sleep(0.02)

    # ---------------------------------------------------------------- main loop
    def run(self):
        restarts = 0
        self.log.emit("agent_start", pid=os.getpid(), nproc=self.nproc)
        try:
            while True:
                a = self.rendezvous()
                if a["state"] == "done":
                    self.log.emit("agent_exit", code=0)
                    return 0
                self.spawn(a)
                out = self.monitor(a)
                if out == DONE:
                    try:
                        self.c.complete(a["gen"])
                    except (ConnectionError, Gone):
                        pass
                    self.log.emit("agent_exit", code=0)
                    return 0
                if out == FAILED:
                    restarts += 1
                    if restarts > self.max_restarts:
                        self.log.emit("agent_give_up", restarts=restarts)
                        try:
                            self.c.call("/leave", {"node": self.node})
                        except (ConnectionError, Gone):
                            pass
                        return 1
        except TimeoutError as e:
            self.log.emit("agent_give_up", why=str(e))
            return 2
        finally:
            self.stop_workers(grace=0.2)
