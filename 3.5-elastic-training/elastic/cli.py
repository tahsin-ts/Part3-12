"""Command line: elastic server | agent | status | demo | bench."""
import argparse
import json
import os
import sys
import time

from .analysis import analyze, fmt

SCENARIOS = ("baseline", "kill-worker", "kill-node", "partition", "pause", "slow", "slow-evict", "scale-up")


def run_scenario(name, workdir, at_step=120, quiet=False, **kw):
    """Run one fault scenario on a local 3-node cluster; returns the analysis report."""
    from .cluster import LocalCluster
    if name == "slow-evict":
        kw.setdefault("evict_lag", 0.02)
    nodes = 2 if name == "scale-up" else kw.pop("nodes", 3)
    with LocalCluster(workdir, nodes=nodes, **kw) as cl:
        cl.wait_step(at_step)
        if name == "kill-worker":
            cl.kill_worker("n2")
        elif name == "kill-node":
            cl.kill_node("n3")
        elif name == "partition":
            cl.partition("n3", 4.0)
        elif name == "pause":
            cl.pause("n3", 4.0)
        elif name in ("slow", "slow-evict"):
            cl.slow("n3", 0.04)
        elif name == "scale-up":
            cl.log.emit("fault", kind="scale_up", node="n3")
            cl.add_node("n3")
        codes = cl.wait_done(timeout=180)
        rep = analyze(cl.events())
        rep["exit_codes"] = codes
    if not quiet:
        print(fmt(rep, f"{name}  (agent exit codes {codes})"))
    return rep


def cmd_server(a):
    from .server import serve
    httpd, co = serve(a.host, a.port, a.events, min_nodes=a.min_nodes, max_nodes=a.max_nodes, lease_ttl=a.lease_ttl,
                      last_call=a.last_call, evict_lag=a.evict_lag)
    port = httpd.server_address[1]
    if a.port_file:
        with open(a.port_file + ".tmp", "w") as f:
            f.write(str(port))
        os.replace(a.port_file + ".tmp", a.port_file)
    print(f"rendezvous listening on {a.host}:{port} (min {a.min_nodes}, max {a.max_nodes}, lease {a.lease_ttl}s)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


def cmd_agent(a):
    from .agent import Agent
    return Agent(a.node, a.endpoint, a.nproc, a.ckpt_dir, a.events, a.faults, a.steps, a.ckpt_every, a.step_sleep, a.seed,
                 a.hb_interval, a.lease_ttl, a.max_restarts, a.join_timeout).run()


def cmd_status(a):
    from .client import Client
    print(json.dumps(Client(a.endpoint).status(), indent=1))
    return 0


def cmd_demo(a):
    names = SCENARIOS if a.scenario == "all" else [a.scenario]
    reps = {n: run_scenario(n, os.path.join(a.workdir, n), steps=a.steps) for n in names}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(reps, f, indent=1)
    return 0


def cmd_bench(a):
    from .bench import run_bench
    run_bench(a.workdir, a.out)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="elastic", description="Fault-tolerant elastic data-parallel training on one machine or many.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("server", help="run the rendezvous / coordination server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=29400)
    s.add_argument("--port-file")
    s.add_argument("--min-nodes", type=int, default=1)
    s.add_argument("--max-nodes", type=int, default=4)
    s.add_argument("--lease-ttl", type=float, default=2.0)
    s.add_argument("--last-call", type=float, default=0.5)
    s.add_argument("--evict-lag", type=float, default=None, help="evict a node whose barrier lag stays above this (s)")
    s.add_argument("--events")
    s.set_defaults(fn=cmd_server)
    g = sub.add_parser("agent", help="run a node agent that launches and supervises workers")
    g.add_argument("--endpoint", required=True)
    g.add_argument("--node", required=True)
    g.add_argument("--nproc", type=int, default=1)
    g.add_argument("--ckpt-dir", default="ckpt")
    g.add_argument("--events")
    g.add_argument("--faults")
    g.add_argument("--steps", type=int, default=300)
    g.add_argument("--ckpt-every", type=int, default=20)
    g.add_argument("--step-sleep", type=float, default=0.01)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--hb-interval", type=float, default=0.25)
    g.add_argument("--lease-ttl", type=float, default=2.0)
    g.add_argument("--max-restarts", type=int, default=3)
    g.add_argument("--join-timeout", type=float, default=60.0)
    g.set_defaults(fn=cmd_agent)
    t = sub.add_parser("status", help="print the coordinator's view")
    t.add_argument("--endpoint", default="127.0.0.1:29400")
    t.set_defaults(fn=cmd_status)
    d = sub.add_parser("demo", help="run a fault scenario on a local 3-node cluster")
    d.add_argument("--scenario", choices=SCENARIOS + ("all",), default="kill-node")
    d.add_argument("--workdir", default="/tmp/elastic-demo")
    d.add_argument("--steps", type=int, default=300)
    d.add_argument("--json")
    d.set_defaults(fn=cmd_demo)
    b = sub.add_parser("bench", help="detection vs lease, lost work vs checkpoint interval, all scenarios")
    b.add_argument("--workdir", default="/tmp/elastic-bench")
    b.add_argument("--out", default="bench_results.json")
    b.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
