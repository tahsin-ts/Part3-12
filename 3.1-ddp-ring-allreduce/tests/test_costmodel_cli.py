import numpy as np

from ringddp import costmodel
from ringddp.cli import main


def test_fit_recovers_alpha_and_bandwidth():
    sizes = np.array([1e3, 1e5, 1e6, 1e7])
    t = [costmodel.ring_time(s, 4, 20e-6, 2e9) for s in sizes]
    alpha, bw = costmodel.fit(sizes, t, 4)
    assert abs(alpha - 20e-6) < 1e-9 and abs(bw - 2e9) / 2e9 < 1e-6


def test_projection_numbers():
    rows = {name: times for name, _, times in costmodel.projection()}
    nvlink, eth = rows["Llama-2-7B"]
    assert 0.09 < nvlink < 0.11 and 2.3 < eth < 2.4
    assert costmodel.ring_time(1e9, 1, 1e-6, 1e9) == 0.0


def test_cli_check_and_project(capsys):
    assert main(["check", "--workers-list", "2", "--epochs", "1", "--samples", "1024", "--batch", "128", "--hidden", "32"]) == 0
    assert main(["project"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "PROJECTED" in out


def test_cli_train_reports_failure(capsys):
    assert main(["train", "--workers", "2", "--epochs", "1", "--samples", "1024", "--batch", "128", "--hidden", "32",
                 "--crash-rank", "0", "--crash-step", "1", "--timeout", "2"]) == 3
    assert "exit code 17" in capsys.readouterr().out


def test_worker_command_runs_as_separate_programs():
    """Two independent OS processes, as on two hosts: only the peer list connects them."""
    import os
    import socket
    import subprocess
    import sys
    ports = []
    for _ in range(2):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        s.close()
    peers = ",".join(f"127.0.0.1:{p}" for p in ports)
    env = dict(os.environ, PYTHONPATH="src")
    args = ["--peers", peers, "--epochs", "1", "--samples", "1024", "--batch", "128", "--hidden", "32"]
    procs = [subprocess.Popen([sys.executable, "-m", "ringddp", "worker", "--rank", str(r)] + args,
                              env=env, stdout=subprocess.PIPE, text=True) for r in range(2)]
    outs = [p.communicate(timeout=60)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs)
    assert outs[0].split("sha256 ")[1][:16] == outs[1].split("sha256 ")[1][:16]
