"""End to end: real processes, real kill -9, real SIGSTOP. Final weights must equal a fault-free run."""
import numpy as np
import pytest

from elastic import model as M
from elastic.analysis import analyze
from elastic.cli import run_scenario

STEPS = 120


def reference_digest(steps=STEPS):
    """Single process, no coordinator: the exact maths every elastic run must reproduce."""
    x, y = M.make_data(0)
    like = M.init_params(0)
    theta, opt = M.flatten(like), M.SGDMomentum(M.flatten(like).size)
    for s in range(steps):
        p, idx = M.unflatten(theta, like), M.batch_indices(s)
        chunks = {c: M.chunk_grad(p, x[idx[c * M.CHUNK:(c + 1) * M.CHUNK]], y[idx[c * M.CHUNK:(c + 1) * M.CHUNK]]) for c in range(M.N_CHUNKS)}
        theta = opt.step(theta, M.reduce_chunks(chunks)[0])
    return M.digest(theta)


REF = reference_digest()


@pytest.mark.parametrize("scenario,world_after", [("kill-worker", 3), ("kill-node", 2), ("pause", 2)])
def test_fault_recovers_and_matches_fault_free_weights(tmp_path, scenario, world_after):
    rep = run_scenario(scenario, str(tmp_path / scenario), at_step=60, quiet=True, steps=STEPS, ckpt_every=10, lease_ttl=1.0)
    f = rep["faults"][0]
    assert rep["digest"] == REF
    assert rep["unique_steps"] == STEPS
    assert f["new_world"] == world_after
    assert 0 <= f["lost_steps"] <= 10                      # never more than one checkpoint interval
    if scenario == "pause":
        assert rep["fenced_requests"] >= 1                   # the frozen node came back and was refused
    if scenario == "kill-node":
        assert 0.6 <= f["detect_s"] <= 2.5                  # one lease TTL minus up to one heartbeat interval
