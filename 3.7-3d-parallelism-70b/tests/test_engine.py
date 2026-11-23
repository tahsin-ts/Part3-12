import os

import pytest

from threed import engine as E
from threed.model import Config
from threed.topology import Grid

CFG = Config()


@pytest.mark.parametrize("grid,sched", [(Grid(2, 2, 2), "1f1b"), (Grid(1, 4, 1), "gpipe"), (Grid(4, 1, 2), "1f1b"),
                                        (Grid(1, 2, 4), "1f1b")])
def test_3d_training_equals_single_process(grid, sched):
    ref_l, ref = E.train_reference(CFG, 3, 16, lr=0.5)
    res = E.train(CFG, grid, 3, 16, 2, lr=0.5, schedule=sched, return_params=True)
    assert max(abs(a - b) for a, b in zip(res[0]["losses"], ref_l)) < 1e-12
    assert E.max_diff(ref, E.gather_full(CFG, grid, [r for r in res if r["coords"][1] == 0])) < 1e-12


def test_measured_bytes_equal_formula():
    g = Grid(2, 2, 2)
    res = E.train(CFG, g, 2, 16, 2)
    for r in res:
        e = E.expected_bytes(CFG, g, r["rank"], 2, r["m"])
        assert {k: r["bytes"][k] // 2 for k in ("tp", "pp", "dp")} == {k: e[k] for k in ("tp", "pp", "dp")}


def test_crash_is_detected_not_hung(tmp_path):
    with pytest.raises(E.WorkerFailed) as ei:
        E.train(CFG, Grid(2, 2, 1), 3, 8, 2, timeout=5, fail_at=(3, 1, str(tmp_path / "m")))
    assert 3 in ei.value.ranks


def test_restart_from_checkpoint_gives_identical_model(tmp_path):
    ref_l, ref = E.train_reference(CFG, 5, 16)
    res, restarts, log = E.train_with_restarts(CFG, Grid(2, 2, 2), 5, 16, 2, str(tmp_path / "ck"),
                                               fail_at=(5, 3, str(tmp_path / "m")), return_params=True, timeout=5)
    assert restarts == 1 and res[0]["start_step"] == 3
    assert E.max_diff(ref, E.gather_full(CFG, Grid(2, 2, 2), [r for r in res if r["coords"][1] == 0])) < 1e-12
    assert sorted(os.listdir(tmp_path / "ck" / "step_000005")) == ["pp0_tp0.npz", "pp0_tp1.npz", "pp1_tp0.npz", "pp1_tp1.npz"]


def test_uneven_batch_rejected():
    with pytest.raises(E.WorkerFailed):
        E.train(CFG, Grid(1, 1, 2), 1, 6, 2, timeout=5)
