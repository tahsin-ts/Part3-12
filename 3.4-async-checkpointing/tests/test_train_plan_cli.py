import math

import numpy as np
import pytest

from ackpt import Checkpointer, ObjectStore, load_latest, shard_state, unshard
from ackpt import planning as pl
from ackpt.cli import main
from ackpt.manifest import step_dir
from ackpt.storage import TransientError
from ackpt.train import Trainer


def test_resume_on_different_world_is_bit_identical():
    store = ObjectStore()
    tr = Trainer(n_params=60_000, layers=3)
    ck = Checkpointer(store, 4)
    for _ in range(12):
        tr.train_step()
        if tr.step % 5 == 0:
            ck.save(tr.step, shard_state(tr.state, 4))
    ck.wait()
    step, ranks, _ = load_latest(store, 3)
    assert step == 10
    resumed = Trainer(n_params=60_000, layers=3)
    resumed.state = unshard(ranks)
    while resumed.step < 25:
        resumed.train_step()
    while tr.step < 25:
        tr.train_step()
    assert all(np.array_equal(tr.state[k], resumed.state[k]) for k in tr.state)


def test_loss_goes_down():
    tr = Trainer(n_params=60_000, layers=3)
    first = tr.train_step()
    for _ in range(30):
        last = tr.train_step()
    assert last < first


def test_planning_numbers():
    assert pl.checkpoint_bytes(7e9) == 98e9
    assert pl.write_seconds(980e9, 8, 2.5e9) == pytest.approx(49.0)
    assert pl.young_daly_interval(40, 3600) == pytest.approx(math.sqrt(2 * 40 * 3600))
    assert pl.job_mtbf(50_000 * 3600, 16_384) / 3600 == pytest.approx(3.05, abs=0.01)


def test_young_daly_minimises_waste():
    c, m = 40.0, 11_000.0
    t = pl.young_daly_interval(c, m)
    w = pl.waste_fraction(t, c, m)
    assert w < pl.waste_fraction(t / 2, c, m) and w < pl.waste_fraction(t * 2, c, m)


def test_objectstore_bandwidth_and_faults():
    s = ObjectStore(bandwidth_bps=10e6, fail_every_n=3)
    import time
    t = time.perf_counter()
    s.put("a", b"x" * 500_000)                                  # 0.05 s at 10 MB/s
    assert time.perf_counter() - t >= 0.045
    assert s.get("a", 10, 5) == b"xxxxx"
    with pytest.raises(TransientError):
        s.get("a")


def test_cli_demo_plan_verify(tmp_path, capsys):
    assert main(["demo", "--dir", str(tmp_path), "--params", "60000"]) == 0
    assert "bit for bit: True" in capsys.readouterr().out
    assert main(["verify", str(tmp_path)]) == 0
    p = tmp_path / step_dir(40) / "rank_00002.bin"
    b = bytearray(p.read_bytes()); b[7] ^= 1; p.write_bytes(bytes(b))
    assert main(["verify", str(tmp_path)]) == 1
    assert "CORRUPT" in capsys.readouterr().out
    main(["plan", "--params", "70e9", "--nodes", "8"])
    assert "980 GB" in capsys.readouterr().out
    main(["gc", str(tmp_path), "--keep", "1"])
    assert "[30]" in capsys.readouterr().out
