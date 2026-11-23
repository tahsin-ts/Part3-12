import pytest

from threed import engine as E
from threed import pipeline as PL
from threed.model import Config
from threed.topology import Grid


@pytest.mark.parametrize("name", ["gpipe", "1f1b"])
@pytest.mark.parametrize("p,m", [(1, 1), (2, 1), (4, 8), (4, 3), (8, 16)])
def test_schedules_valid_and_bubble_matches_formula(name, p, m):
    sched = PL.SCHEDULES[name](p, m)
    assert PL.check_schedule(sched, m)
    _, bubble, _ = PL.simulate(sched, 1.0, 2.0)
    assert abs(bubble - PL.bubble_formula(p, m)) < 1e-12


def test_memory_gpipe_vs_1f1b():
    p, m = 4, 16
    assert [PL.peak_in_flight(o) for o in PL.gpipe(p, m)] == [16] * 4
    assert [PL.peak_in_flight(o) for o in PL.one_f_one_b(p, m)] == [4, 3, 2, 1]


def test_split_layers():
    assert PL.split_layers(80, 8) == [(i * 10, i * 10 + 10) for i in range(8)]
    assert PL.split_layers(5, 2) == [(0, 3), (3, 5)]


def test_measured_bubble_close_to_formula():
    p, m = 4, 8
    res = E.train(Config(layers=p), Grid(1, p, 1), 1, m, 1, schedule="1f1b", pad=(0.01, 0.02), timeout=60)
    span = max(r["timing"][0]["t1"] for r in res) - min(r["timing"][0]["t0"] for r in res)
    busy = sum(r["timing"][0]["busy"] for r in res) / p
    assert abs((1 - busy / span) - PL.bubble_formula(p, m)) < 0.1
    assert res[0]["timing"][0]["peak_live"] == p
