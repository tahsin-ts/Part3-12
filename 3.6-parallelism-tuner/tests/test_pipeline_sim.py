import numpy as np
import pytest

from ptuner import pipeline_sim as ps


@pytest.mark.parametrize("schedule", ["1f1b", "gpipe"])
def test_bubble_matches_formula(schedule):
    for p in (1, 2, 3, 4, 8):
        for m in (1, 2, 5, 8, 32):
            r = ps.simulate(p, m, 1.0, 2.0, schedule)
            assert r.bubble_fraction == pytest.approx(ps.formula(p, m), abs=1e-12)
            assert r.makespan == pytest.approx((m + p - 1) * 3.0)


def test_1f1b_bounds_in_flight_gpipe_does_not():
    p, m = 4, 16
    f = ps.simulate(p, m, schedule="1f1b")
    g = ps.simulate(p, m, schedule="gpipe")
    assert list(f.peak_in_flight) == [min(p - i, m) for i in range(p)]
    assert list(g.peak_in_flight) == [m] * p


def test_order_has_every_op_once():
    for i in range(4):
        o = ps.order_1f1b(i, 4, 6)
        assert sorted(o) == sorted([("F", j) for j in range(6)] + [("B", j) for j in range(6)])


def test_jitter_makes_bubble_worse_than_formula():
    rng = np.random.default_rng(0)
    p, m = 4, 16
    tf = rng.uniform(0.7, 1.3, (p, m))
    r = ps.simulate(p, m, tf, 2 * tf)
    assert r.bubble_fraction > ps.formula(p, m)


def test_communication_delay_extends_makespan():
    a = ps.simulate(4, 8, comm=0.0)
    b = ps.simulate(4, 8, comm=0.1)
    assert b.makespan > a.makespan
