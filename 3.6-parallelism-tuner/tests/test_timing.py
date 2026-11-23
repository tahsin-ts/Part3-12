import pytest

from ptuner.config import Knobs, Plan, get_cluster, get_model
from ptuner import timing, validate


def test_hardware_flops_equal_megatron_equation_3():
    m = get_model("gpt-146b")
    B, s, l, h, V = 2304, m.seq, m.layers, m.hidden, m.vocab
    eq3 = 96 * B * s * l * h * h * (1 + s / (6 * h) + V / (16 * l * h))
    assert timing.hardware_flops(m, B * s, "full") == pytest.approx(eq3, rel=1e-12)


def test_model_flops_close_to_6_n_tokens():
    m = get_model("gpt3-13b")
    assert timing.model_flops(m, 1) == pytest.approx(6 * m.params(), rel=0.15)


def test_ring_allreduce_alpha_beta():
    assert timing.ring_allreduce_s(1e9, 1, 100, 5) == 0
    t = timing.ring_allreduce_s(8e9, 8, 100, 10)
    assert t == pytest.approx(2 * 7 / 8 * 8e9 / 100e9 + 14 * 10e-6)


def test_link_choice_intra_vs_inter():
    c = get_cluster("dgx-a100", 64)
    assert timing.link(c, 8, 1)[0] == c.intra_gbs
    assert timing.link(c, 2, 8)[0] == c.inter_gbs


def test_bubble_fraction_formula_in_step_model():
    m, c = get_model("gpt-39b"), get_cluster("dgx-a100", 512)
    r = timing.step_time(m, c, Plan(32, 8, 2, 1, 1, "full"), 1536)
    assert r.micro_batches == 48
    assert r.bubble_fraction == pytest.approx(1 / 49)
    assert r.bubble / (r.bubble + r.compute + r.tp_comm + r.pp_comm) == pytest.approx(1 / 49)


def test_throughput_consistent_with_step_time():
    m, c = get_model("gpt3-13b"), get_cluster("dgx-a100", 8)
    r = timing.step_time(m, c, Plan(8, 1, 1, 2, 1, "selective"), 64)
    assert r.tokens_per_s_per_gpu == pytest.approx(64 * 2048 / r.total / 8)
    assert 0 < r.mfu < r.hfu < 1


def test_zero3_costs_more_communication_than_zero1():
    m, c = get_model("gpt3-6.7b"), get_cluster("l40s-pcie", 8)
    z1 = timing.step_time(m, c, Plan(8, 1, 1, 1, 1, "full"), 256)
    z3 = timing.step_time(m, c, Plan(8, 1, 1, 3, 1, "full"), 256)
    assert z3.dp_comm > 10 * z1.dp_comm


def test_megatron_table1_error_bounds():
    rows = validate.megatron_rows()
    assert validate.mape(rows) < 5.0
    assert all(abs(r.error_pct) < 10 for r in rows)


def test_leave_one_out_generalises():
    loo = validate.leave_one_out()
    assert sum(abs(e) for _, _, e in loo) / len(loo) < 5.0
    assert all(0.6 < f < 0.7 for _, f, _ in loo)
