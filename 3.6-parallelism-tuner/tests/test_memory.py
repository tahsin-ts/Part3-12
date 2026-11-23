import pytest

from ptuner.config import GIB, Knobs, Model, Plan, get_model
from ptuner import memory, validate


def test_zero_paper_figure1_exact():
    for stage, published, model in validate.zero_check():
        assert model == pytest.approx(published, abs=0.05), stage


def test_param_count_matches_published_sizes():
    for name, size in [("gpt-1.7b", 1.7), ("gpt-18b", 18.4), ("gpt-146b", 145.6), ("gpt-1t", 1008.0)]:
        assert get_model(name).params() / 1e9 == pytest.approx(size, rel=0.03)


def test_activation_formulas_korthikanti():
    m = Model("t", 1, 1024, 16, seq=512)
    sbh = 512 * 2 * 1024
    assert memory.layer_activation_bytes(m, 2, 1, "none") == pytest.approx(sbh * (34 + 5 * 16 * 512 / 1024))
    assert memory.layer_activation_bytes(m, 2, 4, "none", seq_parallel=True) == pytest.approx(sbh * (34 + 40) / 4)
    assert memory.layer_activation_bytes(m, 2, 4, "none", seq_parallel=False) == pytest.approx(sbh * (10 + 6 + 40 / 4))
    assert memory.layer_activation_bytes(m, 2, 4, "selective") == pytest.approx(sbh * 34 / 4)
    assert memory.layer_activation_bytes(m, 2, 4, "full") == pytest.approx(2 * sbh / 4)


def test_1f1b_in_flight_is_min_p_m():
    assert memory.in_flight(Plan(1, 1, 8, 0, 1, "none"), 32) == 8
    assert memory.in_flight(Plan(1, 1, 8, 0, 1, "none"), 3) == 3
    assert memory.in_flight(Plan(1, 1, 8, 0, 1, "none"), 32, stage=7) == 1


def test_pipeline_and_tensor_split_weights():
    m = get_model("gpt3-13b")
    one = memory.stage_params(m, Plan(1, 1, 1, 0, 1, "none"))
    assert one == pytest.approx(m.params(), rel=0.01)
    assert memory.stage_params(m, Plan(1, 2, 4, 0, 1, "none")) < one / 7


def test_175b_needs_pipeline_on_80gb():
    m = get_model("gpt3-175b")
    bad = memory.estimate(m, Plan(16, 8, 1, 0, 1, "full"), 8)
    good = memory.estimate(m, Plan(2, 8, 8, 1, 1, "full"), 64)
    assert bad.total / GIB > 80 and good.total / GIB < 80


def test_published_megatron_configs_fit_in_80gb():
    for r in validate.megatron_rows():
        assert r.mem_gib < 80, r.name


def test_recompute_orders_memory():
    m = get_model("gpt3-6.7b")
    tot = [memory.estimate(m, Plan(8, 1, 1, 1, 4, rc), 8).activations for rc in ("none", "selective", "full")]
    assert tot[0] > tot[1] > tot[2]
