import pytest

from threed import planner as P

MS, H100, A100 = P.ModelSpec(), P.HARDWARE["h100"], P.HARDWARE["a100"]


def test_llama2_70b_parameter_count_is_exact():
    assert MS.params == 68_976_648_192
    assert MS.layer_params == 855_654_400


def test_pure_data_parallel_does_not_fit():
    assert not P.memory(MS, H100, P.Plan(1, 1, 64))["fits"]          # 69 B x 6 bytes alone is 385 GiB


def test_activations_on_stage0_do_not_depend_on_p():
    a = [P.memory(MS, H100, P.Plan(8, p, 1))["activations"] for p in (2, 4, 8)]
    assert max(a) - min(a) < 1e-6 * a[0]


@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024])
def test_best_plan_fits_and_is_sane(n):
    scored, _ = P.best(MS, H100, n)
    _, pl = scored[0]
    assert pl.n == n and P.memory(MS, H100, pl)["fits"] and pl.t <= 8
    s = P.step(MS, H100, pl)
    assert 0.3 < s["mfu"] < 0.6


def test_more_gpus_fewer_days_more_gpu_hours():
    d = [P.days(MS, H100, P.best(MS, H100, n)[0][0][1], 2e12) for n in (64, 1024)]
    assert d[1][0] < d[0][0] / 10 and d[1][1] > d[0][1]


def test_calibration_against_llama2_gpu_hours():
    _, gh = P.days(MS, A100, P.best(MS, A100, 2048)[0][0][1], 2e12)
    assert abs(gh / 1_720_320 - 1) < 0.15                             # Llama 2 paper: 1,720,320 A100-hours for 70B


def test_checkpoint_policy():
    c = P.checkpoint_policy(MS, 1024)
    assert round(c["size_bytes"] / 1e9) == 966
    assert abs(c["interval_min"] * 60 - (2 * c["stall_s"] * c["mtbf_h"] * 3600) ** 0.5) < 1e-6
