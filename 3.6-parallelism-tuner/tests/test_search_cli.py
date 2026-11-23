import json

import pytest

from ptuner.config import GIB, Knobs, Model, get_cluster, get_model
from ptuner import cli, explain, search


def test_divisors():
    assert search.divisors(12) == [1, 2, 3, 4, 6, 12]


def test_every_ranked_plan_is_valid_and_fits():
    m, c = get_model("gpt3-13b"), get_cluster("dgx-a100", 8)
    res = search.search(m, c, 512)
    assert res.ranked and res.examined == len(res.ranked) + sum(res.pruned.values())
    for x in res.ranked:
        p = x.plan
        assert p.dp * p.tp * p.pp == 8 and 512 % (p.dp * p.mbs) == 0
        assert x.mem.total <= c.mem_gib * GIB
    scores = [x.score for x in res.ranked]
    assert scores == sorted(scores, reverse=True)


def test_pruning_reasons():
    c = get_cluster("dgx-a100", 48)
    res = search.search(get_model("gpt3-6.7b"), c, 1536)
    for why in ("tp spans nodes", "tp does not divide heads", "pp does not divide layers", "zero>=2 with pp>1 unsupported", "out of memory"):
        assert res.pruned[why] > 0


def test_slow_interconnect_avoids_tensor_parallelism():
    m = get_model("gpt3-6.7b")
    fast = search.search(m, get_cluster("dgx-a100", 8), 256)
    slow = search.search(m, get_cluster("l40s-pcie", 8), 256)
    best_tp_fast = max(x.plan.tp for x in explain.best_per_shape(fast)[:3])
    assert slow.best.plan.tp == 1
    tp8_fast = [x for x in fast.ranked if x.plan.tp == 8][0].score / fast.best.score
    tp8_slow = [x for x in slow.ranked if x.plan.tp == 8][0].score / slow.best.score
    assert tp8_slow < tp8_fast and best_tp_fast >= 1


def test_impossible_model_has_no_plan():
    res = search.search(get_model("gpt3-175b"), get_cluster("rtx4090", 2), 64)
    assert res.best is None
    assert "No configuration fits" in explain.reasons(res, get_model("gpt3-175b"), get_cluster("rtx4090", 2))[0]


def test_cli_search_text_and_json(capsys):
    assert cli.main(["search", "--model", "gpt3-1.3b", "--cluster", "rtx4090", "--gpus", "2", "--batch", "64"]) == 0
    out = capsys.readouterr().out
    assert "why #1" in out and "PROJECTED" in out
    assert cli.main(["search", "--layers", "12", "--hidden", "768", "--heads", "12", "--gpus", "4", "--batch", "32", "--json", "--top", "3"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 3 and data[0]["tokens_per_s_per_gpu"] >= data[1]["tokens_per_s_per_gpu"]


def test_cli_no_fit_exit_code(capsys):
    assert cli.main(["search", "--model", "gpt3-175b", "--cluster", "rtx4090", "--gpus", "2", "--batch", "64"]) == 1


def test_cli_validate_bubble_list(capsys):
    assert cli.main(["validate"]) == 0
    assert "mean absolute error" in capsys.readouterr().out
    assert cli.main(["bubble", "--p", "4", "--m", "8"]) == 0
    assert "0.2727" in capsys.readouterr().out
    assert cli.main(["list"]) == 0
    assert "dgx-h100" in capsys.readouterr().out


def test_cli_requires_model():
    with pytest.raises(SystemExit):
        cli.main(["search", "--gpus", "8"])


def test_cluster_file(tmp_path, capsys):
    f = tmp_path / "c.json"
    f.write_text('{"name": "x", "gpus": 4, "gpus_per_node": 4, "mem_gib": 24, "peak_tflops": 165, '
                 '"hbm_gbs": 1008, "intra_gbs": 12, "inter_gbs": 10}')
    assert cli.main(["search", "--model", "gpt3-1.3b", "--cluster-file", str(f), "--batch", "64", "--top", "2"]) == 0
    assert "cluster x: 4 GPUs" in capsys.readouterr().out
