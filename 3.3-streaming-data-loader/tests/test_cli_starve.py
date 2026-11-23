import json
import os

import pytest

from streamload import starve
from streamload.cli import main


def test_needed_rate_arithmetic():
    # A100 312 TFLOP/s x 25% / (3 x 2 x 4.1 GFLOP) = 3,170.7 images/s
    assert starve.needed_rate("A100", "resnet50") == pytest.approx(3170.7, rel=1e-4)
    r = starve.starve("H100", "llama-7b", per_core=1000, cores_per_gpu=12)
    assert r["gpu_busy_max"] == 1.0 and r["read_MBps"] < 1
    r = starve.starve("H100", "resnet50", per_core=800, cores_per_gpu=4)
    assert r["gpu_busy_max"] == pytest.approx(3200 / starve.needed_rate("H100", "resnet50"))


def test_cli_make_verify_read_resume(tmp_path, capsys):
    out = str(tmp_path / "ds")
    assert main(["make-data", out, "--records", "96", "--side", "32", "--per-shard", "12"]) == 0
    assert main(["verify", out]) == 0
    state = str(tmp_path / "s.json")
    assert main(["read", out, "--workers", "0", "--batch", "8", "--max-steps", "3", "--state-out", state]) == 0
    assert json.load(open(state))["step"] == 3
    assert main(["resume-check", out, "--workers", "0", "--batch", "8", "--stop-at", "4"]) == 0
    text = capsys.readouterr().out
    assert "exact resume: YES | duplicates: 0" in text and "96 records" in text
    with open(os.path.join(out, "shard-00001.sld"), "r+b") as f:
        f.write(b"XXXX")
    assert main(["verify", out]) == 1
