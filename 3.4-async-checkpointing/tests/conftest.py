import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ackpt import LocalFS, ObjectStore, make_state  # noqa: E402


@pytest.fixture(params=["localfs", "objectstore"])
def store(request, tmp_path):
    return LocalFS(str(tmp_path / "ckpt")) if request.param == "localfs" else ObjectStore()


@pytest.fixture
def state():
    s = make_state(n_params=20_000, seed=3, layers=3)
    s["optim.step"][...] = 7
    s["model.tiny"] = np.arange(3, dtype=np.float32)      # fewer rows than ranks
    return s
