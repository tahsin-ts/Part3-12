import os

import numpy as np
import pytest

from elastic.checkpoint import CheckpointStore, StaleWriter


def test_roundtrip_and_latest(tmp_path):
    st = CheckpointStore(str(tmp_path))
    assert st.load() is None
    st.save(10, {"theta": np.arange(4.0)}, {"gen": 1}, token=1)
    st.save(20, {"theta": np.arange(4.0) * 2}, {"gen": 1}, token=1)
    step, arr, meta = st.load()
    assert step == 20 and arr["theta"].tolist() == [0, 2, 4, 6] and meta == {"gen": 1}


def test_keeps_only_last_n(tmp_path):
    st = CheckpointStore(str(tmp_path), keep=3)
    for s in range(1, 7):
        st.save(s, {"theta": np.zeros(2)}, {}, token=1)
    assert st.steps() == [4, 5, 6]


def test_zombie_writer_is_fenced(tmp_path):
    st = CheckpointStore(str(tmp_path))
    st.save(10, {"theta": np.ones(2)}, {}, token=1)
    st.claim(2)                                   # generation 2 starts
    with pytest.raises(StaleWriter):
        st.save(12, {"theta": np.zeros(2)}, {}, token=1)
    assert st.load()[0] == 10 and st.steps() == [10]
    assert not [f for f in os.listdir(tmp_path) if ".tmp." in f]
    with pytest.raises(StaleWriter):
        st.claim(1)


def test_corrupt_latest_falls_back(tmp_path):
    st = CheckpointStore(str(tmp_path))
    st.save(10, {"theta": np.ones(2)}, {}, token=1)
    st.save(20, {"theta": np.ones(2) * 2}, {}, token=1)
    with open(tmp_path / "ckpt-00000020.npz", "r+b") as f:
        f.seek(100); f.write(b"garbage")
    step, arr, _ = st.load()
    assert step == 10 and arr["theta"].tolist() == [1, 1]


def test_crash_before_rename_leaves_old_state(tmp_path):
    st = CheckpointStore(str(tmp_path))
    st.save(10, {"theta": np.ones(2)}, {}, token=1)
    (tmp_path / ".ckpt-00000020.npz.tmp.999").write_bytes(b"half a file")   # writer died mid-write
    assert st.load()[0] == 10
