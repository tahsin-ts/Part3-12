import json
import os

import numpy as np
import pytest

from ackpt import Checkpointer, CorruptCheckpoint, LocalFS, ObjectStore, load, load_latest, shard_state, unshard
from ackpt.manifest import MANIFEST, step_dir


def two_checkpoints(store, state):
    ck = Checkpointer(store, 4)
    ck.save(1, shard_state(state, 4), blocking=True)
    s2 = {k: v + 1 for k, v in state.items()}
    ck.save(2, shard_state(s2, 4), blocking=True)
    return s2


def test_bit_flip_in_shard_is_detected_and_previous_loads(state):
    store = ObjectStore()
    two_checkpoints(store, state)
    store.corrupt(f"{step_dir(2)}/rank_00001.bin", 100)
    with pytest.raises(CorruptCheckpoint, match="checksum"):
        load(store, 2, 4)
    step, ranks, skipped = load_latest(store, 4)
    assert step == 1 and skipped[0][0] == 2
    assert all(np.array_equal(unshard(ranks)[k], state[k]) for k in state)


def test_bit_flip_in_manifest_is_detected(state):
    store = ObjectStore()
    two_checkpoints(store, state)
    key = f"{step_dir(2)}/{MANIFEST}"
    raw = store.objects[key]
    i = raw.index(b'"step": 2')
    store.objects[key] = raw[:i] + b'"step": 3' + raw[i + 9:]     # still valid JSON, wrong content
    assert load_latest(store, 4)[0] == 1


def test_truncated_shard_on_disk(tmp_path, state):
    fs = LocalFS(str(tmp_path))
    two_checkpoints(fs, state)
    p = tmp_path / step_dir(2) / "rank_00003.bin"
    p.write_bytes(p.read_bytes()[:-10])
    step, _, skipped = load_latest(fs, 2)
    assert step == 1 and "checksum" in skipped[0][1]


def test_missing_shard_file(tmp_path, state):
    fs = LocalFS(str(tmp_path))
    two_checkpoints(fs, state)
    os.remove(tmp_path / step_dir(2) / "rank_00000.bin")
    assert load_latest(fs, 4)[0] == 1


def test_unreadable_manifest(tmp_path, state):
    fs = LocalFS(str(tmp_path))
    two_checkpoints(fs, state)
    (tmp_path / step_dir(2) / MANIFEST).write_text("{not json")
    assert load_latest(fs, 4)[0] == 1


def test_one_rank_reads_only_its_part(state):
    store = ObjectStore()
    Checkpointer(store, 4).save(1, shard_state(state, 4), blocking=True)
    total = sum(c["nbytes"] for c in json.loads(store.objects[f"{step_dir(1)}/{MANIFEST}"])["chunks"])
    ranged, verified = {}, {}
    mine = load(store, 1, 8, verify=False, stats=ranged, ranks=[0])
    load(store, 1, 8, verify=True, stats=verified, ranks=[0])
    assert mine[0] and not any(mine[1:])
    assert ranged["bytes_read"] < 0.2 * total                       # about 1/8 of the bytes
    assert ranged["bytes_read"] < verified["bytes_read"] < 0.4 * total   # CRC needs whole chunks: about 1/4
