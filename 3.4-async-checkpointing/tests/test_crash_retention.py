import os
import subprocess
import sys

import numpy as np
import pytest

from ackpt import Checkpointer, LocalFS, ObjectStore, committed_steps, gc, load_latest, shard_state, unshard
from ackpt.manifest import MANIFEST, step_dir
from ackpt.retention import plan
from ackpt.train import Trainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("point", ["mid_shard", "before_manifest", "mid_manifest"])
def test_sigkill_during_save_keeps_previous_checkpoint(tmp_path, point):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "ackpt.crashsim", str(tmp_path), point], env=env, capture_output=True, timeout=60)
    assert p.returncode == -9                                    # really killed, no cleanup ran
    fs = LocalFS(str(tmp_path))
    assert any(k.startswith(step_dir(20)) for k in fs.list_all())  # debris of the killed save exists
    step, ranks, skipped = load_latest(fs, 4)
    assert step == 10 and skipped == []
    ref = Trainer(n_params=200_000, layers=3)
    for _ in range(10):
        ref.train_step()
    assert all(np.array_equal(ref.state[k], unshard(ranks)[k]) for k in ref.state)
    ck = Checkpointer(fs, 4, keep_last=3)                         # the restarted job saves again
    ck.save(30, shard_state(ref.state, 4), blocking=True)
    assert not any(k.startswith(step_dir(20)) for k in fs.list_all())   # gc removed the debris
    assert committed_steps(fs) == [10, 30]


def test_fault_before_manifest_in_process(state):
    def fault(where, step, rank, **_):
        if where == "before_manifest" and step == 2:
            raise SystemExit("killed")
    store = ObjectStore()
    ck = Checkpointer(store, 4, fault=fault)
    ck.save(1, shard_state(state, 4), blocking=True)
    ck.save(2, shard_state(state, 4))
    with pytest.raises(RuntimeError):
        ck.wait()
    assert committed_steps(store) == [1]
    assert len([k for k in store.list() if k.startswith(step_dir(2))]) == 4   # shards, but no commit


def test_keep_last(state):
    store = ObjectStore()
    ck = Checkpointer(store, 2, keep_last=2)
    for s in range(1, 6):
        ck.save(s, shard_state(state, 2), blocking=True)
    assert committed_steps(store) == [4, 5]
    assert all(k.startswith((step_dir(4), step_dir(5))) for k in store.list())


def test_gc_deletes_manifest_first(state):
    deleted = []

    class Rec(ObjectStore):
        def delete(self, key):
            deleted.append(key)
            super().delete(key)
    store = Rec()
    ck = Checkpointer(store, 3, keep_last=0)
    for s in (1, 2):
        ck.save(s, shard_state(state, 3), blocking=True)
    gc(store, keep_last=1)
    assert deleted[0] == f"{step_dir(1)}/{MANIFEST}" and len(deleted) == 4


def test_gc_plan_protects_in_flight_and_newer_uncommitted():
    keys = [f"{step_dir(s)}/{MANIFEST}" for s in (1, 2, 3)] + [f"{step_dir(s)}/rank_00000.bin" for s in (1, 2, 3, 4, 5)]
    keys.append(f"{step_dir(0)}/rank_00000.bin.tmp.99.1")        # crash debris below the newest commit
    delete, _ = plan(keys, keep_last=2, protect=5)
    assert delete == [0, 1]                                       # 4 is newer than every commit: maybe in flight


def test_keep_last_zero_keeps_everything(state):
    store = ObjectStore()
    ck = Checkpointer(store, 2, keep_last=0)
    for s in (1, 2, 3):
        ck.save(s, shard_state(state, 2), blocking=True)
    assert committed_steps(store) == [1, 2, 3]
