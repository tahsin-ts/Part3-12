import json

import numpy as np
import pytest

from streamload.loader import StreamingLoader, WorkerError, worker_batches
from streamload.storage import HttpStore, LocalStore, ObjectStoreServer


def keys(batches):
    return [k for b in batches for k in b["keys"]]


def same(a, b):
    return len(a) == len(b) and all(x["keys"] == y["keys"] and np.array_equal(x["x"], y["x"])
                                    and np.array_equal(x["y"], y["y"]) for x, y in zip(a, b))


def mk(root_or_store, m, **kw):
    store = LocalStore(root_or_store) if isinstance(root_or_store, str) else root_or_store
    kw.setdefault("batch_size", 16)
    kw.setdefault("shuffle_buffer", 64)
    return StreamingLoader(store, m, **kw)


def test_one_epoch_sees_every_sample_once(data):
    root, m = data
    b = list(mk(root, m, num_workers=2, processes=False))
    k = keys(b)
    assert len(k) == len(set(k)) == 200
    assert b[0]["x"].shape == (16, 24, 24, 3) and b[0]["x"].dtype == np.uint8


def test_processes_match_inline_exactly(data):
    root, m = data
    inline = list(mk(root, m, num_workers=2, processes=False))
    procs = list(mk(root, m, num_workers=2))
    assert same(inline, procs)
    pickled = list(mk(root, m, num_workers=2, transport="pickle"))
    assert same(inline, pickled)


def test_epochs_differ_and_state_rolls_forward(data):
    root, m = data
    L = mk(root, m, processes=False)
    e0 = keys(list(L))
    assert L.state_dict()["epoch"] == 1 and L.state_dict()["step"] == 0
    e1 = keys(list(L))
    assert sorted(e0) == sorted(e1) and e0 != e1


@pytest.mark.parametrize("workers,procs", [(0, False), (3, False), (2, True)])
def test_exact_resume_mid_epoch(data, workers, procs):
    root, m = data
    full = list(mk(root, m, num_workers=workers, processes=procs))
    for stop in (1, 4, len(full) - 1):
        L = mk(root, m, num_workers=workers, processes=procs)
        it = iter(L)
        head = [next(it) for _ in range(stop)]
        state = json.loads(json.dumps(L.state_dict()))       # survives a JSON round trip
        it.close()
        L2 = mk(root, m, num_workers=workers, processes=procs)
        L2.load_state_dict(state)
        assert same(head + list(L2), full)
    assert len(json.dumps(state)) < 64


def test_resume_over_http_matches_local(data):
    root, m = data
    full = list(mk(root, m, num_workers=2, processes=False))
    with ObjectStoreServer(root, latency_ms=1) as srv:
        L = mk(HttpStore(srv.url), m, num_workers=2)
        L.load_state_dict({"epoch": 0, "step": 6, "fp": L.fingerprint()})
        assert same(full[6:], list(L))


def test_state_from_other_config_rejected(data):
    root, m = data
    a, b = mk(root, m, seed=1), mk(root, m, seed=2)
    with pytest.raises(ValueError):
        b.load_state_dict(a.state_dict())


def test_data_parallel_ranks_disjoint_and_same_length(data):
    root, m = data
    per_rank = [list(mk(root, m, num_workers=2, processes=False, world_size=2, rank=r, seed=5)) for r in range(2)]
    k0, k1 = keys(per_rank[0]), keys(per_rank[1])
    assert len(per_rank[0]) == len(per_rank[1])
    assert not set(k0) & set(k1) and len(k0) + len(k1) == 200


def test_augmentation_does_not_depend_on_worker_count(data):
    root, m = data
    one = list(mk(root, m, num_workers=0, processes=False, shuffle=False))
    four = list(mk(root, m, num_workers=4, processes=False, shuffle=False))
    by_key = lambda bs: {k: x.tobytes() for b in bs for k, x in zip(b["keys"], b["x"])}
    assert by_key(one) == by_key(four)


def test_worker_exception_reaches_main_process(data, copy_of):
    root = copy_of(data[0])
    m = data[1]
    name = m["shards"][0]["name"]
    with open(f"{root}/{name}", "r+b") as f:
        f.seek(40)
        f.write(b"\xff" * 16)
    with pytest.raises(WorkerError, match="CorruptRecord"):
        list(mk(root, m, num_workers=2))
    skipped = keys(list(mk(root, m, num_workers=2, on_corrupt="skip")))
    assert 190 <= len(skipped) < 200


def test_float32_option_and_raw_mode(data):
    root, m = data
    b = next(worker_batches(mk(root, m, dtype="float32").cfg, 0, 0))
    assert b["x"].dtype == np.float32 and 0.0 <= b["x"].min() and b["x"].max() <= 1.0
    r = next(worker_batches(mk(root, m, decode="raw").cfg, 0, 0))
    assert isinstance(r["x"][0], bytes)
