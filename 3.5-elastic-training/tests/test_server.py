import threading

import pytest

from elastic.client import Client, Gone, Stale, pack, unpack
from elastic.server import serve


@pytest.fixture()
def endpoint(tmp_path):
    httpd, co = serve("127.0.0.1", 0, str(tmp_path / "ev.jsonl"), min_nodes=2, max_nodes=2, lease_ttl=0.6, last_call=0.0)
    yield f"127.0.0.1:{httpd.server_address[1]}"
    co.stop.set()
    httpd.shutdown()


def test_join_wait_kv_over_http(endpoint):
    a, b = Client(endpoint, "a"), Client(endpoint, "b")
    a.join(1)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("a", a.wait(timeout=3)))
    t.start()
    b.join(1)
    t.join()
    assert out["a"]["gen"] == 1 and out["a"]["world"] == 2
    got = {}
    t = threading.Thread(target=lambda: got.update(a.kv_get(1, ["k"])))   # long poll until b writes
    t.start()
    b.kv_set(1, {"k": pack([1.5, 2.5])})
    t.join()
    assert unpack(got["k"]).tolist() == [1.5, 2.5]


def test_lease_expiry_fences_and_forgets(endpoint):
    a, b = Client(endpoint, "a"), Client(endpoint, "b")
    a.join(1); b.join(1)
    import time
    for _ in range(8):                 # only a keeps heartbeating
        a.heartbeat(1); time.sleep(0.15)
    with pytest.raises(Stale):
        a.kv_set(1, {"k": "v"})
    with pytest.raises(Gone):
        b.heartbeat(1)
    st = a.status()
    assert st["state"] == "open" and st["rejected_writes"] == 1


def test_partition_flag_blocks_calls(endpoint, tmp_path):
    c = Client(endpoint, "a", faults_dir=str(tmp_path))
    c.join(1)
    (tmp_path / "partition-a").touch()
    with pytest.raises(ConnectionError):
        c.join(1)
