import http.client

import pytest

from streamload.reader import ShardReader
from streamload.storage import HttpStore, LocalStore, ObjectStoreServer


def test_range_get_head_and_errors(data):
    root, m = data
    name = m["shards"][0]["name"]
    blob = open(f"{root}/{name}", "rb").read()
    with ObjectStoreServer(root) as srv:
        st = HttpStore(srv.url)
        assert st.size(name) == len(blob)
        assert st.read(name, 10, 100) == blob[10:110]
        assert st.read(name, len(blob) - 5, 5) == blob[-5:]
        c = http.client.HTTPConnection("127.0.0.1", int(srv.url.rsplit(":", 1)[1]))
        c.request("GET", f"/{name}", headers={"Range": f"bytes={len(blob)}-"})
        r = c.getresponse(); r.read()
        assert r.status == 416 and r.getheader("Content-Range") == f"bytes */{len(blob)}"
        c.request("GET", "/../etc/passwd")
        r = c.getresponse(); r.read()
        assert r.status == 404
        with pytest.raises(FileNotFoundError):
            st.size("missing.sld")


def test_keep_alive_one_connection_many_requests(data):
    root, m = data
    with ObjectStoreServer(root) as srv:
        st = HttpStore(srv.url)
        for i in range(20):
            st.read(m["shards"][0]["name"], i, 1)
        assert st.requests == 20 and st.retries == 0


def test_retries_survive_injected_503(data):
    root, m = data
    with ObjectStoreServer(root, fail_every=3) as srv:
        st = HttpStore(srv.url, retries=3, backoff=0.001)
        local = LocalStore(root)
        for s in m["shards"]:
            a = ShardReader(st, s["name"], block_bytes=512)
            b = ShardReader(local, s["name"])
            assert [p for _, p in a.records()] == [p for _, p in b.records()]
        assert st.retries > 0


def test_gives_up_after_max_retries(data):
    root, m = data
    with ObjectStoreServer(root, fail_every=1) as srv:
        st = HttpStore(srv.url, retries=2, backoff=0.001)
        with pytest.raises(IOError, match="after 3 attempts"):
            st.read(m["shards"][0]["name"], 0, 10)


def test_streaming_coalesces_requests(data):
    root, m = data
    name, size = m["shards"][0]["name"], m["shards"][0]["bytes"]
    with ObjectStoreServer(root) as srv:
        st = HttpStore(srv.url)
        r = ShardReader(st, name, block_bytes=1 << 20, size=size)
        list(r.records())
        assert st.requests == 2          # tail (footer + index) and one data read
        st2 = HttpStore(srv.url)
        r2 = ShardReader(st2, name, block_bytes=0, size=size)
        list(r2.records())
        assert st2.requests == 1 + len(r2)
