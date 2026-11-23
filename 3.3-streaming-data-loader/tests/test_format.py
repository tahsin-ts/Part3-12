import json
import os
import struct

import pytest

from streamload.reader import ShardReader
from streamload.shards import FOOTER, CorruptRecord, verify_dataset, write_dataset, write_shard
from streamload.storage import LocalStore


def test_shard_roundtrip_random_access_and_stream(tmp_path):
    payloads = [os.urandom(n) for n in (0, 1, 7, 1000, 70_000, 3)]
    count, size = write_shard(str(tmp_path / "a.sld"), payloads)
    assert count == 6 and size == os.path.getsize(tmp_path / "a.sld")
    r = ShardReader(LocalStore(str(tmp_path)), "a.sld", block_bytes=64)
    assert [r.get(i) for i in range(6)] == payloads
    assert [p for _, p in r.records()] == payloads
    assert [p for _, p in r.records(start=4)] == payloads[4:]


def test_manifest_counts_and_sha256(data):
    root, m = data
    assert m["records"] == 200 and len(m["shards"]) == 8
    assert all(s["records"] == 25 for s in m["shards"])
    assert json.load(open(os.path.join(root, "manifest.json"))) == m
    assert verify_dataset(LocalStore(root), m) == []


def test_corrupt_record_detected_or_skipped(data, copy_of):
    root = copy_of(data[0])
    name = data[1]["shards"][0]["name"]
    r = ShardReader(LocalStore(root), name)
    with open(os.path.join(root, name), "r+b") as f:     # flip one byte inside record 3's payload
        f.seek(r.offsets[3] + 8 + 5)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(CorruptRecord):
        list(ShardReader(LocalStore(root), name).records())
    skip = ShardReader(LocalStore(root), name, on_corrupt="skip")
    got = [p for _, p in skip.records()]
    assert got[3] is None and sum(p is None for p in got) == 1 and skip.skipped == 1
    assert verify_dataset(LocalStore(root), data[1]) == [name]


def test_bad_footer_and_index_rejected(tmp_path):
    write_shard(str(tmp_path / "a.sld"), [b"x" * 10] * 4)
    raw = bytearray((tmp_path / "a.sld").read_bytes())
    bad_magic = raw[:-4] + b"NOPE"
    (tmp_path / "b.sld").write_bytes(bytes(bad_magic))
    with pytest.raises(CorruptRecord):
        ShardReader(LocalStore(str(tmp_path)), "b.sld")
    index_off = struct.unpack_from("<Q", raw, len(raw) - FOOTER.size)[0]
    raw[index_off] ^= 1
    (tmp_path / "c.sld").write_bytes(bytes(raw))
    with pytest.raises(CorruptRecord):
        ShardReader(LocalStore(str(tmp_path)), "c.sld")


def test_write_dataset_last_shard_partial(tmp_path):
    m = write_dataset((bytes([i]) for i in range(10)), str(tmp_path), 4)
    assert [s["records"] for s in m["shards"]] == [4, 4, 2]
