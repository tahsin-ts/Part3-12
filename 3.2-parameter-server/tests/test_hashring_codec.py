import numpy as np

from pskit.codec import decode_rows, encode_keys, encode_push
from pskit.hashring import Ring, mix64, mix64_np


def test_ring_matches_go_vectors():
    # the same vectors are asserted in internal/ps/hashring_test.go
    assert list(Ring(4).owners(np.arange(16))) == [2, 0, 2, 1, 1, 0, 0, 2, 1, 2, 3, 3, 2, 1, 2, 2]
    assert mix64(12345) == 2454886589211414944
    keys = np.array([0, 1, 2**63, 12345, 2**64 - 1], dtype=np.uint64)
    assert [int(x) for x in mix64_np(keys)] == [mix64(int(k)) for k in keys]


def test_adding_a_shard_moves_few_keys():
    keys = np.arange(200_000, dtype=np.uint64)
    old, new = Ring(4).owners(keys), Ring(5).owners(keys)
    moved = old != new
    assert 0.1 < moved.mean() < 0.3
    assert set(new[moved]) == {4}  # keys only move to the new shard


def test_wire_format_sizes_and_roundtrip():
    keys = np.array([1, 2, 3], dtype=np.uint64)
    assert len(encode_keys(keys)) == 4 + 3 * 8
    push = encode_push(keys, np.ones((3, 2), np.float32))
    assert len(push) == 8 + 3 * 8 + 3 * 2 * 4
    reply = np.array([3, 2], "<u4").tobytes() + np.arange(6, dtype="<f4").tobytes()
    rows = decode_rows(reply)
    assert rows.shape == (3, 2) and rows[2, 1] == 5
