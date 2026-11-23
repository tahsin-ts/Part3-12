import numpy as np
import pytest

from ringddp import RingLink, chunk_bounds, launch, ring_allreduce, ring_bytes_per_rank


def _worker(rank, addrs, sock, n, seed):
    link = RingLink(rank, len(addrs), addrs, 10.0, sock)
    try:
        buf = np.random.default_rng(seed + rank).normal(size=n)
        ring_allreduce(buf, link)
        return buf, link.bytes_sent, link.bytes_recv
    finally:
        link.close()


def test_chunk_bounds_cover_everything():
    assert chunk_bounds(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert chunk_bounds(2, 4) == [(0, 1), (1, 2), (2, 2), (2, 2)]     # empty chunks are allowed


@pytest.mark.parametrize("world,n", [(2, 1000), (3, 1001), (4, 7), (5, 3), (4, 1)])
def test_sum_is_correct_and_bitwise_identical(world, n):
    res = launch(_worker, world, (n, 42))
    expected = sum(np.random.default_rng(42 + r).normal(size=n) for r in range(world))
    for buf, _, _ in res:
        np.testing.assert_allclose(buf, expected, rtol=0, atol=1e-12)
        assert buf.tobytes() == res[0][0].tobytes()                   # replicas must not drift apart


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_bytes_match_2_n_minus_1_over_n(world):
    n = 4800                                                          # divisible by 2, 3, 4, 8
    res = launch(_worker, world, (n, 0))
    for _, sent, recv in res:
        assert sent == recv == ring_bytes_per_rank(n * 8, world)
