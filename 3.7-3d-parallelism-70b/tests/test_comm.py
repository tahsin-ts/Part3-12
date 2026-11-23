import numpy as np
import pytest

from threed.comm import Comm, CommTimeout
from threed.engine import launch, ring_bytes


def _ar(comm, rank, n, size, op):
    x = np.arange(size, dtype=float) * (rank + 1) - rank
    return comm.all_reduce(x, list(range(n)), op=op), dict(comm.bytes_sent)


@pytest.mark.parametrize("n,size", [(2, 8), (3, 10), (4, 7), (4, 1024)])
def test_all_reduce_sum_matches_numpy(n, size):
    res = launch(n, _ar, (n, size, "sum"))
    want = sum(np.arange(size, dtype=float) * (r + 1) - r for r in range(n))
    for (got, b), r in zip(res, range(n)):
        assert np.allclose(got, want)
        assert b["tp"] == ring_bytes(size, n, r)          # every byte is accounted for


def test_all_reduce_max_and_identical_results():
    res = launch(3, _ar, (3, 9, "max"))
    want = np.max([np.arange(9, dtype=float) * (r + 1) - r for r in range(3)], axis=0)
    for got, _ in res:
        assert np.array_equal(got, want)
        assert np.array_equal(got, res[0][0])             # bit-identical on every rank


def _gather(comm, rank):
    return comm.all_gather(np.full(3, rank), [0, 1, 2])


def test_all_gather_order():
    for parts in launch(3, _gather):
        assert [int(p[0]) for p in parts] == [0, 1, 2]


def _out_of_order(comm, rank):
    if rank == 0:
        comm.send(1, "b", np.array([2.0]))
        comm.send(1, "a", np.array([1.0]))
        return None
    return float(comm.recv(0, "a")[0]), float(comm.recv(0, "b")[0])


def test_recv_by_tag_out_of_order():
    assert launch(2, _out_of_order)[1] == (1.0, 2.0)


def test_recv_timeout_raises():
    import multiprocessing as mp
    c = Comm(0, 2, [mp.Queue(), mp.Queue()], timeout=0.2)
    with pytest.raises(CommTimeout):
        c.recv(1, "never")
