"""Ring all-reduce = reduce-scatter (N-1 steps) + all-gather (N-1 steps)."""
import numpy as np


def chunk_bounds(n_items, world):
    """Split n_items into `world` nearly equal, contiguous chunks: [(start, stop), ...]."""
    base, extra = divmod(n_items, world)
    sizes = [base + (1 if i < extra else 0) for i in range(world)]
    edges = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(world)]


def ring_allreduce(buf, link, tag=0):
    """Sum `buf` (1-D float64) over all ranks, in place. Every rank ends with identical bits."""
    n, r = link.world, link.rank
    if n == 1:
        return buf
    b = chunk_bounds(buf.size, n)
    view = lambda c: buf[b[c][0]:b[c][1]]
    # 1) reduce-scatter: after step k, chunk (r-k-1) holds the sum of k+2 ranks
    for k in range(n - 1):
        send_c, recv_c = (r - k) % n, (r - k - 1) % n
        link.send(view(send_c), tag + k)
        view(recv_c)[:] += link.recv(b[recv_c][1] - b[recv_c][0], tag + k)
    # now rank r owns the complete sum of chunk (r+1) % n
    # 2) all-gather: pass the finished chunks once around the ring
    for k in range(n - 1):
        send_c, recv_c = (r - k + 1) % n, (r - k) % n
        link.send(view(send_c), tag + n + k)
        view(recv_c)[:] = link.recv(b[recv_c][1] - b[recv_c][0], tag + n + k)
    return buf


def ring_bytes_per_rank(n_bytes, world):
    """Bytes each rank sends (and receives) in one ring all-reduce."""
    return 2 * (world - 1) * n_bytes / world
