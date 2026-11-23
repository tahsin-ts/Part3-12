"""Consistent hashing, bit-for-bit the same as internal/ps/hashring.go."""
import numpy as np

M64 = (1 << 64) - 1


def mix64(x: int) -> int:
    """splitmix64 finalizer for one Python int."""
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def mix64_np(x: np.ndarray) -> np.ndarray:
    """The same finalizer on a uint64 array (NumPy wraps on overflow)."""
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def fnv1a(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode():
        h = ((h ^ b) * 1099511628211) & M64
    return h


class Ring:
    """Each shard owns `vnodes` points; a key goes to the first point clockwise from mix64(key)."""

    def __init__(self, shards, vnodes=64):
        names = [f"shard-{i}" for i in range(shards)] if isinstance(shards, int) else list(shards)
        pts = sorted((mix64(fnv1a(f"{n}#{v}")), i) for i, n in enumerate(names) for v in range(vnodes))
        self.names = names
        self.points = np.array([p for p, _ in pts], dtype=np.uint64)
        self.owner = np.array([o for _, o in pts], dtype=np.int64)

    def owners(self, keys: np.ndarray) -> np.ndarray:
        """Shard index for every key (vectorised binary search)."""
        i = np.searchsorted(self.points, mix64_np(np.asarray(keys, dtype=np.uint64)), side="left")
        i[i == len(self.points)] = 0  # wrap around
        return self.owner[i]
