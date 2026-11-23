"""Find the newest valid checkpoint and load it onto any number of ranks (resharding)."""
import zlib

import numpy as np

from .manifest import MANIFEST, CorruptCheckpoint, parse_manifest, parse_step, step_dir, tensors
from .state import Shard, is_split, owner_of_small, row_ranges


def committed_steps(storage):
    """Steps whose manifest exists. Files without a manifest are invisible."""
    return sorted(parse_step(k) for k in storage.list() if k.endswith("/" + MANIFEST) and parse_step(k) is not None)


def read_manifest(storage, step):
    return parse_manifest(storage.get(f"{step_dir(step)}/{MANIFEST}"))


class _Reader:
    """Reads chunk bytes; with verify=True reads whole chunks once and checks their CRC."""

    def __init__(self, storage, step, verify):
        self.storage, self.dir, self.verify = storage, step_dir(step), verify
        self.cache, self.bytes_read = {}, 0

    def rows(self, c, lo, hi, row_bytes):
        key = f"{self.dir}/{c['file']}"
        if not self.verify:                                   # ranged read: only the rows we need
            start = c["byte_offset"] + (lo - c["row_offset"]) * row_bytes
            raw = self.storage.get(key, start, (hi - lo) * row_bytes)
            self.bytes_read += len(raw)
            return raw
        ck = (key, c["byte_offset"])
        if ck not in self.cache:
            raw = self.storage.get(key, c["byte_offset"], c["nbytes"])
            self.bytes_read += len(raw)
            if len(raw) != c["nbytes"] or zlib.crc32(raw) != c["crc32"]:
                raise CorruptCheckpoint(f"checksum mismatch in {key} ({c['tensor']})")
            self.cache[ck] = raw
        a = (lo - c["row_offset"]) * row_bytes
        return self.cache[ck][a:a + (hi - lo) * row_bytes]


def load(storage, step, world_size, verify=True, stats=None, ranks=None):
    """Return a list of world_size per-rank {name: Shard}, whatever world size saved it.
    ranks: load only these ranks (a real rank loads just itself); the others stay empty."""
    man = read_manifest(storage, step)
    rd = _Reader(storage, step, verify)
    out = [dict() for _ in range(world_size)]
    for name, (shape, dtype, chunks) in tensors(man).items():
        if len(shape) == 0:                                   # scalar: one chunk, one owner
            c = chunks[0]
            if ranks is not None and owner_of_small(name, world_size) not in ranks:
                continue
            arr = np.frombuffer(rd.rows(c, 0, 1, c["nbytes"]), dtype=dtype).reshape(()).copy()
            out[owner_of_small(name, world_size)][name] = Shard(arr, shape, 0)
            continue
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        if is_split(shape, world_size):
            targets = list(enumerate(row_ranges(shape[0], world_size)))
        else:
            targets = [(owner_of_small(name, world_size), (0, shape[0]))]
        for r, (a, b) in targets:
            if ranks is not None and r not in ranks:
                continue
            buf = bytearray()
            for c in chunks:                                  # chunks overlapping rows [a, b)
                lo, hi = max(a, c["row_offset"]), min(b, c["row_offset"] + c["rows"])
                if lo < hi:
                    buf += rd.rows(c, lo, hi, row_bytes)
            if len(buf) != (b - a) * row_bytes:
                raise CorruptCheckpoint(f"{name}: rows {a}-{b} incomplete")
            arr = np.frombuffer(bytes(buf), dtype=dtype).reshape((b - a,) + tuple(shape[1:])).copy()
            out[r][name] = Shard(arr, shape, a)
    if stats is not None:
        stats["bytes_read"] = rd.bytes_read
    return out


def load_latest(storage, world_size, verify=True):
    """Newest checkpoint that loads cleanly. Returns (step, rank_states, skipped[(step, reason)])."""
    skipped = []
    for step in reversed(committed_steps(storage)):
        try:
            return step, load(storage, step, world_size, verify), skipped
        except (CorruptCheckpoint, FileNotFoundError, KeyError) as e:
            skipped.append((step, str(e)))                   # fall back to the previous one
    return None, None, skipped
