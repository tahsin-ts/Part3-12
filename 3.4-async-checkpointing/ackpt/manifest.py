"""The manifest: the single small file that makes a checkpoint exist.

Layout of one checkpoint (any backend):
    step_000000120/rank_00000.bin  ...  rank_00003.bin   raw tensor bytes, one file per rank
    step_000000120/MANIFEST.json                         written LAST = the commit point
"""
import json
import zlib

import numpy as np

FORMAT_VERSION = 1
MANIFEST = "MANIFEST.json"


class CorruptCheckpoint(Exception):
    pass


def step_dir(step):
    return f"step_{step:09d}"


def parse_step(key):
    """'step_000000120/...' -> 120, or None for other keys."""
    head = key.split("/", 1)[0]
    if head.startswith("step_") and head[5:].isdigit():
        return int(head[5:])
    return None


def encode_rank(rank, shards):
    """Pack one rank's shards into bytes + the chunk records that describe them."""
    parts, chunks, off = [], [], 0
    fname = f"rank_{rank:05d}.bin"
    for name in sorted(shards):
        sh = shards[name]
        raw = np.ascontiguousarray(sh.data).tobytes()
        chunks.append({"tensor": name, "file": fname, "byte_offset": off, "nbytes": len(raw),
                       "row_offset": sh.row_offset, "rows": int(sh.data.shape[0]) if sh.data.ndim else 0,
                       "global_shape": list(sh.global_shape), "dtype": sh.data.dtype.str,
                       "crc32": zlib.crc32(raw)})
        parts.append(raw)
        off += len(raw)
    return fname, b"".join(parts), chunks


def build_manifest(step, world_size, rank_chunks, extra=None):
    body = {"format": FORMAT_VERSION, "step": step, "world_size": world_size,
            "chunks": [c for chunks in rank_chunks for c in chunks], "extra": extra or {}}
    canon = json.dumps(body, sort_keys=True).encode()
    body["manifest_crc32"] = zlib.crc32(canon)      # catches bit flips that still parse as JSON
    return json.dumps(body, sort_keys=True, indent=1).encode()


def parse_manifest(raw):
    try:
        body = json.loads(raw)
        crc = body.pop("manifest_crc32")
    except (ValueError, KeyError) as e:
        raise CorruptCheckpoint(f"manifest unreadable: {e}") from None
    if zlib.crc32(json.dumps(body, sort_keys=True).encode()) != crc:
        raise CorruptCheckpoint("manifest checksum mismatch")
    if body["format"] != FORMAT_VERSION:
        raise CorruptCheckpoint(f"unknown format {body['format']}")
    return body


def tensors(manifest):
    """{name: (global_shape, dtype, [chunks sorted by row])}"""
    out = {}
    for c in manifest["chunks"]:
        t = out.setdefault(c["tensor"], (tuple(c["global_shape"]), np.dtype(c["dtype"]), []))
        t[2].append(c)
    for t in out.values():
        t[2].sort(key=lambda c: c["row_offset"])
    return out
