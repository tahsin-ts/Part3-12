"""Shard format: length-prefixed records with a CRC each, an offset index, and a fixed-size footer.

    [magic 4B] [len u32][crc u32][payload] ... [offset u64 x count] [index_off u64][count u32][index_crc u32][magic 4B]

The footer is read first (one small range read), then the index, so any record can be fetched
with one range request and a whole shard can be streamed front to back.
"""
import hashlib
import json
import os
import struct
import zlib

MAGIC = b"SLD1"
REC = struct.Struct("<II")          # payload length, crc32(payload)
FOOTER = struct.Struct("<QII4s")    # index offset, record count, crc32(index), magic
MANIFEST = "manifest.json"


class CorruptRecord(Exception):
    """A record, index or footer failed its checksum or structure check."""


def write_shard(path, payloads):
    """Write one shard; returns (record count, file size)."""
    offsets = []
    with open(path, "wb") as f:
        f.write(MAGIC)
        for p in payloads:
            offsets.append(f.tell())
            f.write(REC.pack(len(p), zlib.crc32(p)))
            f.write(p)
        index_off = f.tell()
        index = struct.pack(f"<{len(offsets)}Q", *offsets)
        f.write(index)
        f.write(FOOTER.pack(index_off, len(offsets), zlib.crc32(index), MAGIC))
        return len(offsets), f.tell()


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(samples, out_dir, records_per_shard, prefix="shard"):
    """Split an iterable of payloads into shards and write manifest.json next to them."""
    os.makedirs(out_dir, exist_ok=True)
    shards, batch = [], []

    def flush():
        name = f"{prefix}-{len(shards):05d}.sld"
        path = os.path.join(out_dir, name)
        count, size = write_shard(path, batch)
        shards.append({"name": name, "records": count, "bytes": size, "sha256": sha256_file(path)})
        batch.clear()

    for p in samples:
        batch.append(p)
        if len(batch) == records_per_shard:
            flush()
    if batch:
        flush()
    manifest = {"format": "SLD1", "shards": shards, "records": sum(s["records"] for s in shards),
                "bytes": sum(s["bytes"] for s in shards)}
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def parse_footer(buf):
    index_off, count, crc, magic = FOOTER.unpack(buf)
    if magic != MAGIC:
        raise CorruptRecord("bad footer magic")
    return index_off, count, crc


def parse_index(buf, count, crc):
    if len(buf) != 8 * count or zlib.crc32(buf) != crc:
        raise CorruptRecord("index checksum mismatch")
    return list(struct.unpack(f"<{count}Q", buf))


def check_record(raw, where=""):
    """raw = header + payload of exactly one record. Returns the payload or raises CorruptRecord."""
    n, crc = REC.unpack_from(raw)
    payload = raw[REC.size:]
    if len(payload) != n or zlib.crc32(payload) != crc:
        raise CorruptRecord(f"crc mismatch {where}")
    return payload


def verify_dataset(store, manifest):
    """Full integrity check: sha256 of every shard against the manifest. Returns list of bad shard names."""
    bad = []
    for s in manifest["shards"]:
        h = hashlib.sha256(store.read(s["name"], 0, s["bytes"])).hexdigest()
        if h != s["sha256"]:
            bad.append(s["name"])
    return bad
