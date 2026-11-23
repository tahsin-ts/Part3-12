"""Sample encoding and decoding: a small header + (optionally zlib-compressed) uint8 pixels.

Decoding is the CPU-heavy part of a real loader (JPEG decode, resize, augment). zlib + crop + flip +
float conversion stand in for it; the augmentation RNG is keyed by the sample, not by the worker, so
the same sample gets the same crop no matter which process decodes it.
"""
import random
import struct
import zlib

import numpy as np

HDR = struct.Struct("<IHHBB")   # label, height, width, channels, codec
RAW, ZLIB, JPEG = 0, 1, 2


def encode_image(arr, label, codec="zlib", level=1, quality=90):
    h, w, c = arr.shape
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if codec == "jpeg":                                  # optional: needs Pillow
        import io
        from PIL import Image
        out = io.BytesIO()
        Image.fromarray(arr).save(out, format="JPEG", quality=quality)
        cid, body = JPEG, out.getvalue()
    elif codec == "zlib":
        cid, body = ZLIB, zlib.compress(arr.tobytes(), level)
    else:
        cid, body = RAW, arr.tobytes()
    return HDR.pack(label, h, w, c, cid) + body


def decode_image(payload):
    label, h, w, c, cid = HDR.unpack_from(payload)
    body = payload[HDR.size:]
    if cid == JPEG:
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(body)).convert("RGB")), label
    if cid == ZLIB:
        body = zlib.decompress(body)
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w, c), label


def augment(img, crop, rng, dtype="uint8"):
    """Random crop of crop x crop and random horizontal flip. uint8 output is 4x smaller to move between
    processes; normalise on the GPU. dtype="float32" scales to [0, 1] on the CPU instead."""
    h, w, _ = img.shape
    crop = min(crop, h, w)
    y, x = rng.randrange(h - crop + 1), rng.randrange(w - crop + 1)
    out = img[y:y + crop, x:x + crop]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if dtype == "float32":
        return out.astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(out)


def sample_rng(seed, epoch, shard, index):
    return random.Random(f"aug:{seed}:{epoch}:{shard}:{index}")


def synth_image(rng, side, channels=3, width=None):
    """Smooth random image (compresses like a photo better than pure noise does)."""
    width = width or side
    small = rng.integers(0, 256, size=(-(-side // 8), -(-width // 8), channels), dtype=np.uint8)
    img = np.repeat(np.repeat(small, 8, axis=0), 8, axis=1)[:side, :width]
    noise = rng.integers(0, 24, size=img.shape, dtype=np.uint8)
    return (img // 2 + noise).astype(np.uint8)


def make_samples(n, side, seed=0, codec="zlib", classes=1000, width=None):
    """n synthetic labelled images of side x width pixels, encoded with codec (raw, zlib or jpeg)."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield encode_image(synth_image(rng, side, width=width), int(rng.integers(classes)), codec)
