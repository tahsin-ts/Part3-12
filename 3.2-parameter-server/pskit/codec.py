"""Binary wire format shared with the Go server (little-endian)."""
import numpy as np

U32, U64, F32 = np.dtype("<u4"), np.dtype("<u8"), np.dtype("<f4")


def encode_keys(keys: np.ndarray) -> bytes:
    return np.array([len(keys)], U32).tobytes() + np.asarray(keys, U64).tobytes()


def decode_rows(b: bytes) -> np.ndarray:
    n, dim = np.frombuffer(b, U32, 2)
    if len(b) != 8 + 4 * int(n) * int(dim):
        raise ValueError("bad pull reply length")
    return np.frombuffer(b, F32, int(n) * int(dim), 8).reshape(int(n), int(dim))


def encode_push(keys: np.ndarray, grads: np.ndarray) -> bytes:
    grads = np.asarray(grads, F32).reshape(len(keys), -1)
    return (np.array([len(keys), grads.shape[1]], U32).tobytes() + np.asarray(keys, U64).tobytes()
            + np.ascontiguousarray(grads).tobytes())
