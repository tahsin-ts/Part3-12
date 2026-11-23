"""Atomic, verified, fenced checkpoints on a shared directory.

save():  write to a temp file, fsync, then under an exclusive lock check the fencing
         token, rename into place, point LATEST at it, fsync the directory.
         A crash at any moment leaves either the old or the new checkpoint, never half of one.
fence:   a writer presents its generation as a token. The store remembers the highest
         token it has seen and refuses older ones, so a zombie from an old generation
         cannot overwrite the checkpoint of the new one (split-brain protection).
load():  verifies the SHA-256 recorded in LATEST; a corrupt newest file falls back to an older one.
"""
import fcntl
import hashlib
import io
import json
import os
import time

import numpy as np


class StaleWriter(Exception):
    """The writer's fencing token is older than one the store has already accepted."""


class CheckpointStore:
    def __init__(self, root, keep=3):
        self.root, self.keep = root, keep
        os.makedirs(root, exist_ok=True)
        self.lockpath = os.path.join(root, ".lock")

    # -------------------------------------------------------------- helpers
    def _locked(self):
        f = open(self.lockpath, "a+")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def _read(self, name, default=None):
        try:
            with open(os.path.join(self.root, name)) as f:
                return f.read().strip()
        except FileNotFoundError:
            return default

    def _write_atomic(self, name, text):
        tmp = os.path.join(self.root, f".{name}.tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(self.root, name))

    def _fsync_dir(self):
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def fence(self):
        return int(self._read("FENCE", "0"))

    # -------------------------------------------------------------- fencing
    def claim(self, token):
        """A new generation announces itself before it reads or writes anything."""
        with self._locked():
            cur = self.fence()
            if token < cur:
                raise StaleWriter(f"token {token} < fence {cur}")
            if token > cur:
                self._write_atomic("FENCE", str(token))
                self._fsync_dir()

    # -------------------------------------------------------------- save / load
    def save(self, step, arrays, meta, token):
        t0 = time.perf_counter()
        buf = io.BytesIO()
        np.savez(buf, meta=np.array(json.dumps(meta)), **arrays)
        data = buf.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        name = f"ckpt-{step:08d}.npz"
        tmp = os.path.join(self.root, f".{name}.tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with self._locked():
            cur = self.fence()
            if token < cur:
                os.unlink(tmp)
                raise StaleWriter(f"token {token} < fence {cur}: checkpoint of step {step} refused")
            if token > cur:
                self._write_atomic("FENCE", str(token))
            os.replace(tmp, os.path.join(self.root, name))
            self._write_atomic("LATEST", json.dumps({"step": step, "file": name, "sha256": digest, "token": token}))
            self._fsync_dir()
            self._prune()
        return name, time.perf_counter() - t0, len(data)

    def _prune(self):
        files = sorted(f for f in os.listdir(self.root) if f.startswith("ckpt-") and f.endswith(".npz"))
        for f in files[:-self.keep]:
            os.unlink(os.path.join(self.root, f))

    def steps(self):
        return sorted(int(f[5:13]) for f in os.listdir(self.root) if f.startswith("ckpt-") and f.endswith(".npz"))

    def latest_step(self):
        info = self._read("LATEST")
        return json.loads(info)["step"] if info else None

    def load(self, step=None):
        """Load `step` (default: LATEST). Returns (step, arrays, meta) or None if nothing usable exists."""
        info = json.loads(self._read("LATEST") or "null")
        want = [step] if step is not None else ([info["step"]] if info else []) + self.steps()[::-1]
        for s in want:
            path = os.path.join(self.root, f"ckpt-{s:08d}.npz")
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                continue
            if info and info["step"] == s and hashlib.sha256(data).hexdigest() != info["sha256"]:
                continue                                    # corrupt newest checkpoint: try an older one
            try:
                z = np.load(io.BytesIO(data))
                arrays = {k: z[k] for k in z.files if k != "meta"}
                return s, arrays, json.loads(str(z["meta"]))
            except Exception:                               # truncated or garbage file
                continue
        return None
