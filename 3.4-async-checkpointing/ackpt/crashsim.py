"""Kill a real process with SIGKILL in the middle of a checkpoint save.

    python -m ackpt.crashsim DIR POINT        POINT = mid_shard | before_manifest | mid_manifest

Step 10 is saved completely; then the trainer runs to step 20 and starts an async save
that is killed at POINT. The parent process then checks that step 10 still loads.
"""
import os
import signal
import sys

from .checkpointer import Checkpointer
from .manifest import MANIFEST, step_dir
from .state import shard_state
from .storage import LocalFS
from .train import Trainer


def main(root, point, world=4):
    fs = LocalFS(root)
    fs.on_crash = lambda: os.kill(os.getpid(), signal.SIGKILL)     # no cleanup, no flush, no finally

    def fault(where, step, rank, key=None, data=None):
        if step != 20:
            return
        if point == "mid_shard" and where == "rank_write" and rank == 2:
            fs.put(key, data, crash_after=len(data) // 2)              # half a file in a temp name
        if point == "before_manifest" and where == "before_manifest":
            os.kill(os.getpid(), signal.SIGKILL)                     # all shards written, no commit
        if point == "mid_manifest" and where == "before_manifest":
            fs.put(f"{step_dir(step)}/{MANIFEST}", b'{"format": 1, "step": 20, "chunks": [', crash_after=20)

    tr = Trainer(n_params=200_000, layers=3)
    ck = Checkpointer(fs, world, keep_last=3, fault=fault)
    for _ in range(20):
        tr.train_step()
        if tr.step % 10 == 0:
            ck.save(tr.step, shard_state(tr.state, world))
    ck.wait()
    print("not killed?!")                                            # never reached
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
