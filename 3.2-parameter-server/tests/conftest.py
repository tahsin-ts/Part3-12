import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def psserver():
    """Path to a built psserver binary (built once with `go build` if needed)."""
    b = os.environ.get("PSSERVER") or os.path.join(ROOT, "bin", "psserver")
    if not os.path.exists(b):
        if not shutil.which("go"):
            pytest.skip("Go toolchain not found and bin/psserver missing")
        subprocess.run(["go", "build", "-o", b, "./cmd/psserver"], cwd=ROOT, check=True)
    os.environ["PSSERVER"] = b
    return b
