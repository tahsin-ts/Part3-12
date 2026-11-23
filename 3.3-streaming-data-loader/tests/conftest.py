import os

import pytest

from streamload.codec import make_samples
from streamload.shards import write_dataset


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    """200 zlib images of 32x32 in 8 shards of 25 records."""
    root = str(tmp_path_factory.mktemp("ds"))
    return root, write_dataset(make_samples(200, 32, seed=0), root, 25)


@pytest.fixture
def copy_of(tmp_path):
    def make(root):
        import shutil
        dst = os.path.join(str(tmp_path), "copy")
        shutil.copytree(root, dst)
        return dst
    return make
