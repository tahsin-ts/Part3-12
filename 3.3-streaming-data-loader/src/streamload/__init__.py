"""streamload: sharded records, range-read streaming, multi-process decode, exact resume."""
from .shards import CorruptRecord, write_dataset, verify_dataset
from .storage import LocalStore, HttpStore, ObjectStoreServer, make_store
from .reader import ShardReader
from .plan import EpochPlan, buffered_shuffle
from .loader import StreamingLoader, WorkerError

__version__ = "0.1.0"
__all__ = ["CorruptRecord", "write_dataset", "verify_dataset", "LocalStore", "HttpStore", "ObjectStoreServer",
           "make_store", "ShardReader", "EpochPlan", "buffered_shuffle", "StreamingLoader", "WorkerError"]
