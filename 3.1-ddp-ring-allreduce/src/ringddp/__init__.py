"""Data-parallel training from scratch: ring all-reduce over TCP, gradient buckets, overlap."""
from .allreduce import chunk_bounds, ring_allreduce, ring_bytes_per_rank
from .ddp import DistributedDataParallel
from .launch import WorkerFailure, launch
from .model import MLP, SGD
from .sampler import DistributedSampler
from .transport import CommError, CommTimeout, PeerClosed, RingLink

__version__ = "1.0.0"
