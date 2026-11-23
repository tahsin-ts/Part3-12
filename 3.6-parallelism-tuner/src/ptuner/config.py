"""Model, cluster and plan descriptions, plus presets.

All bandwidths are *effective bus bandwidths* in GB/s per GPU (what a ring
all-reduce actually achieves), not the datasheet link speed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace

GIB = 1024 ** 3


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    hidden: int
    heads: int
    vocab: int = 51200
    seq: int = 2048

    def params(self) -> float:
        """GPT parameter count (Narayanan et al. 2021, eq. 2)."""
        l, h, V, s = self.layers, self.hidden, self.vocab, self.seq
        return 12 * l * h * h * (1 + 13 / (12 * h) + (V + s) / (12 * l * h))


@dataclass(frozen=True)
class Cluster:
    name: str
    gpus: int
    gpus_per_node: int
    mem_gib: float          # HBM per GPU
    peak_tflops: float      # dense BF16 tensor-core peak
    hbm_gbs: float          # memory bandwidth (optimizer step)
    intra_gbs: float        # effective all-reduce bus bandwidth inside a node
    inter_gbs: float        # effective bandwidth per GPU between nodes
    intra_alpha_us: float = 5.0
    inter_alpha_us: float = 15.0

    def nodes(self) -> int:
        return -(-self.gpus // self.gpus_per_node)


@dataclass(frozen=True)
class Plan:
    dp: int
    tp: int
    pp: int
    zero: int       # 0, 1, 2, 3
    mbs: int        # micro-batch size
    recompute: str  # none | selective | full

    def short(self) -> str:
        return f"dp{self.dp} tp{self.tp} pp{self.pp} z{self.zero} mbs{self.mbs} rc={self.recompute}"


@dataclass(frozen=True)
class Knobs:
    """Assumptions of the step-time model. Change them to calibrate."""
    efficiency: float = 0.65        # fraction of peak reached by the transformer GEMMs
    dp_overlap: float = 0.5         # fraction of the DP gradient sync hidden behind backward
    grad_bytes: int = 2             # bytes per gradient element in the DP reduction
    overhead_gib: float = 3.0       # CUDA context, NCCL buffers, workspaces, fragmentation
    seq_parallel: bool = True       # Megatron sequence parallelism when tp > 1
    gemm_knee: float = 128.0        # GEMM efficiency x w/(w+knee), w = hidden/tp (small GEMMs are slower)
    elem_bytes: float = 40.0        # HBM bytes per hidden element per layer in a forward pass
    score_bytes: float = 6.0        # HBM bytes per attention-score element per layer (fused softmax)


MODELS = {m.name: m for m in [
    Model("gpt-1.7b", 24, 2304, 24), Model("gpt-3.6b", 30, 3072, 32), Model("gpt-7.5b", 36, 4096, 32),
    Model("gpt-18b", 40, 6144, 48), Model("gpt-39b", 48, 8192, 64), Model("gpt-76b", 60, 10240, 80),
    Model("gpt-146b", 80, 12288, 96), Model("gpt-310b", 96, 16384, 128), Model("gpt-530b", 105, 20480, 128),
    Model("gpt-1t", 128, 25600, 160),
    Model("gpt3-1.3b", 24, 2048, 16), Model("gpt3-6.7b", 32, 4096, 32), Model("gpt3-13b", 40, 5120, 40),
    Model("gpt3-175b", 96, 12288, 96),
]}

# Effective bandwidths are assumptions (roughly 75-90% of link peak); see README.
CLUSTERS = {c.name: c for c in [
    Cluster("dgx-a100", 8, 8, 80, 312, 2039, 230, 22, 5, 15),
    Cluster("dgx-h100", 8, 8, 80, 989, 3350, 370, 45, 5, 15),
    Cluster("a100-pcie", 8, 8, 80, 312, 1935, 20, 22, 10, 15),
    Cluster("l40s-pcie", 8, 8, 48, 362, 864, 20, 22, 10, 15),
    Cluster("rtx4090", 4, 4, 24, 165, 1008, 12, 10, 10, 20),
]}


def get_model(name: str) -> Model:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]


def get_cluster(name: str, gpus: int | None = None) -> Cluster:
    if name not in CLUSTERS:
        raise KeyError(f"unknown cluster {name!r}; known: {', '.join(CLUSTERS)}")
    c = CLUSTERS[name]
    return replace(c, gpus=gpus) if gpus else c


def load_cluster(path: str) -> Cluster:
    with open(path) as f:
        return Cluster(**json.load(f))


def to_dict(x) -> dict:
    return asdict(x)
