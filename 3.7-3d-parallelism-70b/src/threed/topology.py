"""The 3D process grid: global rank <-> (pp, dp, tp) coordinates and the three kinds of groups.

Order (Megatron-LM default): tensor-parallel rank changes fastest, then data-parallel, then pipeline.
With tp = GPUs per node, every tensor-parallel group is exactly one node, so the chatty
all-reduces of tensor parallelism stay on NVLink, and only pipeline and data-parallel
traffic crosses the network.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    tp: int
    pp: int
    dp: int

    @property
    def world(self):
        return self.tp * self.pp * self.dp

    def coords(self, rank):
        """rank -> (pp, dp, tp)."""
        if not 0 <= rank < self.world:
            raise ValueError(f"rank {rank} outside world of {self.world}")
        tp = rank % self.tp
        dp = (rank // self.tp) % self.dp
        pp = rank // (self.tp * self.dp)
        return pp, dp, tp

    def rank(self, pp, dp, tp):
        return pp * self.dp * self.tp + dp * self.tp + tp

    def tp_group(self, rank):
        pp, dp, _ = self.coords(rank)
        return [self.rank(pp, dp, t) for t in range(self.tp)]

    def dp_group(self, rank):
        pp, _, tp = self.coords(rank)
        return [self.rank(pp, d, tp) for d in range(self.dp)]

    def pp_group(self, rank):
        _, dp, tp = self.coords(rank)
        return [self.rank(p, dp, tp) for p in range(self.pp)]

    def all_groups(self, kind):
        fn = {"tp": self.tp_group, "dp": self.dp_group, "pp": self.pp_group}[kind]
        seen = []
        for r in range(self.world):
            g = fn(r)
            if g not in seen:
                seen.append(g)
        return seen

    def node_of(self, rank, gpus_per_node=8):
        return rank // gpus_per_node

    def describe(self, gpus_per_node=8):
        """Which groups stay inside one node and which cross the network."""
        out = {}
        for kind in ("tp", "pp", "dp"):
            groups = self.all_groups(kind)
            inside = sum(len({self.node_of(r, gpus_per_node) for r in g}) == 1 for g in groups)
            out[kind] = {"groups": len(groups), "size": len(groups[0]), "inside_one_node": inside}
        return out


def parse_grid(text):
    """'tp=2,pp=2,dp=2' or '2x2x2' (tp x pp x dp) -> Grid."""
    text = text.strip()
    if "=" in text:
        kv = dict(p.split("=") for p in text.split(","))
        return Grid(tp=int(kv.get("tp", 1)), pp=int(kv.get("pp", 1)), dp=int(kv.get("dp", 1)))
    tp, pp, dp = (int(v) for v in text.lower().split("x"))
    return Grid(tp=tp, pp=pp, dp=dp)
