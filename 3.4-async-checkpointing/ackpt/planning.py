"""How big is a checkpoint, how long does it take, and how often should you take one?

Young (1974) / Daly (2006): with checkpoint cost C (time training is paused per checkpoint)
and mean time between failures M, the interval that wastes the least time is about
    T_opt = sqrt(2 C M)            (Daly's refinement adds terms that matter only when C is close to M)
Expected fraction of time wasted ~= C/T + T/(2M) + R/M, where R is the restart time
(detect the failure, reschedule, load the checkpoint).
"""
import math

# bytes stored per parameter for mixed-precision training with Adam
BYTES_PER_PARAM = {"weights_bf16": 2, "master_fp32": 4, "adam_m_fp32": 4, "adam_v_fp32": 4}


def checkpoint_bytes(params, parts=("weights_bf16", "master_fp32", "adam_m_fp32", "adam_v_fp32")):
    return params * sum(BYTES_PER_PARAM[p] for p in parts)


def write_seconds(total_bytes, nodes, per_node_bps):
    """Every node writes its share in parallel; per_node_bps is that node's sustained write bandwidth."""
    return total_bytes / (nodes * per_node_bps)


def young_daly_interval(cost_s, mtbf_s):
    return math.sqrt(2 * cost_s * mtbf_s)


def waste_fraction(interval_s, cost_s, mtbf_s, restart_s=0.0):
    """Share of wall-clock time lost to checkpoint stalls + recomputed work + restarts."""
    return cost_s / interval_s + interval_s / (2 * mtbf_s) + restart_s / mtbf_s


def job_mtbf(per_gpu_mtbf_s, gpus):
    """Failures of independent GPUs add up: N GPUs fail N times as often."""
    return per_gpu_mtbf_s / gpus
