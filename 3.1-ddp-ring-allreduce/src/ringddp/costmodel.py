"""Alpha-beta cost model of ring all-reduce: fit it to measurements, project it to GPU clusters."""
import numpy as np


def ring_time(nbytes, world, alpha, bw):
    """2(N-1) steps; each step pays latency alpha and moves nbytes/N at bandwidth bw (bytes/s)."""
    if world == 1:
        return 0.0
    return 2 * (world - 1) * alpha + 2 * (world - 1) / world * nbytes / bw


def fit(nbytes, seconds, world):
    """Least-squares fit of t = a + c*S; returns (alpha seconds, bw bytes/s)."""
    a_mat = np.vstack([np.ones(len(nbytes)), np.asarray(nbytes, float)]).T
    (a, c), *_ = np.linalg.lstsq(a_mat, np.asarray(seconds, float), rcond=None)
    return a / (2 * (world - 1)), 2 * (world - 1) / world / c


# PROJECTION ONLY: assumptions, not measurements (see README "Benchmark summary").
CLUSTERS = {
    "8x A100 NVLink/NVSwitch": dict(world=8, alpha=5e-6, bw=300e9 * 0.77),   # 300 GB/s per direction, 77% efficiency
    "8 nodes, 100 GbE": dict(world=8, alpha=25e-6, bw=12.5e9 * 0.80),        # 12.5 GB/s line rate, 80% efficiency
}
MODELS = [("ResNet-50", 25.6e6, 4), ("BERT-large", 340e6, 4), ("Llama-2-7B", 6.74e9, 2),
          ("Llama-2-13B", 13.0e9, 2), ("Llama-2-70B", 69.0e9, 2)]       # (name, parameters, gradient bytes each)


def projection():
    rows = []
    for name, n_params, b in MODELS:
        nbytes = n_params * b
        rows.append((name, nbytes, [ring_time(nbytes, c["world"], c["alpha"], c["bw"]) for c in CLUSTERS.values()]))
    return rows
