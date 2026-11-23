"""Draw docs/architecture.png, docs/demo.png (real CLI output) and docs/validation.png."""
import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ptuner import cli, validate  # noqa: E402

DOCS = os.path.join(ROOT, "docs")


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec=color))
    ax.text(x + 0.15, y + h - 0.32, title, color="white", fontsize=10.5, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.78 - 0.36 * i, t, color="white", fontsize=8.6, va="top")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.6))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    box(ax, 0.1, 4.2, 3.2, 1.9, "Inputs", ["model: L, h, heads, V, seq", "cluster: GPUs, memory, FLOP/s,", "NVLink / IB bandwidth, global batch"], "#334155")
    box(ax, 3.9, 4.2, 3.2, 1.9, "search.py", ["enumerate dp x tp x pp x ZeRO", "x micro-batch x recompute", "prune invalid + out-of-memory"], "#2563EB")
    box(ax, 7.7, 4.2, 3.2, 1.9, "explain.py", ["ranked table, tokens/s/GPU", "time + memory breakdown", "why #1 beats the next layout"], "#7C3AED")
    box(ax, 2.0, 1.6, 3.2, 1.9, "memory.py", ["16 B/param + ZeRO sharding", "Korthikanti activations", "1F1B in-flight micro-batches"], "#0F766E")
    box(ax, 5.8, 1.6, 3.2, 1.9, "timing.py", ["FLOPs at an efficiency factor", "TP all-reduce, PP bubble + p2p", "DP alpha-beta ring, overlap"], "#D97706")
    box(ax, 0.1, 0.1, 5.2, 1.1, "pipeline_sim.py (NumPy)", ["1F1B / GPipe event simulation checks (p-1)/(m+p-1)"], "#059669")
    box(ax, 5.7, 0.1, 5.2, 1.1, "validate.py", ["Megatron-LM Table 1 throughput, ZeRO Figure 1 memory"], "#DC2626")
    arrow(ax, 3.3, 5.15, 3.9, 5.15); arrow(ax, 7.1, 5.15, 7.7, 5.15)
    arrow(ax, 5.0, 4.2, 3.8, 3.5); arrow(ax, 6.0, 4.2, 7.2, 3.5)
    arrow(ax, 7.4, 1.6, 8.0, 1.2); arrow(ax, 3.4, 1.6, 2.8, 1.2)
    fig.savefig(os.path.join(DOCS, "architecture.png"), dpi=130, bbox_inches="tight")


def demo():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["search", "--model", "gpt3-13b", "--gpus", "8", "--batch", "512", "--top", "5"])
    text = "$ ptuner search --model gpt3-13b --gpus 8 --batch 512 --top 5\n" + buf.getvalue()
    lines = text.rstrip().split("\n")
    fig = plt.figure(figsize=(12.5, 0.165 * len(lines) + 0.3))
    fig.patch.set_facecolor("#0F172A")
    fig.text(0.01, 0.98, text, family="monospace", fontsize=8.6, color="#E2E8F0", va="top")
    fig.savefig(os.path.join(DOCS, "demo.png"), dpi=120, facecolor=fig.get_facecolor())


def validation():
    rows = validate.megatron_rows()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [r.published for r in rows], 0.4, label="published (Megatron-LM Table 1)", color="#94A3B8")
    ax.bar([i + 0.2 for i in x], [r.predicted for r in rows], 0.4, label="this model", color="#2563EB")
    for i, r in enumerate(rows):
        ax.text(i + 0.2, r.predicted + 2, f"{r.error_pct:+.0f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([r.name.replace("gpt-", "") for r in rows])
    ax.set_ylabel("TFLOP/s per GPU"); ax.set_ylim(0, 190); ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(DOCS, "validation.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    architecture(); demo(); validation()
    print("wrote docs/architecture.png docs/demo.png docs/validation.png")
