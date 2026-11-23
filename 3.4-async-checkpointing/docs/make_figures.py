"""Render docs/architecture.png and docs/demo.png (demo = real output saved in docs/demo_output.txt)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def box(ax, x, y, w, h, color, title, lines=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=color))
    ax.text(x + 0.12, y + h - 0.28, title, color="white", fontsize=10.5, weight="bold", va="top")
    for i, l in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.68 - i * 0.34, l, color="white", fontsize=8.6, va="top")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.6))
    if label:
        ax.text((x1 + x2) / 2 + 0.08, (y1 + y2) / 2 + 0.08, label, fontsize=8, color="#475569", style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    box(ax, 0.2, 4.3, 3.4, 1.7, "#2563EB", "Training loop (N ranks)", ["each rank owns rows of every", "large tensor (Shard)", "ck.save(step, rank_states)"])
    box(ax, 4.3, 4.3, 3.4, 1.7, "#F59E0B", "Phase 1: snapshot (stall)", ["np.copyto into reused", "staging buffers", "~ pinned host memory on GPU"])
    box(ax, 8.4, 4.3, 3.4, 1.7, "#7C3AED", "Phase 2: background thread", ["ranks write rank_XXXXX.bin", "in parallel (CRC32 per chunk)", "then MANIFEST.json = commit"])
    arrow(ax, 3.6, 5.15, 4.3, 5.15); arrow(ax, 7.7, 5.15, 8.4, 5.15)
    box(ax, 0.2, 1.9, 3.4, 1.8, "#0F766E", "LocalFS backend", ["temp file + fsync", "os.replace (atomic rename)", "fsync directory"])
    box(ax, 4.3, 1.9, 3.4, 1.8, "#0891B2", "ObjectStore simulator", ["atomic whole-object put", "latency + shared bandwidth", "injected 503 faults"])
    box(ax, 8.4, 1.9, 3.4, 1.8, "#DC2626", "Retention / GC", ["keep newest K commits", "delete MANIFEST first", "sweep crash debris"])
    arrow(ax, 10.1, 4.3, 2.0, 3.7); arrow(ax, 10.1, 4.3, 6.0, 3.7); arrow(ax, 10.1, 4.3, 10.1, 3.7)
    box(ax, 0.2, 0.1, 11.6, 1.3, "#334155", "Loader: load_latest(storage, world_size)",
        ["newest manifest -> verify CRCs -> map saved row-chunks onto the new world size (4 -> 2, 4 -> 8) -> fall back to the previous step on corruption"])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "architecture.png"), dpi=110)


def demo():
    text = open(os.path.join(HERE, "demo_output.txt")).read().rstrip("\n").split("\n")
    fig = plt.figure(figsize=(11, 0.26 * len(text) + 0.6))
    fig.patch.set_facecolor("#0F172A")
    for i, line in enumerate(text):
        col = "#4ADE80" if line.startswith("$") else "#E2E8F0"
        fig.text(0.02, 1 - (i + 1) / (len(text) + 1.2), line, family="monospace", fontsize=9.6, color=col)
    fig.savefig(os.path.join(HERE, "demo.png"), dpi=110, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    architecture()
    demo()
