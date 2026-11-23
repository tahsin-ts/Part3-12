"""Draw docs/architecture.png and docs/demo.png (a terminal rendering of docs/demo_output.txt, a real run)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def box(ax, x, y, w, h, color, title, lines=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=color, ec="none"))
    ax.text(x + 0.03, y + h - 0.07, title, color="white", fontsize=10, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax.text(x + 0.03, y + h - 0.22 - i * 0.12, t, color="white", fontsize=8, va="top")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))
    if label:
        ax.text((x1 + x2) / 2 + 0.03, (y1 + y2) / 2, label, fontsize=7.5, color="#475569", style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(-0.05, 4.4); ax.set_ylim(0, 2.6); ax.axis("off")
    box(ax, 1.35, 1.85, 1.7, 0.65, "#1B2A4A", "Rendezvous / coordinator (HTTP)",
        ["membership, generations, leases", "fenced key-value store, straggler monitor"])
    for i, n in enumerate(("node n1", "node n2", "node n3")):
        x = 0.1 + i * 1.45
        box(ax, x, 0.75, 1.3, 0.75, "#2F6FED", f"agent ({n})",
            ["join / wait / heartbeat", "spawn + poll workers", "self-fence if lease lost"])
        box(ax, x, 0.1, 1.3, 0.45, "#12A37F", f"worker rank {i}", ["chunked grads -> KV -> sum"])
        arrow(ax, x + 0.65, 0.75, x + 0.65, 0.55)
        arrow(ax, x + 0.65, 1.5, 1.6 + i * 0.6, 1.85, "lease" if i == 0 else None)
    ax.text(3.2, 2.25, "fencing: every request and\ncheckpoint carries the generation;\nold generations get 409", fontsize=8, color="#7B3FE4")
    ax.add_patch(FancyBboxPatch((0.1, 1.85), 1.05, 0.45, boxstyle="round,pad=0.02", fc="#F08A24", ec="none"))
    ax.text(0.13, 2.2, "shared checkpoint dir", color="white", fontsize=9, weight="bold", va="top")
    ax.text(0.13, 2.04, "atomic rename + FENCE token", color="white", fontsize=7.5, va="top")
    ax.plot([0.1, 0.0, 0.0], [0.32, 0.32, 2.0], color="#F08A24", lw=1.4)
    ax.annotate("", (0.08, 2.0), (0.0, 2.0), arrowprops=dict(arrowstyle="->", color="#F08A24", lw=1.4))
    ax.text(0.02, 1.62, "rank 0 saves,\nall ranks load", fontsize=7.5, color="#F08A24", style="italic")
    ax.text(3.2, 1.9, "workers: gradient chunks via the\nfenced key-value store (all-reduce)", fontsize=7.5, color="#12A37F")
    fig.savefig(os.path.join(DOCS, "architecture.png"), dpi=110, bbox_inches="tight")


def demo():
    text = open(os.path.join(DOCS, "demo_output.txt")).read().rstrip("\n").split("\n")
    fig = plt.figure(figsize=(11, 0.6 + 0.19 * len(text)))
    fig.patch.set_facecolor("#0F172A")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#0F172A")
    for i, line in enumerate(text):
        col = "#98C379" if line.startswith("$") else ("#F0A35E" if "fault" in line else "#E2E8F0")
        ax.text(0.01, 1 - (i + 1) / (len(text) + 1.5), line, family="monospace", fontsize=8.2, color=col, va="center")
    fig.savefig(os.path.join(DOCS, "demo.png"), dpi=110, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    architecture()
    demo()
    print("wrote docs/architecture.png and docs/demo.png")
