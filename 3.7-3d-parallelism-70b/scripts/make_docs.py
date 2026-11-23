"""Render docs/architecture.png and docs/demo.png (a terminal screenshot of real command output)."""
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def box(ax, x, y, w, h, color, title, lines=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=color, ec=color))
    ax.text(x + 0.01, y + h - 0.035, title, color="white", fontsize=9.5, weight="bold", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.01, y + h - 0.085 - i * 0.04, line, color="white", fontsize=7.8, va="top")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))
    if label:
        ax.text((x1 + x2) / 2 + 0.005, (y1 + y2) / 2, label, fontsize=7.2, color="#475569", style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 0.975, "threed: tensor x pipeline x data parallelism on CPU processes", fontsize=12, weight="bold", color="#1B2A4A")
    box(ax, 0.0, 0.74, 0.3, 0.19, "#334155", "cli.py  (threed ...)", ["layout | tp-check | pipeline", "train | failover | plan | demo"])
    box(ax, 0.35, 0.74, 0.3, 0.19, "#F08A24", "engine.py  launcher + worker", ["one process per rank, restarts", "schedule loop, DP all-reduce, SGD", "sharded checkpoints"])
    box(ax, 0.70, 0.74, 0.3, 0.19, "#7B3FE4", "planner.py  70B design study", ["memory / GPU, step time, MFU", "comm GB per step, days to train", "checkpoint interval (Young/Daly)"])
    arrow(ax, 0.3, 0.835, 0.35, 0.835); arrow(ax, 0.65, 0.835, 0.70, 0.835)
    box(ax, 0.0, 0.43, 0.23, 0.24, "#2F6FED", "topology.py", ["rank -> (pp, dp, tp)", "tp fastest, then dp, then pp", "tp / dp / pp groups"])
    box(ax, 0.255, 0.43, 0.23, 0.24, "#12A37F", "model.py", ["NumPy GPT block, manual", "backward; vocab-parallel", "embedding + cross-entropy"])
    box(ax, 0.51, 0.43, 0.23, 0.24, "#0E9FC7", "pipeline.py", ["GPipe and 1F1B schedules", "event simulator", "bubble (p-1)/(m+p-1)"])
    box(ax, 0.765, 0.43, 0.235, 0.24, "#E0457B", "comm.py", ["send / recv by tag", "ring all-reduce, all-gather", "bytes counted per kind"])
    for x in (0.115, 0.37, 0.625, 0.88):
        arrow(ax, 0.5, 0.74, x, 0.67)
    ax.text(0, 0.37, "A 2 x 2 x 2 run: 8 processes", fontsize=10, weight="bold", color="#1B2A4A")
    colors = {0: "#93C5FD", 1: "#86EFAC"}
    for pp in range(2):
        for dp in range(2):
            x0 = 0.02 + dp * 0.5 + pp * 0.245
            ax.add_patch(FancyBboxPatch((x0 - 0.01, 0.05), 0.23, 0.27, boxstyle="round,pad=0.005", fc="#F1F5F9", ec="#94A3B8"))
            ax.text(x0, 0.30, f"stage {pp}, replica {dp}", fontsize=8, weight="bold", color="#1E293B")
            for tp in range(2):
                r = pp * 4 + dp * 2 + tp
                ax.add_patch(FancyBboxPatch((x0 + tp * 0.105, 0.1), 0.095, 0.15, boxstyle="round,pad=0.004", fc=colors[pp], ec="#1E293B"))
                ax.text(x0 + tp * 0.105 + 0.047, 0.19, f"rank {r}", ha="center", fontsize=8, weight="bold")
                ax.text(x0 + tp * 0.105 + 0.047, 0.14, f"tp {tp}", ha="center", fontsize=7.5)
            ax.text(x0 + 0.1, 0.065, "TP all-reduce (NVLink)", ha="center", fontsize=6.8, color="#1D4ED8")
    for dp in range(2):
        xa = 0.02 + dp * 0.5 + 0.215
        ax.annotate("", (xa + 0.03, 0.175), (xa - 0.005, 0.175), arrowprops=dict(arrowstyle="<->", color="#C2410C", lw=1.8))
        ax.text(xa - 0.025, 0.335, "PP send/recv", fontsize=7.5, color="#C2410C")
    ax.text(0.2, 0.01, "DP all-reduce of gradients between replicas of the same stage and tp rank (network)", fontsize=7.5, color="#047857")
    fig.savefig(os.path.join(DOCS, "architecture.png"), dpi=130, bbox_inches="tight")


def demo():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    cmds = [["tp-check", "--tp", "1", "2", "4"], ["train", "--grid", "2x2x2", "--steps", "12"], ["plan", "--hw", "h100", "--gpus", "64", "256", "1024"]]
    text = []
    for c in cmds:
        out = subprocess.run([sys.executable, "-m", "threed"] + c, env=env, capture_output=True, text=True, check=True).stdout
        text.append("$ threed " + " ".join(c))
        text += out.rstrip("\n").split("\n")
        text.append("")
    fig = plt.figure(figsize=(12, 0.16 * len(text) + 0.4))
    fig.patch.set_facecolor("#0F172A")
    for i, line in enumerate(text):
        fig.text(0.01, 1 - (i + 1) / (len(text) + 1), line, family="monospace", fontsize=7.6,
                 color="#86EFAC" if line.startswith("$") else "#E2E8F0")
    fig.savefig(os.path.join(DOCS, "demo.png"), dpi=110, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    architecture()
    demo()
    print("wrote docs/architecture.png and docs/demo.png")
