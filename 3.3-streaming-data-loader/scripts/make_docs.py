"""Render docs/architecture.png and docs/demo.png (a terminal screenshot of real command output)."""
import os
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")


def box(ax, x, y, w, h, color, title, lines=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=color, ec=color))
    ax.text(x + 0.01, y + h - 0.03, title, color="white", fontsize=9.5, weight="bold", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 0.01, y + h - 0.08 - i * 0.04, line, color="white", fontsize=7.8, va="top")


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4))
    if label:
        ax.text((x1 + x2) / 2 + 0.005, (y1 + y2) / 2 + 0.01, label, fontsize=7.2, color="#475569", style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, 0.975, "streamload: shards -> range reads -> decode workers -> bounded queues -> training step", fontsize=12,
            weight="bold", color="#1B2A4A")
    box(ax, 0.0, 0.70, 0.22, 0.22, "#334155", "Shard store", ["local dir or HTTP object", "store (Range, 503s, latency)", "manifest.json + sha256"])
    box(ax, 0.27, 0.70, 0.22, 0.22, "#2F6FED", "shards.py / reader.py", ["[len|crc|payload] ... index", "footer read first (1 GET)", "coalesced block reads"])
    box(ax, 0.54, 0.70, 0.21, 0.22, "#7B3FE4", "plan.py (integers only)", ["shard shuffle per epoch", "rank x worker split", "buffered shuffle, schedule"])
    box(ax, 0.79, 0.70, 0.21, 0.22, "#E0457B", "state_dict()", ['{"epoch", "step", "fp"}', "41 bytes of JSON", "exact mid-epoch resume"])
    arrow(ax, 0.22, 0.81, 0.27, 0.81)
    arrow(ax, 0.75, 0.81, 0.79, 0.81)
    for k in range(3):
        x = 0.05 + k * 0.22
        box(ax, x, 0.33, 0.19, 0.22, "#12A37F", f"decode worker {k}", ["own shards only", "shuffle buffer", "decode + crop + flip"])
        arrow(ax, 0.38, 0.70, x + 0.095, 0.55)
        arrow(ax, 0.645, 0.70, x + 0.12, 0.55)
        for j in range(3):
            ax.add_patch(FancyBboxPatch((x + 0.03 + j * 0.045, 0.2), 0.035, 0.06, boxstyle="round,pad=0.003", fc="#FDE68A", ec="#B45309"))
        arrow(ax, x + 0.095, 0.33, x + 0.095, 0.265)
    ax.text(0.05, 0.155, "bounded prefetch queues (depth = prefetch); arrays in shared memory, only names in the pipe", fontsize=8, color="#92400E")
    box(ax, 0.74, 0.12, 0.26, 0.22, "#F08A24", "main process", ["takes batches round-robin", "step counter -> state", "GPU step (here: sleep)"])
    arrow(ax, 0.66, 0.23, 0.74, 0.23)
    ax.text(0.0, 0.03, "Rank r, worker w reads shards order[g::R x W] with g = r x W + w: no sample twice per epoch, same steps on every rank.",
            fontsize=8, color="#1B2A4A")
    fig.savefig(os.path.join(DOCS, "architecture.png"), dpi=130, bbox_inches="tight")


def demo():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "ds")
        cmds = [["make-data", ds, "--records", "2000", "--per-shard", "100"], ["verify", ds],
                ["read", ds, "--workers", "2", "--batch", "64"], ["resume-check", ds, "--workers", "2", "--batch", "64", "--stop-at", "7"],
                ["starve", "--per-core", "300", "--gpus", "H100", "--models", "resnet50,vit-b16"]]
        text = []
        for c in cmds:
            out = subprocess.run([sys.executable, "-m", "streamload"] + c, env=env, capture_output=True, text=True, check=True).stdout
            text.append("$ streamload " + " ".join(c).replace(d, "/tmp"))
            text += [line.replace(d, "/tmp") for line in out.rstrip("\n").split("\n")]
            text.append("")
    fig = plt.figure(figsize=(12, 0.17 * len(text) + 0.4))
    fig.patch.set_facecolor("#0F172A")
    for i, line in enumerate(text):
        fig.text(0.01, 1 - (i + 1) / (len(text) + 1), line, family="monospace", fontsize=7.6,
                 color="#86EFAC" if line.startswith("$") else "#E2E8F0")
    fig.savefig(os.path.join(DOCS, "demo.png"), dpi=110, facecolor=fig.get_facecolor())
    return "\n".join(text)


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    architecture()
    print(demo())
    print("wrote docs/architecture.png and docs/demo.png")
