"""Topological visualization of our causal FIS pathway, TraceRouter Fig.9 style.

Source (onset) node left; one column per downstream layer; top-FIS dense neurons
as nodes. FIS is heavy-tailed and explodes late, so node color/size use PER-LAYER
normalization (the circuit *shape* at each depth); each layer's ABSOLUTE peak FIS
is annotated above its column so the magnitude profile is still visible.

Run: PYTHONPATH=src python3 scripts/viz_pathway.py --model Llama-3.1-8B --knowledge safety
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from know_trans.cspt import pick_onset_layer

DATA = "/share1/zhlu6105/know_trans_data"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Llama-3.1-8B")
ap.add_argument("--knowledge", default="safety")
ap.add_argument("--k", type=int, default=14)
ap.add_argument("--fanout", type=int, default=3)
a = ap.parse_args()

fis = pd.read_parquet(os.path.join(DATA, "pathways", f"{a.model}_{a.knowledge}_fis.parquet"))
adf = pd.read_parquet(os.path.join(DATA, "feature_scores", f"{a.model}_alllayer.parquet"))
onset, _ = pick_onset_layer(adf, a.knowledge, 0.70)
layers = sorted(int(l) for l in fis.layer.unique())
cmap = plt.get_cmap("Purples")

# per-layer top-k nodes with PER-LAYER normalized strength t in [0,1]; strongest low
nodes, peak = {}, {}
for L in layers:
    top = fis[fis.layer == L].nlargest(a.k, "fis")[["neuron", "fis"]].values
    lmax = max(float(top[:, 1].max()), 1e-9); peak[L] = lmax
    top = top[np.argsort(top[:, 1])]
    ys = np.linspace(0.12, 0.97, len(top))[::-1]
    nodes[L] = [(int(n), float(f), float(f) / lmax, float(y)) for (n, f), y in zip(top, ys)]

fig, ax = plt.subplots(figsize=(min(2 + 0.52 * len(layers), 22), 6.4))
xs = {L: i + 1 for i, L in enumerate(layers)}
src_x, src_y = 0.0, 0.5

def edge(x0, y0, x1, y1, t):
    ax.plot([x0, x1], [y0, y1], color=cmap(0.25 + 0.75 * t),
            lw=0.4 + 2.4 * t, alpha=0.12 + 0.8 * t, zorder=1, solid_capstyle="round")

for (n, f, t, y) in nodes[layers[0]]:
    edge(src_x, src_y, xs[layers[0]], y, t)
for Li, Lj in zip(layers[:-1], layers[1:]):
    tgt = sorted(nodes[Lj], key=lambda z: -z[2])[:a.fanout]
    for (_, _, _, ys) in nodes[Li]:
        for (_, _, tj, yj) in tgt:
            edge(xs[Li], ys, xs[Lj], yj, tj)

for L in layers:
    for (n, f, t, y) in nodes[L]:
        ax.scatter(xs[L], y, s=18 + 240 * t, c=[cmap(0.18 + 0.82 * t)],
                   edgecolors=("#1a0a3a" if t > 0.5 else "none"), linewidths=0.7, zorder=3)
    ax.text(xs[L], 1.015, f"{peak[L]:.0f}", ha="center", va="bottom", fontsize=6.2, color="#5b3a8a")

ax.scatter([src_x], [src_y], marker="D", s=430, c="#f3c41a", edgecolors="#7a5c00", linewidths=1.2, zorder=4)
ax.text(src_x, src_y - 0.08, f"L{onset}\nsource", ha="center", va="top", fontsize=8.5, weight="bold")

for L in layers:
    ax.text(xs[L], -0.045, f"L{L}", ha="center", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.22", fc="#e9ecfb", ec="#c3c9ee"))

ax.set_xlim(-0.8, len(layers) + 0.8); ax.set_ylim(-0.13, 1.10); ax.axis("off")
ax.text(0.5, 1.155, f"Causal FIS pathway — {a.model} · {a.knowledge}", transform=ax.transAxes,
        ha="center", fontsize=12.5, weight="bold")
ax.text(0.5, 1.115, f"onset L{onset};  per-layer-normalized node strength;  number above each column = absolute peak FIS",
        transform=ax.transAxes, ha="center", fontsize=8.2, color="#555")

sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01); cb.set_label("FIS (normalized within layer)", weight="bold")

leg = [Line2D([0], [0], marker="D", color="none", markerfacecolor="#f3c41a", markeredgecolor="#7a5c00", markersize=12, label=f"L{onset} source"),
       Line2D([0], [0], color=cmap(0.95), lw=2.6, label="strong (rel. FIS)"),
       Line2D([0], [0], color=cmap(0.35), lw=0.8, label="weak (rel. FIS)")]
ax.legend(handles=leg, loc="lower center", ncol=3, frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, 1.04))

out = os.path.join("report", f"pathway_{a.model}_{a.knowledge}.png")
os.makedirs("report", exist_ok=True)
fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
print("saved", out, "| peak FIS by layer:", {L: round(peak[L], 1) for L in layers if peak[L] > 1})
