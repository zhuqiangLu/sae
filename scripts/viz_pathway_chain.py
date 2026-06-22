"""Connected causal circuit (paper Fig.9 style) from the CHAINED trace.

Draws per-layer WFS sensitive nodes (labeled by neuron id) connected by REAL
one-hop causal edges i(L-1) -> j(L), edge weight = measured Delta from zeroing i.
Edges are normalized WITHIN each layer-transition (heavy-tailed across depth), and
each column's absolute peak node f*mu / each transition's peak Delta is annotated.

Run: PYTHONPATH=src python3 scripts/viz_pathway_chain.py --model Llama-3.1-8B --knowledge safety
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import json

DATA = "/share1/zhlu6105/know_trans_data"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Llama-3.1-8B")
ap.add_argument("--knowledge", default="safety")
ap.add_argument("--top-edges", type=int, default=10, help="strongest edges kept per layer transition")
ap.add_argument("--label-top", type=int, default=2, help="nodes labeled by neuron id per layer")
a = ap.parse_args()

nodes = pd.read_parquet(os.path.join(DATA, "pathways", f"{a.model}_{a.knowledge}_chain_nodes.parquet"))
edges = pd.read_parquet(os.path.join(DATA, "pathways", f"{a.model}_{a.knowledge}_chain_edges.parquet"))
layers = sorted(int(l) for l in nodes.layer.unique())
xs = {L: i for i, L in enumerate(layers)}
cmap = plt.get_cmap("Purples")

# node y-position = neuron IDENTITY (fixed slot by dense coordinate id; NO reallocation).
# A given coordinate sits at the SAME height in every column; strength t = per-layer f*mu
# is used only for node size/color, never position.
d_model = int(json.load(open(os.path.join(DATA, "activations", a.model, "meta.json")))["d_model"])
pos, peak_fmu = {}, {}
for L in layers:
    sub = nodes[nodes.layer == L]
    fmax = max(float(sub.fmu.max()), 1e-12); peak_fmu[L] = fmax
    for _, r in sub.iterrows():
        y = 0.06 + 0.90 * (int(r.neuron) / d_model)   # fixed by coordinate identity
        pos[(L, int(r.neuron))] = (xs[L], float(y), float(r.fmu) / fmax)

fig, ax = plt.subplots(figsize=(min(2 + 0.55 * len(layers), 22), 6.6))

# ---- edges: keep strongest per transition, normalize Delta within transition ----
peak_delta = {}
for p in range(len(layers) - 1):
    Ls, Ld = layers[p], layers[p + 1]
    e = edges[(edges.src_layer == Ls) & (edges.dst_layer == Ld)]
    if e.empty:
        continue
    dmax = max(float(e.delta.max()), 1e-12); peak_delta[Ls] = dmax
    for _, r in e.nlargest(a.top_edges, "delta").iterrows():
        s = pos.get((Ls, int(r.src_neuron))); d = pos.get((Ld, int(r.dst_neuron)))
        if s is None or d is None:
            continue
        t = float(r.delta) / dmax
        ax.plot([s[0], d[0]], [s[1], d[1]], color=cmap(0.30 + 0.70 * t),
                lw=0.4 + 3.0 * t, alpha=0.15 + 0.8 * t, zorder=1, solid_capstyle="round")

# ---- nodes ----
for (L, n), (x, y, t) in pos.items():
    ax.scatter(x, y, s=22 + 230 * t, c=[cmap(0.22 + 0.78 * t)],
               edgecolors=("#1a0a3a" if t > 0.55 else "none"), linewidths=0.7, zorder=3)

# label the strongest nodes per layer with neuron id
for L in layers:
    sub = nodes[nodes.layer == L].nlargest(a.label_top, "fmu")
    for _, r in sub.iterrows():
        x, y, _t = pos[(L, int(r.neuron))]
        ax.text(x, y, str(int(r.neuron)), ha="center", va="center", fontsize=5.0,
                color="white", zorder=4, weight="bold")

# column labels + absolute peak annotations
for L in layers:
    ax.text(xs[L], -0.045, f"L{L}", ha="center", va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.2", fc="#e9ecfb", ec="#c3c9ee"))
    ax.text(xs[L], 1.005, f"{peak_fmu[L]:.1f}", ha="center", va="bottom", fontsize=5.6, color="#7a5c9a")

ax.set_xlim(-0.7, len(layers) - 0.3); ax.set_ylim(-0.12, 1.07); ax.axis("off")
ax.text(0.5, 1.10, f"Connected causal circuit — {a.model} · {a.knowledge}",
        transform=ax.transAxes, ha="center", fontsize=12.5, weight="bold")
ax.text(0.5, 1.06, "y = neuron-id (fixed slot, no reallocation) · nodes = per-layer WFS sensitive coords · "
                   "edges = measured one-hop causal Δ (zero i@L-1 → shift j@L); size/color = strength",
        transform=ax.transAxes, ha="center", fontsize=7.0, color="#555")

sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("edge / node strength (normalized within layer)", weight="bold")

out = os.path.join("report", f"chain_{a.model}_{a.knowledge}.png")
os.makedirs("report", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
print("edges kept per transition:", a.top_edges, "| total edges:", len(edges))
