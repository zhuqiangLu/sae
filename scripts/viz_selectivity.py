"""Bar plot of the greedy route selectivity result: accuracy per subject under
each ablation condition, vs clean and magnitude-matched random. Shows the
'dent-own-topic, spare-general' (surgical) pattern.

Run: PYTHONPATH=src python3 scripts/viz_selectivity.py --route greedy
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PW = "/share1/zhlu6105/know_trans_data/pathways"
ap = argparse.ArgumentParser()
ap.add_argument("--route", default="greedy")
ap.add_argument("--split", default="test", choices=["test", "heldout"])
a = ap.parse_args()

r = json.load(open(os.path.join(PW, f"selectivity_results_{a.route}.json")))
data = r[a.split]
N = r["test_n"] if a.split == "test" else r["heldout_n"]

subjects = ["econometrics", "professional_medicine", "general_avg"]
slabels = ["econometrics\n(N=%d)" % N.get("econometrics", 0),
           "professional_medicine\n(N=%d)" % N.get("professional_medicine", 0),
           "general_avg\n(6 subjects)"]
conds = ["clean", "ablate_econ", "ablate_medical", "ablate_math", "rand_econ"]
clabels = ["clean", "ablate ECON route", "ablate MEDICAL route", "ablate MATH route", "random (matched)"]
colors = ["#444444", "#2c7fb8", "#c51b8a", "#e6a000", "#bbbbbb"]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(subjects)); w = 0.16
for i, (cond, cl, col) in enumerate(zip(conds, clabels, colors)):
    vals = [data[cond][s] for s in subjects]
    bars = ax.bar(x + (i - 2) * w, vals, w, label=cl, color=col,
                  edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.0, color=col)

ax.axhline(0.25, ls="--", lw=1.0, color="#cc0000", alpha=0.7)
ax.text(len(subjects) - 0.5, 0.262, "chance 0.25", color="#cc0000", fontsize=8, ha="right")

# highlight the diagonal (route matching the subject) with a marker
diag = {"econometrics": "ablate_econ", "professional_medicine": "ablate_medical"}
for xi, s in enumerate(subjects):
    if s in diag:
        ci = conds.index(diag[s])
        v = data[diag[s]][s]
        ax.annotate("own route", (xi + (ci - 2) * w, v), xytext=(0, -22),
                    textcoords="offset points", ha="center", fontsize=6.5,
                    color="#111", arrowprops=dict(arrowstyle="->", lw=0.7))

ax.set_xticks(x); ax.set_xticklabels(slabels, fontsize=9)
ax.set_ylabel("MMLU accuracy", fontsize=10)
ax.set_ylim(0, max(0.9, max(data["clean"][s] for s in subjects) + 0.08))
ax.set_title(f"Greedy route selectivity — Llama-3.1-8B ({a.split} split)\n"
             "ablating a route DENTS its own topic, SPARES general (surgical)", fontsize=11)
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
ax.grid(axis="y", alpha=0.25)

out = os.path.join("report", f"selectivity_{a.route}_{a.split}.png")
os.makedirs("report", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
