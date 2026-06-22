"""Cross-layer 'routing' of knowledge: does a concept's detector at layer L fire
on the SAME inputs as its detector at layer L'? Measured by activation-pattern
correlation across layers (index-free), within one model/seed.

- AUC trajectory: per-layer best detector AUC (from saved scores).
- Routing matrix: Pearson corr of each concept's per-layer top detector vs the
  others, over a probe set. High off-diagonal => coherent routing through depth.
  Also reports detector-to-ANY max corr at each other layer (does the pattern
  reappear at all).

Run: PYTHONPATH=src python3 scripts/cross_layer_routing.py --config <cfg> --tag xfamily --role teacher
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import torch

from know_trans.config import load_config
from know_trans.capture import ActivationReader
from know_trans.sae import SAEBundle
from know_trans.utils import get_logger, ensure_dir, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--tag", default="xfamily", help="score-parquet tag to read")
ap.add_argument("--role", default="teacher", choices=["teacher", "student"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--probe-n", type=int, default=40000)
a = ap.parse_args()
log = get_logger("routing")

cfg = load_config(a.config)
data = cfg.paths.data
mcfg = cfg.teacher if a.role == "teacher" else cfg.student
mname = mcfg.name
layers = sorted(int(x) for x in mcfg.layers)

df = pd.read_parquet(os.path.join(data, "feature_scores", f"{mname}_seed{a.seed}_{a.tag}.parquet"))
concepts = sorted(df["concept"].unique())

# Per-(concept, layer) top detector.
top = {}
for c in concepts:
    for L in layers:
        sub = df[(df["concept"] == c) & (df["layer"] == L)]
        if len(sub):
            r = sub.loc[sub["auc"].idxmax()]
            top[(c, L)] = (int(r["feature"]), round(float(r["auc"]), 3))

print(f"\n===== AUC TRAJECTORY ACROSS LAYERS ({mname}, seed{a.seed}) =====")
print(f"{'knowledge':16s} " + " ".join(f"L{L:>2}".rjust(8) for L in layers))
for c in concepts:
    print(f"{c:16s} " + " ".join(f"{top.get((c,L),(0,float('nan')))[1]:>8}" for L in layers))

# Encode probe set through each layer's SAE (full dense codes, kept on GPU).
reader = ActivationReader(os.path.join(data, "activations", mname))
bundle = SAEBundle.load(os.path.join(data, "saes", mname, f"seed{a.seed}"))
g = torch.Generator().manual_seed(0)
samp = torch.randperm(reader.n_tokens, generator=g)[:a.probe_n]

def zcol(x):  # z-score columns
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)

codes = {}
for L in layers:
    acts, _ = reader.read(L)
    xb = acts[samp].to("cuda", dtype=torch.float32)
    codes[L] = bundle[L].to("cuda").encode_dense(xb)  # [N, d_hidden]
    del acts
log.info("encoded probe (%d tokens) at layers %s", a.probe_n, layers)

N = a.probe_n
routing = {}
for c in concepts:
    present = [L for L in layers if (c, L) in top]
    # detector-to-detector correlation matrix
    detvecs = {L: zcol(codes[L][:, top[(c, L)][0]:top[(c, L)][0] + 1]) for L in present}
    mat = {}
    for L in present:
        row = {}
        for Lp in present:
            row[Lp] = round(float((detvecs[L].t() @ detvecs[Lp]).item() / N), 3)
        mat[L] = row
    # detector-to-ANY max corr at each other layer
    toany = {}
    for L in present:
        d = detvecs[L]  # [N,1]
        for Lp in present:
            if Lp == L:
                continue
            z = zcol(codes[Lp])
            toany[f"{L}->{Lp}"] = round(float(((d.t() @ z) / N).max().item()), 3)
    offdiag = [mat[L][Lp] for L in present for Lp in present if L != Lp]
    routing[c] = {"matrix": mat, "to_any": toany,
                  "coherence": round(sum(offdiag) / len(offdiag), 3) if offdiag else None}

save_json({"model": mname, "seed": a.seed, "layers": layers,
           "auc_trajectory": {c: {L: top.get((c, L), (None, None))[1] for L in layers} for c in concepts},
           "routing": routing}, os.path.join(data, "matches", f"routing_{mname}_seed{a.seed}_{a.tag}.json"))

print(f"\n===== CROSS-LAYER ROUTING ({mname}) =====")
print("(detector-to-detector Pearson corr across layers; coherence = mean off-diagonal)\n")
for c in concepts:
    r = routing[c]
    print(f"{c}: coherence={r['coherence']}  detector->any: {r['to_any']}")
    for L, row in r["matrix"].items():
        print(f"    L{L:>2}: " + "  ".join(f"L{Lp}={v:+.2f}" for Lp, v in row.items()))
mc = [routing[c]["coherence"] for c in concepts if routing[c]["coherence"] is not None]
print(f"\nmean routing coherence across knowledge: {round(sum(mc)/len(mc),3) if mc else 'n/a'}")
print(f"saved -> matches/routing_{mname}_seed{a.seed}_{a.tag}.json")
