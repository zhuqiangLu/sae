"""STAR/FIS all-downstream route with the FIXED SAE-space WFS source (Alg.1+2, Eq.5).

Steps 1-3 (source): onset l* (AUROC) -> deltaWFS elbow features -> decoder
back-projection -> elbow dense S_src.
Steps 4-5 (causal probe): zero S_src at l*, FIS = f*mu*Delta_zeroout over ALL
downstream dense neurons (one source ablation).
Step 6 (assemble): route = S_src (root at l*) ∪ elbow-FIS dense neurons per layer.

Writes {model}_{knowledge}_star_nodes.parquet (consumed by eval_selectivity --route star).
Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python3 scripts/run_cspt_star.py --knowledge topic_medical
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.cspt import pick_onset_layer, elbow_k, trace_pathway

DATA = "/share1/zhlu6105/know_trans_data"; FS = os.path.join(DATA, "feature_scores")
OUT = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("run_cspt_star")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Llama-3.1-8B")
ap.add_argument("--knowledge", required=True)
ap.add_argument("--tau", type=float, default=0.70)
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--kmax", type=int, default=16)
ap.add_argument("--kmax-feat", dest="kmax_feat", type=int, default=32)
a = ap.parse_args()
MPATH = f"/share1/zhlu6105/models/{a.model}"
C = a.knowledge

adf = pd.read_parquet(os.path.join(FS, f"{a.model}_alllayer.parquet"))
wdf = pd.read_parquet(os.path.join(FS, f"{a.model}_alllayer_wfs.parquet"))
bundle = SAEBundle.load(os.path.join(DATA, "saes", a.model, "seed0"))
battery = {c.name: c for c in load_battery("data/concepts_pilot")}
if C not in battery:
    raise SystemExit(f"unknown knowledge {C!r}; have {list(battery)}")

# ---- Steps 1-3: SAE-space WFS source -> dense S_src ----
l_star, _ = pick_onset_layer(adf, C, a.tau)
w = wdf[(wdf.concept == C) & (wdf.layer == l_star)].sort_values("delta_wfs", ascending=False)
dwv = w.delta_wfs.to_numpy()
kf = elbow_k(dwv, a.kmin, a.kmax_feat)
feats = w.feature.to_numpy()[:kf].astype(np.int64)
fw = np.clip(dwv[:kf], 0.0, None).astype(np.float32)
Wd = bundle[l_star].W_dec.detach().float().cpu().numpy()
za = np.abs(Wd[:, feats] @ fw)
order = np.argsort(za)[::-1]
ks = elbow_k(za[order], a.kmin, a.kmax)
S_src = np.sort(order[:ks]).astype(np.int64)
log.info("[%s] l*=%d source: %d deltaWFS feats -> %d dense S_src", C, l_star, kf, len(S_src))

# ---- Steps 4-5: zero S_src at l*, FIS over all downstream layers ----
model, tok = load_model_and_tokenizer(MPATH, dtype="bfloat16", device="cuda")
fis_df, _ = trace_pathway(model, tok, bundle, battery[C], l_star, S_src,
                          auroc_df=adf, max_len=256, batch_size=16, dtype="bfloat16")

# ---- Step 6: route = S_src ∪ elbow-FIS per downstream layer ----
rows = [{"layer": int(l_star), "neuron": int(n), "concept": C} for n in S_src]
for L, sub in fis_df.groupby("layer"):
    v = np.sort(sub.fis.to_numpy())[::-1]
    k = elbow_k(v, a.kmin, a.kmax)
    for n in sub.nlargest(k, "fis").neuron:
        rows.append({"layer": int(L), "neuron": int(n), "concept": C})
df = pd.DataFrame(rows)
out = os.path.join(OUT, f"{a.model}_{C}_star_nodes.parquet")
df.to_parquet(out, index=False)
fis_df.to_parquet(os.path.join(OUT, f"{a.model}_{C}_star_fis.parquet"), index=False)
c = df.groupby("layer").neuron.count()
log.info("[%s] saved %s | route nodes=%d over %d layers (incl source)", C, out, len(df), df.layer.nunique())
print(f"saved {out} | l*={l_star} nodes={len(df)} layers={df.layer.nunique()} (source {len(S_src)} + FIS path)")
