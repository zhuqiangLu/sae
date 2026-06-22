"""Greedy causal+differential chain route (de-contaminated, no magnitude selection).

Source at AUROC onset l* by differential (f*mu)_sens-(f*mu)_benign; each downstream
layer selected by score = Delta_ablation(prev pathway) * max(diff,0). Prior-layer
dependence (causal) AND subject-specific (differential); excludes the always-on
backbone that wrecked the magnitude chain.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python3 scripts/run_cspt_greedy.py --knowledge topic_medical
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.concepts import load_battery
from know_trans.sae import SAEBundle
from know_trans.cspt import trace_pathway_greedy

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
OUT = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("run_cspt_greedy")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Llama-3.1-8B")
ap.add_argument("--knowledge", required=True)
ap.add_argument("--tau", type=float, default=0.70)
ap.add_argument("--ksrc", default="elbow", help="'elbow' or fixed int per layer")
ap.add_argument("--ncap", type=int, default=128)
a = ap.parse_args()
MPATH = f"/share1/zhlu6105/models/{a.model}"
ksrc = "elbow" if str(a.ksrc) == "elbow" else int(a.ksrc)

adf = pd.read_parquet(os.path.join(FS, f"{a.model}_alllayer.parquet"))
wdf = pd.read_parquet(os.path.join(FS, f"{a.model}_alllayer_wfs.parquet"))
bundle = SAEBundle.load(os.path.join(DATA, "saes", a.model, "seed0"))
battery = {c.name: c for c in load_battery("data/concepts_pilot")}
if a.knowledge not in battery:
    raise SystemExit(f"unknown knowledge {a.knowledge!r}; have {list(battery)}")

model, tok = load_model_and_tokenizer(MPATH, dtype="bfloat16", device="cuda")
nodes_df, l_star = trace_pathway_greedy(
    model, tok, bundle, wdf, adf, battery[a.knowledge],
    tau_onset=a.tau, k_src=ksrc, n_cap=a.ncap, max_len=256, batch_size=16)

out = os.path.join(OUT, f"{a.model}_{a.knowledge}_greedy_nodes.parquet")
nodes_df.to_parquet(out, index=False)
c = nodes_df.groupby("layer").neuron.count()
log.info("saved %s | l*=%d nodes=%d layers=%d", out, l_star, len(nodes_df), len(c))
print(f"saved {out}")
print(f"onset l*={l_star} | total nodes={len(nodes_df)} over {len(c)} layers")
print("per-layer counts:", c.to_dict())
