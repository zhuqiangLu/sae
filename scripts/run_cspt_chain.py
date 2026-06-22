"""Chained CSPT: connected layer-to-layer causal pathway (paper Fig.9 style).

Per (model, knowledge): per-layer WFS sensitive nodes (source_neurons at EVERY
layer) + per-neuron one-hop edges (zero neuron i at L-1, read its influence on
each node j at L over sensitive tokens). Unlike run_cspt.py (a star from one
onset source), this yields a connected circuit with real L-1 -> L causality.

Run: PYTHONPATH=src python3 scripts/run_cspt_chain.py --model Llama-3.1-8B --knowledge safety
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.cspt import trace_pathway_chain

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
OUT = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("run_cspt_chain")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Llama-3.1-8B")
ap.add_argument("--knowledge", default="safety")
ap.add_argument("--kfeat", type=int, default=10)
ap.add_argument("--ksrc", default="8", help="fixed int nodes/layer, or 'elbow' for per-layer Kneedle K")
ap.add_argument("--ncap", type=int, default=128)
a = ap.parse_args()
MPATH = f"/share1/zhlu6105/models/{a.model}"
ksrc = "elbow" if str(a.ksrc) == "elbow" else int(a.ksrc)

wdf = pd.read_parquet(os.path.join(FS, f"{a.model}_alllayer_wfs.parquet"))
bundle = SAEBundle.load(os.path.join(DATA, "saes", a.model, "seed0"))
battery = load_battery("data/concepts_pilot")
byname = {c.name: c for c in battery}
if a.knowledge not in byname:
    raise SystemExit(f"unknown knowledge {a.knowledge!r}; have {list(byname)}")

model, tok = load_model_and_tokenizer(MPATH, dtype="bfloat16", device="cuda")
nodes_df, edges_df = trace_pathway_chain(
    model, tok, bundle, wdf, byname[a.knowledge],
    k_feat=a.kfeat, k_src=ksrc, n_cap=a.ncap, max_len=256, batch_size=16, dtype="bfloat16")

npath = os.path.join(OUT, f"{a.model}_{a.knowledge}_chain_nodes.parquet")
epath = os.path.join(OUT, f"{a.model}_{a.knowledge}_chain_edges.parquet")
nodes_df.to_parquet(npath, index=False)
edges_df.to_parquet(epath, index=False)

log.info("nodes=%d edges=%d", len(nodes_df), len(edges_df))
print(f"saved {npath}\n      {epath}")
print(f"nodes: {len(nodes_df)} ({len(nodes_df.layer.unique())} layers x {a.ksrc})")
print(f"edges: {len(edges_df)}  | top edges by delta:")
print(edges_df.nlargest(12, "delta")[["src_layer","src_neuron","dst_neuron","delta"]].to_string(index=False))
