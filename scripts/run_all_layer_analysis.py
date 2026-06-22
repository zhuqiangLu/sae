"""All-layer analysis for the 3 models: score the 5 pilot knowledge at EVERY
layer (full AUROC depth curve) + per-knowledge cross-layer routing (full L x L
detector correlation matrix + coherence).

Memory-bounded: encodes one layer at a time, keeps only the small per-knowledge
detector activation vectors for the routing correlations.

Run: PYTHONPATH=src python3 scripts/run_all_layer_analysis.py
"""
from __future__ import annotations
import os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.capture import ActivationReader
from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.score import score_features

DATA = "/share1/zhlu6105/know_trans_data"
BATTERY = "data/concepts_pilot"
PROBE = 40000
log = get_logger("all_layer_analysis")
ensure_dir(os.path.join(DATA, "feature_scores"))
ensure_dir(os.path.join(DATA, "matches"))

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]
battery = load_battery(BATTERY)
names = [c.name for c in battery]

def zc(v):  # z-score a 1-D vector
    return (v - v.mean()) / v.std().clamp_min(1e-6)

for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    layers = sorted(int(l) for l in bundle.layers)

    # ---- score every layer ----
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    df = score_features(model, tok, bundle, battery, layers,
                        os.path.join(DATA, "feature_scores", f"{mname}_alllayer.parquet"),
                        max_len=256, batch_size=32)
    del model; torch.cuda.empty_cache()

    # top detector per (knowledge, layer) + AUC trajectory
    top, traj = {}, {c: {} for c in names}
    for c in names:
        for L in layers:
            sub = df[(df["concept"] == c) & (df["layer"] == L)]
            if len(sub):
                r = sub.loc[sub["auc"].idxmax()]
                top[(c, L)] = int(r["feature"]); traj[c][L] = round(float(r["auc"]), 3)

    # ---- routing: encode one layer at a time, keep only detector vectors ----
    reader = ActivationReader(os.path.join(DATA, "activations", mname))
    g = torch.Generator().manual_seed(0)
    samp = torch.randperm(reader.n_tokens, generator=g)[:PROBE]
    vecs = {c: {} for c in names}
    for L in layers:
        acts, _ = reader.read(L)
        xb = acts[samp].to("cuda", dtype=torch.float32)
        codes = bundle[L].to("cuda").encode_dense(xb)  # [PROBE, H]
        for c in names:
            if (c, L) in top:
                vecs[c][L] = zc(codes[:, top[(c, L)]]).detach()
        del codes, acts, xb; torch.cuda.empty_cache()

    routing = {}
    for c in names:
        Ls = [L for L in layers if L in vecs[c]]
        mat = {int(Li): {int(Lj): round(float((vecs[c][Li] * vecs[c][Lj]).mean()), 3)
                         for Lj in Ls} for Li in Ls}
        off = [mat[Li][Lj] for Li in Ls for Lj in Ls if Li != Lj]
        routing[c] = {"coherence": round(sum(off) / len(off), 3) if off else None,
                      "matrix": mat}
    save_json({"model": mname, "layers": layers, "auc_trajectory": traj, "routing": routing},
              os.path.join(DATA, "matches", f"routing_alllayer_{mname}.json"))
    log.info("[%s] saved all-layer routing", mname)

    print(f"\n#### {mname}: routing coherence (full-depth) ####")
    for c in names:
        print(f"  {c:16s} coherence={routing[c]['coherence']}  "
              f"AUC peak={max(traj[c].values()):.3f}@L{max(traj[c], key=traj[c].get)}")
log.info("ALL-LAYER ANALYSIS DONE")
print("\nsaved -> feature_scores/*_alllayer.parquet, matches/routing_alllayer_*.json")
