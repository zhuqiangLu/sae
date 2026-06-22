"""All-layer token-level WFS for Llama-3.1-8B-Instruct only (Stage 3).

Mirrors scripts/run_wfs.py but restricted to the single Instruct model, loading
its SAEBundle from saes/Llama-3.1-8B-Instruct/seed0 and writing the WFS parquet
to feature_scores/Llama-3.1-8B-Instruct_alllayer_wfs.parquet.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 scripts/run_wfs_instruct.py
"""
from __future__ import annotations
import argparse, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.wfs import wfs_score_features

DATA = "/share1/zhlu6105/know_trans_data"
BATTERY = "data/concepts_pilot"
MODEL_NAME = "Llama-3.1-8B-Instruct"
MODEL_PATH = "/share1/zhlu6105/models/Llama-3.1-8B-Instruct"
log = get_logger("run_wfs_instruct")
ensure_dir(os.path.join(DATA, "feature_scores"))

ap = argparse.ArgumentParser()
ap.add_argument("--skip-existing", action="store_true")
a = ap.parse_args()

battery = load_battery(BATTERY)
out = os.path.join(DATA, "feature_scores", f"{MODEL_NAME}_alllayer_wfs.parquet")
log.info("==== %s ====", MODEL_NAME)
bundle = SAEBundle.load(os.path.join(DATA, "saes", MODEL_NAME, "seed0"))
layers = sorted(int(l) for l in bundle.layers)
log.info("SAE bundle layers: %s", layers)

if a.skip_existing and os.path.exists(out):
    log.info("[%s] WFS parquet exists, skipping compute", MODEL_NAME)
else:
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    wfs_score_features(model, tok, bundle, battery, layers, out,
                       max_len=256, batch_size=16, dtype="bfloat16")
    del model
    torch.cuda.empty_cache()

import pandas as pd
df = pd.read_parquet(out)
print(f"WFS DONE rows={len(df)} layers={df.layer.nunique()} concepts={sorted(df.concept.unique())}")
print(f"safety present: {'safety' in set(df.concept.unique())}")
sub = df[df.concept == 'safety']
print(f"safety rows={len(sub)} delta_wfs range=[{sub.delta_wfs.min():.4f},{sub.delta_wfs.max():.4f}]")
