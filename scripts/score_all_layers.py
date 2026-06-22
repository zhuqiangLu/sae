"""All-layer AUROC scoring for the 3 pilot models (routing-free).

Produces feature_scores/{model}_alllayer.parquet = per-(layer, feature, concept)
detector AUROC, for the CURRENT data/concepts_pilot battery. SAEs are reused
(corpus-trained, not battery-dependent); only the scoring re-runs.

Run: PYTHONPATH=src python3 scripts/score_all_layers.py
"""
from __future__ import annotations
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.score import score_features

DATA = "/share1/zhlu6105/know_trans_data"
log = get_logger("score_all")
ensure_dir(os.path.join(DATA, "feature_scores"))

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]
battery = load_battery("data/concepts_pilot")

for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    layers = sorted(int(l) for l in bundle.layers)
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    score_features(model, tok, bundle, battery, layers,
                   os.path.join(DATA, "feature_scores", f"{mname}_alllayer.parquet"),
                   max_len=256, batch_size=32)
    del model; torch.cuda.empty_cache()
log.info("ALL-LAYER AUROC DONE")
