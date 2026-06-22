"""Index-free stability: match teacher SAE features ACROSS seeds by activation
PATTERN (max Pearson correlation over a probe set), not by index identity.

For each knowledge item, take its seed0 detector features and, for each, find the
best-correlated feature in seed1. Stability = mean of those max correlations.
High => the same pattern re-emerges under a new SAE seed (genuinely stable).

Uses stored activations + the two SAEs only (no model rerun).
Run: PYTHONPATH=src python3 scripts/stability_by_pattern.py
"""
from __future__ import annotations
import os
import pandas as pd
import torch

from know_trans.config import load_config
from know_trans.capture import ActivationReader
from know_trans.sae import SAEBundle
from know_trans.score import concept_feature_sets
from know_trans.utils import get_logger, save_json

CFG = "configs/pair_llama8b_qwen0p6b_pilot.yaml"
N = 40000          # probe tokens
AUC_TH = 0.8
DEV = "cuda"
log = get_logger("stab_pattern")

cfg = load_config(CFG)
data = cfg.paths.data
tname = cfg.teacher.name

reader = ActivationReader(os.path.join(data, "activations", tname))
b0 = SAEBundle.load(os.path.join(data, "saes", tname, "seed0"))
b1 = SAEBundle.load(os.path.join(data, "saes", tname, "seed1"))
df0 = pd.read_parquet(os.path.join(data, "feature_scores", f"{tname}_seed0.parquet"))
sets0 = concept_feature_sets(df0, auc_threshold=AUC_TH, top_k=10)

# Shared probe token indices (same rows for both seeds).
g = torch.Generator().manual_seed(0)
samp = torch.randperm(reader.n_tokens, generator=g)[:N]

def zscore(x: torch.Tensor) -> torch.Tensor:
    m = x.mean(0, keepdim=True)
    s = x.std(0, keepdim=True).clamp_min(1e-6)
    return (x - m) / s

def encode_layer(bundle, layer):
    acts, _ = reader.read(layer)
    xb = acts[samp].to(DEV, dtype=torch.float32)
    sae = bundle[layer].to(DEV)
    with torch.no_grad():
        return sae.encode_dense(xb)   # [N, d_hidden]

# Group concepts by their (single) detector layer to encode each layer once.
by_layer: dict[int, list[str]] = {}
for c, feats in sets0.items():
    if feats:
        by_layer.setdefault(int(feats[0]["layer"]), []).append(c)

stab_pattern: dict[str, float] = {}
detail: dict[str, list[float]] = {}
for layer, concepts in sorted(by_layer.items()):
    log.info("encoding layer %d (%d concepts) ...", layer, len(concepts))
    c0 = encode_layer(b0, layer)
    c1 = encode_layer(b1, layer)
    z1 = zscore(c1)                         # [N, D]
    for cname in concepts:
        idxs = [int(f["feature"]) for f in sets0[cname]]
        z0 = zscore(c0[:, idxs])            # [N, k]
        corr = (z0.t() @ z1) / c0.shape[0]  # [k, D] Pearson
        maxc = corr.max(dim=1).values       # best seed1 match per detector
        detail[cname] = [round(float(v), 3) for v in maxc.tolist()]
        stab_pattern[cname] = round(float(maxc.mean()), 3)
    del c0, c1, z1
    torch.cuda.empty_cache()

# Concepts with no seed0 detector
for c, feats in sets0.items():
    if not feats:
        stab_pattern[c] = None

vals = [v for v in stab_pattern.values() if v is not None]
mean = round(sum(vals) / len(vals), 3) if vals else float("nan")
save_json({"stability_pattern": stab_pattern, "per_feature_maxcorr": detail,
           "mean": mean, "n_probe": N}, os.path.join(data, "matches", "pilot_stability_pattern.json"))

print("\n=== PATTERN-BASED CROSS-SEED STABILITY (teacher) ===")
print("(max Pearson corr of each seed0 detector to ANY seed1 feature, over %d tokens)\n" % N)
print(f"{'knowledge':16s} {'#det':5s} {'pattern-stab':12s} {'per-detector max-corr'}")
print("-" * 70)
for c in ["safety", "topic_math", "topic_economics", "topic_medical", "language_hu"]:
    sp = stab_pattern.get(c)
    nd = len(sets0.get(c) or [])
    print(f"{c:16s} {nd:<5d} {str(sp):12s} {detail.get(c, '')}")
print(f"\nmean pattern-stability: {mean}")
print("(compare: old index-Jaccard was ~0.0 by construction)")
