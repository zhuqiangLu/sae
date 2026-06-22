"""All-layer token-level WFS (TraceRouter §3.1) for the 3 pilot models, plus a
cross-check of the ΔWFS Top-K sensitive set against the detector-AUROC Top-K.

WFS is the COMPLEMENT to AUROC (not a replacement): AUROC stays the primary
anchor; ΔWFS rides alongside and its magnitude feeds distillation weighting.

Run: PYTHONPATH=src python3 scripts/run_wfs.py [--topk 10]
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.wfs import wfs_score_features, wfs_feature_sets

DATA = "/share1/zhlu6105/know_trans_data"
BATTERY = "data/concepts_pilot"
log = get_logger("run_wfs")
ensure_dir(os.path.join(DATA, "feature_scores"))

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--skip-existing", action="store_true")
a = ap.parse_args()

battery = load_battery(BATTERY)
names = [c.name for c in battery]

for mname, mpath in MODELS:
    out = os.path.join(DATA, "feature_scores", f"{mname}_alllayer_wfs.parquet")
    log.info("==== %s ====", mname)
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    layers = sorted(int(l) for l in bundle.layers)
    if a.skip_existing and os.path.exists(out):
        log.info("[%s] WFS parquet exists, skipping compute", mname)
    else:
        model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
        wfs_score_features(model, tok, bundle, battery, layers, out,
                           max_len=256, batch_size=16, dtype="bfloat16")
        del model; torch.cuda.empty_cache()

log.info("ALL WFS COMPUTED")

# ---- cross-check: ΔWFS Top-K vs AUROC Top-K (per knowledge, best layer each) ----
print("\n" + "=" * 78)
print(f"WFS vs AUROC cross-check  (Top-{a.topk} feature sets per knowledge)")
print("=" * 78)
for mname, _ in MODELS:
    wdf = pd.read_parquet(os.path.join(DATA, "feature_scores", f"{mname}_alllayer_wfs.parquet"))
    adf = pd.read_parquet(os.path.join(DATA, "feature_scores", f"{mname}_alllayer.parquet"))
    wsets = wfs_feature_sets(wdf, top_k=a.topk)
    # AUROC top-k per concept at its best (peak-AUC) layer, mirroring score.concept_feature_sets
    print(f"\n#### {mname} ####")
    print(f"{'knowledge':16s} {'WFS layer/peakΔ':22s} {'AUC layer/peak':18s} {'overlap@layer':14s} {'Jaccard':7s}")
    for c in names:
        # AUROC best layer + topk feats
        ca = adf[adf["concept"] == c]
        bestL_a = int(ca.loc[ca["auc"].idxmax(), "layer"])
        a_top = ca[ca["layer"] == bestL_a].nlargest(a.topk, "auc")
        a_feats = set(int(x) for x in a_top["feature"])
        a_peak = float(ca["auc"].max())
        # WFS set
        ws = wsets.get(c, [])
        if ws:
            bestL_w = ws[0]["layer"]; w_peak = ws[0]["delta_wfs"]
            w_feats = set(int(x["feature"]) for x in ws)
        else:
            bestL_w, w_peak, w_feats = None, float("nan"), set()
        # overlap if both picked the SAME layer; else recompute WFS topk AT the AUROC layer
        w_at_aucL = set(int(x) for x in
                        wdf[wdf["concept"] == c][wdf["layer"] == bestL_a].nlargest(a.topk, "delta_wfs")["feature"]) \
            if len(wdf[(wdf["concept"] == c) & (wdf["layer"] == bestL_a)]) else set()
        inter = len(a_feats & w_at_aucL)
        union = len(a_feats | w_at_aucL)
        jac = inter / union if union else 0.0
        print(f"{c:16s} L{str(bestL_w):>3}/Δ={w_peak:7.3f}      "
              f"L{bestL_a:>2}/{a_peak:.3f}      {inter:2d}/{a.topk} (@L{bestL_a})    {jac:.2f}")
print("\nsaved -> feature_scores/*_alllayer_wfs.parquet")
