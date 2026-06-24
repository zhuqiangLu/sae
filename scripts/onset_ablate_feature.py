"""onset_ablate_feature.py — FAITHFUL feature ablation (method B) at the onset.

Same setup as onset_ablate.py (channel zeroing, method A) but instead of zeroing
the dense source channels, we subtract the selected SAE features' reconstruction
from the onset MLP output:

    x' = x - Σ_{j ∈ feats} z_j(x) · W_dec[:, j]          (FeatureAblateHook)

The features = the SAME top-k_feat ΔWFS features the localization used (recomputed
from the cached valloc WFS table), so this ablates exactly the "concept direction"
the pipeline found, scaled per token by how hard it fires — leaving b_pre and all
other features intact.

Conditions: clean / ablate (econ feats) / ablate_random (k_feat random features,
same count). Headline: target_drop large & general_drop ~0 == econ-selective.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.onset_ablate_feature \
        --src-json data/pathways_subject/Llama-3.1-8B_econ_onset_src.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, save_json, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.cspt import FeatureAblateHook
from know_trans.capture import _get_hook_module
from scripts.route_eval_selectivity import (
    MODEL, MODEL_PATH, DOMAIN_TARGET, GENERAL_CATS, LETTERS,
    load_mmlu_pro, _format_q, build_fewshot_prefix, accuracy,
)

log = get_logger("onset_ablate_feature")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=675, help="cap on test items")
    ap.add_argument("--n-general", type=int, default=200, help="per general cat")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bias-thresh", type=float, default=1.0,
                    help="protect (keep) channels with |b_pre| above this in the subtraction")
    ap.add_argument("--feats-json", default=None,
                    help="override: JSON {'feats':[...]} to subtract instead of top-ΔWFS")
    ap.add_argument("--tag", default="", help="suffix for the output json/log")
    ap.add_argument("--out-dir", default="report/diag")
    a = ap.parse_args()

    src = json.load(open(a.src_json))
    domain = src["domain"]
    L = int(src["onset_layer"])
    k_feat = int(src.get("k_feat", 32))
    concept_name, target_cat = DOMAIN_TARGET[domain]
    sae_dir = src["sae_dir"]
    split = json.load(open(src["split_path"]))
    test_idx, val_idx = split["test_idx"], split["val_idx"]
    out_dir = ensure_dir(a.out_dir)
    general_cats = GENERAL_CATS[:4]

    # ---- recover the localization features (top-k_feat ΔWFS at l*) -----------
    wfs_path = os.path.join("data/feature_scores_subject",
                            f"{MODEL}_{domain}_valloc_wfs.parquet")
    wdf = pd.read_parquet(wfs_path)
    if a.feats_json:
        feats = np.array(sorted(json.load(open(a.feats_json))["feats"]), dtype=np.int64)
        log.info("[%s] L%d feats(OVERRIDE %s, n=%d)=%s", domain, L, a.feats_json, len(feats), feats.tolist())
    else:
        w = (wdf[(wdf.concept == concept_name) & (wdf.layer == L)]
             .sort_values("delta_wfs", ascending=False))
        feats = w.feature.to_numpy()[:k_feat].astype(np.int64)
        log.info("[%s] L%d feats(top-%d ΔWFS)=%s", domain, L, k_feat, feats.tolist())

    # ---- data ---------------------------------------------------------------
    econ_rows, _ = load_mmlu_pro(target_cat)
    tgt_test = [econ_rows[i] for i in test_idx][:a.n_target]
    tgt_fewshot = [econ_rows[i] for i in val_idx]                     # disjoint
    gen_rows = {c: load_mmlu_pro(c)[0] for c in general_cats}
    log.info("[%s] L%d ablate %d feats | target test=%d general=%s",
             domain, L, len(feats), len(tgt_test), general_cats)

    # ---- model + SAE --------------------------------------------------------
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = torch.tensor(
        [tok.encode(" " + ch, add_special_tokens=False)[-1] for ch in LETTERS],
        device=model.device)
    bundle = SAEBundle.load(sae_dir, device="cuda")
    sae = bundle[L]
    d_hidden = sae.d_hidden

    tgt_prefix = build_fewshot_prefix(tgt_fewshot, a.nshot)
    gen_prefix = {c: build_fewshot_prefix(gen_rows[c][-(a.nshot + 2):], a.nshot)
                  for c in general_cats}
    gen_eval = {c: gen_rows[c][:a.n_general] for c in general_cats}

    # bias / massive-activation channels to PROTECT (keep) in the subtraction
    b_pre = sae.b_pre.detach().float().cpu().numpy()
    protect = np.where(np.abs(b_pre) > a.bias_thresh)[0].astype(np.int64)
    log.info("[%s] protect %d bias channels (|b_pre|>%.1f): %s",
             domain, len(protect), a.bias_thresh, protect.tolist())

    hook = FeatureAblateHook(_get_hook_module(model, L, "mlp"), sae,
                             np.array([feats[0]], dtype=np.int64))

    def run():
        t, _ = accuracy(model, tok, tgt_test, tgt_prefix, letter_ids, batch=a.batch_size)
        gs = [accuracy(model, tok, gen_eval[c], gen_prefix[c], letter_ids,
                       batch=a.batch_size)[0] for c in general_cats]
        return t, float(np.nanmean(gs))

    # (feats, protect): None=clean; plain subtract; subtract while keeping bias chans
    conds = {"clean": None,
             "ablate": (feats, None),
             "ablate_protect": (feats, protect)}
    res = {"domain": domain, "onset_layer": L, "method": "feature_subtract",
           "feats": feats.tolist(), "k_feat": len(feats), "protect": protect.tolist(),
           "target_category": target_cat, "general": general_cats,
           "n_target": len(tgt_test), "nshot": a.nshot, "src_json": a.src_json,
           "target_acc": {}, "general_acc": {}}
    for name, spec in conds.items():
        if spec is None:
            hook.enabled = False
        else:
            f, prot = spec
            hook.set_feats(f); hook.set_protect(prot); hook.enabled = True
        t, g = run()
        res["target_acc"][name] = t
        res["general_acc"][name] = g
        log.info("[%s L%d] %-14s target=%.3f general=%.3f", domain, L, name, t, g)
    hook.remove()

    ct, cg = res["target_acc"]["clean"], res["general_acc"]["clean"]
    res["headline"] = {
        "target_drop": round(ct - res["target_acc"]["ablate"], 4),
        "general_drop": round(cg - res["general_acc"]["ablate"], 4),
        "target_drop_protect": round(ct - res["target_acc"]["ablate_protect"], 4),
        "general_drop_protect": round(cg - res["general_acc"]["ablate_protect"], 4),
        "clean_target": ct,
    }
    out = os.path.join(out_dir, f"onset_ablate_feature_{domain}_L{L}{a.tag}.json")
    save_json(res, out)

    print("\n" + "=" * 64)
    print(f"FEATURE SUBTRACT (method B)  domain={domain} L{L}  k_feat={len(feats)}")
    print(f"target=econ TEST (n={len(tgt_test)})  general={general_cats}  chance=0.10")
    print(f"protect {len(protect)} bias chans (|b_pre|>{a.bias_thresh}): {protect.tolist()}")
    print("=" * 64)
    print(f"{'condition':16s}{'target':>10s}{'general':>10s}")
    for name in conds:
        print(f"{name:16s}{res['target_acc'][name]:>10.3f}{res['general_acc'][name]:>10.3f}")
    h = res["headline"]
    print("-" * 64)
    print(f"plain  : target_drop={h['target_drop']:+.3f}  general_drop={h['general_drop']:+.3f}")
    print(f"protect: target_drop={h['target_drop_protect']:+.3f}  general_drop={h['general_drop_protect']:+.3f}")
    print("  (want: target_drop LARGE, general_drop ~0)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
