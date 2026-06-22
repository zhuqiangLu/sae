"""onset_locate.py — locate the onset SOURCE neuron(s) from a held-out VAL split.

Cleaner ablation setup (per user):
  1. Split MMLU-Pro <domain> (the eval set) into val:test = 2:8 (seeded).
  2. Locate the onset source neurons using ONLY the val split, so the test split
     stays untouched for the later ablation measurement.

Localization = the (now-fixed) SAE pipeline, ONSET layer only:
  - positives = MMLU-Pro <domain> VAL questions; negatives = MMLU-Pro general
    categories (contrast for ΔWFS / detector AUROC).
  - AUROC over SAE features -> onset layer l* (earliest >= tau, else global best).
  - ΔWFS -> top-k_feat features at l*; UNWEIGHTED decoder back-projection
    Z_proj = W_dec · m_sens (paper §3.2); source = top coords of |Z_proj| chosen
    by MAX-GAP (first sharp drop), NOT the chord-elbow (which can't see k=1).

Writes the split + located source to data/pathways_subject/ for the ablation step.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m scripts.onset_locate --domain econ
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.score import score_features
from know_trans.wfs import wfs_score_features
from know_trans.concepts import Concept
from know_trans.cspt import pick_onset_layer
from scripts.route_eval_selectivity import (
    MODEL, MODEL_PATH, DOMAIN_TARGET, GENERAL_CATS, load_mmlu_pro, _format_q,
)

log = get_logger("onset_locate")
OUT_DIR = "data/pathways_subject"


def max_gap_k(zs_desc: np.ndarray, head: int = 64) -> int:
    """K = rank of the largest absolute drop in the descending curve (first elbow).
    Looks within the first `head` ranks so the long flat tail can't dominate."""
    h = min(head, len(zs_desc) - 1)
    gaps = zs_desc[:h] - zs_desc[1:h + 1]
    return int(np.argmax(gaps)) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["econ", "math", "med"], default="econ")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-feat", type=int, default=32, help="top ΔWFS features at l*")
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sae-dir", default=None)
    a = ap.parse_args()

    concept_name, target_cat = DOMAIN_TARGET[a.domain]
    sae_dir = a.sae_dir or f"data/saes_subject/{MODEL}/{a.domain}"
    ensure_dir(OUT_DIR)
    cache_dir = ensure_dir("data/feature_scores_subject")
    rng = np.random.default_rng(a.seed)

    # ---- 1. split MMLU-Pro <domain> test set into val / test -----------------
    econ_rows, _ = load_mmlu_pro(target_cat)          # list of (q, options, gold)
    n = len(econ_rows)
    perm = rng.permutation(n)
    n_val = int(round(a.val_frac * n))
    val_idx = sorted(int(i) for i in perm[:n_val])
    test_idx = sorted(int(i) for i in perm[n_val:])
    val_rows = [econ_rows[i] for i in val_idx]
    log.info("[%s] %s: %d total -> val=%d test=%d (frac=%.2f, seed=%d)",
             a.domain, target_cat, n, len(val_idx), len(test_idx), a.val_frac, a.seed)

    split_path = os.path.join(OUT_DIR, f"{a.domain}_valtest_split.json")
    json.dump({"domain": a.domain, "category": target_cat, "n": n,
               "val_frac": a.val_frac, "seed": a.seed,
               "val_idx": val_idx, "test_idx": test_idx}, open(split_path, "w"), indent=2)
    log.info("saved split -> %s", split_path)

    # ---- 2. build localization battery (val econ POS vs general NEG) ---------
    pos = [_format_q(q, o) for (q, o, _g) in val_rows]
    neg = []
    for c in GENERAL_CATS:
        rows = load_mmlu_pro(c)[0]
        neg += [_format_q(q, o) for (q, o, _g) in rows]
    rng.shuffle(neg)
    neg = neg[:len(pos)]                                # balance
    concept = Concept(name=concept_name, positives=pos, hard_negatives=neg,
                      source="mmlu_pro_val", group="topic")
    log.info("[%s] localization battery: pos=%d (econ val) neg=%d (general)",
             a.domain, len(pos), len(neg))

    # ---- 3. model + SAE; AUROC + WFS on the val battery ----------------------
    bundle = SAEBundle.load(sae_dir, device="cuda")
    layers = sorted(int(l) for l in bundle.layers)
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")

    adf = score_features(model, tok, bundle, [concept], layers,
                         os.path.join(cache_dir, f"{MODEL}_{a.domain}_valloc_auroc.parquet"),
                         max_len=a.max_len, batch_size=a.batch_size, dtype="bfloat16")
    wdf = wfs_score_features(model, tok, bundle, [concept], layers,
                             os.path.join(cache_dir, f"{MODEL}_{a.domain}_valloc_wfs.parquet"),
                             max_len=a.max_len, batch_size=a.batch_size, dtype="bfloat16")

    # ---- 4. onset layer ------------------------------------------------------
    l_star, peak = pick_onset_layer(adf, concept_name, tau_onset=a.tau)
    log.info("[%s] onset l*=%d (peak AUROC=%.3f, tau=%.2f)", a.domain, l_star, peak, a.tau)

    # ---- 5. source: ΔWFS feats -> UNWEIGHTED back-projection -> max-gap K -----
    w = (wdf[(wdf.concept == concept_name) & (wdf.layer == l_star)]
         .sort_values("delta_wfs", ascending=False))
    feats = w.feature.to_numpy()[:a.k_feat].astype(np.int64)
    W_dec = bundle[l_star].W_dec.detach().float().cpu().numpy()      # [d_in, H]
    z = np.abs(W_dec[:, feats].sum(axis=1))                          # unweighted W_dec·m_sens
    order = np.argsort(z)[::-1]
    zs = z[order]
    k = max_gap_k(zs)
    src = sorted(int(c) for c in order[:k])

    log.info("[%s] top-8 |z|: %s", a.domain, np.round(zs[:8], 3).tolist())
    log.info("[%s] max-gap K = %d  (|z| #1=%.3f, #2=%.3f, ratio=%.2f)",
             a.domain, k, zs[0], zs[1], zs[0] / max(zs[1], 1e-9))
    log.info("[%s] onset source neurons (layer %d): %s", a.domain, l_star, src)

    src_path = os.path.join(OUT_DIR, f"{MODEL}_{a.domain}_onset_src.json")
    json.dump({"domain": a.domain, "concept": concept_name, "onset_layer": int(l_star),
               "onset_peak_auroc": float(peak), "k_feat": int(a.k_feat),
               "max_gap_k": int(k), "source_neurons": src,
               "zproj_top16": [float(x) for x in zs[:16]],
               "sae_dir": sae_dir, "split_path": split_path}, open(src_path, "w"), indent=2)

    print("\n" + "=" * 64)
    print(f"ONSET SOURCE LOCATED FROM VAL  domain={a.domain} ({target_cat})")
    print(f"  onset layer l*      = {l_star}  (peak AUROC {peak:.3f})")
    print(f"  |z| top-8           = {np.round(zs[:8], 3).tolist()}")
    print(f"  max-gap K           = {k}")
    print(f"  source neuron(s)    = {src}")
    print(f"  saved -> {src_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
