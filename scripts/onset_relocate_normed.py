"""onset_relocate_normed.py — re-locate the onset SOURCE channels with a SCALE-
NORMALIZED back-projection, to stop the high-bias / massive-activation channels
from automatically winning the raw unweighted sum.

Diagnosis (diag_backproj_commonmode): raw z_proj = Σ_{j∈feats} W_dec[:,j] ranks
channels ≈ by |b_pre| (activation-mean magnitude). The top-3 (788/1384/4062) are
the model's bias / massive-activation channels (b_pre ≈ ±5 vs median 0.015), not
econ — every high-firing econ feature carries a same-sign loading on them, so they
dominate the sum in RAW units. Fix two ways and compare:

  * NORM : pick top |z_proj[c]| / std_c   (push relative to the channel's own
           natural variation — bias/massive channels have huge std, so they sink)
  * MASK : drop channels with |b_pre| > thresh, then pick top |z_proj| as before.

Writes ..._onset_src_normed.json / ..._onset_src_masked.json (same schema as
onset_locate) so onset_ablate.py can ablate the new channels directly.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m scripts.onset_relocate_normed --domain econ
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, batched
from know_trans.sae import SAEBundle
from know_trans.capture import MLPHook, _get_hook_module
from scripts.route_eval_selectivity import (
    MODEL, MODEL_PATH, DOMAIN_TARGET, GENERAL_CATS, load_mmlu_pro, _format_q,
)

log = get_logger("relocate_normed")
OUT_DIR = "data/pathways_subject"
CHANS = [788, 1384, 4062]


def max_gap_k(zs_desc: np.ndarray, head: int = 64) -> int:
    h = min(head, len(zs_desc) - 1)
    gaps = zs_desc[:h] - zs_desc[1:h + 1]
    return int(np.argmax(gaps)) + 1


@torch.no_grad()
def channel_stats(model, tok, texts, L, max_len=384, batch_size=16):
    """Per-channel mean/std of the layer-L MLP output over a battery of texts."""
    hook = MLPHook(_get_hook_module(model, L, "mlp"), L, to_cpu=False)
    n = 0
    s = s2 = None
    for tb in batched(list(texts), batch_size):
        enc = tok(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        attn = enc.get("attention_mask")
        model(**enc)
        a = hook.pop().float()                                   # [B,S,d]
        bb, ss, d = a.shape
        m = attn.bool().reshape(-1) if attn is not None else torch.ones(bb * ss, dtype=torch.bool, device=a.device)
        af = a.reshape(bb * ss, d)[m]                            # [T,d]
        s = af.sum(0) if s is None else s + af.sum(0)
        s2 = (af * af).sum(0) if s2 is None else s2 + (af * af).sum(0)
        n += af.shape[0]
        torch.cuda.empty_cache()
    hook.remove()
    mean = (s / n).cpu().numpy()
    std = np.sqrt(np.maximum((s2 / n).cpu().numpy() - mean ** 2, 1e-12))
    return mean, std, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["econ", "math", "med"], default="econ")
    ap.add_argument("--k-feat", type=int, default=32)
    ap.add_argument("--bias-thresh", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=200)
    a = ap.parse_args()

    concept_name, target_cat = DOMAIN_TARGET[a.domain]
    src0 = json.load(open(os.path.join(OUT_DIR, f"{MODEL}_{a.domain}_onset_src.json")))
    L = int(src0["onset_layer"]); sae_dir = src0["sae_dir"]; split_path = src0["split_path"]
    val_idx = json.load(open(split_path))["val_idx"]
    ensure_dir(OUT_DIR)

    # ---- localization features (top-k_feat ΔWFS at l*) ----------------------
    wdf = pd.read_parquet(f"data/feature_scores_subject/{MODEL}_{a.domain}_valloc_wfs.parquet")
    w = (wdf[(wdf.concept == concept_name) & (wdf.layer == L)]
         .sort_values("delta_wfs", ascending=False))
    feats = w.feature.to_numpy()[:a.k_feat].astype(np.int64)

    bundle = SAEBundle.load(sae_dir, device="cuda")
    sae = bundle[L]
    W_dec = sae.W_dec.detach().float().cpu().numpy()             # [d_in, H]
    b_pre = sae.b_pre.detach().float().cpu().numpy()             # [d_in]
    z_raw = W_dec[:, feats].sum(1)                               # [d_in]
    zabs = np.abs(z_raw)

    # ---- battery for per-channel std ----------------------------------------
    econ_rows, _ = load_mmlu_pro(target_cat)
    pos = [_format_q(q, o) for (q, o, _g) in (econ_rows[i] for i in val_idx)][:a.cap]
    neg = []
    for c in GENERAL_CATS:
        neg += [_format_q(q, o) for (q, o, _g) in load_mmlu_pro(c)[0]]
    rng = np.random.default_rng(0); rng.shuffle(neg); neg = neg[:a.cap]

    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mean, std, ntok = channel_stats(model, tok, pos + neg, L)
    z_norm = zabs / (std + 1e-6)

    # ---- report + pick ------------------------------------------------------
    def pick(name, score, mask=None):
        sc = score.astype(np.float64).copy()
        if mask is not None:
            sc[mask] = -np.inf
        order = np.argsort(sc)[::-1]
        k = max_gap_k(sc[order])
        src = sorted(int(c) for c in order[:k])
        log.info("[%s] %-14s top8=%s k=%d src=%s", a.domain, name, order[:8].tolist(), k, src)
        return src, order[:8].tolist()

    print("=" * 72)
    print(f"RE-LOCATE {a.domain} source @ L{L}  (k_feat={len(feats)}, std over {ntok} tokens)")
    print("=" * 72)
    print(f"b_pre @{CHANS} = {np.round(b_pre[CHANS],2).tolist()}  (median |b_pre|={np.median(np.abs(b_pre)):.3f})")
    print(f"std   @{CHANS} = {np.round(std[CHANS],2).tolist()}  (median std={np.median(std):.3f})")
    print(f"raw |z|@{CHANS} = {np.round(zabs[CHANS],2).tolist()}")
    print(f"z/std  @{CHANS} = {np.round(z_norm[CHANS],3).tolist()}")
    print("-" * 72)
    src_raw, top_raw = pick("RAW |z|", zabs)
    src_norm, top_norm = pick("NORM z/std", z_norm)
    bias_mask = np.abs(b_pre) > a.bias_thresh
    src_mask, top_mask = pick(f"MASK|bpre|>{a.bias_thresh}", zabs, mask=bias_mask)
    print("-" * 72)
    print(f"RAW   src = {src_raw}")
    print(f"NORM  src = {src_norm}")
    print(f"MASK  src = {src_mask}   (masked {int(bias_mask.sum())} bias chans: "
          f"{sorted(np.where(bias_mask)[0].tolist())})")

    for tag, src in [("normed", src_norm), ("masked", src_mask)]:
        out = os.path.join(OUT_DIR, f"{MODEL}_{a.domain}_onset_src_{tag}.json")
        json.dump({"domain": a.domain, "concept": concept_name, "onset_layer": L,
                   "k_feat": len(feats), "method": tag, "source_neurons": src,
                   "sae_dir": sae_dir, "split_path": split_path}, open(out, "w"), indent=2)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
