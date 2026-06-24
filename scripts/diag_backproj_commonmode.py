"""diag_backproj_commonmode.py — is the back-projection onto {788,1384,4062} an
econ signal, or a common-mode (sink) artifact of the unweighted decoder sum?

User's objection: ΔWFS is a contrast, so sink FEATURES (fire equally on econ &
general) get ΔWFS≈0 and aren't selected. True. But the CHANNELS come from the
unweighted SUM of the selected features' decoder columns. A sum amplifies the
direction the columns SHARE and cancels idiosyncratic parts. Test: do RANDOM
feature sets back-project to the same channels? If yes -> common-mode artifact,
independent of the econ contrast.

CPU only (no model forward). Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES= python -m scripts.diag_backproj_commonmode
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from know_trans.sae import TopKSAE

SAE_PATH = "data/saes_subject/Llama-3.1-8B/econ/layer1.safetensors"
WFS_PATH = "data/feature_scores_subject/Llama-3.1-8B_econ_valloc_wfs.parquet"
CONCEPT = "topic_economics"
LAYER = 1
CHANS = [788, 1384, 4062]


def main() -> None:
    sae = TopKSAE.load(SAE_PATH, device="cpu")
    W = sae.W_dec.detach().float()                       # [d_in, H], unit-norm cols
    b_pre = sae.b_pre.detach().float().numpy()           # [d_in]
    d_in, H = W.shape
    print(f"W_dec {tuple(W.shape)}  (unit-norm cols: mean |col|={W.norm(dim=0).mean():.3f})")

    wdf = pd.read_parquet(WFS_PATH)
    w = (wdf[(wdf.concept == CONCEPT) & (wdf.layer == LAYER)]
         .sort_values("delta_wfs", ascending=False))
    econ_feats = w.feature.to_numpy()[:32].astype(np.int64)

    z_econ = W[:, econ_feats].sum(1).numpy()             # the localization vector
    print("\n--- ECON feats (top-32 ΔWFS) back-projection at the located channels ---")
    for c in CHANS:
        col = W[c, econ_feats].numpy()                   # 32 per-feature components
        cm_ratio = abs(col.sum()) / (np.abs(col).sum() + 1e-9)   # 1.0 == all same sign
        same = float(np.mean(np.sign(col) == np.sign(col.sum())))
        print(f"  ch{c:5d}: z_proj={z_econ[c]:+.3f}  per-feat mean={col.mean():+.4f} "
              f"std={col.std():.4f}  common-mode={cm_ratio:.2f}  same-sign frac={same:.2f}")

    # ---- RANDOM feature sets: do they hit the SAME channels? -----------------
    rng = np.random.default_rng(0)
    n_draw = 200
    rand_z = np.zeros((n_draw, len(CHANS)))
    rand_top = []                                        # top-3 |z| channels per draw
    for i in range(n_draw):
        rf = rng.choice(H, size=32, replace=False)
        z = W[:, rf].sum(1).numpy()
        rand_z[i] = [z[c] for c in CHANS]
        rand_top.append(set(np.argsort(np.abs(z))[::-1][:3].tolist()))
    print("\n--- RANDOM 32-feature sets (200 draws): z_proj at the SAME channels ---")
    for j, c in enumerate(CHANS):
        col = rand_z[:, j]
        print(f"  ch{c:5d}: random z_proj mean={col.mean():+.3f} std={col.std():.3f} "
              f"|range|=[{col.min():+.2f},{col.max():+.2f}]   (econ={z_econ[c]:+.3f})")
    hit = np.mean([len({788, 1384, 4062} & s) for s in rand_top])
    frac_all3 = np.mean([{788, 1384, 4062}.issubset(s) for s in rand_top])
    print(f"  random draws whose top-3 |z| channels intersect {{788,1384,4062}}: "
          f"avg {hit:.2f}/3 ;  all-3 match in {frac_all3*100:.0f}% of draws")

    # ---- global common-mode: mean decoder column over ALL features -----------
    gmean = W.mean(1).numpy()                            # [d_in]
    gtop = np.argsort(np.abs(gmean))[::-1][:8]
    print("\n--- global mean decoder column (common mode over ALL 16384 feats) ---")
    print(f"  top-8 |mean| channels: {gtop.tolist()}")
    print(f"  |mean| at those       : {np.round(np.abs(gmean[gtop]), 4).tolist()}")
    print(f"  mean at {CHANS}: {np.round(gmean[CHANS], 4).tolist()}  "
          f"(median |mean| over all ch = {np.median(np.abs(gmean)):.4f})")

    # ---- b_pre (data mean) at these channels ---------------------------------
    print("\n--- b_pre (≈ activation data mean) ---")
    print(f"  b_pre at {CHANS}: {np.round(b_pre[CHANS], 2).tolist()}")
    print(f"  |b_pre| median over all ch = {np.median(np.abs(b_pre)):.3f}; "
          f"max = {np.abs(b_pre).max():.1f} at ch {int(np.abs(b_pre).argmax())}")
    btop = np.argsort(np.abs(b_pre))[::-1][:8]
    print(f"  top-8 |b_pre| channels: {btop.tolist()}  vals={np.round(b_pre[btop],1).tolist()}")

    # ---- where does ECON's idiosyncratic (contrast-surviving) signal point? --
    # subtract the global common mode from each econ column, then sum.
    z_econ_decm = (W[:, econ_feats] - gmean[:, None]).sum(1).numpy()
    top_decm = np.argsort(np.abs(z_econ_decm))[::-1][:8]
    print("\n--- econ back-projection AFTER removing global common mode ---")
    print(f"  top-8 |z| channels: {top_decm.tolist()}")
    print(f"  (vs raw econ top channels {np.argsort(np.abs(z_econ))[::-1][:8].tolist()})")


if __name__ == "__main__":
    main()
