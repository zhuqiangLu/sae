"""Downstream-K study: (1) the paper-faithful ELBOW cutoff per downstream layer,
and (2) a sweep of fixed k_down to test whether the SB/DR architecture labels are
stable to the (arbitrary) downstream-K choice.

Reuses the existing FIS parquets (AUROC onset); runs the suppression in LIGHT mode
(conditions clean/src/path/src+randD only — enough to classify architecture).

Run: PYTHONPATH=src python3 scripts/run_cspt_ksweep.py
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.cspt import pick_onset_layer, source_neurons, suppress_pathway_validation

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
PW = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("ksweep")

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]
KVALS = [5, 10, 25, 50, "elbow"]
battery = load_battery("data/concepts_pilot")
byname = {c.name: c for c in battery}
names = [c.name for c in battery]


def classify(clean, src, path, srd):
    if clean < 0.62: return "weak"
    sb = (clean - src) > 0.10
    dr = (srd - path) > 0.10
    return "SB" if (sb and not dr) else "DR" if (dr and not sb) else "SB+DR" if (sb and dr) else "none"


summary = {}
for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    adf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer.parquet"))
    wdf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer_wfs.parquet"))
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    layers = sorted(int(l) for l in bundle.layers)
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    summary[mname] = {}
    for C in names:
        fpath = os.path.join(PW, f"{mname}_{C}_fis.parquet")
        if not os.path.exists(fpath):
            summary[mname][C] = {"skipped": True}; continue
        fis_df = pd.read_parquet(fpath)
        lstar, _ = pick_onset_layer(adf, C, tau_onset=0.70)
        if lstar >= max(layers):
            summary[mname][C] = {"skipped": "no downstream"}; continue
        src, _, _ = source_neurons(bundle[lstar], wdf, C, lstar, k_feat=10, k_src=64)
        summary[mname][C] = {"onset": int(lstar), "k": {}}
        for k in KVALS:
            v = suppress_pathway_validation(model, tok, bundle, byname[C], lstar, src, fis_df, adf,
                                            k_down=k, light=True, max_len=256, batch_size=16, dtype="bfloat16")
            if not len(v):
                continue
            m = {c: round(float(v[f"auc_{c}"].mean()), 3) for c in ["clean", "src", "path", "src+randD"]}
            arch = classify(m["clean"], m["src"], m["path"], m["src+randD"])
            mk = round(float(v.attrs.get("mean_k", k if isinstance(k, int) else 0)), 1)
            summary[mname][C]["k"][str(k)] = {**m, "arch": arch, "mean_k": mk}
            log.info("[%s/%s] k=%-5s meanK=%-5.1f arch=%-5s clean=%.3f src=%.3f path=%.3f src+randD=%.3f",
                     mname, C, k, mk, arch, m["clean"], m["src"], m["path"], m["src+randD"])
    del model; torch.cuda.empty_cache()

save_json(summary, os.path.join(PW, "cspt_ksweep_summary.json"))

# ---- elbow result + stability tables ----
print("\n" + "=" * 96)
print("ELBOW (paper-faithful) per-downstream-layer k  +  k_down SWEEP architecture stability")
print("=" * 96)
for mname, _ in MODELS:
    print(f"\n#### {mname} ####")
    print(f"{'knowledge':16s} {'elbow meanK':12s} " + " ".join(f"k={k!s:<5s}" for k in KVALS) + "  stable?")
    for C in names:
        s = summary[mname].get(C, {})
        if "k" not in s or not s["k"]:
            print(f"{C:16s} (skipped)"); continue
        archs = [s["k"].get(str(k), {}).get("arch", "-") for k in KVALS]
        elbow_k = s["k"].get("elbow", {}).get("mean_k", "-")
        stable = "yes" if len(set(a for a in archs if a != "-")) == 1 else "NO"
        print(f"{C:16s} {str(elbow_k):12s} " + " ".join(f"{a:<7s}" for a in archs) + f"  {stable}")
print("\nsaved -> pathways/cspt_ksweep_summary.json")
