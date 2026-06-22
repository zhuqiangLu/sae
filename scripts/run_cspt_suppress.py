"""§3.3 path-suppression test: compare clean vs source-only vs source+path
interventions, for all 3 models on the 5 pilot knowledge.

Reuses the FIS parquets from run_cspt.py (the identified path) and re-scores the
knowledge detector under each condition. The headline is the MARGIN: does zeroing
the full path (64 @ onset + k_down @ each downstream layer) collapse knowledge
MORE than zeroing the source alone?

Run: PYTHONPATH=src python3 scripts/run_cspt_suppress.py [--kdown 10 --kfeat 10 --ksrc 64 --tau 0.70]
"""
from __future__ import annotations
import argparse, os, json
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.cspt import pick_onset_layer, source_neurons, suppress_pathway_validation

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
PW = os.path.join(DATA, "pathways")
log = get_logger("cspt_suppress")

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--kdown", type=int, default=10)
ap.add_argument("--kfeat", type=int, default=10)
ap.add_argument("--ksrc", type=int, default=64)
ap.add_argument("--tau", type=float, default=0.70)
a = ap.parse_args()

battery = load_battery("data/concepts_pilot")
byname = {c.name: c for c in battery}
names = [c.name for c in battery]

summary = {}
for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    adf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer.parquet"))
    wdf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer_wfs.parquet"))
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    summary[mname] = {}
    for C in names:
        fpath = os.path.join(PW, f"{mname}_{C}_fis.parquet")
        if not os.path.exists(fpath):
            summary[mname][C] = {"skipped": "no fis"}; continue
        fis_df = pd.read_parquet(fpath)
        lstar, _ = pick_onset_layer(adf, C, tau_onset=a.tau)
        src, _, _ = source_neurons(bundle[lstar], wdf, C, lstar, k_feat=a.kfeat, k_src=a.ksrc)
        v = suppress_pathway_validation(model, tok, bundle, byname[C], lstar, src, fis_df, adf,
                                        k_down=a.kdown, max_len=256, batch_size=16, dtype="bfloat16")
        v.to_parquet(os.path.join(PW, f"{mname}_{C}_suppress.parquet"), index=False)
        CONDS = ["clean", "src", "path", "src+path", "src+randD", "randS+path"]
        means = {c: (round(float(v[f"auc_{c}"].mean()), 3) if len(v) else None) for c in CONDS}
        summary[mname][C] = {"onset": int(lstar), **means}
        log.info("[%s/%s] " + " ".join(f"{c}=%.3f" for c in CONDS), mname, C,
                 *[means[c] or 0 for c in CONDS])
    del model; torch.cuda.empty_cache()

save_json(summary, os.path.join(PW, "cspt_suppress_summary.json"))

CONDS = ["clean", "src", "path", "src+path", "src+randD", "randS+path"]
print("\n" + "=" * 100)
print(f"FACTORIAL INTERVENTION — mean downstream detector AUC (src=64@onset, path/rand={a.kdown}@downstream)")
print("clean=original | lower AUC = more collapse | src<randS+path => source real | src+path<src+randD => path real")
print("=" * 100)
for mname, _ in MODELS:
    print(f"\n#### {mname} ####")
    print(f"{'knowledge':16s} " + " ".join(f"{c:>10s}" for c in CONDS))
    for C in names:
        s = summary[mname].get(C, {})
        if "skipped" in s:
            print(f"{C:16s} (skipped)"); continue
        print(f"{C:16s} " + " ".join(f"{(s.get(c) or 0):>10.3f}" for c in CONDS))
print("\nsaved -> pathways/*_suppress.parquet, cspt_suppress_summary.json")
