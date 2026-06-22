"""Robustness: re-run the full CSPT pipeline (FIS trace + factorial suppression)
using a CHOSEN onset method, so we can test whether the source-bottlenecked /
distributed-relay architecture labels depend on how l* is located.

--onset auroc : first AUROC crossing (the main report's method)
--onset attn  : relaxed attention-divergence onset (pathways/onset_attention.json)

Run: PYTHONPATH=src python3 scripts/run_cspt_onset.py --onset attn
"""
from __future__ import annotations
import argparse, os, json
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.cspt import pick_onset_layer, source_neurons, trace_pathway, suppress_pathway_validation

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
PW = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("cspt_onset")

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--onset", choices=["auroc", "attn"], default="attn")
ap.add_argument("--kfeat", type=int, default=10)
ap.add_argument("--ksrc", type=int, default=64)
ap.add_argument("--kdown", type=int, default=10)
ap.add_argument("--tau", type=float, default=0.70)
a = ap.parse_args()

attn_onsets = {}
if a.onset == "attn":
    j = json.load(open(os.path.join(PW, "onset_attention.json")))
    attn_onsets = j["summary"]

battery = load_battery("data/concepts_pilot")
byname = {c.name: c for c in battery}
names = [c.name for c in battery]


def classify(clean, src, path, src_randd):
    if clean < 0.62:  # weak detector -> no clean concept to trace
        return "weak"
    sb = (clean - src) > 0.10            # source alone collapses
    dr = (src_randd - path) > 0.10       # path beats magnitude-matched random
    if sb and not dr: return "SB"
    if dr and not sb: return "DR"
    if sb and dr:     return "SB+DR"
    return "none"


summary = {}
for mname, mpath in MODELS:
    log.info("==== %s (onset=%s) ====", mname, a.onset)
    adf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer.parquet"))
    wdf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer_wfs.parquet"))
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    layers = sorted(int(l) for l in bundle.layers)
    summary[mname] = {}
    for C in names:
        if a.onset == "attn":
            lstar = int(attn_onsets[mname][C]["onset_attn"])
        else:
            lstar, _ = pick_onset_layer(adf, C, tau_onset=a.tau)
        if lstar >= max(layers):
            summary[mname][C] = {"onset": lstar, "skipped": "no downstream"}; continue
        src, _, _ = source_neurons(bundle[lstar], wdf, C, lstar, k_feat=a.kfeat, k_src=a.ksrc)
        fis_df, _ = trace_pathway(model, tok, bundle, byname[C], lstar, src,
                                  auroc_df=adf, max_len=256, batch_size=16, dtype="bfloat16")
        v = suppress_pathway_validation(model, tok, bundle, byname[C], lstar, src, fis_df, adf,
                                        k_down=a.kdown, max_len=256, batch_size=16, dtype="bfloat16")
        CN = ["clean", "src", "path", "src+path", "src+randD", "randS+path"]
        m = {c: round(float(v[f"auc_{c}"].mean()), 3) if len(v) else None for c in CN}
        arch = classify(m["clean"], m["src"], m["path"], m["src+randD"]) if len(v) else "n/a"
        summary[mname][C] = {"onset": int(lstar), "arch": arch, **m}
        log.info("[%s/%s] L%d %s | clean=%.3f src=%.3f path=%.3f src+randD=%.3f",
                 mname, C, lstar, arch, m["clean"], m["src"], m["path"], m["src+randD"])
    del model; torch.cuda.empty_cache()

save_json(summary, os.path.join(PW, f"cspt_suppress_{a.onset}onset_summary.json"))
print("\n" + "=" * 96)
print(f"FULL CSPT under onset={a.onset}  (arch: SB=source-bottlenecked, DR=distributed-relay)")
print("=" * 96)
for mname, _ in MODELS:
    print(f"\n#### {mname} ####")
    print(f"{'knowledge':16s} {'onset':6s} {'arch':6s} {'clean':7s} {'src':7s} {'path':7s} {'src+randD':10s}")
    for C in names:
        s = summary[mname].get(C, {})
        if "skipped" in s:
            print(f"{C:16s} L{s['onset']:<4d} (skipped)"); continue
        print(f"{C:16s} L{s['onset']:<4d} {s['arch']:6s} {s['clean']:<7.3f} {s['src']:<7.3f} "
              f"{s['path']:<7.3f} {s['src+randD']:<10.3f}")
print(f"\nsaved -> pathways/cspt_suppress_{a.onset}onset_summary.json")
