"""Causal Semantic Pathway Tracing (TraceRouter §3.2) for all 3 pilot models on
the 5 pilot knowledge, with the zero-out validation.

Per (model, knowledge): onset l* (AUROC first-crossing) -> WFS source features ->
back-project to S_src -> zero-out + FIS over downstream dense neurons -> validate
(downstream knowledge AUROC clean vs intervened).

Run: PYTHONPATH=src python3 scripts/run_cspt.py [--tau 0.70 --kfeat 10 --ksrc 64]
"""
from __future__ import annotations
import argparse, os, json
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.cspt import pick_onset_layer, source_neurons, trace_pathway

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
OUT = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("run_cspt")

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, default=0.70)
ap.add_argument("--kfeat", type=int, default=10)
ap.add_argument("--ksrc", type=int, default=64)
ap.add_argument("--top-pathway", type=int, default=10, help="top-FIS neurons/layer kept in summary")
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
        lstar, auc = pick_onset_layer(adf, C, tau_onset=a.tau)
        all_layers = sorted(int(l) for l in bundle.layers)
        if lstar >= max(all_layers):
            log.warning("[%s/%s] onset l*=%d is last layer; skipping (no downstream).", mname, C, lstar)
            summary[mname][C] = {"onset": lstar, "skipped": "no downstream"}
            continue
        src, zproj, feats = source_neurons(bundle[lstar], wdf, C, lstar, k_feat=a.kfeat, k_src=a.ksrc)
        fis_df, val_df = trace_pathway(model, tok, bundle, byname[C], lstar, src,
                                       auroc_df=adf, max_len=256, batch_size=16, dtype="bfloat16")
        fis_df.to_parquet(os.path.join(OUT, f"{mname}_{C}_fis.parquet"), index=False)
        val_df.to_parquet(os.path.join(OUT, f"{mname}_{C}_validation.parquet"), index=False)

        # pathway = top-FIS neurons per downstream layer
        pathway = {}
        for l in sorted(fis_df.layer.unique()):
            top = fis_df[fis_df.layer == l].nlargest(a.top_pathway, "fis")
            pathway[int(l)] = [(int(r.neuron), round(float(r.fis), 4)) for r in top.itertuples()]
        mean_drop = float(val_df["drop"].mean()) if len(val_df) else None
        max_drop = float(val_df["drop"].max()) if len(val_df) else None
        summary[mname][C] = {
            "onset": lstar, "onset_auc": round(auc, 3),
            "source_features": feats, "n_src": int(len(src)),
            "n_downstream_layers": int(len(pathway)),
            "mean_auc_drop": None if mean_drop is None else round(mean_drop, 3),
            "max_auc_drop": None if max_drop is None else round(max_drop, 3),
            "pathway_top": {l: v[:3] for l, v in pathway.items()},  # compact: top-3/layer
        }
        log.info("[%s/%s] l*=%d auc=%.3f down=%d meanDrop=%.3f maxDrop=%.3f",
                 mname, C, lstar, auc, len(pathway),
                 mean_drop or 0.0, max_drop or 0.0)
    del model; torch.cuda.empty_cache()

save_json(summary, os.path.join(OUT, "cspt_summary.json"))

print("\n" + "=" * 80)
print("CAUSAL SEMANTIC PATHWAY TRACING — summary (validation = downstream AUROC drop)")
print("=" * 80)
for mname, _ in MODELS:
    print(f"\n#### {mname} ####")
    print(f"{'knowledge':16s} {'onset':6s} {'onsetAUC':9s} {'#down':6s} {'meanDrop':9s} {'maxDrop':8s}")
    for C in names:
        s = summary[mname].get(C, {})
        if "skipped" in s:
            print(f"{C:16s} L{s['onset']:<4d} (skipped: no downstream)")
            continue
        print(f"{C:16s} L{s['onset']:<4d} {s['onset_auc']:<9.3f} {s['n_downstream_layers']:<6d} "
              f"{s['mean_auc_drop']:<9.3f} {s['max_auc_drop']:<8.3f}")
print("\nsaved -> pathways/*_fis.parquet, *_validation.parquet, cspt_summary.json")
