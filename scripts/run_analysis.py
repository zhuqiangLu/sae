"""Consolidated cross-family analysis: score -> match -> gates, with the CORRECT
(index-free) pattern-based stability and a softened (margin-aware) threshold.

Parameterized by --config and --battery so it runs for any teacher/student pair.

Run: PYTHONPATH=src python3 scripts/run_analysis.py --config <cfg> --tag <name>
"""
from __future__ import annotations
import argparse, os
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.config import load_config
from know_trans.capture import ActivationReader
from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.score import score_features, concept_feature_sets
from know_trans.match import run_matching, null_control

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--battery", default="data/concepts_pilot")
ap.add_argument("--tau", type=float, default=0.70, help="softened detector AUC threshold")
ap.add_argument("--probe-n", type=int, default=40000)
ap.add_argument("--tag", required=True, help="label for output files")
a = ap.parse_args()
log = get_logger("analysis")

cfg = load_config(a.config)
data = cfg.paths.data
battery = load_battery(a.battery)
names = [c.name for c in battery]
fsdir = ensure_dir(os.path.join(data, "feature_scores"))
mdir = ensure_dir(os.path.join(data, "matches"))
sdir = lambda m, s: os.path.join(data, "saes", m, f"seed{s}")
log.info("analysis tag=%s battery=%d items config=%s", a.tag, len(battery), a.config)

def raw_best(df, name):
    sub = df[df["concept"] == name]
    if len(sub) == 0:
        return (None, None)
    r = sub.loc[sub["auc"].idxmax()]
    return (int(r["layer"]), round(float(r["auc"]), 3))

def zscore(x):
    return (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)

def pattern_stability(reader, b0, b1, sets0, probe_n):
    """Index-free cross-seed stability: max activation-correlation of each seed0
    detector latent to ANY seed1 latent over a probe set. Returns {concept: mean}."""
    g = torch.Generator().manual_seed(0)
    samp = torch.randperm(reader.n_tokens, generator=g)[:probe_n]
    by_layer = {}
    for c, feats in sets0.items():
        if feats:
            by_layer.setdefault(int(feats[0]["layer"]), []).append(c)
    out = {c: None for c in sets0}
    for layer, concepts in sorted(by_layer.items()):
        acts, _ = reader.read(layer)
        xb = acts[samp].to("cuda", dtype=torch.float32)
        c0 = b0[layer].to("cuda").encode_dense(xb)
        c1 = b1[layer].to("cuda").encode_dense(xb)
        z1 = zscore(c1)
        for cname in concepts:
            idxs = [int(f["feature"]) for f in sets0[cname]]
            z0 = zscore(c0[:, idxs])
            corr = (z0.t() @ z1) / c0.shape[0]
            out[cname] = round(float(corr.max(dim=1).values.mean()), 3)
        del c0, c1, z1; torch.cuda.empty_cache()
    return out

# ---- score teacher (all available seeds) -----------------------------------
tname, tlayers = cfg.teacher.name, [int(x) for x in cfg.teacher.layers]
log.info("scoring teacher %s", tname)
tm, tt = load_model_and_tokenizer(cfg.teacher.path, dtype=cfg.teacher.dtype, device="cuda")
t_df, t_sets = {}, {}
for seed in (0, 1):
    if not os.path.isdir(sdir(tname, seed)):
        continue
    b = SAEBundle.load(sdir(tname, seed))
    df = score_features(tm, tt, b, battery, tlayers,
                        os.path.join(fsdir, f"{tname}_seed{seed}_{a.tag}.parquet"),
                        max_len=256, batch_size=32)
    t_df[seed] = df
    t_sets[seed] = concept_feature_sets(df, auc_threshold=a.tau, top_k=10)
del tm; torch.cuda.empty_cache()

# ---- score student ---------------------------------------------------------
sname, slayers = cfg.student.name, [int(x) for x in cfg.student.layers]
log.info("scoring student %s", sname)
sm, st = load_model_and_tokenizer(cfg.student.path, dtype=cfg.student.dtype, device="cuda")
sb = SAEBundle.load(sdir(sname, 0))
s_df = score_features(sm, st, sb, battery, slayers,
                      os.path.join(fsdir, f"{sname}_seed0_{a.tag}.parquet"),
                      max_len=256, batch_size=32)
s_sets = concept_feature_sets(s_df, auc_threshold=a.tau, top_k=10)
del sm; torch.cuda.empty_cache()

# ---- gates -----------------------------------------------------------------
stab = {}
if 1 in t_sets:
    reader = ActivationReader(os.path.join(data, "activations", tname))
    stab = pattern_stability(reader, SAEBundle.load(sdir(tname, 0)),
                             SAEBundle.load(sdir(tname, 1)), t_sets[0], a.probe_n)
matches = run_matching(t_sets[0], s_sets)
null = null_control(t_sets[0], s_sets, n_shuffle=100)

# ---- softened classification -----------------------------------------------
def classify(ta, sa):
    if ta is None: ta = 0.0
    if sa is None: sa = 0.0
    if ta >= a.tau and sa >= a.tau: return "shared"
    if ta >= a.tau and sa < a.tau:  return "teacher>student"
    return "weak/neither"

rows = []
for n in names:
    tl, ta = raw_best(t_df[0], n)
    sl, sa = raw_best(s_df, n)
    rows.append({"knowledge": n, "teacher": (tl, ta), "student": (sl, sa),
                 "gap": None if (ta is None or sa is None) else round(ta - sa, 3),
                 "class": classify(ta, sa), "pattern_stab": stab.get(n)})
save_json({"tag": a.tag, "tau": a.tau, "rows": rows, "null": null},
          os.path.join(mdir, f"analysis_{a.tag}.json"))

print(f"\n========= ANALYSIS [{a.tag}]  (tau={a.tau}) =========")
print(f"{'knowledge':16s} {'teacher(L,AUC)':18s} {'student(L,AUC)':18s} {'gap':6s} {'class':16s} {'stab':5s}")
print("-" * 86)
for r in rows:
    print(f"{r['knowledge']:16s} {str(r['teacher']):18s} {str(r['student']):18s} "
          f"{str(r['gap']):6s} {r['class']:16s} {r['pattern_stab']}")
sm_mean = [r['pattern_stab'] for r in rows if r['pattern_stab'] is not None]
print(f"\nmean pattern-stability: {round(sum(sm_mean)/len(sm_mean),3) if sm_mean else 'n/a'}")
print(f"null control: real={null.get('real_mean'):.3f} null={null.get('null_mean'):.3f} z={null.get('z'):.2f}")
print(f"saved -> matches/analysis_{a.tag}.json")
