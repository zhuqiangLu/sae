"""Pilot analysis: score the 5 knowledge items through each model's SAEs, then
match teacher<->student and run the validation gates. Seed-aware (SAEs live in
saes/<model>/seed{N}/). Produces the first cross-family knowledge table.

Run: PYTHONPATH=src python3 scripts/run_pilot_analysis.py
"""
from __future__ import annotations
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.config import load_config
from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.concepts import load_battery
from know_trans.score import score_features, concept_feature_sets
from know_trans.match import run_matching, stability_score, null_control

CFG = "configs/pair_llama8b_qwen0p6b_pilot.yaml"
BATTERY_DIR = "data/concepts_pilot"
AUC_TH = 0.8
log = get_logger("pilot")

cfg = load_config(CFG)
data = cfg.paths.data
battery = load_battery(BATTERY_DIR)
names = [c.name for c in battery]
log.info("battery: %d items: %s", len(battery), names)

sdir = lambda m, s: os.path.join(data, "saes", m, f"seed{s}")
fsdir = ensure_dir(os.path.join(data, "feature_scores"))
mdir = ensure_dir(os.path.join(data, "matches"))

def raw_best(df, name):
    sub = df[df["concept"] == name]
    if len(sub) == 0:
        return (None, None)
    r = sub.loc[sub["auc"].idxmax()]
    return (int(r["layer"]), round(float(r["auc"]), 3))

# ---- Teacher: score seed0 + seed1 (model run twice, one per SAE seed) -------
tname, tlayers = cfg.teacher.name, [int(x) for x in cfg.teacher.layers]
log.info("loading teacher %s", tname)
tm, tt = load_model_and_tokenizer(cfg.teacher.path, dtype=cfg.teacher.dtype, device="cuda")
t_sets, t_df = {}, {}
for seed in (0, 1):
    b = SAEBundle.load(sdir(tname, seed))
    df = score_features(tm, tt, b, battery, tlayers,
                        os.path.join(fsdir, f"{tname}_seed{seed}.parquet"),
                        max_len=256, batch_size=32)
    t_df[seed] = df
    t_sets[seed] = concept_feature_sets(df, auc_threshold=AUC_TH, top_k=10)
    log.info("teacher seed%d scored", seed)
del tm; torch.cuda.empty_cache()

# ---- Student: score seed0 --------------------------------------------------
sname, slayers = cfg.student.name, [int(x) for x in cfg.student.layers]
log.info("loading student %s", sname)
sm, st = load_model_and_tokenizer(cfg.student.path, dtype=cfg.student.dtype, device="cuda")
sb = SAEBundle.load(sdir(sname, 0))
s_df = score_features(sm, st, sb, battery, slayers,
                      os.path.join(fsdir, f"{sname}_seed0.parquet"),
                      max_len=256, batch_size=32)
s_sets = concept_feature_sets(s_df, auc_threshold=AUC_TH, top_k=10)
del sm; torch.cuda.empty_cache()

# ---- Match + gates ---------------------------------------------------------
matches = run_matching(t_sets[0], s_sets)
stab = stability_score(t_sets[0], t_sets[1])      # cross-seed teacher stability
null = null_control(t_sets[0], s_sets, n_shuffle=100)
save_json(matches, os.path.join(mdir, "pilot_matches.json"))
save_json({"stability": stab, "null": null}, os.path.join(mdir, "pilot_gates.json"))

# ---- Table -----------------------------------------------------------------
print("\n================ PILOT CROSS-FAMILY KNOWLEDGE TABLE ================")
print(f"AUC threshold for a 'detector' = {AUC_TH}\n")
hdr = f"{'knowledge':16s} {'teacher best(L,AUC)':22s} {'student best(L,AUC)':22s} {'shared':7s} {'stab(Jacc)':10s}"
print(hdr); print("-" * len(hdr))
for n in names:
    tb = raw_best(t_df[0], n)
    sbst = raw_best(s_df, n)
    shared = matches.get(n, {}).get("shared")
    sj = stab.get(n)
    sj = "" if sj is None else round(sj, 2)
    print(f"{n:16s} {str(tb):22s} {str(sbst):22s} {str(shared):7s} {sj}")
print()
print("teacher cross-seed stability (mean Jaccard):", round(stab.get("__mean__", float('nan')), 3))
print("null control:", null)
print("\nsaved -> matches/pilot_matches.json, matches/pilot_gates.json")
