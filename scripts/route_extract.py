"""route_extract.py — Route separation for the SUBJECT-SPECIALIZED SAEs.

Extract a subject's causal cross-layer route (dense MLP-output coords per layer)
for the teacher Llama-3.1-8B, using the NEW subject-specialized SAEs
(data/saes_subject/Llama-3.1-8B/<domain>). The CSPT engine (know_trans.cspt) is
reused verbatim; we only orchestrate it and compute the SAE-specific anchor
tables (AUROC + WFS) inline, because the existing feature_scores/*.parquet are
bound to the GENERAL SAE and must NOT be reused.

Per domain (econ|math|med), the matching topic concept is
topic_economics|topic_math|topic_medical, whose positives are that subject's
MMLU questions and whose hard negatives are other-subject MMLU questions (the
concept battery in data/concepts_pilot). We:

  1. load the base model + the specialized SAEBundle;
  2. compute the AUROC table (know_trans.score.score_features) for THIS SAE,
     scoped to the matching concept (cached to disk);
  3. compute the WFS table (know_trans.wfs.wfs_score_features) for THIS SAE,
     scoped to the matching concept (cached to disk);
  4. extract the route with cspt.trace_pathway_greedy (default) or the
     star/FIS path (cspt.pick_onset_layer + trace_pathway), and
  5. write data/pathways_subject/Llama-3.1-8B_<domain>_<method>_nodes.parquet
     with columns [layer, neuron, concept] (+ stat columns).

Routes live in DENSE MLP-output space; the SAE is used only to select the
source at the onset layer (its W_dec back-projection).

Run (full):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/route_extract.py --domain econ --method greedy
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/route_extract.py --domain math --method star

Smoke (tiny, proves end-to-end, NOT the real run):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/route_extract.py --domain econ --smoke
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.sae import SAEBundle
from know_trans.score import score_features
from know_trans.wfs import wfs_score_features
from know_trans.concepts import Concept, load_battery
from know_trans.cspt import (
    pick_onset_layer,
    elbow_k,
    trace_pathway,
    trace_pathway_greedy,
)

log = get_logger("route_extract")

MODEL = "Llama-3.1-8B"
MODEL_PATH = "models/Llama-3.1-8B"  # BASE model (specialized SAEs are base-space)
BATTERY_DIR = "data/concepts_pilot"

# domain -> concept name in the battery
DOMAIN_CONCEPT = {
    "econ": "topic_economics",
    "math": "topic_math",
    "med": "topic_medical",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["econ", "math", "med"], required=True)
    ap.add_argument("--method", choices=["greedy", "star"], default="greedy")
    ap.add_argument("--sae-dir", default=None,
                    help="default data/saes_subject/Llama-3.1-8B/<domain>")
    ap.add_argument("--out-dir", default="data/pathways_subject")
    ap.add_argument("--cache-dir", default="data/feature_scores_subject",
                    help="where the per-SAE AUROC/WFS anchor tables are cached")
    ap.add_argument("--tau", type=float, default=0.70, help="onset AUROC threshold")
    ap.add_argument("--ncap", type=int, default=128, help="prompts per side for the trace")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--kmax-feat", dest="kmax_feat", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--recompute", action="store_true",
                    help="force recompute of the cached AUROC/WFS tables")
    # hidden smoke knobs: cap layers + examples so it finishes in minutes
    ap.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--smoke-layers", type=int, default=8, help=argparse.SUPPRESS)
    ap.add_argument("--smoke-cap", type=int, default=24, help=argparse.SUPPRESS)
    a = ap.parse_args()

    domain = a.domain
    concept_name = DOMAIN_CONCEPT[domain]
    sae_dir = a.sae_dir or f"data/saes_subject/{MODEL}/{domain}"
    out_dir = ensure_dir(a.out_dir)
    cache_dir = ensure_dir(a.cache_dir)
    suffix = "_smoke" if a.smoke else ""

    # ---- battery: keep ONLY the matching topic concept -----------------------
    battery = {c.name: c for c in load_battery(BATTERY_DIR)}
    if concept_name not in battery:
        raise SystemExit(f"concept {concept_name!r} absent from {BATTERY_DIR}; "
                         f"have {sorted(battery)}")
    concept: Concept = battery[concept_name]
    if a.smoke:
        # shrink the battery so AUROC/WFS/trace all finish fast
        concept = Concept(
            name=concept.name,
            positives=concept.positives[:a.smoke_cap],
            hard_negatives=concept.hard_negatives[:a.smoke_cap],
            source=concept.source,
            group=concept.group,
        )
    log.info("[%s] concept=%s pos=%d neg=%d", domain, concept.name,
             concept.n_pos, concept.n_neg)

    # ---- SAE bundle for THIS domain -----------------------------------------
    bundle = SAEBundle.load(sae_dir, device="cuda")
    all_layers = sorted(int(l) for l in bundle.layers)
    if a.smoke:
        # use the first contiguous block of layers (onset is typically early/mid;
        # for a smoke test we only need >=2 layers so the trace has a hop)
        layers = all_layers[:a.smoke_layers]
    else:
        layers = all_layers
    log.info("[%s] SAE dir=%s layers=%d (using %d)", domain, sae_dir,
             len(all_layers), len(layers))

    # ---- model (base) --------------------------------------------------------
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")

    # ---- anchor tables for THIS SAE (cached) --------------------------------
    auroc_path = os.path.join(cache_dir, f"{MODEL}_{domain}_auroc{suffix}.parquet")
    wfs_path = os.path.join(cache_dir, f"{MODEL}_{domain}_wfs{suffix}.parquet")

    if a.recompute or not os.path.exists(auroc_path):
        log.info("[%s] computing AUROC table -> %s", domain, auroc_path)
        adf = score_features(model, tok, bundle, [concept], layers, auroc_path,
                             max_len=a.max_len, batch_size=a.batch_size,
                             dtype="bfloat16")
    else:
        log.info("[%s] reusing cached AUROC table %s", domain, auroc_path)
        adf = pd.read_parquet(auroc_path)

    if a.recompute or not os.path.exists(wfs_path):
        log.info("[%s] computing WFS table -> %s", domain, wfs_path)
        wdf = wfs_score_features(model, tok, bundle, [concept], layers, wfs_path,
                                 max_len=a.max_len, batch_size=a.batch_size,
                                 dtype="bfloat16")
    else:
        log.info("[%s] reusing cached WFS table %s", domain, wfs_path)
        wdf = pd.read_parquet(wfs_path)

    # onset (for logging + the star path)
    l_star, peak = pick_onset_layer(adf, concept_name, tau_onset=a.tau)
    log.info("[%s] onset l*=%d (peak AUROC=%.3f, tau=%.2f)", domain, l_star, peak, a.tau)

    # ---- extract the route ---------------------------------------------------
    if a.method == "greedy":
        nodes_df, l_star = trace_pathway_greedy(
            model, tok, bundle, wdf, adf, concept,
            tau_onset=a.tau, k_src="elbow", kmin=a.kmin, kmax=a.kmax,
            kmax_feat=a.kmax_feat, n_cap=a.ncap, max_len=a.max_len,
            batch_size=a.batch_size,
        )
    else:  # star / FIS
        # Steps 1-3: SAE-space WFS source -> dense S_src (mirror run_cspt_star.py)
        w = (wdf[(wdf.concept == concept_name) & (wdf.layer == l_star)]
             .sort_values("delta_wfs", ascending=False))
        if w.empty:
            raise SystemExit(f"no WFS rows for {concept_name!r} at l*={l_star}")
        dwv = w.delta_wfs.to_numpy()
        kf = elbow_k(dwv, a.kmin, a.kmax_feat)
        feats = w.feature.to_numpy()[:kf].astype(np.int64)
        fw = np.clip(dwv[:kf], 0.0, None).astype(np.float32)
        Wd = bundle[l_star].W_dec.detach().float().cpu().numpy()
        za = np.abs(Wd[:, feats] @ fw)
        order = np.argsort(za)[::-1]
        ks = elbow_k(za[order], a.kmin, a.kmax)
        S_src = np.sort(order[:ks]).astype(np.int64)
        log.info("[%s] star source l*=%d: %d deltaWFS feats -> %d dense S_src",
                 domain, l_star, kf, len(S_src))

        fis_df, _ = trace_pathway(
            model, tok, bundle, concept, l_star, S_src,
            auroc_df=adf, max_len=a.max_len, batch_size=a.batch_size, dtype="bfloat16",
        )
        # route = S_src (root @ l*) U elbow-FIS dense neurons per downstream layer
        rows = [{"layer": int(l_star), "neuron": int(n), "concept": concept_name,
                 "fis": float("nan"), "l_star": int(l_star)} for n in S_src]
        for L, sub in fis_df.groupby("layer"):
            v = np.sort(sub.fis.to_numpy())[::-1]
            k = elbow_k(v, a.kmin, a.kmax)
            for r in sub.nlargest(k, "fis").itertuples(index=False):
                rows.append({"layer": int(L), "neuron": int(r.neuron),
                             "concept": concept_name, "fis": float(r.fis),
                             "l_star": int(l_star)})
        nodes_df = pd.DataFrame(rows)
        fis_df.to_parquet(
            os.path.join(out_dir, f"{MODEL}_{domain}_star_fis{suffix}.parquet"),
            index=False)

    # ---- persist -------------------------------------------------------------
    out = os.path.join(out_dir, f"{MODEL}_{domain}_{a.method}_nodes{suffix}.parquet")
    nodes_df.to_parquet(out, index=False)
    per_layer = nodes_df.groupby("layer").neuron.count()
    log.info("[%s] saved %s | l*=%d nodes=%d over %d layers",
             domain, out, l_star, len(nodes_df), nodes_df.layer.nunique())
    print(f"saved {out}")
    print(f"domain={domain} concept={concept_name} method={a.method}")
    print(f"onset l*={l_star} | total route nodes={len(nodes_df)} "
          f"over {nodes_df.layer.nunique()} layers")
    print("per-layer counts:", {int(k): int(v) for k, v in per_layer.to_dict().items()})


if __name__ == "__main__":
    main()
