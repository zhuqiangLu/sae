"""onset_locate_split.py — per-subject AUROC onset-layer localization on the new
down_proj_in SAEs, using the single-subject splits. Memory-bounded: the SAEs are
~3.3 GB each (d_hidden=28672 x d_in=14336), so we keep ONE on the GPU at a time.

For each domain:
  positives = subject VAL questions (split), negatives = unrelated MMLU subjects.
  1. one forward pass -> mean-pooled down_proj_in per (layer, example)  [E, 14336]
  2. per layer: load SAE -> encode_dense -> per-feature detector AUROC -> best
  onset = earliest layer whose best AUROC >= tau, else global-best layer.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.onset_locate_split
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from know_trans.utils import load_model_and_tokenizer, get_logger, batched, ensure_dir
from know_trans.capture import MLPHook, _get_hook_module
from know_trans.sae import TopKSAE

log = get_logger("onset_split")
MODEL = "Llama-3.1-8B"
MODEL_PATH = "models/Llama-3.1-8B"
BENCH = "benchmarks/MMLU"
LETTERS = ["A", "B", "C", "D"]
GENERAL = ["professional_law", "high_school_world_history", "philosophy", "high_school_psychology"]


def fmt(q: str, ch) -> str:
    s = q.strip() + "\n"
    for i, c in enumerate(ch):
        s += f"{LETTERS[i]}. {str(c).strip()}\n"
    return s + "Answer:"


def load_subject_texts(subj: str) -> list[str]:
    out, seen = [], set()
    for sp in ("test", "validation", "dev"):
        for p in glob.glob(f"{BENCH}/{subj}/{sp}-*.parquet"):
            df = pd.read_parquet(p)
            if "question" not in df.columns:
                continue
            for q, ch in zip(df["question"], df["choices"]):
                ch = list(ch)
                if isinstance(q, str) and len(ch) == 4 and q.strip() not in seen:
                    seen.add(q.strip()); out.append(fmt(q, ch))
    return out


@torch.no_grad()
def pooled_down_proj_in(model, tok, texts, layers, batch_size, max_len):
    """One forward pass -> {layer: [E, d_in] mean-pooled down_proj input (CPU fp32)}."""
    hooks = {L: MLPHook(_get_hook_module(model, L, "down_proj_in"), L,
                        to_cpu=False, capture_input=True) for L in layers}
    pooled = {L: [] for L in layers}
    for tb in batched(list(texts), batch_size):
        enc = tok(list(tb), return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        attn = enc["attention_mask"].unsqueeze(-1).float()
        model(**enc)
        for L in layers:
            act = hooks[L].pop().float()                       # [B,S,d]
            mean = (act * attn).sum(1) / attn.sum(1).clamp_min(1.0)  # [B,d]
            pooled[L].append(mean.cpu())
    for h in hooks.values():
        h.remove()
    return {L: torch.cat(v, 0) for L, v in pooled.items()}


@torch.no_grad()
def locate(model, tok, domain, tau, batch_size, max_len, seed):
    rng = np.random.default_rng(seed)
    split = json.load(open(f"data/pathways_subject/mmlu_{domain}_split.json"))
    subject = split["subject"]
    pos = [fmt(r["question"], r["choices"]) for r in split["val"]]
    neg = []
    for s in GENERAL:
        neg += load_subject_texts(s)
    rng.shuffle(neg); neg = neg[:len(pos)]
    texts = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))

    sae_dir = f"data/saes_subject/{MODEL}/{domain}"
    layers = sorted(int(os.path.basename(p)[len("layer"):-len(".safetensors")])
                    for p in glob.glob(f"{sae_dir}/layer*.safetensors"))
    log.info("[%s/%s] pos=%d neg=%d layers=%d", domain, subject, len(pos), len(neg), len(layers))

    pooled = pooled_down_proj_in(model, tok, texts, layers, batch_size, max_len)

    rows = []
    for L in layers:
        sae = TopKSAE.load(f"{sae_dir}/layer{L}.safetensors", device="cuda")
        codes = sae.encode_dense(pooled[L].cuda().float()).float().cpu().numpy()  # [E,H]
        del sae; torch.cuda.empty_cache()
        cmin, cmax = codes.min(0), codes.max(0)
        active = np.nonzero(cmax > cmin)[0]
        aucs = np.full(codes.shape[1], 0.5)
        for j in active:
            aucs[j] = roc_auc_score(labels, codes[:, j])
        best = float(aucs.max())                              # best positive-detector
        n_above = int((aucs >= tau).sum())
        rows.append({"layer": L, "best_auc": round(best, 4), "n_above": n_above,
                     "n_active_feat": int(len(active))})

    per = pd.DataFrame(rows).set_index("layer").sort_index()
    crossed = per[per.best_auc >= tau]
    if len(crossed):
        onset, reason = int(crossed.index[0]), f"first layer >= {tau}"
    else:
        onset, reason = int(per.best_auc.idxmax()), f"global-best (none >= {tau})"
    return subject, per, onset, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", default="econ,math,med")
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = ensure_dir("report/diag")

    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    summary = {}
    for domain in [d for d in a.domains.split(",") if d.strip()]:
        subject, per, onset, reason = locate(model, tok, domain, a.tau,
                                             a.batch_size, a.max_len, a.seed)
        summary[domain] = {"subject": subject, "onset": onset, "reason": reason,
                           "onset_auc": float(per.loc[onset, "best_auc"]),
                           "global_best_auc": float(per.best_auc.max()),
                           "global_best_layer": int(per.best_auc.idxmax()),
                           "per_layer": per.reset_index().to_dict("records")}
        json.dump(summary[domain], open(os.path.join(out_dir, f"onset_split_{domain}.json"), "w"), indent=2)

        print("\n" + "=" * 60)
        print(f"ONSET  {domain} ({subject})   tau={a.tau}")
        print("=" * 60)
        print("  L : best_auc  (#feat>=tau)")
        for r in per.reset_index().itertuples(index=False):
            mark = " <-- onset" if r.layer == onset else ""
            print(f"  {r.layer:2d}: {r.best_auc:.3f}      ({r.n_above}){mark}")
        print(f"  => onset L{onset} (auc={per.loc[onset,'best_auc']:.3f}), "
              f"global-best L{per.best_auc.idxmax()} auc={per.best_auc.max():.3f}  [{reason}]")

    print("\n" + "#" * 60)
    print("SUMMARY (onset layer per subject)")
    for d, s in summary.items():
        print(f"  {d:5s} {s['subject']:32s} onset=L{s['onset']:<2d} "
              f"auc={s['onset_auc']:.3f}  global-best=L{s['global_best_layer']} {s['global_best_auc']:.3f}")


if __name__ == "__main__":
    main()
