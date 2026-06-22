"""Controlled experiment: subject-specialized SAEs (econ / math / med only).

Question: if the SAE's training corpus actually CONTAINS the domain, does it
reconstruct that domain's activations well (and does specialization hold)?

Design — the ONLY variable is the training corpus:
  - same recipe as the existing general teacher SAE: d_hidden=16384, k=32,
    lr=4e-4, batch=4096, aux_k=256, aux_coef=1/32  (general seed0 used steps=4000).
  - train ONE SAE per domain on that domain's text (MMLU subjects + MMLU-Pro
    category), questions split train/eval so FVU is measured on HELD-OUT tokens
    (controls for overfitting).
  - negatives/contrast = the OTHER domains (per user): we report a 3x3 cross-FVU
    matrix (domain-SAE x eval-domain) + the existing general SAE as a baseline row.

Headline comparison (teacher, layer 16):
  general-SAE FVU on econ-eval (~0.59 measured earlier)  vs
  econ-SAE   FVU on econ-eval  (should drop a lot if coverage is the issue).

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/exp_subject_sae.py
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil

import pandas as pd
import torch
from safetensors.torch import load_file as st_load

from know_trans import utils
from know_trans.capture import ActivationReader, capture_activations
from know_trans.config import SAECfg
from know_trans.sae import TopKSAE, train_sae

BENCH = "benchmarks"
MODEL_PATH = "models/Llama-3.1-8B"
GEN_SAE_DIR = "data/saes/Llama-3.1-8B/seed0"     # existing general SAE (baseline)
TMP = "data/_subj_exp"                            # scratch for captured acts
OUT = "report/diag/subject_sae.json"

CAP_LAYERS = [8, 16, 24]      # capture these (eval against general SAE at each)
TRAIN_LAYERS = [16]           # train specialized SAEs at this/these layer(s)
MAX_LEN = 384
CAP_BATCH = 8
N_TRAIN = 2500                # max texts per domain for SAE training
N_EVAL = 300                  # held-out texts per domain for FVU eval (taken from the tail)
STEPS = 3000                  # specialized SAE training steps (vs general 4000)

DOMAINS = {
    "econ": {
        "mmlu": ["high_school_macroeconomics", "high_school_microeconomics", "econometrics"],
        "pro": ["economics"],
    },
    "math": {
        "mmlu": ["abstract_algebra", "college_mathematics", "high_school_mathematics",
                 "elementary_mathematics", "high_school_statistics"],
        "pro": ["math"],
    },
    "med": {
        "mmlu": ["clinical_knowledge", "college_medicine", "professional_medicine",
                 "anatomy", "medical_genetics", "virology", "nutrition"],
        "pro": ["health"],
    },
}

rng = random.Random(0)


def _clean(s) -> str:
    return " ".join(str(s).split())


def _fmt(q, choices) -> str:
    q = _clean(q)
    try:
        opts = " ".join(_clean(c) for c in list(choices))
    except Exception:
        opts = ""
    return (q + " " + opts).strip()


def build_texts(domain: str) -> list[str]:
    spec = DOMAINS[domain]
    texts: list[str] = []
    # full MMLU subjects
    mp = os.path.join(BENCH, "MMLU", "all", "test-00000-of-00001.parquet")
    df = pd.read_parquet(mp, columns=["question", "subject", "choices"])
    df = df[df["subject"].isin(spec["mmlu"])]
    texts += [_fmt(q, c) for q, c in zip(df["question"], df["choices"])]
    # MMLU-Pro categories
    pp = os.path.join(BENCH, "MMLU-Pro", "data", "test-00000-of-00001.parquet")
    dp = pd.read_parquet(pp, columns=["question", "options", "category"])
    dp = dp[dp["category"].isin(spec["pro"])]
    texts += [_fmt(q, o) for q, o in zip(dp["question"], dp["options"])]
    texts = [t for t in texts if t]
    rng.shuffle(texts)
    return texts


@torch.inference_mode()
def fvu(sae: TopKSAE, x: torch.Tensor, device) -> dict:
    res_sq = var_sq = cos_sum = 0.0
    n = 0
    mu = x.mean(0, keepdim=True)
    for i in range(0, x.shape[0], 8192):
        xb = x[i:i + 8192].to(device).float()
        vals, idx = sae.encode(xb)
        recon = sae.decode(vals, idx)
        res_sq += ((xb - recon) ** 2).sum().item()
        var_sq += ((xb - mu.to(device)) ** 2).sum().item()
        cos_sum += torch.nn.functional.cosine_similarity(xb, recon, dim=-1).sum().item()
        n += xb.shape[0]
    return {"fvu": res_sq / var_sq, "cos": cos_sum / n, "n_tok": n}


def load_general(layer, device) -> TopKSAE:
    cfg = json.load(open(os.path.join(GEN_SAE_DIR, f"layer{layer}.safetensors.config.json")))
    sae = TopKSAE(cfg["d_in"], cfg["d_hidden"], cfg["k"])
    sae.load_state_dict(st_load(os.path.join(GEN_SAE_DIR, f"layer{layer}.safetensors")), strict=True)
    return sae.to(device).eval()


def main():
    log = utils.get_logger("exp_subject_sae")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    device = utils.get_device()

    # ---- 1. capture domain activations (train + eval) --------------------- #
    log.info("loading model %s", MODEL_PATH)
    model, tok = utils.load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device=device)
    text_counts = {}
    for d in DOMAINS:
        texts = build_texts(d)
        ev = texts[-N_EVAL:]               # held-out tail for FVU
        train = texts[:-N_EVAL][:N_TRAIN]  # the rest (capped) for SAE training
        text_counts[d] = {"train": len(train), "eval": len(ev)}
        log.info("domain %s: %d train / %d eval texts", d, len(train), len(ev))
        capture_activations(model, tok, train, CAP_LAYERS, f"{TMP}/{d}_train",
                            batch_size=CAP_BATCH, max_len=MAX_LEN)
        capture_activations(model, tok, ev, CAP_LAYERS, f"{TMP}/{d}_eval",
                            batch_size=CAP_BATCH, max_len=MAX_LEN)
    del model
    torch.cuda.empty_cache()

    # ---- 2. train one specialized SAE per domain (at TRAIN_LAYERS) -------- #
    cfg = SAECfg(d_hidden=16384, k=32, lr=4e-4, batch_size=4096, steps=STEPS,
                 aux_k=256, aux_coef=1 / 32, activation="topk")
    spec_saes = {}  # (domain, layer) -> TopKSAE
    for d in DOMAINS:
        reader = ActivationReader(f"{TMP}/{d}_train")
        for L in TRAIN_LAYERS:
            log.info("training specialized SAE: domain=%s layer=%d steps=%d", d, L, STEPS)
            sae = train_sae(reader, L, cfg, seed=0, out_path=f"{TMP}/sae_{d}_L{L}.safetensors")
            spec_saes[(d, L)] = sae.to(device).eval()

    # ---- 3. preload eval activations + general SAEs ----------------------- #
    eval_acts = {}  # (domain, layer) -> tensor
    for d in DOMAINS:
        r = ActivationReader(f"{TMP}/{d}_eval")
        for L in CAP_LAYERS:
            eval_acts[(d, L)] = r.read(L)[0]
    gen = {L: load_general(L, device) for L in CAP_LAYERS}

    # ---- 4. metrics ------------------------------------------------------- #
    results = {"steps": STEPS, "d_hidden": 16384, "train_layers": TRAIN_LAYERS,
               "text_counts": text_counts, "general_baseline": {}, "specialized_matrix": {}}

    # general SAE FVU on each domain eval, at every captured layer
    for d in DOMAINS:
        results["general_baseline"][d] = {}
        for L in CAP_LAYERS:
            m = fvu(gen[L], eval_acts[(d, L)], device)
            results["general_baseline"][d][str(L)] = m
            log.info("GENERAL  SAE  eval=%s L%-2d  FVU=%.4f cos=%.4f", d, L, m["fvu"], m["cos"])

    # specialized SAE (trained on domain D) FVU on each eval domain, at TRAIN_LAYERS
    for (dtrain, L), sae in spec_saes.items():
        results["specialized_matrix"].setdefault(dtrain, {})
        for deval in DOMAINS:
            m = fvu(sae, eval_acts[(deval, L)], device)
            results["specialized_matrix"][dtrain].setdefault(str(L), {})[deval] = m
            tag = "  <== own" if dtrain == deval else ""
            log.info("SPECIAL[%s] SAE  eval=%s L%-2d  FVU=%.4f cos=%.4f%s",
                     dtrain, deval, L, m["fvu"], m["cos"], tag)

    json.dump(results, open(OUT, "w"), indent=2)

    # ---- 5. summary tables ------------------------------------------------ #
    L = TRAIN_LAYERS[0]
    print(f"\n===== HEADLINE (layer {L}): does domain-trained SAE fix its own FVU? =====")
    print("domain      general_FVU   specialized_FVU   drop")
    for d in DOMAINS:
        g = results["general_baseline"][d][str(L)]["fvu"]
        s = results["specialized_matrix"][d][str(L)][d]["fvu"]
        print(f"{d:<10}  {g:.4f}        {s:.4f}            {g - s:+.4f}")

    print(f"\n===== specialization matrix (layer {L}) rows=trained-on, cols=eval-on =====")
    print("trained\\eval   " + "   ".join(f"{d:>8}" for d in DOMAINS))
    for dt in DOMAINS:
        row = f"{dt:<12} "
        for de in DOMAINS:
            row += f"   {results['specialized_matrix'][dt][str(L)][de]['fvu']:.4f}  "
        print(row)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
