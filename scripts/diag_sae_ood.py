"""OOD confirmation diagnostic: is the teacher SAE out-of-distribution on the
full-MMLU / MMLU-Pro math/eco/med text that the topic concepts are built and
evaluated on?

Mechanism under test: the SAE training corpus (_harvest_benchmark_texts) is
AdvBench + HarmBench + Global-MMLU-Lite ONLY. The topic concepts come from full
canonical MMLU, which the SAE never trained on. If true, the SAE should
reconstruct full-MMLU / MMLU-Pro activations markedly worse than its own
training distribution -> the topic concept-activation dims are noise -> null
distillation.

We measure per-token reconstruction FVU (fraction of variance unexplained) of
the trained teacher SAE on four text groups, captured identically:
  ID_safety   AdvBench behaviors                      (in training distribution)
  ID_gmmlu    Global-MMLU-Lite questions (multiling)  (in training distribution)
  OOD_mmlu    full MMLU math/eco/med (English)        (topic concept source)
  OOD_mmlupro MMLU-Pro math/economics/health          (transfer eval set)

FVU = sum||x-recon||^2 / sum||x-mean(x)||^2 over per-token MLP activations.
Higher = more OOD. Run: PYTHONPATH=src python scripts/diag_sae_ood.py
"""
from __future__ import annotations

import glob
import json
import os
import random

import pandas as pd
import torch
from safetensors.torch import load_file as st_load

from know_trans import utils
from know_trans.sae import TopKSAE

BENCH = "benchmarks"
MODEL_PATH = "models/Llama-3.1-8B"
SAE_DIR = "data/saes/Llama-3.1-8B/seed0"
LAYERS = [8, 16, 24]            # pilot layers (present in both seeds)
MAX_LEN = 256
BATCH = 16
TOK_CAP = 20000                 # per-token activations collected per group
N_TEXT_CAP = 400
OUT = "report/diag/sae_ood_fvu.json"

MMLU_MATH = ["abstract_algebra", "college_mathematics", "high_school_mathematics",
             "elementary_mathematics", "high_school_statistics"]
MMLU_ECON = ["high_school_macroeconomics", "high_school_microeconomics", "econometrics"]
MMLU_MED = ["clinical_knowledge", "college_medicine", "professional_medicine",
            "anatomy", "medical_genetics", "virology", "nutrition"]
MMLU_TOPIC_SUBJECTS = set(MMLU_MATH + MMLU_ECON + MMLU_MED)
MMLUPRO_CATS = {"math", "economics", "health"}

rng = random.Random(0)


def _clean(s) -> str:
    return " ".join(str(s).split())


def load_safety(n):
    out = []
    path = os.path.join(BENCH, "AdvBench", "advbench_behaviors.jsonl")
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            v = o.get("Behavior") or o.get("behavior")
            if v:
                out.append(_clean(v))
    # top up from HarmBench so the safety group isn't tiny
    hb = os.path.join(BENCH, "HarmBench", "metadata.csv")
    if os.path.isfile(hb):
        df = pd.read_csv(hb)
        if "Behavior" in df.columns:
            out += [_clean(v) for v in df["Behavior"].dropna().tolist()]
    rng.shuffle(out)
    return out[:n]


def load_gmmlu(n):
    out = []
    pqs = sorted(glob.glob(os.path.join(BENCH, "Global-MMLU-Lite", "**", "*.parquet"),
                           recursive=True))
    rng.shuffle(pqs)
    for pq in pqs:
        try:
            df = pd.read_parquet(pq)
        except Exception:
            continue
        col = next((c for c in ("question", "Question", "text") if c in df.columns), None)
        if col is None:
            continue
        out += [_clean(v) for v in df[col].dropna().tolist()]
        if len(out) >= n * 2:
            break
    rng.shuffle(out)
    return out[:n]


def load_mmlu_topic(n):
    paths = sorted(glob.glob(os.path.join(BENCH, "MMLU", "all", "test-*.parquet")))
    rows = []
    for p in paths:
        df = pd.read_parquet(p, columns=["question", "subject"])
        df = df[df["subject"].isin(MMLU_TOPIC_SUBJECTS)]
        rows += [_clean(q) for q in df["question"].dropna().tolist()]
    rng.shuffle(rows)
    return rows[:n]


def load_mmlupro_topic(n):
    p = os.path.join(BENCH, "MMLU-Pro", "data", "test-00000-of-00001.parquet")
    df = pd.read_parquet(p, columns=["question", "category"])
    df = df[df["category"].isin(MMLUPRO_CATS)]
    rows = [_clean(q) for q in df["question"].dropna().tolist()]
    rng.shuffle(rows)
    return rows[:n]


GROUPS = {
    "ID_safety": load_safety,
    "ID_gmmlu": load_gmmlu,
    "OOD_mmlu": load_mmlu_topic,
    "OOD_mmlupro": load_mmlupro_topic,
}


@torch.inference_mode()
def capture_group(model, tok, texts, device):
    """Return {layer: tensor[N_tok, d_model] float32} of per-token MLP outputs."""
    blocks = model.model.layers
    store: dict[int, list[torch.Tensor]] = {L: [] for L in LAYERS}
    counts = {L: 0 for L in LAYERS}
    handles = []
    cur = {}

    def mk(L):
        def hook(_m, _i, out):
            cur[L] = out[0] if isinstance(out, tuple) else out
        return hook

    for L in LAYERS:
        handles.append(blocks[L].mlp.register_forward_hook(mk(L)))
    try:
        for i in range(0, len(texts), BATCH):
            if all(counts[L] >= TOK_CAP for L in LAYERS):
                break
            batch = texts[i:i + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN)
            enc = {k: v.to(device) for k, v in enc.items()}
            cur.clear()
            model(**enc)
            mask = enc["attention_mask"].bool()           # [B, T]
            for L in LAYERS:
                act = cur[L]                               # [B, T, d]
                sel = act[mask].float().cpu()              # [n_tok, d]
                if counts[L] < TOK_CAP:
                    store[L].append(sel)
                    counts[L] += sel.shape[0]
    finally:
        for h in handles:
            h.remove()
    return {L: torch.cat(store[L])[:TOK_CAP] for L in LAYERS}


def load_sae(layer, device):
    cfg = json.load(open(os.path.join(SAE_DIR, f"layer{layer}.safetensors.config.json")))
    sae = TopKSAE(cfg["d_in"], cfg["d_hidden"], cfg["k"])
    sd = st_load(os.path.join(SAE_DIR, f"layer{layer}.safetensors"))
    sae.load_state_dict(sd, strict=True)
    sae.to(device).eval()
    return sae


@torch.inference_mode()
def fvu_for(sae, x, device):
    """Per-token FVU + mean cosine(recon, x) over x [N, d]."""
    res_sq = 0.0
    var_sq = 0.0
    cos_sum = 0.0
    n = 0
    mu = x.mean(0, keepdim=True)
    for i in range(0, x.shape[0], 8192):
        xb = x[i:i + 8192].to(device)
        vals, idx = sae.encode(xb)
        recon = sae.decode(vals, idx)
        res_sq += ((xb - recon) ** 2).sum().item()
        var_sq += ((xb - mu.to(device)) ** 2).sum().item()
        cos_sum += torch.nn.functional.cosine_similarity(xb, recon, dim=-1).sum().item()
        n += xb.shape[0]
    return {"fvu": res_sq / var_sq, "cos": cos_sum / n, "n_tok": n}


def main():
    log = utils.get_logger("diag_sae_ood")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    device = utils.get_device()
    log.info("loading model %s on %s", MODEL_PATH, device)
    model, tok = utils.load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device=device)

    saes = {L: load_sae(L, device) for L in LAYERS}

    results = {"layers": LAYERS, "max_len": MAX_LEN, "tok_cap": TOK_CAP, "groups": {}}
    for gname, loader in GROUPS.items():
        texts = loader(N_TEXT_CAP)
        log.info("group %s: %d texts", gname, len(texts))
        acts = capture_group(model, tok, texts, device)
        results["groups"][gname] = {"n_texts": len(texts), "per_layer": {}}
        for L in LAYERS:
            m = fvu_for(saes[L], acts[L], device)
            results["groups"][gname]["per_layer"][str(L)] = m
            log.info("  %s L%-2d  FVU=%.4f  cos=%.4f  n_tok=%d",
                     gname, L, m["fvu"], m["cos"], m["n_tok"])

    json.dump(results, open(OUT, "w"), indent=2)

    # pretty summary table
    print("\n===== SAE reconstruction FVU (lower = better fit; higher = more OOD) =====")
    hdr = "group".ljust(14) + "".join(f"  L{L:<8}" for L in LAYERS)
    print(hdr)
    for g, d in results["groups"].items():
        row = g.ljust(14)
        for L in LAYERS:
            row += f"  {d['per_layer'][str(L)]['fvu']:.4f}    "
        print(row)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
