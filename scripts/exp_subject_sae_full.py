"""Experiment B (full): subject-specialized SAEs across 3 models x ALL layers.

Split (per user decision): hold out MMLU-Pro entirely.
  TRAIN: full-MMLU domain subjects (test+validation+dev)  -> SAE training
  EVAL : MMLU-Pro same domain (held out)                  -> FVU measurement
This makes "specialized beats general" a GENERALIZATION claim and keeps MMLU-Pro
(the Goal-2 transfer eval set) uncontaminated by SAE training.

For each (model, domain, layer): train a TopK SAE (d_hidden=16384, k=32, 2000
steps -- same recipe as the general SAE, only the corpus differs), then measure
its held-out MMLU-Pro FVU vs the existing general SAE's FVU on the same tokens.

Resumable: each SAE -> data/saes_subject/<model>/<domain>/layer{L}.safetensors;
results -> report/diag/subjectB_<model>_<domain>.json (merged per layer). Re-running
skips any (layer) already trained + recorded. Shard across GPUs via CUDA_VISIBLE_DEVICES
and --tasks; each (model,domain) task unit writes a distinct file (no races).

Run (per GPU process):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.exp_subject_sae_full \
      --tasks Llama-3.1-8B:econ,Llama-3.1-8B:math,Qwen3-0.6B:econ,Qwen3-0.6B:math
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import time

import pandas as pd
import torch
from safetensors.torch import load_file as st_load

from know_trans import utils
from know_trans.capture import ActivationReader, capture_activations
from know_trans.config import SAECfg
from know_trans.sae import TopKSAE, train_sae
from scripts.exp_subject_sae import DOMAINS, _fmt

# Narrow single-subject domains (cut training data to ONE specific topic).
DOMAINS = {**DOMAINS,
           "micro": {"mmlu": ["high_school_microeconomics"], "pro": ["economics"]}}

BENCH = "benchmarks"
SAE_ROOT = "data/saes"                 # existing general SAEs (baseline)
SPEC_ROOT = "data/saes_subject"        # specialized SAEs we train here
TMP_ROOT = "data/_subjB"
RES_DIR = "report/diag"
MAX_LEN = 384
CAP_BATCH = 8
N_EVAL_PRO = 300                       # held-out MMLU-Pro eval texts per domain
STEPS = 1000                           # per user: 1000 steps, expansion factor 2, lr 1e-4
# Hook the SwiGLU intermediate (down_proj input, dim=intermediate_size) — the MLP
# "neuron" / key-value-memory space — not the 4096-dim MLP output (which is the
# massive-activation-dominated residual write). Knowledge localization lives here.
HOOK_POINT = "down_proj_in"
SPLIT_TPL = "data/pathways_subject/mmlu_{domain}_split.json"  # single-subject train/val/test


def n_layers(model: str) -> int:
    return json.load(open(f"models/{model}/config.json"))["num_hidden_layers"]


def _split_texts(domain: str, key: str):
    """Formatted texts from the single-subject split (mmlu_{domain}_split.json), or None."""
    sp = SPLIT_TPL.format(domain=domain)
    if not os.path.exists(sp):
        return None
    return [_fmt(r["question"], r["choices"]) for r in json.load(open(sp))[key]]


def build_train(domain: str) -> list[str]:
    """SAE training corpus: split's TRAIN rows (single subject) if present, else legacy pool."""
    sp = _split_texts(domain, "train")
    if sp is not None:
        return sp
    spec = DOMAINS[domain]
    texts: list[str] = []
    for subj in spec["mmlu"]:
        for split in ("test", "validation", "dev"):
            for p in glob.glob(f"{BENCH}/MMLU/{subj}/{split}-*.parquet"):
                df = pd.read_parquet(p)
                if "question" not in df.columns:
                    continue
                ch = df["choices"] if "choices" in df.columns else [None] * len(df)
                texts += [_fmt(q, c) for q, c in zip(df["question"], ch)]
    texts = [t for t in texts if t]
    random.Random(hash(domain) & 0xFFFF).shuffle(texts)
    return texts


def build_eval(domain: str) -> list[str]:
    """FVU eval: split's held-out TEST rows (in-distribution) if present, else MMLU-Pro."""
    sp = _split_texts(domain, "test")
    if sp is not None:
        return sp
    spec = DOMAINS[domain]
    dp = pd.read_parquet(f"{BENCH}/MMLU-Pro/data/test-00000-of-00001.parquet",
                         columns=["question", "options", "category"])
    dp = dp[dp["category"].isin(spec["pro"])]
    texts = [_fmt(q, o) for q, o in zip(dp["question"], dp["options"])]
    texts = [t for t in texts if t]
    random.Random(1234).shuffle(texts)
    return texts[:N_EVAL_PRO]


@torch.inference_mode()
def fvu(sae: TopKSAE, x: torch.Tensor, device) -> dict:
    res_sq = var_sq = cos_sum = 0.0
    n = 0
    active = torch.zeros(sae.d_hidden, dtype=torch.bool, device=device)
    mu = x.mean(0, keepdim=True).to(device)
    for i in range(0, x.shape[0], 8192):
        xb = x[i:i + 8192].to(device).float()
        vals, idx = sae.encode(xb)
        recon = sae.decode(vals, idx)
        res_sq += ((xb - recon) ** 2).sum().item()
        var_sq += ((xb - mu) ** 2).sum().item()
        cos_sum += torch.nn.functional.cosine_similarity(xb, recon, dim=-1).sum().item()
        active[idx.reshape(-1)] = True
        n += xb.shape[0]
    return {"fvu": res_sq / var_sq, "cos": cos_sum / n, "n_tok": n,
            "frac_feat_active": active.float().mean().item()}


def load_general(model: str, layer: int, device) -> TopKSAE:
    base = os.path.join(SAE_ROOT, model, "seed0", f"layer{layer}.safetensors")
    cfg = json.load(open(base + ".config.json"))
    sae = TopKSAE(cfg["d_in"], cfg["d_hidden"], cfg["k"])
    sae.load_state_dict(st_load(base), strict=True)
    return sae.to(device).eval()


def res_path(model: str, domain: str, tag: str = "") -> str:
    suf = f"_{tag}" if tag else ""
    return os.path.join(RES_DIR, f"subjectB_{model}_{domain}{suf}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True,
                    help="comma list of Model:domain units, e.g. Llama-3.1-8B:econ,Qwen3-0.6B:math")
    ap.add_argument("--layers", default=None,
                    help="subset of layers e.g. '0-15' or '0,5,9' (default: all). Shard GPUs by layer range.")
    ap.add_argument("--tag", default="",
                    help="suffix for tmp/results paths; use distinct tags per shard to avoid races")
    args = ap.parse_args()

    def _parse_layers(s):
        if not s:
            return None
        out = set()
        for part in s.split(","):
            if "-" in part:
                a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        return sorted(out)
    requested = _parse_layers(args.layers)
    log = utils.get_logger("exp_subjB")
    os.makedirs(RES_DIR, exist_ok=True)
    device = utils.get_device()

    units = [t.split(":") for t in args.tasks.split(",") if t.strip()]
    by_model: dict[str, list[str]] = {}
    for model, domain in units:
        by_model.setdefault(model, []).append(domain)

    cfg = SAECfg(expansion=2, d_hidden=None, k=32, lr=1e-4, batch_size=4096, steps=STEPS,
                 aux_k=256, aux_coef=1 / 32, activation="topk")  # d_hidden = 2 * d_in

    for model, domains in by_model.items():
        nl = n_layers(model)
        layers = list(range(nl))
        if requested is not None:
            layers = [L for L in layers if L in requested]
        log.info("=== MODEL %s (%d layers); training=%s; domains=%s ===",
                 model, nl, (f"L{layers[0]}-{layers[-1]}" if layers else "none"), domains)

        # which (domain,layer) still need work?
        pending = {}
        for d in domains:
            done = {}
            rp = res_path(model, d, args.tag)
            if os.path.exists(rp):
                done = json.load(open(rp)).get("layers", {})
            need = [L for L in layers
                    if str(L) not in done
                    or not os.path.exists(f"{SPEC_ROOT}/{model}/{d}/layer{L}.safetensors")]
            pending[d] = need
            log.info("  domain %s: %d/%d layers pending", d, len(need), nl)
        if not any(pending.values()):
            log.info("  nothing to do for %s", model)
            continue

        mdl, tok = utils.load_model_and_tokenizer(f"models/{model}", dtype="bfloat16",
                                                  device=device)
        gen_cache: dict[int, TopKSAE] = {}

        for d in domains:
            if not pending[d]:
                continue
            tmp = f"{TMP_ROOT}/{model}_{d}" + (f"_{args.tag}" if args.tag else "")
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)
            tr_txt, ev_txt = build_train(d), build_eval(d)
            log.info("  [%s/%s] train=%d MMLU texts, eval=%d MMLU-Pro texts",
                     model, d, len(tr_txt), len(ev_txt))
            mdl.to(device)                                    # model on GPU for capture
            capture_activations(mdl, tok, tr_txt, layers, f"{tmp}/train",
                                batch_size=CAP_BATCH, max_len=MAX_LEN, hook_point=HOOK_POINT)
            capture_activations(mdl, tok, ev_txt, layers, f"{tmp}/eval",
                                batch_size=CAP_BATCH, max_len=MAX_LEN, hook_point=HOOK_POINT)
            r_tr = ActivationReader(f"{tmp}/train")
            r_ev = ActivationReader(f"{tmp}/eval")
            mdl.to("cpu"); torch.cuda.empty_cache()           # free ~16GB; SAE training doesn't need the model

            rp = res_path(model, d, args.tag)
            results = json.load(open(rp)) if os.path.exists(rp) else \
                {"model": model, "domain": d, "n_layers": nl, "steps": STEPS,
                 "split": "train=MMLU(test+val+dev), eval=MMLU-Pro(held-out)",
                 "n_train_texts": len(tr_txt), "n_eval_texts": len(ev_txt), "layers": {}}

            for L in pending[d]:
                t0 = time.time()
                out = f"{SPEC_ROOT}/{model}/{d}/layer{L}.safetensors"
                if os.path.exists(out):
                    sae = TopKSAE.load(out, device=str(device))
                else:
                    sae = train_sae(r_tr, L, cfg, seed=0, out_path=out)
                    sae.to(device).eval()
                x_ev = r_ev.read(L)[0]
                m_spec = fvu(sae, x_ev, device)
                # General-SAE baseline only comparable when it lives in the SAME
                # space; the general SAEs are d_in=4096 (MLP output), so with the
                # down_proj_in hook (d_in=14336) the comparison is skipped.
                m_gen = None
                if L not in gen_cache:
                    try:
                        gen_cache[L] = load_general(model, L, device)
                    except Exception:
                        gen_cache[L] = None
                if gen_cache[L] is not None and gen_cache[L].d_in == x_ev.shape[-1]:
                    m_gen = fvu(gen_cache[L], x_ev, device)
                results["layers"][str(L)] = {
                    "fvu_spec": m_spec["fvu"], "fvu_gen": (m_gen["fvu"] if m_gen else None),
                    "cos_spec": m_spec["cos"], "cos_gen": (m_gen["cos"] if m_gen else None),
                    "frac_active_spec": m_spec["frac_feat_active"],
                    "n_eval_tok": m_spec["n_tok"], "d_in": int(x_ev.shape[-1]),
                    "hook_point": HOOK_POINT,
                }
                json.dump(results, open(rp, "w"), indent=2)
                log.info("  [%s/%s L%-2d] spec_FVU=%.3f gen_FVU=%s active=%.1f%% %.0fs",
                         model, d, L, m_spec["fvu"],
                         ("%.3f" % m_gen["fvu"]) if m_gen else "n/a",
                         100 * m_spec["frac_feat_active"], time.time() - t0)
                del sae
                torch.cuda.empty_cache()

            shutil.rmtree(tmp, ignore_errors=True)
            gen_cache.clear()

        del mdl
        torch.cuda.empty_cache()

    log.info("ALL ASSIGNED TASKS DONE")


if __name__ == "__main__":
    main()
