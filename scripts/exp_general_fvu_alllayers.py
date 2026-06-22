"""Experiment A: OOD map of the EXISTING general SAEs across all layers x 3 models.

No training. For each model (teacher + both students) and EVERY layer, measure
the trained general SAE's per-token reconstruction FVU on held-out econ/math/med
activations. Shows where, across depth and across model families, the current
SAE is out-of-distribution on the topic domains that distillation targets.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/exp_general_fvu_alllayers.py
"""
from __future__ import annotations

import json
import os
import shutil

import torch
from safetensors.torch import load_file as st_load

from know_trans import utils
from know_trans.capture import ActivationReader, capture_activations
from know_trans.sae import TopKSAE
from scripts.exp_subject_sae import DOMAINS, build_texts, fvu

MODELS = ["Llama-3.1-8B", "Qwen3-0.6B", "Llama-3.2-1B"]
SAE_ROOT = "data/saes"
TMP = "data/_fvu_map"
OUT = "report/diag/general_fvu_alllayers.json"
N_EVAL = 250
MAX_LEN = 320
CAP_BATCH = 8


def n_layers(model: str) -> int:
    return json.load(open(f"models/{model}/config.json"))["num_hidden_layers"]


def load_sae(model: str, layer: int, device) -> TopKSAE:
    base = os.path.join(SAE_ROOT, model, "seed0", f"layer{layer}.safetensors")
    cfg = json.load(open(base + ".config.json"))
    sae = TopKSAE(cfg["d_in"], cfg["d_hidden"], cfg["k"])
    sae.load_state_dict(st_load(base), strict=True)
    return sae.to(device).eval()


def main():
    log = utils.get_logger("exp_general_fvu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    device = utils.get_device()
    results = {"n_eval": N_EVAL, "max_len": MAX_LEN, "models": {}}

    # held-out eval texts per domain (same for every model)
    eval_texts = {d: build_texts(d)[-N_EVAL:] for d in DOMAINS}
    for d in DOMAINS:
        log.info("domain %s: %d held-out eval texts", d, len(eval_texts[d]))

    for model in MODELS:
        nl = n_layers(model)
        layers = list(range(nl))
        if os.path.isdir(TMP):
            shutil.rmtree(TMP)
        log.info("=== model %s (%d layers) ===", model, nl)
        mdl, tok = utils.load_model_and_tokenizer(f"models/{model}", dtype="bfloat16",
                                                  device=device)
        for d in DOMAINS:
            capture_activations(mdl, tok, eval_texts[d], layers, f"{TMP}/{d}",
                                batch_size=CAP_BATCH, max_len=MAX_LEN)
        del mdl
        torch.cuda.empty_cache()

        results["models"][model] = {"n_layers": nl, "domains": {d: {} for d in DOMAINS}}
        for d in DOMAINS:
            reader = ActivationReader(f"{TMP}/{d}")
            for L in layers:
                x = reader.read(L)[0]
                sae = load_sae(model, L, device)
                m = fvu(sae, x, device)
                results["models"][model]["domains"][d][str(L)] = m
                del sae
                torch.cuda.empty_cache()
            row = " ".join(f"{results['models'][model]['domains'][d][str(L)]['fvu']:.2f}"
                           for L in layers)
            log.info("%s %-5s FVU/layer: %s", model, d, row)
        json.dump(results, open(OUT, "w"), indent=2)

    if os.path.isdir(TMP):
        shutil.rmtree(TMP)

    # summary: mean FVU over layers, and min/best-layer per (model, domain)
    print("\n===== general-SAE FVU on topic domains (mean over layers / best layer) =====")
    print(f"{'model':<16}{'domain':<7}{'mean_fvu':>9}{'min_fvu':>9}{'best_L':>8}")
    for model in MODELS:
        for d in DOMAINS:
            per = results["models"][model]["domains"][d]
            vals = {int(L): v["fvu"] for L, v in per.items()}
            mean = sum(vals.values()) / len(vals)
            bestL = min(vals, key=vals.get)
            print(f"{model:<16}{d:<7}{mean:>9.3f}{vals[bestL]:>9.3f}{bestL:>8}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
