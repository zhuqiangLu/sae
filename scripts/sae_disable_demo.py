"""sae_disable_demo.py — per-example outputs under the econ SAE-neuron disable.

Loads the elbow-K ΔWFS critical features (report/diag/sae_ablate_econ.json), disables
them at the onset layer, and prints for 10 target (macro) + 10 general (law) questions:
gold letter, clean vs disabled MC prediction, and a short greedy generation.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.sae_disable_demo
"""
from __future__ import annotations
import json
import numpy as np
import torch

from know_trans.utils import load_model_and_tokenizer
from know_trans.sae import TopKSAE
from scripts.sae_neuron_ablate import SAEFeatureDisableHook, fmt, load_subject_rows, fewshot, LETTERS

DOM = "econ"
GEN_SUBJ = "professional_law"


@torch.no_grad()
def mc_pred(model, tok, letter_ids, prefix, q, ch):
    enc = tok(prefix + fmt(q, ch), return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1]
    return LETTERS[int(logits[letter_ids].argmax().item())]


@torch.no_grad()
def gen_text(model, tok, prefix, q, ch, n=24):
    enc = tok(prefix + fmt(q, ch), return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ")[:90]


def main():
    s = json.load(open(f"report/diag/sae_ablate_{DOM}.json"))
    L = int(s["onset"]); K = int(s["elbow_K"]); feats = np.array(s["critical_feats"], dtype=np.int64)
    split = json.load(open(f"data/pathways_subject/mmlu_{DOM}_split.json"))

    model, tok = load_model_and_tokenizer("models/Llama-3.1-8B", dtype="bfloat16", device="cuda")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = torch.tensor([tok.encode(" " + c, add_special_tokens=False)[-1] for c in LETTERS],
                              device=model.device)
    sae = TopKSAE.load(f"data/saes_subject/Llama-3.1-8B/{DOM}/layer{L}.safetensors", device="cuda")
    hook = SAEFeatureDisableHook(model.model.layers[L].mlp.down_proj, sae)
    hook.set_feats(feats)

    tgt_prefix = fewshot([(r["question"], r["choices"], r["answer"]) for r in split["train"]], 5)
    tgt = [(r["question"], r["choices"], r["answer"]) for r in split["test"]][:10]
    gen_rows = load_subject_rows(GEN_SUBJ)
    gen_prefix = fewshot(gen_rows[-7:], 5)
    gen = gen_rows[:10]

    def show(tag, prefix, items):
        print("\n" + "#" * 78)
        print(f"{tag}  —  disable {K} econ ΔWFS features @ L{L}")
        print("#" * 78)
        for (q, ch, a) in items:
            hook.enabled = False; cp = mc_pred(model, tok, letter_ids, prefix, q, ch); cg = gen_text(model, tok, prefix, q, ch)
            hook.enabled = True;  dp = mc_pred(model, tok, letter_ids, prefix, q, ch); dg = gen_text(model, tok, prefix, q, ch)
            hook.enabled = False
            gold = LETTERS[a]
            print(f"\ngold={gold} | clean={cp}{'✓' if cp == gold else '✗'}  disabled={dp}{'✓' if dp == gold else '✗'}")
            print(f"  Q: {q[:85]}")
            print(f"  clean   gen: {cg!r}")
            print(f"  disabled gen: {dg!r}")

    show("TARGET (high_school_macroeconomics test)", tgt_prefix, tgt)
    show(f"GENERAL ({GEN_SUBJ})", gen_prefix, gen)
    hook.remove()


if __name__ == "__main__":
    main()
