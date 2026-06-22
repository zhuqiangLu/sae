"""gen_ablate_demo.py — show actual generations clean vs ablating the located onset
source neurons, for TARGET (econ) and GENERAL questions.

Reveals HOW the model breaks under ablation (gibberish vs coherent-but-wrong).
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.gen_ablate_demo \
        --src-json data/pathways_subject/Llama-3.1-8B_econ_onset_src_L1.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger
from know_trans.cspt import ZeroOutHook
from know_trans.capture import _get_hook_module
from scripts.route_eval_selectivity import (
    MODEL_PATH, DOMAIN_TARGET, GENERAL_CATS, LETTERS,
    load_mmlu_pro, _format_q, build_fewshot_prefix,
)

log = get_logger("gen_demo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=40)
    a = ap.parse_args()

    src = json.load(open(a.src_json))
    domain = src["domain"]; L = int(src["onset_layer"])
    coords = np.array(sorted(int(c) for c in src["source_neurons"]), dtype=np.int64)
    _, target_cat = DOMAIN_TARGET[domain]
    split = json.load(open(src["split_path"]))
    test_idx, val_idx = split["test_idx"], split["val_idx"]

    econ_rows, _ = load_mmlu_pro(target_cat)
    tgt = [econ_rows[i] for i in test_idx][:a.n]
    tgt_prefix = build_fewshot_prefix([econ_rows[i] for i in val_idx], a.nshot)

    # general = spread across 4 general cats, each with its own few-shot prefix
    gen_cats = GENERAL_CATS[:4]
    gen_items = []  # (cat, (q,o,gold))
    per = a.n // len(gen_cats) + 1
    gprefix = {}
    for c in gen_cats:
        rows = load_mmlu_pro(c)[0]
        gprefix[c] = build_fewshot_prefix(rows[-(a.nshot + 2):], a.nshot)
        for r in rows[:per]:
            gen_items.append((c, r))
    gen_items = gen_items[:a.n]

    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hook = ZeroOutHook(_get_hook_module(model, L, "mlp"), coords)

    @torch.no_grad()
    def gen(prefix, q, o):
        prompt = prefix + _format_q(q, o)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def show(tag, prefix, q, o, gold):
        hook.enabled = False
        clean = gen(prefix, q, o).replace("\n", " ⏎ ")[:160]
        hook.enabled = True
        abl = gen(prefix, q, o).replace("\n", " ⏎ ")[:160]
        print(f"\n[{tag}] gold={LETTERS[gold]}  Q: {q[:90]}")
        print(f"  CLEAN  : {clean!r}")
        print(f"  ABLATED: {abl!r}")

    print("=" * 80)
    print(f"GENERATION clean vs ablate {domain} L{L} coords={coords.tolist()} (nshot={a.nshot})")
    print("=" * 80)
    print("\n########## TARGET (econ) ##########")
    for (q, o, g) in tgt:
        show("econ", tgt_prefix, q, o, g)
    print("\n########## GENERAL ##########")
    for (c, (q, o, g)) in gen_items:
        show(c, gprefix[c], q, o, g)
    hook.remove()


if __name__ == "__main__":
    main()
