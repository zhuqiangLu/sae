"""onset_ablate_perneuron.py — ablate located onset source neurons ONE AT A TIME.

Given a source json (onset_locate.py) + val/test split, ablates each source neuron
INDIVIDUALLY (plus clean and the full set) at the onset layer, and measures few-shot
MMLU-Pro accuracy on the TEST split (target = domain category) vs GENERAL categories.

Distinguishes: (a) one econ-selective neuron, (b) each neuron globally critical,
(c) only-jointly catastrophic. For each single neuron we report target/general drop.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.onset_ablate_perneuron \
        --src-json data/pathways_subject/Llama-3.1-8B_econ_onset_src_L1.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, save_json, ensure_dir
from know_trans.cspt import ZeroOutHook
from know_trans.capture import _get_hook_module
from scripts.route_eval_selectivity import (
    MODEL, MODEL_PATH, DOMAIN_TARGET, GENERAL_CATS, LETTERS,
    load_mmlu_pro, _format_q, build_fewshot_prefix, accuracy,
)

log = get_logger("perneuron")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=300)
    ap.add_argument("--n-general", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default="report/diag")
    a = ap.parse_args()

    src = json.load(open(a.src_json))
    domain = src["domain"]
    L = int(src["onset_layer"])
    neurons = [int(c) for c in src["source_neurons"]]
    concept_name, target_cat = DOMAIN_TARGET[domain]
    split = json.load(open(src["split_path"]))
    test_idx, val_idx = split["test_idx"], split["val_idx"]
    out_dir = ensure_dir(a.out_dir)
    general_cats = GENERAL_CATS[:4]

    econ_rows, _ = load_mmlu_pro(target_cat)
    tgt_test = [econ_rows[i] for i in test_idx][:a.n_target]
    tgt_fewshot = [econ_rows[i] for i in val_idx]
    gen_rows = {c: load_mmlu_pro(c)[0] for c in general_cats}
    log.info("[%s] L%d neurons=%s | target test=%d general=%s",
             domain, L, neurons, len(tgt_test), general_cats)

    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = torch.tensor(
        [tok.encode(" " + ch, add_special_tokens=False)[-1] for ch in LETTERS],
        device=model.device)

    tgt_prefix = build_fewshot_prefix(tgt_fewshot, a.nshot)
    gen_prefix = {c: build_fewshot_prefix(gen_rows[c][-(a.nshot + 2):], a.nshot)
                  for c in general_cats}
    gen_eval = {c: gen_rows[c][:a.n_general] for c in general_cats}

    hook = ZeroOutHook(_get_hook_module(model, L, "mlp"), np.array([0], dtype=np.int64))

    def run():
        t, _ = accuracy(model, tok, tgt_test, tgt_prefix, letter_ids, batch=a.batch_size)
        gs = [accuracy(model, tok, gen_eval[c], gen_prefix[c], letter_ids,
                       batch=a.batch_size)[0] for c in general_cats]
        return t, float(np.nanmean(gs))

    # conditions: clean, each single neuron, full set
    conds = {"clean": None}
    for nrn in neurons:
        conds[f"n{nrn}"] = np.array([nrn], dtype=np.int64)
    if len(neurons) > 1:
        conds["ALL"] = np.array(neurons, dtype=np.int64)

    res = {"domain": domain, "onset_layer": L, "neurons": neurons,
           "target_category": target_cat, "general": general_cats,
           "n_target": len(tgt_test), "nshot": a.nshot, "rows": {}}
    for name, c in conds.items():
        if c is None:
            hook.enabled = False
        else:
            hook.set_coords(c); hook.enabled = True
        t, g = run()
        res["rows"][name] = {"target": t, "general": g}
        log.info("[%s L%d] %-10s target=%.3f general=%.3f", domain, L, name, t, g)
    hook.remove()

    ct = res["rows"]["clean"]["target"]; cg = res["rows"]["clean"]["general"]
    out = os.path.join(out_dir, f"perneuron_{domain}_L{L}.json")
    save_json(res, out)

    print("\n" + "=" * 60)
    print(f"PER-NEURON ABLATION  {domain} L{L}  (target=test n={len(tgt_test)}, chance=0.10)")
    print("=" * 60)
    print(f"{'cond':10s}{'target':>9s}{'general':>9s}{'tgt_drop':>10s}{'gen_drop':>10s}")
    for name in conds:
        r = res["rows"][name]
        td = ct - r["target"] if name != "clean" else 0.0
        gd = cg - r["general"] if name != "clean" else 0.0
        print(f"{name:10s}{r['target']:>9.3f}{r['general']:>9.3f}"
              f"{td:>+10.3f}{gd:>+10.3f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
