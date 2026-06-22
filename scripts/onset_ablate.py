"""onset_ablate.py — ablate the located onset SOURCE neurons and measure selectivity.

Reads a source json (from onset_locate.py) = {onset_layer, source_neurons} and the
val/test split. Ablates those dense coords at the onset layer (ZeroOutHook on .mlp)
and measures few-shot MMLU-Pro accuracy on the TEST split (target = domain category)
vs GENERAL categories. Controls: clean + magnitude-matched random coords (same count,
same layer).

Headline: target_drop = clean_target - ablate_target (want LARGE & >> random),
general_drop (want ~0).

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.onset_ablate \
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
    layer_magnitudes, random_matched,
)

log = get_logger("onset_ablate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--n-target", type=int, default=675, help="cap on test items")
    ap.add_argument("--n-general", type=int, default=200, help="per general cat")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="report/diag")
    a = ap.parse_args()

    src = json.load(open(a.src_json))
    domain = src["domain"]
    L = int(src["onset_layer"])
    coords = np.array(sorted(int(c) for c in src["source_neurons"]), dtype=np.int64)
    concept_name, target_cat = DOMAIN_TARGET[domain]
    split = json.load(open(src["split_path"]))
    test_idx, val_idx = split["test_idx"], split["val_idx"]
    out_dir = ensure_dir(a.out_dir)
    general_cats = GENERAL_CATS[:4]

    # ---- data ---------------------------------------------------------------
    econ_rows, _ = load_mmlu_pro(target_cat)
    tgt_test = [econ_rows[i] for i in test_idx][:a.n_target]
    tgt_fewshot = [econ_rows[i] for i in val_idx]                     # disjoint from test
    gen_rows = {c: load_mmlu_pro(c)[0] for c in general_cats}
    log.info("[%s] L%d ablate %d coords=%s | target test=%d general=%s",
             domain, L, len(coords), coords.tolist(), len(tgt_test), general_cats)

    # ---- model --------------------------------------------------------------
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = torch.tensor(
        [tok.encode(" " + ch, add_special_tokens=False)[-1] for ch in LETTERS],
        device=model.device)
    d_model = model.config.hidden_size

    tgt_prefix = build_fewshot_prefix(tgt_fewshot, a.nshot)
    gen_prefix = {c: build_fewshot_prefix(gen_rows[c][-(a.nshot + 2):], a.nshot)
                  for c in general_cats}
    gen_eval = {c: gen_rows[c][:a.n_general] for c in general_cats}

    # ---- magnitude-matched random control at the onset layer ----------------
    neutral = []
    for c in general_cats:
        neutral += [_format_q(q, o) for (q, o, _g) in gen_rows[c][:8]]
    mags = layer_magnitudes(model, tok, neutral, [L], n=48)
    route = {L: coords}
    rand = random_matched(route, mags, d_model, seed=a.seed)

    hook = ZeroOutHook(_get_hook_module(model, L, "mlp"), np.array([0], dtype=np.int64))

    def run():
        t, _ = accuracy(model, tok, tgt_test, tgt_prefix, letter_ids, batch=a.batch_size)
        gs = [accuracy(model, tok, gen_eval[c], gen_prefix[c], letter_ids,
                       batch=a.batch_size)[0] for c in general_cats]
        return t, float(np.nanmean(gs))

    conds = {"clean": None, "ablate": coords, "ablate_random": rand[L]}
    res = {"domain": domain, "onset_layer": L, "coords": coords.tolist(),
           "k": len(coords), "target_category": target_cat,
           "general": general_cats, "n_target": len(tgt_test), "nshot": a.nshot,
           "src_json": a.src_json, "target_acc": {}, "general_acc": {}}
    for name, c in conds.items():
        if c is None:
            hook.enabled = False
        else:
            hook.set_coords(c); hook.enabled = True
        t, g = run()
        res["target_acc"][name] = t
        res["general_acc"][name] = g
        log.info("[%s L%d] %-14s target=%.3f general=%.3f", domain, L, name, t, g)
    hook.remove()

    ct, cg = res["target_acc"]["clean"], res["general_acc"]["clean"]
    res["headline"] = {
        "target_drop": round(ct - res["target_acc"]["ablate"], 4),
        "general_drop": round(cg - res["general_acc"]["ablate"], 4),
        "target_drop_random": round(ct - res["target_acc"]["ablate_random"], 4),
        "general_drop_random": round(cg - res["general_acc"]["ablate_random"], 4),
        "clean_target": ct,
    }
    out = os.path.join(out_dir, f"onset_ablate_{domain}_L{L}.json")
    save_json(res, out)

    print("\n" + "=" * 64)
    print(f"ONSET ABLATION  domain={domain} L{L}  k={len(coords)} coords={coords.tolist()}")
    print(f"target=econ TEST (n={len(tgt_test)})  general={general_cats}  chance=0.10")
    print("=" * 64)
    print(f"{'condition':16s}{'target':>10s}{'general':>10s}")
    for name in conds:
        print(f"{name:16s}{res['target_acc'][name]:>10.3f}{res['general_acc'][name]:>10.3f}")
    h = res["headline"]
    print("-" * 64)
    print(f"target_drop        = {h['target_drop']:+.3f}   (want LARGE)")
    print(f"general_drop       = {h['general_drop']:+.3f}   (want ~0)")
    print(f"target_drop_random = {h['target_drop_random']:+.3f}   (want ~0)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
