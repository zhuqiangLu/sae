"""necessity_sweep.py — is a subject's knowledge carried by a SPARSE set of MLP
coords, or is it distributed? (Distinguishes "route method too weak" from
"knowledge is distributed", independent of any SAE route.)

Method (per domain, base Llama-3.1-8B):
  1. Rank every MLP-output coord at each band layer by DOMAIN DIFFERENTIAL
     activation: standardized |mean_target - mean_general| (Cohen-d-like),
     computed on CALIBRATION items disjoint from the eval items.
  2. Sweep k (coords ablated per layer): for each k, zero the top-k domain-ranked
     coords at every band layer and measure target-category vs general-category
     few-shot MMLU-Pro accuracy. Compare against a uniform-RANDOM top-k control.
  3. If target accuracy drops SELECTIVELY (much more than general, and much more
     than the random control) at some k -> a sparse necessary set exists (method
     can be improved). If even huge k only hurts target and general together ->
     knowledge is distributed.

Reuses the scorer + ablation primitive from route_eval_selectivity.py.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/necessity_sweep.py --domain econ
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, save_json, ensure_dir
from know_trans.cspt import ZeroOutHook
from know_trans.capture import MLPHook, _get_hook_module
from scripts.route_eval_selectivity import (
    DOMAIN_TARGET, GENERAL_CATS, LETTERS, MODEL_PATH,
    load_mmlu_pro, _format_q, build_fewshot_prefix, accuracy,
)

log = get_logger("necessity_sweep")


def parse_layers(spec: str) -> list[int]:
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi) + 1))


@torch.no_grad()
def rank_coords(model, tok, target_prompts, general_prompts, layers, batch=8, max_len=512):
    """Per layer: rank MLP-output coords by standardized |E[a|target]-E[a|general]|.

    Returns {layer: np.ndarray[int] coord indices sorted by descending domain
    differential} and the raw score array per layer.
    """
    def stats(prompts):
        hooks = {L: MLPHook(_get_hook_module(model, L, "mlp"), L, to_cpu=False) for L in layers}
        tok.padding_side = "right"
        s = {L: None for L in layers}      # sum over tokens
        sq = {L: None for L in layers}     # sum of squares
        n = 0
        for i in range(0, len(prompts), batch):
            e = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
            e = {k: v.to(model.device) for k, v in e.items()}
            model(**e)
            m = e["attention_mask"].bool().reshape(-1)
            cnt = int(m.sum().item())
            for j, L in enumerate(layers):
                a = hooks[L].pop().float()
                d = a.shape[-1]
                af = a.reshape(-1, d)[m]              # [ntok, d]
                ssum = af.sum(0)
                ssq = (af * af).sum(0)
                s[L] = ssum if s[L] is None else s[L] + ssum
                sq[L] = ssq if sq[L] is None else sq[L] + ssq
            n += cnt
            torch.cuda.empty_cache()
        for h in hooks.values():
            h.remove()
        mean = {L: (s[L] / max(n, 1)).cpu().numpy() for L in layers}
        var = {L: (sq[L] / max(n, 1)).cpu().numpy() - mean[L] ** 2 for L in layers}
        return mean, var

    tmean, tvar = stats(target_prompts)
    gmean, gvar = stats(general_prompts)
    ranking, scores = {}, {}
    for L in layers:
        pooled = np.sqrt(0.5 * (np.clip(tvar[L], 0, None) + np.clip(gvar[L], 0, None)) + 1e-6)
        sc = np.abs(tmean[L] - gmean[L]) / pooled
        scores[L] = sc
        ranking[L] = np.argsort(-sc).astype(np.int64)
    return ranking, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["econ", "math", "med"], required=True)
    ap.add_argument("--layers", default="8-31", help="band of layers to ablate")
    ap.add_argument("--ks", default="16,64,256,1024,4096", help="coords/layer to ablate")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--general", default="history,law,philosophy,psychology", help="general cats")
    ap.add_argument("--out-dir", default="report/diag")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.smoke:
        a.n_eval, a.n_calib = 20, 32
        a.ks = "16,256"
        a.general = "history,law"

    layers = parse_layers(a.layers)
    ks = [int(x) for x in a.ks.split(",")]
    general_cats = a.general.split(",")
    concept_name, target_cat = DOMAIN_TARGET[a.domain]
    out_dir = ensure_dir(a.out_dir)
    rng = np.random.default_rng(a.seed)

    # ---- data: split MMLU-Pro test into [eval | calib] (disjoint) ----------
    tgt_test, _ = load_mmlu_pro(target_cat)
    gen_test = {c: load_mmlu_pro(c)[0] for c in general_cats}
    tgt_eval, tgt_calib = tgt_test[:a.n_eval], tgt_test[a.n_eval:a.n_eval + a.n_calib]
    gen_eval = {c: gen_test[c][:a.n_eval] for c in general_cats}
    gen_calib = []
    for c in general_cats:
        gen_calib += gen_test[c][a.n_eval:a.n_eval + a.n_calib // len(general_cats) + 1]

    log.info("[%s] target=%s eval=%d calib=%d | general=%s",
             a.domain, target_cat, len(tgt_eval), len(tgt_calib), general_cats)

    # ---- model -------------------------------------------------------------
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    d_model = model.config.hidden_size
    letter_ids = torch.tensor(
        [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS],
        device=model.device)

    # few-shot prefixes (exemplars from calibration slice, disjoint from eval)
    tgt_prefix = build_fewshot_prefix(tgt_calib, a.nshot)
    gen_prefix = {c: build_fewshot_prefix(gen_test[c][a.n_eval:a.n_eval + a.nshot + 2], a.nshot)
                  for c in general_cats}

    # ---- rank coords by domain differential (on calibration items) ---------
    tgt_calib_txt = [_format_q(q, o) for (q, o, _ai) in tgt_calib]
    gen_calib_txt = [_format_q(q, o) for (q, o, _ai) in gen_calib]
    ranking, _scores = rank_coords(model, tok, tgt_calib_txt, gen_calib_txt, layers,
                                   batch=a.batch_size)

    # ---- ablation hooks on the band ----------------------------------------
    hooks = {L: ZeroOutHook(_get_hook_module(model, L, "mlp"), np.array([0], dtype=np.int64))
             for L in layers}

    def set_coords(coords_by_layer):
        for L, h in hooks.items():
            if L in coords_by_layer and len(coords_by_layer[L]):
                h.set_coords(coords_by_layer[L]); h.enabled = True
            else:
                h.enabled = False

    def eval_all():
        t, _ = accuracy(model, tok, tgt_eval, tgt_prefix, letter_ids, batch=a.batch_size)
        gs = [accuracy(model, tok, gen_eval[c], gen_prefix[c], letter_ids, batch=a.batch_size)[0]
              for c in general_cats]
        return t, float(np.nanmean(gs))

    results = {"domain": a.domain, "target_category": target_cat, "general": general_cats,
               "layers": a.layers, "ks": ks, "n_eval": a.n_eval, "nshot": a.nshot,
               "rows": {}}

    # clean
    set_coords({})
    ct, cg = eval_all()
    results["clean"] = {"target": ct, "general": cg}
    log.info("[%s] clean target=%.3f general=%.3f", a.domain, ct, cg)

    for k in ks:
        ranked = {L: ranking[L][:min(k, d_model)] for L in layers}
        rand = {L: rng.choice(d_model, size=min(k, d_model), replace=False).astype(np.int64)
                for L in layers}
        set_coords(ranked); rt, rg = eval_all()
        set_coords(rand); zt, zg = eval_all()
        results["rows"][str(k)] = {"ranked_target": rt, "ranked_general": rg,
                                   "random_target": zt, "random_general": zg}
        log.info("[%s] k=%-5d ranked: t=%.3f g=%.3f | random: t=%.3f g=%.3f",
                 a.domain, k, rt, rg, zt, zg)

    for h in hooks.values():
        h.remove()

    out_path = os.path.join(out_dir, f"necessity_{a.domain}.json")
    save_json(results, out_path)

    # ---- table -------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"NECESSITY SWEEP  domain={a.domain} ({target_cat})  layers={a.layers}  "
          f"n_eval={a.n_eval} nshot={a.nshot}  chance=0.10")
    print(f"general={general_cats}")
    print("=" * 76)
    print(f"{'k/layer':>8}{'ranked_tgt':>12}{'ranked_gen':>12}{'rand_tgt':>10}{'rand_gen':>10}"
          f"{'tgt_drop':>10}{'gen_drop':>10}")
    print(f"{'clean':>8}{ct:>12.3f}{cg:>12.3f}{'-':>10}{'-':>10}{'-':>10}{'-':>10}")
    for k in ks:
        r = results["rows"][str(k)]
        print(f"{k:>8}{r['ranked_target']:>12.3f}{r['ranked_general']:>12.3f}"
              f"{r['random_target']:>10.3f}{r['random_general']:>10.3f}"
              f"{ct - r['ranked_target']:>+10.3f}{cg - r['ranked_general']:>+10.3f}")
    print("-" * 76)
    print("SELECTIVE if at some k: ranked tgt_drop >> gen_drop AND >> random's target drop.")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
