"""route_eval_selectivity.py — selectivity of a subject route on MMLU-Pro.

The core eval: ABLATING a subject's route should DROP that subject's test
accuracy while leaving GENERAL (unrelated-subject) accuracy ~unchanged. A
magnitude-matched RANDOM control rules out "any ablation of that size hurts".

We reuse the route format (dense MLP-output coords per layer, from
route_extract.py) and the cspt.ZeroOutHook ablation primitive. The only new
machinery here is a few-shot MMLU-Pro multiple-choice scorer (MMLU-Pro has up to
10 options A-J), scored by the log-prob of each option's letter under a few-shot
prompt (nshot exemplars from validation; eval on test).

Conditions, each for TARGET (the domain's MMLU-Pro category) and GENERAL (the
mean over a fixed set of unrelated categories):
  * clean          — no intervention
  * ablate_route   — zero the route's dense coords at every route layer
  * ablate_random  — magnitude-matched random coords (same per-layer count),
                     sampled to match the route's activation magnitude on
                     neutral text (mirrors eval_selectivity.py random_matched)

Headline: target_drop = clean_target - ablate_route_target (want LARGE),
general_drop = clean_general - ablate_route_general (want ~0), and the random
control (want ~0 for both).

Run (full):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/route_eval_selectivity.py \
      --domain econ --method greedy --bench mmlu_pro --nshot 5 --n-eval 200

Smoke (tiny):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/route_eval_selectivity.py \
      --domain econ --n-eval 20 --nshot 5 --smoke
"""
from __future__ import annotations

import argparse
import glob
import os
import string

import numpy as np
import pandas as pd
import torch

torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, save_json, ensure_dir
from know_trans.cspt import ZeroOutHook
from know_trans.capture import MLPHook, _get_hook_module

log = get_logger("route_selectivity")

MODEL = "Llama-3.1-8B"
MODEL_PATH = "models/Llama-3.1-8B"
LETTERS = list(string.ascii_uppercase[:10])  # A..J (MMLU-Pro: up to 10 options)

# domain -> (route concept, MMLU-Pro target category)
DOMAIN_TARGET = {
    "econ": ("topic_economics", "economics"),
    "math": ("topic_math", "math"),
    "med": ("topic_medical", "health"),
}
# fixed unrelated set for the GENERAL average (MMLU-Pro categories)
GENERAL_CATS = ["history", "law", "philosophy", "psychology", "biology", "business"]

MMLU_PRO = "benchmarks/MMLU-Pro/data"
MMLU = "benchmarks/MMLU"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _mmlu_pro_split(split: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(MMLU_PRO, f"{split}-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no MMLU-Pro {split} parquet in {MMLU_PRO}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_mmlu_pro(category: str):
    """Return (test_rows, val_rows) for one MMLU-Pro category. Each row is
    (question, options[list], answer_index[int])."""
    def _rows(df):
        sub = df[df["category"] == category]
        out = []
        for _, r in sub.iterrows():
            out.append((str(r["question"]), list(r["options"]), int(r["answer_index"])))
        return out
    return _rows(_mmlu_pro_split("test")), _rows(_mmlu_pro_split("validation"))


def load_mmlu(subject: str):
    """MMLU fallback: (test, val) rows for one subject (4 options)."""
    def _rows(splits):
        frames = []
        for s in splits:
            for f in glob.glob(os.path.join(MMLU, subject, f"{s}-*.parquet")):
                frames.append(pd.read_parquet(f))
        if not frames:
            return []
        df = pd.concat(frames, ignore_index=True)
        return [(str(r["question"]), list(r["choices"]), int(r["answer"]))
                for _, r in df.iterrows()]
    return _rows(["test"]), _rows(["validation", "dev"])


# --------------------------------------------------------------------------- #
# Few-shot MC prompt + log-prob scorer
# --------------------------------------------------------------------------- #
def _format_q(question: str, options: list, with_answer: int | None = None) -> str:
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    s = f"Question: {question}\nOptions:\n{body}\nAnswer:"
    if with_answer is not None:
        s += f" {LETTERS[with_answer]}\n\n"
    return s


def build_fewshot_prefix(exemplars, nshot: int) -> str:
    head = ("The following are multiple choice questions. "
            "Answer with the letter of the correct option.\n\n")
    shots = exemplars[:nshot]
    return head + "".join(_format_q(q, o, ai) for (q, o, ai) in shots)


@torch.no_grad()
def accuracy(model, tok, rows, prefix, letter_ids, batch=8, max_len=2048):
    """Few-shot MC accuracy by argmax over option-letter next-token logits.

    rows: list of (question, options, gold_index). Only the letters that exist
    for each item (len(options)) compete, so 4- and 10-option items are handled.
    """
    if not rows:
        return float("nan"), 0
    tok.padding_side = "left"
    preds, gold = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [prefix + _format_q(q, o) for (q, o, _ai) in chunk]
        nopt = [len(o) for (_q, o, _ai) in chunk]
        gold.extend(ai for (_q, _o, ai) in chunk)
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :]  # [B, vocab]
        cand = logits.index_select(-1, letter_ids)  # [B, 10]
        for b in range(len(chunk)):
            row = cand[b].clone()
            if nopt[b] < len(LETTERS):  # mask letters beyond this item's options
                row[nopt[b]:] = float("-inf")
            preds.append(int(row.argmax().item()))
        del logits, cand
        torch.cuda.empty_cache()
    acc = float((np.array(preds) == np.array(gold)).mean())
    return acc, len(gold)


# --------------------------------------------------------------------------- #
# Route + magnitude-matched random control
# --------------------------------------------------------------------------- #
def load_route(route_dir, domain, method, smoke):
    suffix = "_smoke" if smoke else ""
    path = os.path.join(route_dir, f"{MODEL}_{domain}_{method}_nodes{suffix}.parquet")
    if not os.path.exists(path):  # fall back to non-smoke nodes if present
        alt = os.path.join(route_dir, f"{MODEL}_{domain}_{method}_nodes.parquet")
        if os.path.exists(alt):
            path = alt
    nodes = pd.read_parquet(path)
    route = {int(L): sub["neuron"].to_numpy().astype(np.int64)
             for L, sub in nodes.groupby("layer")}
    return route, path


@torch.no_grad()
def layer_magnitudes(model, tok, prompts, layers, n=48, batch=8, max_len=512):
    """Mean |MLP-output| per coordinate at each route layer (for mag-matched random)."""
    hooks = {L: MLPHook(_get_hook_module(model, L, "mlp"), L, to_cpu=False) for L in layers}
    tok.padding_side = "right"
    summ = {L: None for L in layers}
    ntok = 0
    for i in range(0, min(len(prompts), n), batch):
        e = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len)
        e = {k: v.to(model.device) for k, v in e.items()}
        model(**e)
        m = e["attention_mask"].bool().reshape(-1)
        first = True
        for L in layers:
            a = hooks[L].pop().float()
            d = a.shape[-1]
            af = a.reshape(-1, d)[m].abs().sum(0)
            summ[L] = af if summ[L] is None else summ[L] + af
            if first:
                ntok += int(m.sum().item()); first = False
        torch.cuda.empty_cache()
    for h in hooks.values():
        h.remove()
    return {L: (summ[L] / max(ntok, 1)).cpu().numpy() for L in layers}


def random_matched(route, mags, d_model, seed=0):
    """Per layer: same #coords as the route, random coords with magnitude >= the
    route nodes' min magnitude at that layer (magnitude-matched control)."""
    rng = np.random.default_rng(seed)
    out = {}
    for L, nodes in route.items():
        k = len(nodes)
        mag = mags[L]
        thr = float(np.min(mag[nodes])) if k else 0.0
        alln = np.arange(d_model)
        pool = alln[(mag >= thr) & (~np.isin(alln, nodes))]
        if len(pool) < k:
            pool = alln[~np.isin(alln, nodes)]
        out[L] = rng.choice(pool, size=min(k, len(pool)), replace=False).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=["econ", "math", "med"], required=True)
    ap.add_argument("--method", choices=["greedy", "star"], default="greedy")
    ap.add_argument("--bench", choices=["mmlu_pro", "mmlu"], default="mmlu_pro")
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=200, help="items per category")
    ap.add_argument("--route-dir", default="data/pathways_subject")
    ap.add_argument("--out-dir", default="report/diag")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: also shrinks general set + caps eval")
    a = ap.parse_args()

    if a.smoke:
        a.n_eval = min(a.n_eval, 20)

    concept_name, target_cat = DOMAIN_TARGET[a.domain]
    general_cats = GENERAL_CATS[:3] if a.smoke else GENERAL_CATS
    out_dir = ensure_dir(a.out_dir)

    # ---- load eval data ------------------------------------------------------
    loader = (lambda c: load_mmlu_pro(c)) if a.bench == "mmlu_pro" else (lambda c: load_mmlu(c))
    # map MMLU-Pro general cat names to MMLU subjects if --bench mmlu (best-effort)
    target_test, target_val = loader(target_cat)
    gen_data = {c: loader(c) for c in general_cats}

    def cap_test(rows):
        return rows[:a.n_eval]

    log.info("[%s] target=%s test=%d val=%d | general=%s",
             a.domain, target_cat, len(target_test), len(target_val), general_cats)
    if len(target_val) < a.nshot:
        log.warning("only %d validation exemplars for %s (< nshot=%d)",
                    len(target_val), target_cat, a.nshot)

    # ---- model ---------------------------------------------------------------
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    d_model = model.config.hidden_size
    # token id of " A".." J" (leading space, as the model emits after "Answer:")
    letter_ids = torch.tensor(
        [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS],
        device=model.device)

    # few-shot prefixes: target uses its own validation exemplars; each general
    # category uses its own validation exemplars (so the shots are in-domain).
    target_prefix = build_fewshot_prefix(target_val, a.nshot)
    gen_prefix = {c: build_fewshot_prefix(gen_data[c][1], a.nshot) for c in general_cats}

    # ---- route + controls ----------------------------------------------------
    route, route_path = load_route(a.route_dir, a.domain, a.method, a.smoke)
    route_layers = sorted(route)
    log.info("[%s] route %s | %d layers, sizes=%s", a.domain, route_path,
             len(route_layers), {int(L): int(len(v)) for L, v in route.items()})

    # magnitude prepass on neutral text (general categories' test questions)
    neutral = []
    for c in general_cats:
        neutral += [_format_q(q, o) for (q, o, _ai) in gen_data[c][0][:8]]
    mags = layer_magnitudes(model, tok, neutral, route_layers, n=48)
    rand = random_matched(route, mags, d_model, seed=a.seed)

    # one ZeroOutHook per route layer; per condition set coords + enable
    hooks = {L: ZeroOutHook(_get_hook_module(model, L, "mlp"), np.array([0], dtype=np.int64))
             for L in route_layers}

    def set_condition(coords_by_layer):
        for L, h in hooks.items():
            if L in coords_by_layer and len(coords_by_layer[L]):
                h.set_coords(coords_by_layer[L]); h.enabled = True
            else:
                h.enabled = False

    conditions = {"clean": {}, "ablate_route": route, "ablate_random": rand}

    # ---- run all conditions --------------------------------------------------
    tgt_test = cap_test(target_test)
    results = {"domain": a.domain, "concept": concept_name, "method": a.method,
               "bench": a.bench, "nshot": a.nshot, "n_eval": a.n_eval,
               "target_category": target_cat, "general_categories": general_cats,
               "route_path": route_path,
               "route_sizes": {int(L): int(len(v)) for L, v in route.items()},
               "target_acc": {}, "general_acc": {}, "general_breakdown": {}}

    for cond, coords in conditions.items():
        set_condition(coords)
        t_acc, t_n = accuracy(model, tok, tgt_test, target_prefix, letter_ids,
                              batch=a.batch_size)
        gen_accs = {}
        for c in general_cats:
            g_acc, _ = accuracy(model, tok, cap_test(gen_data[c][0]),
                                gen_prefix[c], letter_ids, batch=a.batch_size)
            gen_accs[c] = g_acc
        g_mean = float(np.nanmean(list(gen_accs.values())))
        results["target_acc"][cond] = t_acc
        results["general_acc"][cond] = g_mean
        results["general_breakdown"][cond] = gen_accs
        log.info("[%s] %-14s target=%.3f general=%.3f", a.domain, cond, t_acc, g_mean)

    for h in hooks.values():
        h.remove()
    results["target_n"] = len(tgt_test)

    # ---- headline drops ------------------------------------------------------
    ct, cg = results["target_acc"]["clean"], results["general_acc"]["clean"]
    chance = round(1.0 / 10.0 if a.bench == "mmlu_pro" else 0.25, 3)
    results["chance"] = chance
    # "informative" = clean target meaningfully above chance AND enough items
    # for the gap to be measurable (a tiny n gives a noisy estimate that can even
    # invert the drop sign, as the smoke run does).
    informative = bool(ct > chance + 0.15 and results["target_n"] >= 100)
    results["headline"] = {
        "target_drop_route": round(ct - results["target_acc"]["ablate_route"], 4),
        "general_drop_route": round(cg - results["general_acc"]["ablate_route"], 4),
        "target_drop_random": round(ct - results["target_acc"]["ablate_random"], 4),
        "general_drop_random": round(cg - results["general_acc"]["ablate_random"], 4),
        "clean_above_chance": bool(ct > chance + 0.05),
        "informative": informative,
    }

    out_path = os.path.join(out_dir, f"route_selectivity_{a.domain}_{a.method}.json")
    save_json(results, out_path)

    # ---- print table ---------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"ROUTE SELECTIVITY  domain={a.domain} ({target_cat}) method={a.method} "
          f"bench={a.bench} nshot={a.nshot}")
    print(f"target_n={results['target_n']} general={general_cats} chance={chance}")
    print("=" * 72)
    print(f"{'condition':16s}{'target_acc':>14s}{'general_acc':>14s}")
    for cond in conditions:
        print(f"{cond:16s}{results['target_acc'][cond]:>14.3f}"
              f"{results['general_acc'][cond]:>14.3f}")
    h = results["headline"]
    print("-" * 72)
    print(f"target_drop (route)   = {h['target_drop_route']:+.3f}   (want LARGE)")
    print(f"general_drop (route)  = {h['general_drop_route']:+.3f}   (want ~0)")
    print(f"target_drop (random)  = {h['target_drop_random']:+.3f}   (want ~0)")
    print(f"general_drop (random) = {h['general_drop_random']:+.3f}   (want ~0)")
    if not h["informative"]:
        print(f"\nWARNING: this run is likely UNINFORMATIVE — clean target acc "
              f"{ct:.3f} (chance {chance}) on n={results['target_n']}. Use n-eval "
              f">=100 and confirm clean target acc is well above chance before "
              f"trusting the drops.")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
