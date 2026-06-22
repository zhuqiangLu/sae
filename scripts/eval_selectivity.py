"""Route knowledge-selectivity: does each topic route CARRY that subject's
knowledge (necessity) WITHOUT damaging other/general knowledge (specificity)?

Ablate each topic's chain route (zero its dense nodes at every route layer) and
measure MMLU multiple-choice accuracy across target + general subjects, vs a
magnitude-matched random ablation control.

Targets are evaluated on HELD-OUT validation+dev (the routes were built on the
battery = test split, so test is contaminated and reported only as a secondary
high-N view). General subjects are evaluated on test (never used in any route).

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_selectivity.py
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, save_json
from know_trans.cspt import ZeroOutHook
from know_trans.capture import MLPHook, _get_hook_module

DATA = "/share1/zhlu6105/know_trans_data"
PW = os.path.join(DATA, "pathways")
BENCH = "/share1/zhlu6105/benchmarks/MMLU"
MODEL = "Llama-3.1-8B"; MPATH = f"/share1/zhlu6105/models/{MODEL}"
log = get_logger("selectivity")
LETTERS = ["A", "B", "C", "D"]

TARGETS = {"topic_math": "abstract_algebra",
           "topic_economics": "econometrics",
           "topic_medical": "professional_medicine"}
GENERAL = ["world_religions", "marketing", "high_school_geography",
           "sociology", "nutrition", "us_foreign_policy"]
HELDOUT_SPLITS = ["validation", "dev"]   # clean for targets (routes built on test)


def mc_prompt(q, ch):
    return ("The following is a multiple choice question. Answer with the letter of "
            "the correct option.\n\n" + q + "\n" +
            "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ch)) + "\nAnswer:")


def load_subject(subj, splits):
    frames = []
    for s in splits:
        for f in glob.glob(os.path.join(BENCH, subj, f"{s}-*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        return [], []
    df = pd.concat(frames, ignore_index=True)
    prompts = [mc_prompt(r["question"], list(r["choices"])) for _, r in df.iterrows()]
    gold = [int(a) for a in df["answer"]]
    return prompts, gold


@torch.no_grad()
def accuracy(model, tok, prompts, gold, lid_t, batch=8):
    if not prompts:
        return float("nan"), 0
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(prompts), batch):
        e = tok(prompts[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=384)
        e = {k: v.to(model.device) for k, v in e.items()}
        cand = model(**e).logits[:, -1, :].index_select(-1, lid_t)
        preds += cand.argmax(-1).cpu().tolist()
        torch.cuda.empty_cache()
    return float((np.array(preds) == np.array(gold)).mean()), len(gold)


ROUTE = "greedy"   # set by --route; "greedy" (causal+differential) or "chain" (magnitude)


def load_route(knowledge):
    """route nodes -> {layer: np.array(neuron coords)}."""
    nf = os.path.join(PW, f"{MODEL}_{knowledge}_{ROUTE}_nodes.parquet")
    nodes = pd.read_parquet(nf)
    return {int(L): sub["neuron"].to_numpy().astype(np.int64)
            for L, sub in nodes.groupby("layer")}


@torch.no_grad()
def layer_magnitudes(model, tok, prompts, layers, n=64, batch=8):
    """Mean |MLP-output| per coordinate at each layer (for magnitude-matched random)."""
    hooks = {L: MLPHook(_get_hook_module(model, L, "mlp"), L, to_cpu=False) for L in layers}
    tok.padding_side = "right"
    summ = {L: None for L in layers}; ntok = 0
    for i in range(0, min(len(prompts), n), batch):
        e = tok(prompts[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=384)
        e = {k: v.to(model.device) for k, v in e.items()}; model(**e)
        m = e["attention_mask"].bool().reshape(-1)
        first = True
        for L in layers:
            a = hooks[L].pop().float(); d = a.shape[-1]
            af = a.reshape(-1, d)[m].abs().sum(0)
            summ[L] = af if summ[L] is None else summ[L] + af
            if first:
                ntok += int(m.sum().item()); first = False
        torch.cuda.empty_cache()
    for h in hooks.values():
        h.remove()
    return {L: (summ[L] / max(ntok, 1)).cpu().numpy() for L in layers}


def random_matched(route, mags, d_model, seed=0):
    """Per layer: same count as route nodes, random coords with magnitude >= the
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


def main():
    model, tok = load_model_and_tokenizer(MPATH, dtype="bfloat16", device="cuda")
    tok.pad_token = tok.pad_token or tok.eos_token
    lid_t = torch.tensor([tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS],
                         device=model.device)
    d_model = model.config.hidden_size

    # routes + magnitude-matched random controls
    routes = {K: load_route(K) for K in TARGETS}
    all_layers = sorted({L for r in routes.values() for L in r})
    # magnitude prepass on a neutral mix (general subjects' test prompts)
    magmix = []
    for s in GENERAL:
        p, _ = load_subject(s, ["test"]); magmix += p[:12]
    mags = layer_magnitudes(model, tok, magmix, all_layers, n=64)
    rands = {K: random_matched(routes[K], mags, d_model, seed=0) for K in TARGETS}

    # eval datasets
    targets_heldout = {TARGETS[K]: load_subject(TARGETS[K], HELDOUT_SPLITS) for K in TARGETS}
    targets_test = {TARGETS[K]: load_subject(TARGETS[K], ["test"]) for K in TARGETS}
    general = {s: load_subject(s, ["test"]) for s in GENERAL}

    # one ZeroOutHook per layer; per condition set coords+enable on the right layers
    hooks = {L: ZeroOutHook(_get_hook_module(model, L, "mlp"), np.array([0], dtype=np.int64))
             for L in all_layers}

    def set_condition(coords_by_layer):
        for L, h in hooks.items():
            if L in coords_by_layer and len(coords_by_layer[L]):
                h.set_coords(coords_by_layer[L]); h.enabled = True
            else:
                h.enabled = False

    conditions = {
        "clean": {},
        "ablate_math": routes["topic_math"],
        "ablate_econ": routes["topic_economics"],
        "ablate_medical": routes["topic_medical"],
        "rand_math": rands["topic_math"],
        "rand_econ": rands["topic_economics"],
        "rand_medical": rands["topic_medical"],
    }

    res = {"heldout": {}, "test": {}, "route_sizes": {K: {int(L): int(len(v)) for L, v in routes[K].items()} for K in TARGETS}}
    for cond, coords in conditions.items():
        set_condition(coords)
        # targets on held-out
        row_h, row_t = {}, {}
        for K, subj in TARGETS.items():
            ph, gh = targets_heldout[subj]; row_h[subj] = accuracy(model, tok, ph, gh, lid_t)[0]
            pt, gt = targets_test[subj]; row_t[subj] = accuracy(model, tok, pt, gt, lid_t)[0]
        # general on test
        gen_acc = {}
        for s, (p, g) in general.items():
            gen_acc[s] = accuracy(model, tok, p, g, lid_t)[0]
        row_h["general_avg"] = float(np.nanmean(list(gen_acc.values())))
        row_t["general_avg"] = row_h["general_avg"]
        res["heldout"][cond] = row_h
        res["test"][cond] = {**row_t, "general_breakdown": gen_acc}
        log.info("[%s] heldout=%s gen_avg=%.3f", cond,
                 {k: round(v, 3) for k, v in row_h.items() if k != "general_avg"}, row_h["general_avg"])

    for h in hooks.values():
        h.remove()

    # held-out N per target
    res["heldout_n"] = {TARGETS[K]: len(targets_heldout[TARGETS[K]][1]) for K in TARGETS}
    res["test_n"] = {TARGETS[K]: len(targets_test[TARGETS[K]][1]) for K in TARGETS}
    res["route_type"] = ROUTE
    save_json(res, os.path.join(PW, f"selectivity_results_{ROUTE}.json"))

    # ---- print the matrix ----
    subj_cols = [TARGETS[k] for k in TARGETS] + ["general_avg"]
    print("\n" + "=" * 90)
    print("ROUTE SELECTIVITY — MMLU accuracy (targets=HELD-OUT val+dev; general=test). chance=0.25")
    print(f"held-out N: {res['heldout_n']}")
    print("=" * 90)
    hdr = f"{'condition':16s}" + "".join(f"{c[:14]:>15s}" for c in subj_cols)
    print(hdr)
    for cond in conditions:
        r = res["heldout"][cond]
        print(f"{cond:16s}" + "".join(f"{r[c]:>15.3f}" for c in subj_cols))
    print("\n(secondary, CONTAMINATED high-N test-split targets:)")
    print(f"{'condition':16s}" + "".join(f"{c[:14]:>15s}" for c in subj_cols))
    for cond in conditions:
        r = res["test"][cond]
        print(f"{cond:16s}" + "".join(f"{r[c]:>15.3f}" for c in subj_cols))
    print(f"\nsaved -> pathways/selectivity_results_{ROUTE}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="greedy", choices=["greedy", "chain", "star", "srconly"])
    a = ap.parse_args()
    ROUTE = a.route
    main()
