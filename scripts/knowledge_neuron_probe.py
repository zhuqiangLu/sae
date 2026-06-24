"""knowledge_neuron_probe.py — SAE-free knowledge-neuron localization + ablation for
ONE narrow MMLU subject, in the down_proj-INPUT space (14336-dim SwiGLU intermediate
= the MLP "neurons" / key-value-memory coefficients).

Locate: per-(layer,neuron) mean activation on subject tokens vs general (unrelated
subjects); differential = |mean_subj - mean_gen|. Pick the layer with the strongest
top-K separation; those top-K neurons are the candidates. No SAE involved.

Ablate: zero those neurons in the down_proj input (forward_pre_hook) and measure
4-choice MMLU accuracy on a HELD-OUT slice of the SAME subject (target, should drop)
vs unrelated subjects (general, should hold). Control: random-K neurons (matched).

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.knowledge_neuron_probe \
       --subject high_school_microeconomics
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, batched, ensure_dir, save_json

log = get_logger("kn_probe")
MODEL_PATH = "models/Llama-3.1-8B"
BENCH = "benchmarks/MMLU"
LETTERS = ["A", "B", "C", "D"]
GENERAL_SUBJECTS = ["professional_law", "philosophy", "high_school_biology",
                    "high_school_psychology"]


def load_subject(subj: str) -> list[tuple]:
    rows = []
    for sp in ("test", "validation", "dev"):
        for p in glob.glob(f"{BENCH}/{subj}/{sp}-*.parquet"):
            df = pd.read_parquet(p)
            if "question" not in df.columns or "answer" not in df.columns:
                continue
            for q, ch, a in zip(df["question"], df["choices"], df["answer"]):
                ch = list(ch)
                if isinstance(q, str) and len(ch) == 4:
                    rows.append((q, ch, int(a)))
    return rows


def fmt(q: str, ch: list) -> str:
    s = q.strip() + "\n"
    for i, c in enumerate(ch):
        s += f"{LETTERS[i]}. {str(c).strip()}\n"
    return s + "Answer:"


def fewshot(rows: list, n: int) -> str:
    return "".join(fmt(q, ch) + f" {LETTERS[a]}\n\n" for (q, ch, a) in rows[:n])


@torch.no_grad()
def accuracy(model, tok, rows, prefix, letter_ids, batch=8, max_len=1024) -> float:
    if not rows:
        return float("nan")
    cor = tot = 0
    for tb in batched(rows, batch):
        prompts = [prefix + fmt(q, ch) for (q, ch, _a) in tb]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        logits = model(**enc).logits[:, -1, :]          # left-padded -> last col
        pick = logits[:, letter_ids].argmax(1)
        for j, (_q, _ch, a) in enumerate(tb):
            cor += int(pick[j].item() == a); tot += 1
    return cor / max(tot, 1)


@torch.no_grad()
def neuron_means(model, tok, texts, layers, batch=8, max_len=384):
    """Per-(layer,neuron) mean of the down_proj input over all real tokens."""
    from know_trans.capture import MLPHook, _get_hook_module
    hooks = {L: MLPHook(_get_hook_module(model, L, "down_proj_in"), L,
                        to_cpu=False, capture_input=True) for L in layers}
    ssum = {L: None for L in layers}
    ntok = 0
    for tb in batched(texts, batch):
        enc = tok(tb, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        attn = enc["attention_mask"]
        model(**enc)
        m = attn.bool().reshape(-1)
        for L in layers:
            a = hooks[L].pop().float()
            bb, s, d = a.shape
            af = a.reshape(bb * s, d)[m]
            ssum[L] = af.sum(0) if ssum[L] is None else ssum[L] + af.sum(0)
        ntok += int(m.sum().item())
    for h in hooks.values():
        h.remove()
    return {L: (ssum[L] / max(ntok, 1)).cpu().numpy() for L in layers}, ntok


class NeuronAblateHook:
    """Zero selected down_proj-INPUT units (forward_pre_hook on down_proj)."""
    def __init__(self, down_proj, neurons):
        self.set(neurons); self.enabled = False
        self._h = down_proj.register_forward_pre_hook(self._pre)

    def set(self, neurons):
        self.neurons = torch.as_tensor(np.ascontiguousarray(neurons), dtype=torch.long)

    def _pre(self, module, args):
        if not self.enabled:
            return None
        x = args[0].clone()
        x[..., self.neurons.to(x.device)] = 0.0
        return (x,) + tuple(args[1:])

    def remove(self):
        self._h.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="high_school_microeconomics")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--n-general", type=int, default=80, help="test Qs per general subject")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="report/diag")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    subj_rows = load_subject(a.subject)
    perm = rng.permutation(len(subj_rows))
    n_test = int(round(a.test_frac * len(subj_rows)))
    test_rows = [subj_rows[i] for i in perm[:n_test]]
    loc_rows = [subj_rows[i] for i in perm[n_test:]]            # locate + few-shot
    gen_rows = {s: load_subject(s) for s in GENERAL_SUBJECTS}
    log.info("[%s] %d Qs -> locate=%d test=%d ; general subjects=%s",
             a.subject, len(subj_rows), len(loc_rows), len(test_rows), GENERAL_SUBJECTS)

    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = torch.tensor(
        [tok.encode(" " + c, add_special_tokens=False)[-1] for c in LETTERS],
        device=model.device)
    n_layers = int(model.config.num_hidden_layers)
    layers = list(range(n_layers))

    # ---- locate: differential mean activation, subject vs general ------------
    subj_txt = [fmt(q, ch) for (q, ch, _a) in loc_rows]
    gen_txt = []
    for s in GENERAL_SUBJECTS:
        gen_txt += [fmt(q, ch) for (q, ch, _a) in gen_rows[s][:len(loc_rows)//len(GENERAL_SUBJECTS)+1]]
    m_subj, n1 = neuron_means(model, tok, subj_txt, layers)
    m_gen, n2 = neuron_means(model, tok, gen_txt, layers)
    log.info("located over subj_tok=%d gen_tok=%d", n1, n2)

    best_L, best_score, best_neurons = None, -1, None
    per_layer = {}
    for L in layers:
        diff = np.abs(m_subj[L] - m_gen[L])
        topk = np.sort(diff)[::-1][:a.k]
        score = float(topk.sum())
        per_layer[L] = score
        if score > best_score:
            best_score, best_L = score, L
            best_neurons = np.argsort(diff)[::-1][:a.k].astype(np.int64)
    top5 = sorted(per_layer, key=per_layer.get, reverse=True)[:5]
    log.info("layer separability (top-K |Δmean| sum): best L%d; top-5 layers=%s",
             best_L, [(L, round(per_layer[L], 2)) for L in top5])
    neurons = np.sort(best_neurons)

    # ---- ablation eval at the chosen layer -----------------------------------
    tgt_prefix = fewshot(loc_rows, a.nshot)
    gen_eval = {s: gen_rows[s][:a.n_general] for s in GENERAL_SUBJECTS}
    gen_prefix = {s: fewshot(gen_rows[s][-a.nshot - 2:], a.nshot) for s in GENERAL_SUBJECTS}
    down_proj = model.model.layers[best_L].mlp.down_proj
    hook = NeuronAblateHook(down_proj, neurons)
    rand = np.sort(rng.choice(np.setdiff1d(np.arange(model.config.intermediate_size), neurons),
                              size=a.k, replace=False)).astype(np.int64)

    def evalall():
        t = accuracy(model, tok, test_rows, tgt_prefix, letter_ids)
        gs = [accuracy(model, tok, gen_eval[s], gen_prefix[s], letter_ids) for s in GENERAL_SUBJECTS]
        return t, float(np.nanmean(gs))

    res = {"subject": a.subject, "layer": best_L, "k": a.k, "neurons": neurons.tolist(),
           "n_test": len(test_rows), "n_layers": n_layers, "rows": {}}
    for name, n in [("clean", None), ("ablate", neurons), ("ablate_random", rand)]:
        if n is None:
            hook.enabled = False
        else:
            hook.set(n); hook.enabled = True
        t, g = evalall()
        res["rows"][name] = {"target": t, "general": g}
        log.info("[%s L%d] %-14s target=%.3f general=%.3f", a.subject, best_L, name, t, g)
    hook.remove()

    ct, cg = res["rows"]["clean"]["target"], res["rows"]["clean"]["general"]
    res["headline"] = {
        "target_drop": round(ct - res["rows"]["ablate"]["target"], 4),
        "general_drop": round(cg - res["rows"]["ablate"]["general"], 4),
        "target_drop_random": round(ct - res["rows"]["ablate_random"]["target"], 4),
        "general_drop_random": round(cg - res["rows"]["ablate_random"]["general"], 4),
    }
    out = os.path.join(ensure_dir(a.out_dir), f"kn_probe_{a.subject}.json")
    save_json(res, out)

    print("\n" + "=" * 64)
    print(f"KNOWLEDGE-NEURON PROBE  {a.subject}  (down_proj_in, L{best_L}, k={a.k})")
    print(f"target=held-out {a.subject} (n={len(test_rows)}, 4-choice chance=0.25)")
    print(f"general={GENERAL_SUBJECTS}")
    print("=" * 64)
    print(f"{'condition':16s}{'target':>10s}{'general':>10s}")
    for name in ("clean", "ablate", "ablate_random"):
        r = res["rows"][name]
        print(f"{name:16s}{r['target']:>10.3f}{r['general']:>10.3f}")
    h = res["headline"]
    print("-" * 64)
    print(f"target_drop        = {h['target_drop']:+.3f}   (want LARGE)")
    print(f"general_drop       = {h['general_drop']:+.3f}   (want ~0)")
    print(f"target_drop_random = {h['target_drop_random']:+.3f}   (want ~0)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
