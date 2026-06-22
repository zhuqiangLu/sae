"""Stage 1 of pathway distillation: precompute & cache the FROZEN teacher targets.

Per knowledge: pick the teacher layer L_T (best AUROC among FIS-available layers),
the pathway neurons there (top-FIS dense neurons), and the teacher's pooled
activation at L_T on the TRAIN inputs. Also save the train/eval split so the
student stage uses identical inputs.

math/medical -> MMLU MC prompts (eval = accuracy); language -> Hungarian
sentences (eval = perplexity). Cached to pathways/distill/<knowledge>.{npz,json}.

Run: PYTHONPATH=src python3 scripts/distill_precompute.py
"""
from __future__ import annotations
import os, glob, json
import numpy as np
import pandas as pd
import torch

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir
from know_trans.concepts import load_battery
from know_trans.capture import MLPHook, _get_hook_module

DATA = "/share1/zhlu6105/know_trans_data"; FS = os.path.join(DATA, "feature_scores")
PW = os.path.join(DATA, "pathways"); OUT = ensure_dir(os.path.join(PW, "distill"))
BENCH = "/share1/zhlu6105/benchmarks/MMLU"
log = get_logger("distill_pre")
TEACHER = "/share1/zhlu6105/models/Llama-3.1-8B"; TNAME = "Llama-3.1-8B"
KPATH = 50; LETTERS = ["A", "B", "C", "D"]
KNOW = {"topic_math": ("mmlu", "abstract_algebra"),
        "topic_medical": ("mmlu", "professional_medicine"),
        "language_hu": ("lang", None)}


def mc_prompt(q, ch):
    return ("The following is a multiple choice question. Answer with the letter of "
            "the correct option.\n\n" + q + "\n" +
            "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ch)) + "\nAnswer:")


def build_items(kind, arg):
    """Return (train_inputs, eval_inputs, eval_gold). For mmlu inputs are MC prompts
    + gold letter ints; for lang inputs are raw sentences, gold=None."""
    if kind == "mmlu":
        df = pd.read_parquet(glob.glob(os.path.join(BENCH, arg, "test-*.parquet"))[0])
        prompts = [mc_prompt(r["question"], list(r["choices"])) for _, r in df.iterrows()]
        gold = [int(a) for a in df["answer"]]
        h = len(prompts) // 2
        return prompts[:h], prompts[h:], gold[h:]
    else:  # language: Hungarian sentences
        bat = {c.name: c for c in load_battery("data/concepts_pilot")}
        sents = [s for s in bat["language_hu"].positives if s.strip()]
        h = len(sents) // 2
        return sents[:h], sents[h:], None


@torch.no_grad()
def pooled_acts(model, tok, inputs, layer, max_len=256, batch=8):
    """Mean-pooled MLP-output activation at `layer`, one vector per input."""
    hook = MLPHook(_get_hook_module(model, layer, "mlp"), layer, to_cpu=False)
    out = []
    for i in range(0, len(inputs), batch):
        enc = tok(inputs[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}; am = enc["attention_mask"]
        model(**enc)
        a = hook.pop().float()                      # [B,S,d]
        m = am.float().unsqueeze(-1)
        pooled = (a * m).sum(1) / m.sum(1).clamp_min(1)   # [B,d]
        out.append(pooled.cpu())
    hook.remove()
    return torch.cat(out).numpy()


model, tok = load_model_and_tokenizer(TEACHER, dtype="bfloat16", device="cuda")
tok.padding_side = "right"; tok.pad_token = tok.pad_token or tok.eos_token
adf = pd.read_parquet(os.path.join(FS, f"{TNAME}_alllayer.parquet"))

for C, (kind, arg) in KNOW.items():
    fis = pd.read_parquet(os.path.join(PW, f"{TNAME}_{C}_fis.parquet"))
    fis_layers = sorted(int(l) for l in fis["layer"].unique())
    sub = adf[(adf["concept"] == C) & (adf["layer"].isin(fis_layers))]
    L_T = int(sub.loc[sub["auc"].idxmax(), "layer"])
    path_idx = fis[fis["layer"] == L_T].nlargest(KPATH, "fis")["neuron"].to_numpy().astype(np.int64)

    tr, ev, gold = build_items(kind, arg)
    acts = pooled_acts(model, tok, tr, L_T)        # [N_tr, d_T]
    np.savez(os.path.join(OUT, f"{C}.npz"),
             acts=acts.astype(np.float16), path_idx=path_idx, L_T=np.int64(L_T))
    json.dump({"knowledge": C, "kind": kind, "subject": arg, "L_T": L_T,
               "n_train": len(tr), "n_eval": len(ev), "kpath": int(len(path_idx)),
               "train": tr, "eval": ev, "eval_gold": gold},
              open(os.path.join(OUT, f"{C}.json"), "w"))
    log.info("[%s] L_T=%d kpath=%d n_train=%d n_eval=%d acts=%s",
             C, L_T, len(path_idx), len(tr), len(ev), acts.shape)

print("saved teacher targets -> pathways/distill/<knowledge>.{npz,json}")
