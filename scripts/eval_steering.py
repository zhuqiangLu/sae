"""SUFFICIENCY test (go/no-go for pathway distillation): does ADDING the
SAE-identified knowledge pathway raise task accuracy? (Necessity we proved by
ablation; this tests the converse.)

Steering vector = normalized sum of the medical detector features' decoder
directions (W_dec columns = feature directions in dense space), added to the
MLP output at the knowledge's best AUROC layer, scaled by alpha * mean token
MLP-output norm. Conditions: clean / steer(alpha sweep) / random-direction control.
Metric: MMLU professional_medicine accuracy.

If steered > clean AND > random for some alpha -> the pathway is injectable
(distillation worth building). If flat -> capability isn't a steerable direction.

Run: PYTHONPATH=src python3 scripts/eval_steering.py
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, save_json
from know_trans.sae import SAEBundle
from know_trans.capture import MLPHook, _get_hook_module

DATA = "/share1/zhlu6105/know_trans_data"; FS = os.path.join(DATA, "feature_scores")
PW = os.path.join(DATA, "pathways"); BENCH = "/share1/zhlu6105/benchmarks/MMLU"
log = get_logger("steer")

MODELS = [("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
          ("Qwen3-0.6B", "/share1/zhlu6105/models/Qwen3-0.6B"),
          ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B")]
C, SUBJ = "topic_medical", "professional_medicine"
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0]
KFEAT = 16
LETTERS = ["A", "B", "C", "D"]


class SteerHook:
    """Add a fixed vector to a module's (MLP) output."""
    def __init__(self, module):
        self.vec = None  # [d] on device, or None=off
        self._h = module.register_forward_hook(self._hook)
    def _hook(self, m, i, o):
        if self.vec is None: return o
        out = o[0] if isinstance(o, (tuple, list)) else o
        out = out + self.vec.to(out.dtype)
        return (out, *o[1:]) if isinstance(o, (tuple, list)) else out
    def remove(self): self._h.remove()


def prompt(q, ch):
    return ("The following is a multiple choice question. Answer with the letter of "
            "the correct option.\n\n" + q + "\n" +
            "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ch)) + "\nAnswer:")


@torch.no_grad()
def accuracy(model, tok, prompts, gold, lid, batch=4):
    lid_t = torch.tensor(lid, device=model.device); preds = []
    for i in range(0, len(prompts), batch):
        e = tok(prompts[i:i+batch], return_tensors="pt", padding=True, truncation=True, max_length=384)
        e = {k: v.to(model.device) for k, v in e.items()}
        cand = model(**e).logits[:, -1, :].index_select(-1, lid_t)
        preds += cand.argmax(-1).cpu().tolist()
    torch.cuda.empty_cache()
    return float((np.array(preds) == np.array(gold)).mean())


results = {}
for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    adf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer.parquet"))
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    lid = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]

    df = pd.read_parquet(glob.glob(os.path.join(BENCH, SUBJ, "test-*.parquet"))[0])
    prompts = [prompt(r["question"], list(r["choices"])) for _, r in df.iterrows()]
    gold = [int(a) for a in df["answer"]]

    # medical detector features at best AUROC layer
    sub = adf[adf["concept"] == C]
    bestL = int(sub.loc[sub["auc"].idxmax(), "layer"])
    feats = sub[sub["layer"] == bestL].nlargest(KFEAT, "auc")["feature"].to_numpy()
    sae = bundle[bestL]; Wdec = sae.W_dec.detach().float()  # [d, H] unit-norm cols
    d_unit = Wdec[:, feats].sum(1); d_unit = (d_unit / d_unit.norm()).to(model.device)
    g = torch.Generator().manual_seed(0)
    r_unit = torch.randn(Wdec.shape[0], generator=g); r_unit = (r_unit / r_unit.norm()).to(model.device)

    # mean MLP-output token norm at bestL (scale reference)
    rd = MLPHook(_get_hook_module(model, bestL, "mlp"), bestL, to_cpu=False)
    norms = []
    for i in range(0, min(len(prompts), 64), 4):
        e = tok(prompts[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=384)
        e = {k: v.to(model.device) for k, v in e.items()}; model(**e)
        a = rd.pop().float(); m = e["attention_mask"].bool()
        norms.append(a[m].norm(dim=-1).mean().item())
    rd.remove()
    base_norm = float(np.mean(norms))

    sh = SteerHook(_get_hook_module(model, bestL, "mlp"))
    res = {"bestL": bestL, "base_norm": round(base_norm, 2), "n": len(gold), "steer": {}, "rand": {}}
    for a in ALPHAS:
        sh.vec = (a * base_norm) * d_unit if a > 0 else None
        res["steer"][str(a)] = round(accuracy(model, tok, prompts, gold, lid), 3)
        sh.vec = (a * base_norm) * r_unit if a > 0 else None
        res["rand"][str(a)] = round(accuracy(model, tok, prompts, gold, lid), 3)
    sh.vec = None; sh.remove()
    results[mname] = res
    log.info("[%s] L%d clean=%.3f steer=%s rand=%s", mname, bestL,
             res["steer"]["0.0"], res["steer"], res["rand"])
    del model; torch.cuda.empty_cache()

save_json(results, os.path.join(PW, "steering_sufficiency.json"))
print("\n" + "=" * 78)
print(f"SUFFICIENCY: steer the medical pathway -> MMLU {SUBJ} accuracy (chance=0.25)")
print("=" * 78)
for mname, _ in MODELS:
    r = results[mname]
    print(f"\n#### {mname} (best L{r['bestL']}, n={r['n']}) ####")
    print(f"{'alpha':8s} " + " ".join(f"{a:>7.1f}" for a in ALPHAS))
    print(f"{'steer':8s} " + " ".join(f"{r['steer'][str(a)]:>7.3f}" for a in ALPHAS))
    print(f"{'random':8s} " + " ".join(f"{r['rand'][str(a)]:>7.3f}" for a in ALPHAS))
print("\nsaved -> pathways/steering_sufficiency.json")
