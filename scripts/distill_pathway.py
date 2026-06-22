"""Stage 2: pathway self-distillation (student->teacher OT projection).

Per (student, knowledge):
  1. L_S = student's best-AUROC layer; load cached teacher target (L_T, pooled
     acts, pathway neurons) from stage 1.
  2. z-score both sides (train stats); fit a FROZEN OT coupling M: student->teacher
     (Sinkhorn on 1-corr neuron cost).  [knowledge-fit OK in this direction:
     target a_T is the real teacher, so closing the residual = importing teacher.]
  3. Train the student (full params, AdamW) to minimize
        ||  z(a_S(x)) @ M  [pathway]  -  z(a_T(x))[pathway] ||^2   +   lambda*KL(student||orig)
     KL-to-frozen-original is the collapse leash.
  4. Eval ORIGINAL vs DISTILLED: MMLU accuracy (math/medical) or perplexity (language).

Run: PYTHONPATH=src python3 scripts/distill_pathway.py --student Llama-3.2-1B --knowledge topic_medical
"""
from __future__ import annotations
import argparse, os, json, glob
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.capture import _get_hook_module

DATA = "/share1/zhlu6105/know_trans_data"; FS = os.path.join(DATA, "feature_scores")
PW = os.path.join(DATA, "pathways"); TG = os.path.join(PW, "distill")
OUT = ensure_dir(os.path.join(PW, "distill_results"))
log = get_logger("distill")
LETTERS = ["A", "B", "C", "D"]

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--knowledge", required=True)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lam-kl", type=float, default=1.0)
ap.add_argument("--direction", choices=["s2t", "t2s"], default="s2t")
a = ap.parse_args()
SPATH = f"/share1/zhlu6105/models/{a.student}"


class Grab:
    def __init__(self, mod): self.out = None; self.h = mod.register_forward_hook(self._f)
    def _f(self, m, i, o): self.out = o[0] if isinstance(o, (tuple, list)) else o
    def remove(self): self.h.remove()


def pool(act, am):  # [B,S,d],[B,S] -> [B,d]
    m = am.float().unsqueeze(-1)
    return (act * m).sum(1) / m.sum(1).clamp_min(1)


def sinkhorn(C, eps=0.05, iters=200):
    dS, dT = C.shape
    K = torch.exp(-C / eps)
    a = torch.full((dS,), 1.0 / dS, device=C.device); b = torch.full((dT,), 1.0 / dT, device=C.device)
    v = torch.ones(dT, device=C.device)
    for _ in range(iters):
        u = a / (K @ v).clamp_min(1e-30)
        v = b / (K.t() @ u).clamp_min(1e-30)
    Pi = u[:, None] * K * v[None, :]
    return Pi / Pi.sum(0, keepdim=True).clamp_min(1e-30)   # M: column(teacher)-normalized [dS,dT]


@torch.no_grad()
def mmlu_acc(model, tok, prompts, gold, lid):
    tok.padding_side = "left"; preds = []
    for i in range(0, len(prompts), 4):
        e = tok(prompts[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=384)
        e = {k: v.to(model.device) for k, v in e.items()}
        preds += model(**e).logits[:, -1, lid].argmax(-1).cpu().tolist()
    return float((np.array(preds) == np.array(gold)).mean())


@torch.no_grad()
def perplexity(model, tok, sents):
    tok.padding_side = "right"; tot, ntok = 0.0, 0
    for i in range(0, len(sents), 4):
        e = tok(sents[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=128)
        e = {k: v.to(model.device) for k, v in e.items()}
        logits = model(**e).logits[:, :-1].float(); tgt = e["input_ids"][:, 1:]
        m = e["attention_mask"][:, 1:].bool()
        ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
        ll = ll[m.reshape(-1)]; tot += ll.sum().item(); ntok += ll.numel()
    return float(np.exp(tot / max(ntok, 1)))


# ---- load cached teacher target + split ----
tg = np.load(os.path.join(TG, f"{a.knowledge}.npz"))
meta = json.load(open(os.path.join(TG, f"{a.knowledge}.json")))
L_T = int(tg["L_T"]); path_idx = torch.tensor(tg["path_idx"], dtype=torch.long)
aT = torch.tensor(tg["acts"].astype(np.float32))           # [N_tr, d_T]
train_in = meta["train"]; eval_in = meta["eval"]; kind = meta["kind"]

# ---- student L_S (+ student pathway for t2s) ----
adf = pd.read_parquet(os.path.join(FS, f"{a.student}_alllayer.parquet"))
sub = adf[adf["concept"] == a.knowledge]
path_S = None
if a.direction == "t2s":
    sfis = pd.read_parquet(os.path.join(PW, f"{a.student}_{a.knowledge}_fis.parquet"))
    sfl = sorted(int(l) for l in sfis["layer"].unique())
    sub_f = sub[sub["layer"].isin(sfl)]
    L_S = int(sub_f.loc[sub_f["auc"].idxmax(), "layer"])
    path_S = torch.tensor(sfis[sfis["layer"] == L_S].nlargest(int(len(path_idx)), "fis")["neuron"].to_numpy(), dtype=torch.long)
else:
    L_S = int(sub.loc[sub["auc"].idxmax(), "layer"])
log.info("[%s/%s] dir=%s L_S=%d L_T=%d n_train=%d", a.student, a.knowledge, a.direction, L_S, L_T, len(train_in))

model, tok = load_model_and_tokenizer(SPATH, dtype="bfloat16", device="cuda")
orig, _ = load_model_and_tokenizer(SPATH, dtype="bfloat16", device="cuda")
for p in orig.parameters(): p.requires_grad_(False)
orig.eval()
tok.pad_token = tok.pad_token or tok.eos_token
lid = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
gb = Grab(_get_hook_module(model, L_S, "mlp")); go = Grab(_get_hook_module(orig, L_S, "mlp"))
dev = model.device

@torch.no_grad()
def student_acts(mdl, grab, inputs):
    tok.padding_side = "right"; out = []
    for i in range(0, len(inputs), 8):
        e = tok(inputs[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: v.to(dev) for k, v in e.items()}; mdl(**e); out.append(pool(grab.out.float(), e["attention_mask"]).cpu())
    return torch.cat(out)

aS0 = student_acts(orig, go, train_in)                     # [N_tr, d_S]
# z-stats (train)
muS, sdS = aS0.mean(0), aS0.std(0).clamp_min(1e-4)
muT, sdT = aT.mean(0), aT.std(0).clamp_min(1e-4)
zS0 = ((aS0 - muS) / sdS).to(dev)                          # [N_tr, d_S]
zTf = ((aT - muT) / sdT).to(dev)                           # [N_tr, d_T]
muS, sdS = muS.to(dev), sdS.to(dev)
N = len(aS0)

if a.direction == "s2t":   # map student->teacher, target = real teacher pathway
    M = sinkhorn((1.0 - (zS0.t() @ zTf) / N))              # [d_S, d_T]
    pidx = path_idx.to(dev)
    tgt_path = zTf[:, pidx]                                # [N, k]
    def predict(zS): return (zS @ M)[:, pidx]
else:                      # map teacher->student, target = mapped teacher at student pathway
    M = sinkhorn((1.0 - (zTf.t() @ zS0) / N))             # [d_T, d_S]
    pidx = path_S.to(dev)
    tgt_path = (zTf @ M)[:, pidx]                          # [N, k]  (mapped teacher, frozen)
    def predict(zS): return zS[:, pidx]
log.info("OT fit dir=%s: M=%s  pathway=%d", a.direction, tuple(M.shape), len(pidx))

def proj_dist(grab, mdl):  # mean ||predict(z(aS)) - tgt|| over train (no grad)
    aS = student_acts(mdl, grab, train_in).to(dev)
    pred = predict((aS - muS) / sdS)
    return float(((pred - tgt_path) ** 2).mean().item())

# ---- eval BEFORE ----
gold = meta["eval_gold"]
metric_name = "acc" if kind == "mmlu" else "ppl"
before = mmlu_acc(orig, tok, eval_in, gold, lid) if kind == "mmlu" else perplexity(orig, tok, eval_in)
d_before = proj_dist(gb, model)

# ---- train ----
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
model.train(); order = np.arange(len(train_in)); step = 0; g = torch.Generator().manual_seed(0)
while step < a.steps:
    perm = torch.randperm(len(train_in), generator=g).numpy()
    for s in range(0, len(perm), a.batch):
        idx = perm[s:s+a.batch]; bin_ = [train_in[i] for i in idx]
        tok.padding_side = "right"
        e = tok(bin_, return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: v.to(dev) for k, v in e.items()}
        out = model(**e); aS = pool(gb.out.float(), e["attention_mask"])     # [b,d_S]
        pred = predict((aS - muS) / sdS)                                      # [b,k]
        loss_proj = ((pred - tgt_path[idx]) ** 2).mean()
        with torch.no_grad():
            lo = orig(**e).logits.float()
        ls = out.logits.float()
        m = e["attention_mask"].bool()
        kl = (F.softmax(ls, -1) * (F.log_softmax(ls, -1) - F.log_softmax(lo, -1))).sum(-1)[m].mean()
        loss = loss_proj + a.lam_kl * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1
        if step % 25 == 0:
            log.info("  step %d  proj=%.4f kl=%.4f", step, float(loss_proj), float(kl))
        if step >= a.steps: break
model.eval()

# ---- eval AFTER ----
after = mmlu_acc(model, tok, eval_in, gold, lid) if kind == "mmlu" else perplexity(model, tok, eval_in)
d_after = proj_dist(gb, model)
res = {"student": a.student, "knowledge": a.knowledge, "direction": a.direction, "kind": kind,
       "L_S": L_S, "L_T": L_T, "metric": metric_name, "before": round(before, 4), "after": round(after, 4),
       "proj_before": round(d_before, 4), "proj_after": round(d_after, 4),
       "n_eval": len(eval_in), "steps": a.steps, "lr": a.lr, "lam_kl": a.lam_kl}
save_json(res, os.path.join(OUT, f"{a.student}_{a.knowledge}_{a.direction}.json"))
log.info("[%s/%s] %s: %.4f -> %.4f | proj %.4f -> %.4f",
         a.student, a.knowledge, metric_name, before, after, d_before, d_after)
print(json.dumps(res))
