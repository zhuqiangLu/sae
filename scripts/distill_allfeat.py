"""All-SAE-feature pathway distillation (path-only) — isolates the SPACE variable.

Instead of matching dense FIS pathway neurons, match the FULL 16384-dim SAE
feature code (sparsity selects the knowledge-relevant features for free). The
cross-family bridge is a SQUARE 16384<->16384 OT alignment over the two
(matched-size) SAE dictionaries. Still path-only (no task anchor) — the controlled
test of whether the dense-pathway null was caused by the space (predicted: no).

direction s2t (default): map student features -> teacher features, target = real
teacher feature code (frozen).  t2s: map teacher -> student, target = mapped teacher.

Run: PYTHONPATH=src python3 scripts/distill_allfeat.py --student Llama-3.2-1B --knowledge topic_medical
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.capture import _get_hook_module

DATA = "/share1/zhlu6105/know_trans_data"; FS = os.path.join(DATA, "feature_scores")
PW = os.path.join(DATA, "pathways"); TG = os.path.join(PW, "distill")
OUT = ensure_dir(os.path.join(PW, "distill_results")); log = get_logger("distill_af")
TNAME = "Llama-3.1-8B"; LETTERS = ["A", "B", "C", "D"]

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--knowledge", required=True)
ap.add_argument("--direction", choices=["s2t", "t2s"], default="s2t")
ap.add_argument("--steps", type=int, default=150); ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lam-kl", type=float, default=1.0)
a = ap.parse_args()
SPATH = f"/share1/zhlu6105/models/{a.student}"


class Grab:
    def __init__(self, mod): self.out = None; self.h = mod.register_forward_hook(self._f)
    def _f(self, m, i, o): self.out = o[0] if isinstance(o, (tuple, list)) else o
    def remove(self): self.h.remove()

def pool(act, am): m = am.float().unsqueeze(-1); return (act * m).sum(1) / m.sum(1).clamp_min(1)

def sinkhorn(C, eps=0.05, iters=200):
    dS, dT = C.shape; K = torch.exp(-C / eps)
    aa = torch.full((dS,), 1.0 / dS, device=C.device); bb = torch.full((dT,), 1.0 / dT, device=C.device)
    v = torch.ones(dT, device=C.device)
    for _ in range(iters):
        u = aa / (K @ v).clamp_min(1e-30); v = bb / (K.t() @ u).clamp_min(1e-30)
    Pi = u[:, None] * K * v[None, :]
    return Pi / Pi.sum(0, keepdim=True).clamp_min(1e-30)

@torch.no_grad()
def mmlu_acc(model, tok, prompts, gold, lid):
    tok.padding_side = "left"; pr = []
    for i in range(0, len(prompts), 4):
        e = tok(prompts[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=384)
        e = {k: v.to(model.device) for k, v in e.items()}; pr += model(**e).logits[:, -1, lid].argmax(-1).cpu().tolist()
    return float((np.array(pr) == np.array(gold)).mean())

@torch.no_grad()
def perplexity(model, tok, sents):
    tok.padding_side = "right"; tot, n = 0.0, 0
    for i in range(0, len(sents), 4):
        e = tok(sents[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=128)
        e = {k: v.to(model.device) for k, v in e.items()}
        lg = model(**e).logits[:, :-1].float(); tg = e["input_ids"][:, 1:]; mk = e["attention_mask"][:, 1:].bool()
        ll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="none")[mk.reshape(-1)]
        tot += ll.sum().item(); n += ll.numel()
    return float(np.exp(tot / max(n, 1)))


# ---- cached teacher dense acts at L_T + split ----
tg = np.load(os.path.join(TG, f"{a.knowledge}.npz")); meta = json.load(open(os.path.join(TG, f"{a.knowledge}.json")))
L_T = int(tg["L_T"]); aT = torch.tensor(tg["acts"].astype(np.float32))
train_in, eval_in, kind, gold = meta["train"], meta["eval"], meta["kind"], meta["eval_gold"]

# ---- teacher features z_T (encode cached acts thru teacher SAE), then free teacher SAE ----
tsae = SAEBundle.load(os.path.join(DATA, "saes", TNAME, "seed0"))[L_T]
with torch.no_grad():
    p = next(tsae.parameters()); zT = tsae.encode_dense(aT.to(p.device, p.dtype)).float().cpu()   # [N,16384]
del tsae; torch.cuda.empty_cache()

# ---- student L_S, SAE, models ----
adf = pd.read_parquet(os.path.join(FS, f"{a.student}_alllayer.parquet"))
sub = adf[adf["concept"] == a.knowledge]; L_S = int(sub.loc[sub["auc"].idxmax(), "layer"])
ssae = SAEBundle.load(os.path.join(DATA, "saes", a.student, "seed0"))[L_S]
for q in ssae.parameters(): q.requires_grad_(False)
log.info("[%s/%s] dir=%s allfeat L_S=%d L_T=%d H=%d", a.student, a.knowledge, a.direction, L_S, L_T, zT.shape[1])

model, tok = load_model_and_tokenizer(SPATH, dtype="bfloat16", device="cuda")
orig, _ = load_model_and_tokenizer(SPATH, dtype="bfloat16", device="cuda")
for q in orig.parameters(): q.requires_grad_(False)
orig.eval(); tok.pad_token = tok.pad_token or tok.eos_token
lid = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
gb = Grab(_get_hook_module(model, L_S, "mlp")); go = Grab(_get_hook_module(orig, L_S, "mlp"))
dev = model.device; ssae = ssae.to(dev)

@torch.no_grad()
def student_feats(mdl, grab, inputs):
    tok.padding_side = "right"; out = []
    for i in range(0, len(inputs), 8):
        e = tok(inputs[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: v.to(dev) for k, v in e.items()}; mdl(**e)
        pooled = pool(grab.out.float(), e["attention_mask"])
        out.append(ssae.encode_dense(pooled.to(next(ssae.parameters()).dtype)).float().cpu())
    return torch.cat(out)                                       # [N,16384]

zS0 = student_feats(orig, go, train_in)                         # [N,16384]
muS, sdS = zS0.mean(0), zS0.std(0).clamp_min(1e-4)
muT, sdT = zT.mean(0), zT.std(0).clamp_min(1e-4)
zS0z = ((zS0 - muS) / sdS).to(dev); zTz = ((zT - muT) / sdT).to(dev); N = len(zS0)
muS, sdS = muS.to(dev), sdS.to(dev)

if a.direction == "s2t":
    P = sinkhorn((1.0 - (zS0z.t() @ zTz) / N))                 # [H,H] student->teacher
    tgt = zTz                                                   # [N,H] real teacher (frozen)
    def predict(zfeat): return zfeat @ P
else:
    P = sinkhorn((1.0 - (zTz.t() @ zS0z) / N))                 # [H,H] teacher->student
    tgt = (zTz @ P)                                             # [N,H] mapped teacher (frozen)
    def predict(zfeat): return zfeat
log.info("OT(feat) P=%s", tuple(P.shape))

def zfeat_of(grab, mdl_feats):  # helper: z-scored student feature code [N,H]
    return ((mdl_feats - muS) / sdS)

def proj_dist(mdl, grab):
    zf = (student_feats(mdl, grab, train_in).to(dev) - muS) / sdS
    return float(((predict(zf) - tgt) ** 2).mean().item())

metric = "acc" if kind == "mmlu" else "ppl"
before = mmlu_acc(orig, tok, eval_in, gold, lid) if kind == "mmlu" else perplexity(orig, tok, eval_in)
d_before = proj_dist(model, gb)

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
model.train(); step = 0; g = torch.Generator().manual_seed(0)
while step < a.steps:
    perm = torch.randperm(N, generator=g).numpy()
    for s in range(0, N, a.batch):
        idx = perm[s:s+a.batch]; bin_ = [train_in[i] for i in idx]
        tok.padding_side = "right"
        e = tok(bin_, return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: v.to(dev) for k, v in e.items()}
        out = model(**e); pooled = pool(gb.out.float(), e["attention_mask"])
        zf = (ssae.encode_dense(pooled.to(next(ssae.parameters()).dtype)).float() - muS) / sdS   # [b,H]
        loss_proj = ((predict(zf) - tgt[idx]) ** 2).mean()
        with torch.no_grad(): lo = orig(**e).logits.float()
        ls = out.logits.float(); m = e["attention_mask"].bool()
        kl = (F.softmax(ls, -1) * (F.log_softmax(ls, -1) - F.log_softmax(lo, -1))).sum(-1)[m].mean()
        loss = loss_proj + a.lam_kl * kl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1
        if step % 25 == 0: log.info("  step %d proj=%.4f kl=%.4f", step, float(loss_proj), float(kl))
        if step >= a.steps: break
model.eval()

after = mmlu_acc(model, tok, eval_in, gold, lid) if kind == "mmlu" else perplexity(model, tok, eval_in)
d_after = proj_dist(model, gb)
res = {"student": a.student, "knowledge": a.knowledge, "direction": a.direction, "target": "allfeat",
       "kind": kind, "L_S": L_S, "L_T": L_T, "metric": metric, "before": round(before, 4), "after": round(after, 4),
       "proj_before": round(d_before, 4), "proj_after": round(d_after, 4), "n_eval": len(eval_in)}
save_json(res, os.path.join(OUT, f"{a.student}_{a.knowledge}_{a.direction}_allfeat.json"))
log.info("[%s/%s] allfeat %s: %.4f -> %.4f | proj %.4f -> %.4f", a.student, a.knowledge, metric, before, after, d_before, d_after)
print(json.dumps(res))
