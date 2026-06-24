"""sae_neuron_ablate.py — disable critical (ΔWFS) SAE neurons at the onset layer and
measure subject vs general accuracy drop. Pure SAE-feature-space intervention: no
dense ablation, no AUROC, no sink screening (per user).

Per subject, at its onset layer l* (from onset_split_{domain}.json):
  1. LOCATE: rank the 28672 SAE features by ΔWFS = WFS_subj - WFS_gen
     (token-level f*mu contrast: subject VAL questions vs general MMLU subjects).
  2. DISABLE: subtract the top-k features' contribution from the down_proj INPUT
     (forward_pre_hook): x' = x - Σ_{j∈crit} z_j(x)·W_dec[:,j]  -- only those
     features removed; b_pre / other features / residual kept exact (no SAE
     reconstruction-error confound).
  3. MEASURE on held-out TEST: subject (split test) vs general (other subjects),
     clean -> disabled; k-sweep + matched random-feature control.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m scripts.sae_neuron_ablate
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
from know_trans.capture import MLPHook, _get_hook_module
from know_trans.sae import TopKSAE
from know_trans.cspt import elbow_k

log = get_logger("sae_ablate")
MODEL = "Llama-3.1-8B"
MODEL_PATH = "models/Llama-3.1-8B"
BENCH = "benchmarks/MMLU"
LETTERS = ["A", "B", "C", "D"]
GENERAL = ["professional_law", "high_school_world_history", "philosophy", "high_school_psychology"]
KS = [1, 4, 16, 64]


def fmt(q, ch):
    s = q.strip() + "\n"
    for i, c in enumerate(ch):
        s += f"{LETTERS[i]}. {str(c).strip()}\n"
    return s + "Answer:"


def load_subject_rows(subj):
    rows, seen = [], set()
    for sp in ("test", "validation", "dev"):
        for p in glob.glob(f"{BENCH}/{subj}/{sp}-*.parquet"):
            df = pd.read_parquet(p)
            if "question" not in df.columns or "answer" not in df.columns:
                continue
            for q, ch, a in zip(df["question"], df["choices"], df["answer"]):
                ch = list(ch)
                if isinstance(q, str) and len(ch) == 4 and q.strip() not in seen:
                    seen.add(q.strip()); rows.append((q.strip(), [str(c) for c in ch], int(a)))
    return rows


def fewshot(rows, n):
    return "".join(fmt(q, ch) + f" {LETTERS[a]}\n\n" for (q, ch, a) in rows[:n])


class SAEFeatureDisableHook:
    """forward_pre_hook on down_proj: subtract selected SAE features' contribution
    from the down_proj INPUT (disable those features)."""
    def __init__(self, down_proj, sae):
        self.sae = sae; self.feats = torch.zeros(0, dtype=torch.long); self.enabled = False
        self._h = down_proj.register_forward_pre_hook(self._pre)

    def set_feats(self, feats):
        self.feats = torch.as_tensor(np.ascontiguousarray(feats), dtype=torch.long)

    @torch.no_grad()
    def _pre(self, module, args):
        if not self.enabled or self.feats.numel() == 0:
            return None
        x = args[0]; shape = x.shape; dt = x.dtype
        p = next(self.sae.parameters()); feats = self.feats.to(p.device)
        xf = x.reshape(-1, shape[-1])
        z = self.sae.encode_dense(xf.to(p.device, p.dtype))         # [N, H]
        contrib = z.index_select(1, feats).float() @ self.sae.W_dec.index_select(1, feats).float().t()
        new = (xf - contrib.to(device=x.device, dtype=dt)).reshape(shape)
        return (new,) + tuple(args[1:])

    def remove(self):
        self._h.remove()


@torch.no_grad()
def wfs(model, tok, sae, L, texts, batch, max_len):
    """Per-feature WFS = f*mu over all real tokens of `texts` at layer L (down_proj_in)."""
    hook = MLPHook(_get_hook_module(model, L, "down_proj_in"), L, to_cpu=False, capture_input=True)
    p = next(sae.parameters())
    cnt = ssum = None; ntok = 0
    for tb in batched(list(texts), batch):
        enc = tok(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
        attn = enc["attention_mask"]; model(**enc)
        af = hook.pop().float().reshape(-1, sae.d_in)[attn.bool().reshape(-1)]
        z = sae.encode_dense(af.to(p.device, p.dtype)).float()      # [T, H]
        act = (z > 0).float()
        cnt = act.sum(0) if cnt is None else cnt + act.sum(0)
        ssum = (z * act).sum(0) if ssum is None else ssum + (z * act).sum(0)
        ntok += af.shape[0]
    hook.remove()
    cnt = cnt.cpu().numpy(); ssum = ssum.cpu().numpy()
    f = cnt / max(ntok, 1); mu = np.where(cnt > 0, ssum / np.maximum(cnt, 1.0), 0.0)
    return f * mu


@torch.no_grad()
def accuracy(model, tok, rows, prefix, letter_ids, batch=8, max_len=1024):
    if not rows:
        return float("nan")
    cor = tot = 0
    for tb in batched(rows, batch):
        prompts = [prefix + fmt(q, ch) for (q, ch, _a) in tb]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        pick = logits[:, letter_ids].argmax(1)
        for j, (_q, _ch, a) in enumerate(tb):
            cor += int(pick[j].item() == a); tot += 1
    return cor / max(tot, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", default="econ,math,med")
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--n-general", type=int, default=80, help="general test Qs per subject")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out_dir = ensure_dir("report/diag")

    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype="bfloat16", device="cuda")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = torch.tensor([tok.encode(" " + c, add_special_tokens=False)[-1] for c in LETTERS],
                              device=model.device)
    gen_rows = {s: load_subject_rows(s) for s in GENERAL}

    summary = {}
    for domain in [d for d in a.domains.split(",") if d.strip()]:
        split = json.load(open(f"data/pathways_subject/mmlu_{domain}_split.json"))
        subject = split["subject"]
        L = int(json.load(open(f"report/diag/onset_split_{domain}.json"))["onset"])
        sae = TopKSAE.load(f"data/saes_subject/{MODEL}/{domain}/layer{L}.safetensors", device="cuda")

        # ---- locate: ΔWFS at onset layer (val subject vs general) ----
        val_txt = [fmt(q, ch) for (q, ch, _a) in [(r["question"], r["choices"], r["answer"]) for r in split["val"]]]
        gen_txt = []
        for s in GENERAL:
            gen_txt += [fmt(q, ch) for (q, ch, _a) in gen_rows[s][:len(split["val"]) // len(GENERAL) + 1]]
        wfs_s = wfs(model, tok, sae, L, val_txt, a.batch_size, 384)
        wfs_g = wfs(model, tok, sae, L, gen_txt, a.batch_size, 384)
        dwfs = wfs_s - wfs_g
        order = np.argsort(dwfs)[::-1].copy()   # contiguous (avoid negative-stride -> torch)
        log.info("[%s/%s] onset L%d | top ΔWFS feats=%s (Δ=%s)", domain, subject, L,
                 order[:8].tolist(), np.round(dwfs[order[:8]], 4).tolist())

        # ---- eval sets ----
        tgt_test = [(r["question"], r["choices"], r["answer"]) for r in split["test"]]
        tgt_prefix = fewshot([(r["question"], r["choices"], r["answer"]) for r in split["train"]], a.nshot)
        gen_eval = {s: gen_rows[s][:a.n_general] for s in GENERAL}
        gen_prefix = {s: fewshot(gen_rows[s][-a.nshot - 2:], a.nshot) for s in GENERAL}

        hook = SAEFeatureDisableHook(model.model.layers[L].mlp.down_proj, sae)

        def evalall():
            t = accuracy(model, tok, tgt_test, tgt_prefix, letter_ids, batch=a.batch_size)
            gs = [accuracy(model, tok, gen_eval[s], gen_prefix[s], letter_ids, batch=a.batch_size) for s in GENERAL]
            return t, float(np.nanmean(gs))

        # ---- elbow-K on the ΔWFS descending curve (positive head = subject-selective dir) ----
        dwfs_desc = dwfs[order]
        n_pos = int((dwfs_desc > 0).sum())
        head = dwfs_desc[:max(n_pos, 4)]
        K = int(elbow_k(head, kmin=2, kmax=256))
        crit = order[:K]
        log.info("[%s L%d] elbow-K=%d (n_pos=%d, ΔWFS #1=%.4f #K=%.4f #K+1=%.4f)", domain, L, K, n_pos,
                 float(dwfs_desc[0]), float(dwfs_desc[K - 1]), float(dwfs_desc[min(K, len(dwfs_desc) - 1)]))

        hook.enabled = False
        ct, cg = evalall()
        log.info("[%s L%d] clean target=%.3f general=%.3f", domain, L, ct, cg)
        hook.set_feats(crit); hook.enabled = True
        t, g = evalall()
        pool = np.setdiff1d(np.arange(sae.d_hidden), crit)
        rnd = np.sort(rng.choice(pool, size=K, replace=False))
        hook.set_feats(rnd)
        tr, gr = evalall()
        hook.enabled = False; hook.remove()
        log.info("[%s L%d] elbow-K=%d disable: tgt=%.3f(drop%+.3f) gen=%.3f(drop%+.3f) | rand tgt=%.3f gen=%.3f",
                 domain, L, K, t, ct - t, g, cg - g, tr, gr)
        summary[domain] = {"subject": subject, "onset": L, "elbow_K": K, "n_pos": n_pos,
                           "clean": {"target": ct, "general": cg},
                           "disable": {"target": t, "general": g, "tgt_drop": round(ct - t, 4),
                                       "gen_drop": round(cg - g, 4)},
                           "random": {"target": tr, "general": gr},
                           "critical_feats": crit.tolist()}
        save_json(summary[domain], os.path.join(out_dir, f"sae_ablate_{domain}.json"))

    print("\n" + "#" * 72)
    print("SAE-NEURON DISABLE (ΔWFS, elbow-K, onset layer)  chance=0.25")
    print("#" * 72)
    for d, s in summary.items():
        dd, rr = s["disable"], s["random"]
        print(f"\n=== {d} ({s['subject']}) L{s['onset']}  elbow-K={s['elbow_K']} (n_pos={s['n_pos']}) ===")
        print(f"  clean   : tgt={s['clean']['target']:.3f} gen={s['clean']['general']:.3f}")
        print(f"  disable : tgt={dd['target']:.3f}({dd['tgt_drop']:+.3f}) gen={dd['general']:.3f}({dd['gen_drop']:+.3f})")
        print(f"  random  : tgt={rr['target']:.3f} gen={rr['general']:.3f}")


if __name__ == "__main__":
    main()
