"""Relaxed attention-divergence onset (TraceRouter §3.1 SPIRIT, feasible for
whole-prompt text batteries).

The paper locates the onset layer l* via a Sensitivity Score SS(l) built from
attention from sensitive-modifier tokens to target-entity nouns, minus a
contextual-disturbance term from minimal pairs. Our knowledge battery has neither
token roles nor minimal pairs, so we use a role-free / pair-free relaxation:

  * per prompt, per layer: mean attention ENTROPY (over heads & query positions)
    = how diffuse vs concentrated the attention pattern is.
  * attn-divergence(l) = ROC-AUC of that scalar separating the knowledge item's
    POSITIVE prompts from its NEGATIVE prompts (attention-pattern detectability).
  * onset_attn = the earliest layer where attn-divergence first LOCALLY PEAKS
    above background (the paper's first-local-peak rule), else argmax.

This is "onset located by how attention to the content differs between sensitive
and non-sensitive prompts" — the paper's kind of signal — vs our AUROC onset
which is MLP-activation detectability. Output compares the two.

Run: PYTHONPATH=src python3 scripts/onset_attention.py
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

from know_trans.concepts import load_battery
from know_trans.score import _build_example_texts
from know_trans.cspt import pick_onset_layer
from know_trans.utils import get_logger, ensure_dir, save_json

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
OUT = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("onset_attn")

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]
battery = load_battery("data/concepts_pilot")
names = [c.name for c in battery]
MAXLEN, BS = 128, 8


def first_local_peak(layers, vals):
    bg = float(np.mean(vals))
    for i, l in enumerate(layers):
        left = vals[i - 1] if i > 0 else -1e9
        right = vals[i + 1] if i < len(layers) - 1 else -1e9
        if vals[i] >= left and vals[i] >= right and vals[i] > bg:
            return int(l)
    return int(layers[int(np.argmax(vals))])


@torch.no_grad()
def attn_entropy_by_layer(model, tok, texts, device):
    """Return [n_texts, n_layers] mean attention entropy per prompt per layer."""
    rows = []
    for i in range(0, len(texts), BS):
        tb = texts[i:i + BS]
        enc = tok(tb, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        am = enc["attention_mask"]  # [B,T]
        out = model(**enc, output_attentions=True)
        atts = out.attentions  # tuple L x [B,H,T,T]
        qmask = am.bool()  # [B,T] valid query positions
        per_layer = []
        for A in atts:
            A = A.float().clamp_min(1e-12)  # [B,H,T,T]
            ent = -(A * A.log()).sum(-1)    # [B,H,T] entropy over keys per query
            ent = ent.mean(1)               # [B,T] mean over heads
            # mean over valid query positions
            ent = (ent * qmask).sum(1) / qmask.sum(1).clamp_min(1)  # [B]
            per_layer.append(ent.cpu().numpy())
        rows.append(np.stack(per_layer, axis=1))  # [B, L]
        del out, atts
        torch.cuda.empty_cache()
    return np.concatenate(rows, axis=0)  # [n_texts, L]


summary = {}
curves = {}
for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    adf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer.parquet"))
    tok = AutoTokenizer.from_pretrained(mpath)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mpath, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()

    texts, spans = _build_example_texts(battery)
    ent = attn_entropy_by_layer(model, tok, list(texts), "cuda")  # [n_texts, L]
    n_layers = ent.shape[1]
    layers = list(range(n_layers))

    summary[mname] = {}
    curves[mname] = {}
    for c in names:
        eids, labels, _ = spans[c]  # type: ignore[misc]
        eids = np.asarray(eids); labels = np.asarray(labels)
        # attention-divergence per layer = AUROC of entropy separating pos vs neg
        div = []
        for L in layers:
            x = ent[eids, L]
            try:
                a = roc_auc_score(labels, x)
            except ValueError:
                a = 0.5
            div.append(abs(a - 0.5) * 2.0)  # symmetric separation in [0,1]
        onset_attn = first_local_peak(layers, div)
        onset_auc, auc_at = pick_onset_layer(adf, c, tau_onset=0.70)
        summary[mname][c] = {
            "onset_attn": int(onset_attn),
            "onset_auroc": int(onset_auc),
            "attn_div_at_onset": round(float(div[onset_attn]), 3),
            "peak_attn_div": round(float(max(div)), 3),
            "peak_attn_layer": int(layers[int(np.argmax(div))]),
        }
        curves[mname][c] = [round(float(d), 3) for d in div]
        log.info("[%s/%s] onset_attn=L%d  onset_auroc=L%d  (attn-div peak %.2f@L%d)",
                 mname, c, onset_attn, onset_auc,
                 max(div), layers[int(np.argmax(div))])
    del model; torch.cuda.empty_cache()

save_json({"summary": summary, "curves": curves}, os.path.join(OUT, "onset_attention.json"))

print("\n" + "=" * 78)
print("ONSET LOCALIZATION: attention-divergence (paper-spirit) vs AUROC (ours)")
print("=" * 78)
for mname, _ in MODELS:
    print(f"\n#### {mname} ####")
    print(f"{'knowledge':16s} {'onset_attn':11s} {'onset_auroc':12s} {'agree?':7s} {'attnDiv@peak':12s}")
    for c in names:
        s = summary[mname][c]
        agree = "yes" if s["onset_attn"] == s["onset_auroc"] else f"Δ={s['onset_attn']-s['onset_auroc']:+d}"
        print(f"{c:16s} L{s['onset_attn']:<10d} L{s['onset_auroc']:<11d} {agree:7s} "
              f"{s['peak_attn_div']:.2f}@L{s['peak_attn_layer']}")
print("\nsaved -> pathways/onset_attention.json")
