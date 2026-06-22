"""TRUE task-performance test: does suppressing the identified pathway actually
destroy the model's CAPABILITY (MMLU multiple-choice accuracy), not just the
linear detectability (AUROC)?

Per (model, topic-subject), evaluate MMLU MC accuracy under three conditions:
  * clean        — no intervention (baseline capability)
  * src+path     — zero the source (64 @ onset) AND the full FIS path
                   (elbow-k FIS neurons @ each downstream layer)  [the intervention]
  * rand         — magnitude-matched random control at BOTH sites (same counts)

A real causal pathway => clean >> src+path, while rand ~ clean. Topics only
(safety/language have no MC-accuracy metric). Path k_down = elbow (paper-faithful,
the full path) since top-10 captures only a fraction of FIS mass.

Run: PYTHONPATH=src python3 scripts/eval_task_accuracy.py
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.utils import load_model_and_tokenizer, get_logger, ensure_dir, save_json
from know_trans.sae import SAEBundle
from know_trans.cspt import pick_onset_layer, source_neurons, ZeroOutHook, elbow_k, _get_hook_module, MLPHook

DATA = "/share1/zhlu6105/know_trans_data"
FS = os.path.join(DATA, "feature_scores")
PW = os.path.join(DATA, "pathways")
BENCH = "/share1/zhlu6105/benchmarks/MMLU"
log = get_logger("eval_acc")

MODELS = [
    ("Llama-3.1-8B", "/share1/zhlu6105/models/Llama-3.1-8B"),
    ("Qwen3-0.6B",   "/share1/zhlu6105/models/Qwen3-0.6B"),
    ("Llama-3.2-1B", "/share1/zhlu6105/models/Llama-3.2-1B"),
]
SUBJECTS = {"topic_math": "abstract_algebra",
            "topic_economics": "econometrics",
            "topic_medical": "professional_medicine"}
LETTERS = ["A", "B", "C", "D"]


def load_mmlu_items(subject):
    paths = glob.glob(os.path.join(BENCH, subject, "test-*.parquet"))
    df = pd.read_parquet(paths[0])
    return df  # question, choices(4), answer(int)


def make_prompt(q, choices):
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"The following is a multiple choice question. "
            f"Answer with the letter of the correct option.\n\n{q}\n{body}\nAnswer:")


@torch.no_grad()
def accuracy(model, tok, prompts, gold, lid, batch=4, max_len=384):
    """MC accuracy: argmax over the A/B/C/D next-token logits."""
    lid_t = torch.tensor(lid, device=model.device)
    preds = []
    for i in range(0, len(prompts), batch):
        pb = prompts[i:i + batch]
        enc = tok(pb, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc)
        cand = out.logits[:, -1, :].index_select(-1, lid_t)  # [B, 4]
        preds.extend(cand.argmax(-1).cpu().tolist())
        del out, cand
    torch.cuda.empty_cache()
    preds = np.array(preds)
    return float((preds == np.array(gold)).mean())


results = {}
for mname, mpath in MODELS:
    log.info("==== %s ====", mname)
    adf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer.parquet"))
    wdf = pd.read_parquet(os.path.join(FS, f"{mname}_alllayer_wfs.parquet"))
    bundle = SAEBundle.load(os.path.join(DATA, "saes", mname, "seed0"))
    layers = sorted(int(l) for l in bundle.layers)
    model, tok = load_model_and_tokenizer(mpath, dtype="bfloat16", device="cuda")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lid = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    rng = np.random.default_rng(0)
    results[mname] = {}

    for C, subj in SUBJECTS.items():
        items = load_mmlu_items(subj)
        prompts = [make_prompt(r["question"], list(r["choices"])) for _, r in items.iterrows()]
        gold = [int(a) for a in items["answer"]]
        lstar, _ = pick_onset_layer(adf, C, tau_onset=0.70)
        down = [l for l in layers if l > lstar]
        src, _, _ = source_neurons(bundle[lstar], wdf, C, lstar, k_feat=10, k_src=64)
        fis = pd.read_parquet(os.path.join(PW, f"{mname}_{C}_fis.parquet"))

        # path = elbow-k FIS neurons per downstream layer; randD = magnitude-matched
        path, randD = {}, {}
        for l in down:
            sub = fis[fis["layer"] == l]
            neur = sub["neuron"].to_numpy(); mag = (sub["f"] * sub["mu"]).to_numpy()
            k = elbow_k(np.sort(sub["fis"].to_numpy())[::-1])
            p = sub.nlargest(k, "fis")["neuron"].to_numpy(); path[l] = p
            pset = set(int(x) for x in p)
            thr = float(np.min(mag[np.isin(neur, list(pset))])) if len(pset) else 0.0
            pool = neur[(mag >= thr) & (~np.isin(neur, list(pset)))]
            if len(pool) < k: pool = neur[~np.isin(neur, list(pset))]
            randD[l] = rng.choice(pool, size=min(k, len(pool)), replace=False)

        # magnitude-matched randS at l* (clean pre-pass over the eval prompts)
        rd = MLPHook(_get_hook_module(model, lstar, "mlp"), lstar, to_cpu=False)
        d_l = bundle[lstar].W_dec.shape[0]
        cnt = torch.zeros(d_l, device=model.device); ssum = torch.zeros(d_l, device=model.device); nt = 0.0
        for i in range(0, len(prompts), 4):
            enc = tok(prompts[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=384)
            enc = {k: v.to(model.device) for k, v in enc.items()}; am = enc["attention_mask"]
            model(**enc)
            a = rd.pop().float(); b, s, d = a.shape
            mf = am.bool().reshape(-1); af = a.reshape(b*s, d)[mf]; act = (af > 0).float()
            cnt += act.sum(0); ssum += (af*act).sum(0); nt += float(af.shape[0])
            del a, af, act
        rd.remove(); torch.cuda.empty_cache()
        magl = (ssum / max(nt, 1.0)).cpu().numpy()
        thr_s = float(np.min(magl[src])) if len(src) else 0.0
        alln = np.arange(d_l); poolS = alln[(magl >= thr_s) & (~np.isin(alln, src))]
        if len(poolS) < len(src): poolS = alln[~np.isin(alln, src)]
        randS = rng.choice(poolS, size=min(len(src), len(poolS)), replace=False)

        # hooks
        zh = {lstar: ZeroOutHook(_get_hook_module(model, lstar, "mlp"), src)}
        for l in down: zh[l] = ZeroOutHook(_get_hook_module(model, l, "mlp"), path[l])
        conds = {
            "clean":      {},
            "src":        {lstar: src},
            "path":       {**path},
            "src+path":   {lstar: src, **path},
            "src+randD":  {lstar: src, **randD},
            "randS+path": {lstar: randS, **path},
            "rand":       {lstar: randS, **randD},
        }
        accs = {}
        for cn, cmap in conds.items():
            for l, h in zh.items():
                if l in cmap: h.set_coords(cmap[l]); h.enabled = True
                else: h.enabled = False
            accs[cn] = round(accuracy(model, tok, prompts, gold, lid), 3)
        for h in zh.values(): h.remove()
        mean_k = round(float(np.mean([len(path[l]) for l in down])), 1)
        results[mname][C] = {"n": len(gold), "onset": int(lstar), "mean_path_k": mean_k, "chance": round(1/4, 3), **accs}
        log.info("[%s/%s] n=%d L%d k~%.0f  clean=%.3f src=%.3f path=%.3f src+path=%.3f src+randD=%.3f randS+path=%.3f rand=%.3f",
                 mname, C, len(gold), lstar, mean_k, accs["clean"], accs["src"], accs["path"],
                 accs["src+path"], accs["src+randD"], accs["randS+path"], accs["rand"])
    del model; torch.cuda.empty_cache()

save_json(results, os.path.join(PW, "task_accuracy.json"))
CN = ["clean", "src", "path", "src+path", "src+randD", "randS+path", "rand"]
print("\n" + "=" * 110)
print("TRUE TASK PERFORMANCE — MMLU accuracy under the full intervention factorial (chance=0.25; lower=more loss)")
print("path real => src+path < src+randD ; source real => src+path < randS+path")
print("=" * 110)
for mname, _ in MODELS:
    print(f"\n#### {mname} ####")
    print(f"{'subject':16s} {'n':>4s} " + " ".join(f"{c:>10s}" for c in CN))
    for C in SUBJECTS:
        r = results[mname][C]
        print(f"{C:16s} {r['n']:>4d} " + " ".join(f"{r[c]:>10.3f}" for c in CN))
print("\nsaved -> pathways/task_accuracy.json")
