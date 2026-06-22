"""Aggregate Experiment-B results (subject-specialized SAE FVU across models/layers).

Reads report/diag/subjectB_<model>_<domain>.json (partial or complete) and prints
a per-(model,domain) summary plus the key comparison: at the mid/late layers where
the general SAE is most OOD, does the domain-specialized SAE reconstruct the
held-out MMLU-Pro activations better?

Run: PYTHONPATH=src python -m scripts.agg_subjectB
"""
import glob
import json
import os

RES = "report/diag"
MODELS = ["Llama-3.1-8B", "Qwen3-0.6B", "Llama-3.2-1B"]
DOMAINS = ["econ", "math", "med"]


def main():
    files = sorted(glob.glob(f"{RES}/subjectB_*.json"))
    if not files:
        print("no subjectB_*.json yet")
        return
    data = {}
    for f in files:
        d = json.load(open(f))
        data[(d["model"], d["domain"])] = d

    print(f"{'model':<15}{'dom':<5}{'done':>6}  "
          f"{'gen_FVU(mid+late)':>17}  {'spec_FVU(mid+late)':>18}  {'win%':>6}  {'active%':>8}")
    for m in MODELS:
        nl = None
        for dom in DOMAINS:
            d = data.get((m, dom))
            if not d:
                continue
            nl = d["n_layers"]
            layers = {int(k): v for k, v in d["layers"].items()}
            done = len(layers)
            # "mid+late" = layers >= 1/3 depth (early layers reconstruct trivially)
            cut = nl // 3
            ml = {L: v for L, v in layers.items() if L >= cut}
            if ml:
                gen = sum(v["fvu_gen"] for v in ml.values()) / len(ml)
                spec = sum(v["fvu_spec"] for v in ml.values()) / len(ml)
                wins = sum(1 for v in ml.values() if v["fvu_spec"] < v["fvu_gen"])
                winpct = 100 * wins / len(ml)
                act = 100 * sum(v.get("frac_active_spec", 0) for v in ml.values()) / len(ml)
                print(f"{m:<15}{dom:<5}{done:>3}/{nl:<2}  "
                      f"{gen:>17.3f}  {spec:>18.3f}  {winpct:>5.0f}%  {act:>7.1f}%")
            else:
                print(f"{m:<15}{dom:<5}{done:>3}/{nl:<2}  (no mid/late layers yet)")

    # per-layer detail for the teacher (most informative), all domains
    print("\n--- teacher Llama-3.1-8B per-layer (gen | spec) FVU ---")
    for dom in DOMAINS:
        d = data.get(("Llama-3.1-8B", dom))
        if not d:
            continue
        layers = {int(k): v for k, v in d["layers"].items()}
        gen = " ".join(f"{layers[L]['fvu_gen']:.2f}" for L in sorted(layers))
        spc = " ".join(f"{layers[L]['fvu_spec']:.2f}" for L in sorted(layers))
        print(f"{dom:<5} L{min(layers) if layers else '-'}..{max(layers) if layers else '-'}")
        print(f"  gen : {gen}")
        print(f"  spec: {spc}")


if __name__ == "__main__":
    main()
