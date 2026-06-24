"""make_mmlu_split.py — single-subject train/val/test split of one MMLU subject.

train = N_TRAIN, val = N_VAL, test = the rest (disjoint, de-duplicated by question
text, seeded). One file per domain: data/pathways_subject/mmlu_{domain}_split.json,
so econ/math/med each get an isolated specific subject (4-choice, self-contained).

Run: PYTHONPATH=src python -m scripts.make_mmlu_split \
       --domain econ --subject high_school_macroeconomics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import pandas as pd

BENCH = "benchmarks/MMLU"


def load_subject(subj: str) -> list[dict]:
    rows, seen = [], set()
    for sp in ("test", "validation", "dev"):
        for p in glob.glob(f"{BENCH}/{subj}/{sp}-*.parquet"):
            df = pd.read_parquet(p)
            if "question" not in df.columns or "answer" not in df.columns:
                continue
            for q, ch, a in zip(df["question"], df["choices"], df["answer"]):
                ch = [str(c) for c in list(ch)]
                key = q.strip() if isinstance(q, str) else None
                if key and len(ch) == 4 and key not in seen:
                    seen.add(key)
                    rows.append({"subject": subj, "question": key, "choices": ch, "answer": int(a)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_subject(a.subject)
    need = a.n_train + a.n_val + 1
    if len(rows) < need:
        raise SystemExit(f"{a.subject}: only {len(rows)} Qs; need > {a.n_train + a.n_val}")
    random.Random(a.seed).shuffle(rows)
    train = rows[:a.n_train]
    val = rows[a.n_train:a.n_train + a.n_val]
    test = rows[a.n_train + a.n_val:]

    out = f"data/pathways_subject/mmlu_{a.domain}_split.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"domain": a.domain, "subject": a.subject, "seed": a.seed,
               "n_total": len(rows), "n_train": len(train), "n_val": len(val),
               "n_test": len(test), "train": train, "val": val, "test": test},
              open(out, "w"), indent=2)
    print(f"[{a.domain}] {a.subject}: total={len(rows)} -> "
          f"train={len(train)} val={len(val)} test={len(test)}  -> {out}")


if __name__ == "__main__":
    main()
