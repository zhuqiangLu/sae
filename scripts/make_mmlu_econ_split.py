"""make_mmlu_econ_split.py — build a self-contained train/test split of MMLU econ.

Replaces the train-MMLU / eval-MMLU-Pro setup (which mixed distributions and risked
MMLU->MMLU-Pro contamination). Here train and test come from the SAME MMLU econ pool
(4-choice), de-duplicated, seeded.

Saves data/pathways_subject/mmlu_econ_split.json with the actual rows:
  {subjects, seed, test_frac, n, n_train, n_test,
   train:[{subject,question,choices,answer}], test:[...]}

Run: PYTHONPATH=src python -m scripts.make_mmlu_econ_split \
       --subjects high_school_microeconomics,high_school_macroeconomics,econometrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

BENCH = "benchmarks/MMLU"
OUT = "data/pathways_subject/mmlu_econ_split.json"
DEFAULT_SUBJECTS = "high_school_microeconomics,high_school_macroeconomics,econometrics"


def load_subject(subj: str) -> list[dict]:
    rows = []
    for sp in ("test", "validation", "dev"):
        for p in glob.glob(f"{BENCH}/{subj}/{sp}-*.parquet"):
            df = pd.read_parquet(p)
            if "question" not in df.columns or "answer" not in df.columns:
                continue
            for q, ch, a in zip(df["question"], df["choices"], df["answer"]):
                ch = [str(c) for c in list(ch)]
                if isinstance(q, str) and len(ch) == 4:
                    rows.append({"subject": subj, "question": q.strip(),
                                 "choices": ch, "answer": int(a)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default=DEFAULT_SUBJECTS)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import random
    subjects = [s for s in a.subjects.split(",") if s.strip()]
    rows: list[dict] = []
    seen: set = set()
    per_subj = {}
    for s in subjects:
        sr = load_subject(s)
        # de-duplicate by question text (MMLU has a few repeats across splits)
        uniq = []
        for r in sr:
            key = r["question"]
            if key in seen:
                continue
            seen.add(key); uniq.append(r); rows.append(r)
        per_subj[s] = len(uniq)

    rng = random.Random(a.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_test = int(round(a.test_frac * n))
    test, train = rows[:n_test], rows[n_test:]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"subjects": subjects, "seed": a.seed, "test_frac": a.test_frac,
               "n": n, "n_train": len(train), "n_test": len(test),
               "per_subject": per_subj, "train": train, "test": test},
              open(a.out, "w"), indent=2)

    print(f"MMLU econ split -> {a.out}")
    print(f"  subjects: {per_subj}  (deduped total={n})")
    print(f"  train={len(train)}  test={len(test)}  (test_frac={a.test_frac}, seed={a.seed})")
    # per-subject breakdown of the test split (sanity)
    from collections import Counter
    print(f"  test by subject: {dict(Counter(r['subject'] for r in test))}")


if __name__ == "__main__":
    main()
