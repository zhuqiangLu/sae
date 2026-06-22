"""Build the reduced LLM-pilot concept battery into data/concepts_pilot/.

Selection (per design decisions):
  - safety                     (harmful vs benign)
  - topic_math                 (single MMLU subject: abstract_algebra)
  - topic_economics            (single MMLU subject: econometrics)
  - topic_medical              (single MMLU subject: professional_medicine)
  - language_<TARGET_LANG>     (one language vs the others; Turkish absent -> substitute)

NOTE: topics are SINGLE MMLU subjects (not pooled aggregates) so each is a
coherent concept; hard negatives are questions from any OTHER subject (balanced).

Run:  PYTHONPATH=src python3 scripts/build_pilot_battery.py [TARGET_LANG]
"""
from __future__ import annotations
import sys, logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from know_trans.concepts import (
    _load_advbench, _load_harmbench, _load_mmlu, _mmlu_lang_dirs,
    _build_safety_concept, _build_language_concepts,
    _dedup, _balanced_negatives, Concept, save_battery,
)

BENCH = "/share1/zhlu6105/benchmarks"
OUT = "data/concepts_pilot"
N = 200
SEED = 0
TARGET_LANG = sys.argv[1] if len(sys.argv) > 1 else "hu"  # Turkish unavailable

# Single MMLU subjects (one coherent concept each), not pooled aggregates.
MATH = {"abstract_algebra"}
ECON = {"econometrics"}
MED = {"professional_medicine"}


def topic_aggregate(mmlu, name, subjects):
    """One topic concept: positives = questions in the given subject(s), hard
    negatives = questions from any other subject (balanced)."""
    pos = _dedup(r["question"] for r in mmlu if r["subject"] in subjects)[:N]
    neg_pool = [r["question"] for r in mmlu if r["subject"] not in subjects]
    neg = _balanced_negatives(neg_pool, set(pos), len(pos), SEED)
    return Concept(f"topic_{name}", pos, neg, "mmlu", "topic")


def main():
    advbench = _load_advbench(BENCH)
    hb = _load_harmbench(BENCH)
    harmful = _dedup(advbench + [r["behavior"] for r in hb])
    mmlu = _load_mmlu(BENCH)
    benign = _dedup(r["question"] for r in mmlu)

    battery = []
    safety = _build_safety_concept(harmful, benign, N, SEED)
    if safety:
        battery.append(safety)
    battery.append(topic_aggregate(mmlu, "math", MATH))
    battery.append(topic_aggregate(mmlu, "economics", ECON))
    battery.append(topic_aggregate(mmlu, "medical", MED))

    langs = _mmlu_lang_dirs(BENCH)
    if TARGET_LANG not in langs:
        raise SystemExit(f"language {TARGET_LANG!r} not available; have {langs}")
    all_lang = _build_language_concepts(BENCH, langs, N, SEED)
    lang = next((c for c in all_lang if c.name == f"language_{TARGET_LANG}"), None)
    if lang:
        battery.append(lang)

    battery = [c for c in battery if c is not None and c.is_usable()]
    save_battery(battery, OUT)
    print(f"\n=== pilot battery: {len(battery)} concepts -> {OUT} ===")
    for c in battery:
        print(f"  {c.name:22s} pos={c.n_pos:4d}  neg={c.n_neg:4d}  [{c.group}]")


if __name__ == "__main__":
    main()
