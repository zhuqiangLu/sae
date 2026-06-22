"""Build the held-out safety benchmark TEST set with contamination control (Stage 4a).

The safety WFS / route is built from the battery's safety positives
(load_battery('data/concepts_pilot').safety.positives -- the first 200 of the
deduped advbench+harmbench harmful list). The benchmark must NOT reuse those
exact prompts. We therefore build:

  TEST = (AdvBench behaviors NOT in battery safety positives, exact-string match)
         + (HarmBench DirectRequest test behaviors)

and dedup, recording provenance counts and how many AdvBench rows were excluded
as train-overlap. Saved to {DATA}/pathways/safety_benchmark_heldout.json.

Run: PYTHONPATH=src python3 scripts/build_safety_heldout.py
"""
from __future__ import annotations
import json, os
import pandas as pd

from know_trans.concepts import load_battery
from know_trans.utils import ensure_dir, get_logger

DATA = "/share1/zhlu6105/know_trans_data"
BENCH = "/share1/zhlu6105/benchmarks"
BATTERY = "data/concepts_pilot"
OUT = ensure_dir(os.path.join(DATA, "pathways"))
log = get_logger("build_safety_heldout")


def _clean(s) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    return "" if t.lower() in {"nan", "none", "null"} else t


def main() -> None:
    # --- battery safety positives = the WFS/route TRAIN prompts ---
    battery = load_battery(BATTERY)
    safety = next((c for c in battery if c.name == "safety"), None)
    if safety is None:
        raise SystemExit("no 'safety' concept in battery")
    train_pos = set(_clean(p) for p in safety.positives if _clean(p))
    log.info("battery safety positives (TRAIN): %d", len(train_pos))

    # --- AdvBench behaviors ---
    adv_path = os.path.join(BENCH, "AdvBench", "advbench_behaviors.jsonl")
    adv_rows = []
    with open(adv_path, "r", encoding="utf-8-sig") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                adv_rows.append(json.loads(ln))
    adv_behaviors = [_clean(r.get("Behavior")) for r in adv_rows]
    adv_behaviors = [b for b in adv_behaviors if b]
    # dedup preserving order
    seen = set()
    adv_unique = []
    for b in adv_behaviors:
        if b not in seen:
            seen.add(b)
            adv_unique.append(b)
    adv_kept = [b for b in adv_unique if b not in train_pos]
    adv_excluded = [b for b in adv_unique if b in train_pos]
    log.info("AdvBench: %d rows, %d unique, %d excluded(train-overlap), %d kept",
             len(adv_behaviors), len(adv_unique), len(adv_excluded), len(adv_kept))

    # --- HarmBench DirectRequest test ---
    hb_path = os.path.join(BENCH, "HarmBench", "DirectRequest", "test-00000-of-00001.parquet")
    hb = pd.read_parquet(hb_path)
    hb_behaviors = [_clean(b) for b in hb["Behavior"].tolist()]
    hb_behaviors = [b for b in hb_behaviors if b]
    seen_hb = set()
    hb_unique = []
    for b in hb_behaviors:
        if b not in seen_hb:
            seen_hb.add(b)
            hb_unique.append(b)
    hb_excluded = [b for b in hb_unique if b in train_pos]
    hb_kept = [b for b in hb_unique if b not in train_pos]
    log.info("HarmBench DR test: %d rows, %d unique, %d excluded(train-overlap), %d kept",
             len(hb_behaviors), len(hb_unique), len(hb_excluded), len(hb_kept))

    # --- combined TEST set (dedup across sources, track source) ---
    test = []
    test_seen = set()
    for b in adv_kept:
        if b not in test_seen:
            test_seen.add(b)
            test.append({"behavior": b, "source": "advbench"})
    for b in hb_kept:
        if b not in test_seen:
            test_seen.add(b)
            test.append({"behavior": b, "source": "harmbench_directrequest"})

    out = {
        "description": ("Held-out harmful-prompt TEST set for the Instruct safety "
                        "benchmark. AdvBench behaviors not present in the battery "
                        "safety positives (exact-string match) + HarmBench "
                        "DirectRequest test behaviors. Contamination control vs the "
                        "WFS/route TRAIN prompts (battery safety positives)."),
        "provenance": {
            "battery_safety_positives_train": len(train_pos),
            "advbench_total_rows": len(adv_behaviors),
            "advbench_unique": len(adv_unique),
            "advbench_excluded_train_overlap": len(adv_excluded),
            "advbench_kept": len(adv_kept),
            "harmbench_dr_test_total_rows": len(hb_behaviors),
            "harmbench_dr_test_unique": len(hb_unique),
            "harmbench_dr_test_excluded_train_overlap": len(hb_excluded),
            "harmbench_dr_test_kept": len(hb_kept),
            "test_total_after_cross_source_dedup": len(test),
        },
        "advbench_excluded_examples": adv_excluded[:10],
        "test": test,
    }
    out_path = os.path.join(OUT, "safety_benchmark_heldout.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    log.info("wrote %s", out_path)
    print("HELDOUT DONE")
    print(json.dumps(out["provenance"], indent=2))


if __name__ == "__main__":
    main()
