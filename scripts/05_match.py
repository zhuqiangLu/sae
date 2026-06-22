#!/usr/bin/env python3
"""Stage 05 — label-anchored teacher<->student concept matching.

Thin wrapper around ``know_trans.cli match``. All flags are forwarded::

    python scripts/05_match.py --config configs/pair_llama8b_qwen0p6b.yaml
    python scripts/05_match.py --config <cfg> --auc-threshold 0.8 --top-k 10
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from know_trans.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["match", *sys.argv[1:]]))
