#!/usr/bin/env python3
"""Stage 06 — concept-space distillation of the student.

Thin wrapper around ``know_trans.cli distill``. All flags are forwarded::

    python scripts/06_distill.py --config configs/pair_llama8b_qwen0p6b.yaml
    python scripts/06_distill.py --config <cfg> --matches data/matches/matches.json
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from know_trans.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["distill", *sys.argv[1:]]))
