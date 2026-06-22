#!/usr/bin/env python3
"""Stage 02 — train TopK SAEs.

Thin wrapper around ``know_trans.cli train-sae``. All flags are forwarded::

    python scripts/02_train_sae.py --config configs/pair_llama8b_qwen0p6b.yaml
    python scripts/02_train_sae.py --config <cfg> --model student --steps 1000
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from know_trans.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["train-sae", *sys.argv[1:]]))
