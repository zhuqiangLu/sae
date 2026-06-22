#!/usr/bin/env python3
"""Stage 01 — capture activations.

Thin wrapper around ``know_trans.cli capture``. All flags are forwarded, so::

    python scripts/01_capture.py --config configs/pair_llama8b_qwen0p6b.yaml
    python scripts/01_capture.py --config <cfg> --model teacher --layers 4,8,12

is equivalent to ``python -m know_trans.cli capture ...``.
"""

from __future__ import annotations

import os
import sys

# Make ``know_trans`` importable when running from a fresh checkout (no install).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from know_trans.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["capture", *sys.argv[1:]]))
