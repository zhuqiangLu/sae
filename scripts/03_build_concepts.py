#!/usr/bin/env python3
"""Stage 03 — build the concept battery from benchmarks.

Thin wrapper around ``know_trans.cli build-concepts``. All flags are forwarded::

    python scripts/03_build_concepts.py --config configs/pair_llama8b_qwen0p6b.yaml
    python scripts/03_build_concepts.py --config <cfg> --n-per-concept 200
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from know_trans.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["build-concepts", *sys.argv[1:]]))
