#!/usr/bin/env bash
#
# run_all.sh — end-to-end know_trans pipeline (documented order).
#
# SAE-based cross-family knowledge distillation. Each stage reads/writes the
# data/ artifacts the next stage consumes, so the order below is mandatory:
#
#   01 capture         teacher + student MLP activations on a corpus
#                          -> data/activations/<model>/{layer*.safetensors,
#                                                        index.parquet, meta.json}
#   02 train-sae       one TopK SAE per (model, captured layer)
#                          -> data/saes/<model>/layer*.safetensors
#   03 build-concepts  labeled concept battery from the real benchmark files
#                          -> data/concepts/*.jsonl
#   04 score           score every SAE feature as a concept detector (ROC-AUC)
#                          -> data/feature_scores/<model>.parquet
#   05 match           label-anchored teacher<->student concept matching
#                          -> data/matches/matches.json
#   06 distill         train the student with the concept-distillation loss
#                          (teacher frozen; both SAEs frozen by cfg)
#
# NOTE: stages 01, 02, 04, 06 are heavy (model loads / training / GPU). This
# script only orchestrates them; it never downloads models or data. Point CONFIG
# at a YAML whose model paths already exist locally.
#
# Usage:
#   scripts/run_all.sh [CONFIG] [PYTHON]
#     CONFIG  path to the run YAML  (default: configs/pair_llama8b_qwen0p6b.yaml)
#     PYTHON  python interpreter    (default: python3)
#
# Run a single stage instead via the numbered wrappers, e.g.:
#   python scripts/01_capture.py --config <CONFIG> --model teacher
# or directly through the unified CLI:
#   python -m know_trans.cli capture --config <CONFIG>

set -euo pipefail

# --- resolve paths ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-${REPO_ROOT}/configs/pair_llama8b_qwen0p6b.yaml}"
PYTHON="${2:-python3}"

# Make the in-tree package importable without an editable install.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "error: config not found: ${CONFIG}" >&2
  echo "usage: scripts/run_all.sh [CONFIG] [PYTHON]" >&2
  exit 1
fi

echo "==> know_trans pipeline"
echo "    repo   : ${REPO_ROOT}"
echo "    config : ${CONFIG}"
echo "    python : ${PYTHON}"
echo

run_stage () {
  local label="$1"; shift
  echo "==> [${label}] $*"
  "${PYTHON}" -m know_trans.cli "$@" --config "${CONFIG}"
  echo "<== [${label}] done"
  echo
}

run_stage "01 capture"        capture
run_stage "02 train-sae"      train-sae
run_stage "03 build-concepts" build-concepts
run_stage "04 score"          score
run_stage "05 match"          match
run_stage "06 distill"        distill

echo "==> pipeline complete."
