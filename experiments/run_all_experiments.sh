#!/usr/bin/env bash
# 在已有 dev 分数上跑实验 A/B/C
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AASIST="${ROOT}/Baseline-AASIST"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
RESULTS="${ROOT}/experiments/results"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u

SCORES="${SCORES:-${1:-}}"
CKPT="${CKPT:-${2:-${AASIST}/models/weights/AASIST/best.pth}}"

if [[ -z "${SCORES}" ]]; then
  SCORES=$(find "${AASIST}/exp_result" -name 'eval_scores_dev_full.txt' 2>/dev/null | sort -r | head -1)
fi
if [[ ! -f "${SCORES}" ]]; then
  echo "No score file. Run scripts/run_baseline_eval.sh first." >&2
  exit 1
fi

mkdir -p "${RESULTS}/exp_a" "${RESULTS}/exp_b" "${RESULTS}/exp_c"

echo "=== Experiment A ==="
python "${ROOT}/experiments/run_exp_a_eval.py" \
  --scores "${SCORES}" \
  --out-dir "${RESULTS}/exp_a" \
  --tag aasist --no-swanlab

echo "=== Experiment B ==="
python "${ROOT}/experiments/run_exp_b_attack_compare.py" \
  --scores "${SCORES}" \
  --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
  --out-dir "${RESULTS}/exp_b"

if [[ -f "${CKPT}" && -f "${DATA}/ASVspoof5.dev.metainfor.txt" ]]; then
  echo "=== Experiment C (5000 utts) ==="
  python "${ROOT}/experiments/robustness_eval.py" \
    --checkpoint "${CKPT}" \
    --data-dir "${DATA}" \
    --out-dir "${RESULTS}/exp_c" \
    --max-utts 5000
fi

echo "All done under ${RESULTS}"
