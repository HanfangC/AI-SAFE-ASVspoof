#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
AASIST="${ROOT}/Baseline-AASIST"
SCORES="${SCORES:-${AASIST}/exp_result/AASIST_ASVspoof5_eval_ep100_bs24/eval_scores_dev_full.txt}"
OUT="${ROOT}/experiments/results/exp_b"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u
mkdir -p "${OUT}"
python "${ROOT}/experiments/run_exp_b_attack_compare.py" \
  --scores "${SCORES}" \
  --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
  --out-dir "${OUT}"
echo "Done: ${OUT}/exp_b_attack_eer.json"
