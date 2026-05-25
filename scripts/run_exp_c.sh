#!/usr/bin/env bash
# 实验 C：鲁棒性（5000 条子集，计算量大，需 GPU）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
AASIST="${ROOT}/Baseline-AASIST"
CKPT="${CKPT:-${AASIST}/models/weights/AASIST/best.pth}"
OUT="${ROOT}/experiments/results/exp_c"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u
mkdir -p "${OUT}/metrics"
python "${ROOT}/experiments/robustness_eval.py" \
  --checkpoint "${CKPT}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --max-utts 5000
echo "Done: ${OUT}/metrics/"
