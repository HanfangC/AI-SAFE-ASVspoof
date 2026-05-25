#!/usr/bin/env bash
# 预训练 best.pth 全量 dev 推理，生成 eval_scores_dev_full.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AASIST="${ROOT}/Baseline-AASIST"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"
CKPT="${CKPT:-${AASIST}/models/weights/AASIST/best.pth}"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u

cd "${AASIST}"
export USE_SWANLAB="${USE_SWANLAB:-0}"
python ./main.py --config ./config/AASIST_ASVspoof5_eval.conf --eval \
  --eval_model_weights "${CKPT}"

SCORE=$(find exp_result -name 'eval_scores_dev_full.txt' 2>/dev/null | sort -r | head -1)
if [[ -z "${SCORE}" || ! -f "${SCORE}" ]]; then
  echo "未找到 eval_scores_dev_full.txt" >&2
  exit 1
fi
echo "Scores written: ${SCORE}"
echo "下一步: SCORES=${SCORE} bash scripts/run_postprocess.sh"
