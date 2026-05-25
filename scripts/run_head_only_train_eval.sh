#!/usr/bin/env bash
# Head-only 微调 + 各 epoch 全量 dev 评测
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AASIST="${ROOT}/Baseline-AASIST"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
OUT="${ROOT}/experiments/results/innovation/head_only"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"
EVAL_CONF="${AASIST}/config/AASIST_ASVspoof5_eval_fast.conf"
BASELINE_EER="${BASELINE_EER:-0.152}"
MAX_EPOCH="${MAX_EPOCH:-9}"
TRAIN_CONF="${AASIST}/config/AASIST_ASVspoof5_head_only.conf"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u
cd "${AASIST}"
unset CUDA_VISIBLE_DEVICES
export USE_SWANLAB="${USE_SWANLAB:-0}"
mkdir -p "${OUT}" "${ROOT}/logs"

echo "=== Training head-only ==="
python ./main.py --config "${TRAIN_CONF}" --output_dir ./exp_result \
  --comment head_only_innovation 2>&1 | tee "${ROOT}/logs/head_only_train.log"

WEIGHTS_DIR=$(find "${AASIST}/exp_result" -path '*head_only*' -type d -name weights 2>/dev/null | head -1)
if [[ -z "${WEIGHTS_DIR}" || ! -d "${WEIGHTS_DIR}" ]]; then
  echo "未找到 head_only weights 目录" >&2
  exit 1
fi
echo "Weights: ${WEIGHTS_DIR}"

bad_streak=0
best_eer="${BASELINE_EER}"
best_ckpt=""

shopt -s nullglob
for ckpt in "${WEIGHTS_DIR}"/epoch_*.pth; do
  base=$(basename "${ckpt}" .pth)
  epoch_num=$(echo "${base}" | sed -n 's/epoch_\([0-9]*\)_.*/\1/p')
  [[ -n "${epoch_num}" ]] || continue
  if [[ "${epoch_num}" -gt "${MAX_EPOCH}" ]]; then
    continue
  fi
  tag="epoch_${epoch_num}"
  echo "=== full dev eval: ${tag} ==="
  python ./main.py --config "${EVAL_CONF}" --eval \
    --eval_model_weights "${ckpt}" \
    --output_dir ./exp_result --comment "head_only_${tag}"
  SCORE="${AASIST}/exp_result/AASIST_ASVspoof5_eval_fast_ep100_bs64_head_only_${tag}/eval_scores_dev_full.txt"
  python "${ROOT}/experiments/run_exp_a_eval.py" \
    --scores "${SCORE}" --out-dir "${OUT}/${tag}" --tag "${tag}" --no-swanlab
  eer=$(python3 -c "import json; print(json.load(open('${OUT}/${tag}/${tag}_metrics.json'))['EER'])")
  echo "epoch ${epoch_num} EER=${eer}"
  if python3 -c "import sys; sys.exit(0 if float('${eer}') < float('${best_eer}') else 1)"; then
    best_eer="${eer}"
    best_ckpt="${ckpt}"
    bad_streak=0
  else
    bad_streak=$((bad_streak + 1))
    if python3 -c "import sys; sys.exit(0 if float('${eer}') > float('${BASELINE_EER}') else 1)"; then
      if [[ "${bad_streak}" -ge 2 ]]; then
        echo "Early stop: EER above baseline twice."
        break
      fi
    fi
  fi
done

python3 <<PY
import json
from pathlib import Path
out = Path("${OUT}")
summary = {
    "best_eer": float("${best_eer}"),
    "best_ckpt": "${best_ckpt}",
    "baseline_eer": float("${BASELINE_EER}"),
}
out.mkdir(parents=True, exist_ok=True)
out.joinpath("head_only_eval.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Done: ${OUT}"
