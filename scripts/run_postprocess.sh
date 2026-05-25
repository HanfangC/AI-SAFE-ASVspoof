#!/usr/bin/env bash
# 对预训练 dev 分数做校准与分族平移
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
AASIST="${ROOT}/Baseline-AASIST"
SCORES="${SCORES:-${AASIST}/exp_result/AASIST_ASVspoof5_eval_ep100_bs24/eval_scores_dev_full.txt}"
OUT="${ROOT}/experiments/results/postprocess"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u

mkdir -p "${OUT}"

echo "=== Baseline (raw scores) ==="
python "${ROOT}/experiments/run_exp_a_eval.py" \
  --scores "${SCORES}" \
  --out-dir "${OUT}/baseline" \
  --tag baseline \
  --no-swanlab
cp "${OUT}/baseline/baseline_metrics.json" "${OUT}/baseline.json"

echo "=== Platt calibration ==="
python "${ROOT}/experiments/calibrate_scores.py" \
  --scores "${SCORES}" \
  --out-scores "${OUT}/eval_scores_dev_full_calibrated.txt" \
  --out-dir "${OUT}/calibrated" \
  --tag calibrated
cp "${OUT}/calibrated/calibrated_metrics.json" "${OUT}/calibrated.json"

echo "=== Per-family score shift search ==="
python "${ROOT}/experiments/per_family_threshold.py" \
  --scores "${SCORES}" \
  --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
  --out-dir "${OUT}" \
  --write-scores

if [[ -f "${OUT}/eval_scores_dev_full_family_adjusted.txt" ]]; then
  python "${ROOT}/experiments/run_exp_a_eval.py" \
    --scores "${OUT}/eval_scores_dev_full_family_adjusted.txt" \
    --out-dir "${OUT}/family_adjusted" \
    --tag family_adjusted \
    --no-swanlab
  cp "${OUT}/family_adjusted/family_adjusted_metrics.json" "${OUT}/family_adjusted.json"
fi

python3 <<PY
import json
from pathlib import Path

out = Path("${OUT}")
summary = {
    "scores_file": "${SCORES}",
    "baseline": json.loads((out / "baseline.json").read_text()),
    "calibrated": json.loads((out / "calibrated.json").read_text()),
    "per_family_threshold": json.loads((out / "per_family_threshold.json").read_text()),
}
fa = out / "family_adjusted.json"
if fa.is_file():
    summary["family_adjusted"] = json.loads(fa.read_text())
path = out / "postprocess_summary.json"
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(path.read_text())
PY

echo "Done: ${OUT}/postprocess_summary.json"
