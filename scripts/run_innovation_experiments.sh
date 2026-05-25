#!/usr/bin/env bash
# 完整创新实验：A1 TTA、A4、B1/B2、B stack；可选 A2 head-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
AASIST="${ROOT}/Baseline-AASIST"
PRETRAINED="${AASIST}/models/weights/AASIST/best.pth"
SCORES_BASE="${SCORES:-${AASIST}/exp_result/AASIST_ASVspoof5_eval_ep100_bs24/eval_scores_dev_full.txt}"
INNO="${ROOT}/experiments/results/innovation"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"
BASELINE_EER="0.152"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u
mkdir -p "${INNO}"

bash "${ROOT}/scripts/run_innovation_fast.sh"

echo "=== A1: Full dev TTA n_aug=3 ==="
TTA3="${INNO}/tta_full_n3"
mkdir -p "${TTA3}"
python "${ROOT}/experiments/tta_eval.py" \
  --checkpoint "${PRETRAINED}" \
  --data-dir "${DATA}" \
  --out-dir "${TTA3}" \
  --full-dev --n-aug 3 --batch-size 4 --run-metrics
EER_TTA3=$(python3 -c "import json; print(json.load(open('${TTA3}/tta_n3_metrics.json'))['EER'])")
python3 -c "
import json
from pathlib import Path
p = Path('${INNO}/summary.json')
d = json.loads(p.read_text(encoding='utf-8'))
d['A1_tta_n3'] = {'EER': float('${EER_TTA3}'), 'dir': '${TTA3}'}
d['A1_best'] = {'EER': float('${EER_TTA3}'), 'scores': '${TTA3}/dev_scores_tta.txt'}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
"

TTA_SCORES="${TTA3}/dev_scores_tta.txt" bash "${ROOT}/scripts/run_innovation_fast.sh"

if [[ "${RUN_HEAD_ONLY:-0}" == "1" ]]; then
  bash "${ROOT}/scripts/run_head_only_train_eval.sh"
  if [[ -f "${INNO}/head_only/head_only_eval.json" ]]; then
    python3 -c "
import json
from pathlib import Path
ho = json.loads(Path('${INNO}/head_only/head_only_eval.json').read_text())
p = Path('${INNO}/summary.json')
d = json.loads(p.read_text(encoding='utf-8'))
d['A2_head_only'] = {'EER': ho['best_eer'], 'dir': '${INNO}/head_only'}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
"
  fi
fi

echo "Innovation done: ${INNO}/summary.json"
