#!/usr/bin/env bash
# 创新实验快速轨：B1、B2、分族+攻击叠加、A4（需已有 TTA 分数则设置 TTA_SCORES）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
AASIST="${ROOT}/Baseline-AASIST"
SCORES_BASE="${SCORES:-${AASIST}/exp_result/AASIST_ASVspoof5_eval_ep100_bs24/eval_scores_dev_full.txt}"
INNO="${ROOT}/experiments/results/innovation"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"

source "${CONDA_SH}"
set +u
conda activate "${ENV_NAME}"
set -u
mkdir -p "${INNO}"

python3 <<INIT
import json
from pathlib import Path
p = Path("${INNO}/summary.json")
d = {
    "baseline_eer": 0.152,
    "family_adjusted_eer": 0.1407,
    "target_eer": 0.1407,
}
if p.is_file():
    d.update(json.loads(p.read_text(encoding="utf-8")))
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
INIT

metric_eer() { python3 -c "import json; print(json.load(open('$1'))['EER'])"; }
update_summary() {
  python3 -c "
import json
from pathlib import Path
p = Path('${INNO}/summary.json')
d = json.loads(p.read_text(encoding='utf-8'))
d.update($1)
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
"
}

echo "=== B1: per-attack calibration ==="
B1="${INNO}/per_attack"
python "${ROOT}/experiments/per_attack_calibration.py" \
  --scores "${SCORES_BASE}" --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
  --out-dir "${B1}" --write-scores
python "${ROOT}/experiments/run_exp_a_eval.py" \
  --scores "${B1}/eval_scores_dev_full_per_attack.txt" \
  --out-dir "${B1}/metrics" --tag per_attack --no-swanlab
EER_B1=$(metric_eer "${B1}/metrics/per_attack_metrics.json")
update_summary "{'B1_per_attack': {'EER': ${EER_B1}}}"

echo "=== B2: Platt + per-attack ==="
CAL="${INNO}/platt_per_attack"
python "${ROOT}/experiments/calibrate_scores.py" \
  --scores "${SCORES_BASE}" --out-scores "${CAL}/eval_scores_platt.txt" \
  --out-dir "${CAL}" --tag platt
python "${ROOT}/experiments/per_attack_calibration.py" \
  --scores "${CAL}/eval_scores_platt.txt" \
  --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
  --out-dir "${CAL}/per_attack" --write-scores
python "${ROOT}/experiments/run_exp_a_eval.py" \
  --scores "${CAL}/per_attack/eval_scores_dev_full_per_attack.txt" \
  --out-dir "${CAL}/metrics" --tag platt_per_attack --no-swanlab
EER_B2=$(metric_eer "${CAL}/metrics/platt_per_attack_metrics.json")
update_summary "{'B2_platt_per_attack': {'EER': ${EER_B2}}}"

echo "=== B stack: family + per-attack ==="
STACK="${INNO}/stack_baseline"
mkdir -p "${STACK}"
python "${ROOT}/experiments/stack_postprocess.py" \
  --scores "${SCORES_BASE}" \
  --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
  --out-dir "${STACK}" --tag baseline
FINAL="${STACK}/eval_scores_baseline_family_attack.txt"
python "${ROOT}/experiments/run_exp_a_eval.py" \
  --scores "${FINAL}" \
  --out-dir "${STACK}/metrics" --tag stack_family_attack --no-swanlab
EER_STACK=$(metric_eer "${STACK}/metrics/stack_family_attack_metrics.json")
update_summary "{'B_stack_family_attack': {'EER': ${EER_STACK}, 'scores': '${FINAL}'}}"

TTA_SCORES="${TTA_SCORES:-${INNO}/tta_full_n3/dev_scores_tta.txt}"
if [[ -f "${TTA_SCORES}" ]]; then
  echo "=== A4: ensemble pretrained + TTA ==="
  ENS="${INNO}/ensemble_pretrained_tta"
  mkdir -p "${ENS}"
  python "${ROOT}/experiments/score_ensemble.py" \
    --scores-a "${SCORES_BASE}" \
    --scores-b "${TTA_SCORES}" \
    --metainfor "${DATA}/ASVspoof5.dev.metainfor.txt" \
    --out-dir "${ENS}" --tag ensemble_pretrained_tta
  python "${ROOT}/experiments/run_exp_a_eval.py" \
    --scores "${ENS}/eval_scores_dev_full_ensemble_pretrained_tta.txt" \
    --out-dir "${ENS}/metrics" --tag ensemble_pretrained_tta --no-swanlab
  EER_ENS=$(metric_eer "${ENS}/metrics/ensemble_pretrained_tta_metrics.json")
  update_summary "{'A4_ensemble': {'EER': ${EER_ENS}}}"
else
  echo "跳过 A4（未找到 TTA 分数: ${TTA_SCORES}）"
fi

python3 <<PY
import json
from pathlib import Path
p = Path("${INNO}/summary.json")
d = json.loads(p.read_text(encoding="utf-8"))
cands = [(k, v["EER"]) for k, v in d.items() if isinstance(v, dict) and "EER" in v]
best = min(cands, key=lambda x: x[1])
d["best_track"] = {"id": best[0], "EER": best[1]}
a_cands = [(k, v["EER"]) for k, v in d.items()
           if isinstance(v, dict) and "EER" in v and k.startswith(("A1", "A4", "A2"))]
if a_cands:
    ta = min(a_cands, key=lambda x: x[1])
    d["track_A_best"] = {"id": ta[0], "EER": ta[1]}
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Best:", best)
PY

echo "Fast track done: ${INNO}/summary.json"
