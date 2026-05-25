#!/usr/bin/env bash
# 用 conda 创建/更新 ASVspoof5 AASIST 环境
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AASIST="${ROOT}/Baseline-AASIST"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${AASIST_CONDA_ENV:-aasist}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "未找到 conda: ${CONDA_SH}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${SKIP_CONDA_UPDATE:-0}" == "1" ]]; then
    echo "跳过 conda update (SKIP_CONDA_UPDATE=1): ${ENV_NAME}"
  else
    echo "更新已有环境: ${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f "${AASIST}/environment.yml" --prune
  fi
else
  echo "创建环境: ${ENV_NAME}"
  conda env create -f "${AASIST}/environment.yml"
fi

set +u
conda activate "${ENV_NAME}"
set -u
pip install -q 'numpy<2'
python - <<'PY'
import torch
import soundfile
print("Python OK")
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("soundfile:", soundfile.__version__)
PY

echo ""
echo "环境就绪。激活: source ${CONDA_SH} && conda activate ${ENV_NAME}"
echo "配置数据路径: export ASVSPOOF5_DATA_DIR=/path/to/ASVspoof5 && bash scripts/patch_config_paths.sh"
