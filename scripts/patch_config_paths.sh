#!/usr/bin/env bash
# Set database_path in all Baseline-AASIST configs from ASVSPOOF5_DATA_DIR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ASVSPOOF5_DATA_DIR:-/path/to/ASVspoof5}"
CONF_DIR="${ROOT}/Baseline-AASIST/config"

if [[ ! -d "${DATA}" ]]; then
  echo "警告: 数据目录不存在: ${DATA}" >&2
  echo "请 export ASVSPOOF5_DATA_DIR=/your/ASVspoof5 后重试" >&2
fi

for f in "${CONF_DIR}"/*.conf; do
  [[ -f "${f}" ]] || continue
  python3 -c "
import json, sys
p = sys.argv[1]
data = sys.argv[2]
with open(p, encoding='utf-8') as fh:
    c = json.load(fh)
c['database_path'] = data
with open(p, 'w', encoding='utf-8') as fh:
    json.dump(c, fh, indent=4)
    fh.write('\n')
" "${f}" "${DATA}"
done
echo "已更新 ${CONF_DIR} 中 database_path -> ${DATA}"
