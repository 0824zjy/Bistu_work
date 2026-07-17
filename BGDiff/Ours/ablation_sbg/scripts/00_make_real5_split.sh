#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
ABLATION_ROOT="${PROJECT_DIR}/Ours/ablation_sbg"
LOG_ROOT="${ABLATION_ROOT}/logs/split"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

export SEED=0
export RATIO=0.05
export SRC_JSON="${PROJECT_DIR}/data/prompt_PH2_train.json"
export DST_JSON="${PROJECT_DIR}/data/prompt_PH2_train_5p_seed0.json"

nohup python -u - <<'PY' \
  > "${LOG_ROOT}/make_real5_split_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
import os
import random
from pathlib import Path

seed = int(os.environ.get("SEED", "0"))
ratio = float(os.environ.get("RATIO", "0.05"))

src = Path(os.environ["SRC_JSON"])
dst = Path(os.environ["DST_JSON"])

lines = src.read_text().strip().splitlines()
random.Random(seed).shuffle(lines)

n = max(1, int(len(lines) * ratio))
subset = lines[:n]

dst.write_text("\n".join(subset) + "\n")

print(f"[OK] total={len(lines)}, selected={n}, saved={dst}")
PY

echo "[OK] Real-5 split generation started."
echo "     output: ${DST_JSON}"
echo "     logs:   ${LOG_ROOT}"
