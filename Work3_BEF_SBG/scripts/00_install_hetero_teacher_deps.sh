#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_DIR="${PRETRAIN_DIR:-/data/zjy_work/pretrained}"
SAM_CKPT="${SAM_CKPT:-${PRETRAIN_DIR}/sam_vit_b_01ec64.pth}"
mkdir -p "${PRETRAIN_DIR}"

python -m pip install --upgrade "git+https://github.com/facebookresearch/segment-anything.git"

if [ ! -f "${SAM_CKPT}" ]; then
  echo "[DOWNLOAD] official SAM ViT-B checkpoint -> ${SAM_CKPT}"
  python - <<PY
from urllib.request import urlretrieve
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
out = r"${SAM_CKPT}"
urlretrieve(url, out)
print(out)
PY
else
  echo "[OK] SAM checkpoint exists: ${SAM_CKPT}"
fi

python - <<'PY'
import torch
import torchvision
from segment_anything import sam_model_registry
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("SAM models:", sorted(sam_model_registry.keys()))
PY
