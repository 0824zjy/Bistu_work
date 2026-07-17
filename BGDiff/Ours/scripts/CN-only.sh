#修改logger.py中        root = os.path.join(save_dir, "image_log", split)
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
EXP_NAME="exp_mask2img_cn_only"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=0

# 可复现
export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

export RESUME_PATH="${PROJECT_DIR}/Ours/models/control_sd15_init.pth"
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"

export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8
export LR=1e-5
export MAX_STEPS=30000
export LOGGER_FREQ=2000
export LOG_ROOT="${LOG_ROOT}"
export EXP_NAME="${EXP_NAME}"
export IMG_SIZE=384
export EMPTY_PROMPT_PROB=0.02
export PRECISION=16-mixed
export NUM_WORKERS=0

export TRAIN_UNET_DECODER=0
export SD_LOCKED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nohup python -u "${PROJECT_DIR}/tutorial_train.py" \
  > "${LOG_ROOT}/${EXP_NAME}/train_1gpu_cn_only_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] training started (1 GPU, CN-only). logs in ${LOG_ROOT}/${EXP_NAME}"
