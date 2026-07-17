# #!/usr/bin/env bash
# set -e

# PROJECT_DIR="/data/zjy_work/BGDiff"
# LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
# OUT_DIR="/data/zjy_work/BGDiff/Ours/results/exp_mask2img_stage1_2"
# EXP_NAME="exp_mask2img"

# mkdir -p "${LOG_ROOT}/${EXP_NAME}"
# mkdir -p "${OUT_DIR}"
# cd "${PROJECT_DIR}"

# export CUDA_VISIBLE_DEVICES=0

# # 推理配置
# export CKPT_PATH="/data/zjy_work/BGDiff/Ours/models/merged_stage1_2_model.pth"
# export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"
# export OUT_DIR="${OUT_DIR}"
# export DEVICE="cuda:0"

# # 吞吐：一次只读1条样本（确保“每条样本生成 n_samples 张”严格对应）
# export BATCH_SIZE=1
# # 多样性：每条样本生成1张
# export N_SAMPLES=1

# export NUM_WORKERS=4

# export SAMPLER="ddim"
# export DDIM_STEPS=70
# export DPM_STEPS=20
# export HYBRID_SPLIT=0.5
# export CFG=9.0

# # mask-only 默认：0；若要 mask+image 增强，改成 1
# export USE_IMAGE_CONTROL=0

# nohup python -u /data/zjy_work/BGDiff/tutorial_inference.py \
#   > "${LOG_ROOT}/${EXP_NAME}/infer_stage1_2_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# echo "[OK] inference started. results in ${OUT_DIR}"


#4,14
#!/usr/bin/env bash
set -e

PROJECT_DIR="/data/zjy_work/BGDiff"
LOG_ROOT="/data/zjy_work/BGDiff/Ours/logs"
OUT_DIR="/data/zjy_work/BGDiff/Ours/results/exp_mask2img_region_boundary"
EXP_NAME="exp_mask2img_region_boundary_infer"

mkdir -p "${LOG_ROOT}/${EXP_NAME}"
mkdir -p "${OUT_DIR}"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES=0

export CKPT_PATH="/data/zjy_work/BGDiff/Ours/models/merged_stage1_2_model.pth"
export PROMPT_JSON="${PROJECT_DIR}/data/prompt_train.json"
export OUT_DIR="${OUT_DIR}"
export DEVICE="cuda:0"

export BATCH_SIZE=1
export N_SAMPLES=1
export NUM_WORKERS=4

export SAMPLER="ddim"
export DDIM_STEPS=70
export DPM_STEPS=20
export HYBRID_SPLIT=0.5
export CFG=9.0

# mask-only 推理，boundary 由 dataset 自动生成
export USE_IMAGE_CONTROL=0

nohup python -u /data/zjy_work/BGDiff/tutorial_inference.py \
  > "${LOG_ROOT}/${EXP_NAME}/infer_region_boundary_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "[OK] region-boundary inference started. results in ${OUT_DIR}"
