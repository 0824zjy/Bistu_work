#!/usr/bin/env bash

# ============================================================
# Common config for Work3 BEF-SBG
# ============================================================

export WORK_ROOT="${WORK_ROOT:-/data/zjy_work/Work3_BEF_SBG}"

export BGDNET_ROOT="${BGDNET_ROOT:-/data/zjy_work/BGDNet}"
export BGDIFF_ROOT="${BGDIFF_ROOT:-/data/zjy_work/BGDiff}"

export ISIC_TRAIN_ROOT="${ISIC_TRAIN_ROOT:-/data/zjy_work/ISIC2018/train}"
export ISIC_TEST_ROOT="${ISIC_TEST_ROOT:-/data/zjy_work/ISIC2018/test}"

export ISIC_TRAIN_IMAGE_DIR="${ISIC_TRAIN_IMAGE_DIR:-${ISIC_TRAIN_ROOT}/Images}"
export ISIC_TRAIN_MASK_DIR="${ISIC_TRAIN_MASK_DIR:-${ISIC_TRAIN_ROOT}/Masks}"

export RATIO_TAG="${RATIO_TAG:-5p}"
export SEED="${SEED:-0}"
export N_FOLDS="${N_FOLDS:-5}"

# Recommended unified split dir.
# Example:
#   /data/zjy_work/Work3_BEF_SBG/splits/ISIC2018_seed0/low_5p_all.txt
export SPLIT_DIR="${SPLIT_DIR:-${WORK_ROOT}/splits/ISIC2018_seed${SEED}}"

export LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"
export PID_DIR="${PID_DIR:-${WORK_ROOT}/logs/pids}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Teacher OOF outputs.
export TEACHER_CKPT_ROOT="${TEACHER_CKPT_ROOT:-${WORK_ROOT}/teacher/checkpoints/ISIC2018_${RATIO_TAG}}"
export OOF_OUT_DIR="${OOF_OUT_DIR:-${WORK_ROOT}/results/oof_teacher/ISIC2018_${RATIO_TAG}}"

# Boundary feedback outputs.
export FEEDBACK_OUT_DIR="${FEEDBACK_OUT_DIR:-${WORK_ROOT}/results/boundary_feedback/ISIC2018_${RATIO_TAG}}"

# BEF diffusion json.
export BEF_PROMPT_JSON="${BEF_PROMPT_JSON:-${WORK_ROOT}/feedback/prompt_bef_train_${RATIO_TAG}.json}"

# Diffusion logs.
export DIFF_LOG_ROOT="${DIFF_LOG_ROOT:-${WORK_ROOT}/logs/diffusion}"

export DIFF_STAGE1_EXP="${DIFF_STAGE1_EXP:-stage1_bef_sbg_${RATIO_TAG}}"
export DIFF_STAGE2_EXP="${DIFF_STAGE2_EXP:-stage2_bef_sbg_decoder_${RATIO_TAG}}"

export DIFF_STAGE1_LOG_DIR="${DIFF_LOG_ROOT}/${DIFF_STAGE1_EXP}"
export DIFF_STAGE2_LOG_DIR="${DIFF_LOG_ROOT}/${DIFF_STAGE2_EXP}"

# Generated samples.
export GEN_OUT_DIR="${GEN_OUT_DIR:-${WORK_ROOT}/results/generated_bef_sbg/${RATIO_TAG}}"

# Scoring outputs.
export GEN_SCORE_CSV="${GEN_SCORE_CSV:-${WORK_ROOT}/results/generated_bef_sbg_scores_${RATIO_TAG}.csv}"
export GEN_ACCEPTED_JSONL="${GEN_ACCEPTED_JSONL:-${WORK_ROOT}/results/generated_bef_sbg_accepted_${RATIO_TAG}.jsonl}"

# Final segmentation json and outputs.
export WEIGHTED_TRAIN_JSON="${WEIGHTED_TRAIN_JSON:-${WORK_ROOT}/results/weighted_train_${RATIO_TAG}_bef.jsonl}"
export FINAL_SEG_SAVE_DIR="${FINAL_SEG_SAVE_DIR:-${WORK_ROOT}/results/final_bgdnet_bef_${RATIO_TAG}}"
export FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${WORK_ROOT}/results/eval_final_bef_${RATIO_TAG}}"

mkdir -p "${PID_DIR}"
mkdir -p "${LOG_ROOT}"/{splits,teacher,infer,feedback,visual,diffusion,scoring,segmentation}
mkdir -p "${WORK_ROOT}"/{splits,feedback,results,teacher}
mkdir -p "${WORK_ROOT}/teacher/checkpoints"
mkdir -p "${WORK_ROOT}/results"
