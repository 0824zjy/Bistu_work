#!/usr/bin/env bash

# ============================================================
# Standard BGDNet environment
# Swin-T: embed_dim=96, patch_size=4
# ============================================================

export BGDNET_ARCH_NAME="standard_swin96_patch4"
export BGDNET_SWIN_VARIANT="standard"

# Current environment has confirmed cuDNN failure.
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"

# Required Python paths:
# /data/zjy_work          -> import BGDNet.models...
# /data/zjy_work/BGDNet   -> import models...
# /data/zjy_work/env_patch -> load sitecustomize.py
export PYTHONPATH="/data/zjy_work:/data/zjy_work/BGDNet:/data/zjy_work/env_patch:${PYTHONPATH:-}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments"* ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
fi
