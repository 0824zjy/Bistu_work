#!/usr/bin/env bash
set -e

cd /data/zjy_work/BGDNet
export CUDA_VISIBLE_DEVICES=3

python - <<'PY'
from utils.dataloader_BGDiff import build_pairs_from_txt

def check(image_root, mask_root, txt, mode):
    imgs, gts = build_pairs_from_txt(
        image_root=image_root,
        gt_root=mask_root,
        list_txt=txt,
        mode=mode,
        mask_suffix="_segmentation",
        verbose=True
    )
    print(f"==> FINAL: {txt} -> pairs={len(imgs)}\n")

# 真实训练（ISIC train：mask 通常 *_segmentation.png）
check(
    image_root="/data/zjy_work/ISIC2018/train/Images",
    mask_root="/data/zjy_work/ISIC2018/train/Masks",
    txt="/data/zjy_work/data_txt/train_5p.txt",
    mode="isic"
)

# 生成训练（cn_only）
check(
    image_root="/data/zjy_work/ISIC2018/exp_mask2img_cn_only/images",
    mask_root="/data/zjy_work/ISIC2018/exp_mask2img_cn_only/masks",
    txt="/data/zjy_work/data_txt/mix_ratio/train_5p__cn_only__real1_gen1__nreal130_ngen130.txt",
    mode="paired"
)

# 测试（ISIC test：mask 是同名 .jpg）
check(
    image_root="/data/zjy_work/ISIC2018/test/Images",
    mask_root="/data/zjy_work/ISIC2018/test/Masks",
    txt="/data/zjy_work/data_txt/test.txt",
    mode="isic"
)
PY


# [TXT LOAD] /data/zjy_work/data_txt/train_5p.txt
#   image_root: /data/zjy_work/ISIC2018/train/Images
#   gt_root   : /data/zjy_work/ISIC2018/train/Masks
#   mode      : isic
#   ids=130, pairs=130, miss_img=0, miss_mask=0
# ==> FINAL: /data/zjy_work/data_txt/train_5p.txt -> pairs=130

# [TXT LOAD] /data/zjy_work/data_txt/mix_ratio/train_5p__cn_only__real1_gen1__nreal130_ngen130.txt
#   image_root: /data/zjy_work/ISIC2018/exp_mask2img_cn_only/images
#   gt_root   : /data/zjy_work/ISIC2018/exp_mask2img_cn_only/masks
#   mode      : paired
#   ids=130, pairs=130, miss_img=0, miss_mask=0
# ==> FINAL: /data/zjy_work/data_txt/mix_ratio/train_5p__cn_only__real1_gen1__nreal130_ngen130.txt -> pairs=130

# [TXT LOAD] /data/zjy_work/data_txt/test.txt
#   image_root: /data/zjy_work/ISIC2018/test/Images
#   gt_root   : /data/zjy_work/ISIC2018/test/Masks
#   mode      : isic
#   ids=1000, pairs=1000, miss_img=0, miss_mask=0
# ==> FINAL: /data/zjy_work/data_txt/test.txt -> pairs=1000