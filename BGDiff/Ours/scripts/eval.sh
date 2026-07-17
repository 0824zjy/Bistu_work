# #!/bin/bash
# set -e

# LOG_DIR=/data/zjy_work/BGDiff/Ours/logs/eval
# mkdir -p $LOG_DIR

# echo "Running cn_only..."
# python /data/zjy_work/BGDiff/Ours/eval_quality.py \
#   --real_images /data/zjy_work/ISIC2018/train/Images \
#   --real_masks /data/zjy_work/ISIC2018/train/Masks \
#   --gen_images /data/zjy_work/BGDiff/Ours/results/exp_mask2img_cn_only/images \
#   --gen_masks /data/zjy_work/BGDiff/Ours/results/exp_mask2img_cn_only/masks \
#   --device cuda \
#   --out_json /data/zjy_work/BGDiff/Ours/data_txt/gen_quality_metrics_cn_only.json \
#   | tee $LOG_DIR/eval_cn_only.log

echo "Running stage1_2..."
python /data/zjy_work/BGDiff/Ours/eval_quality.py \
  --real_images /data/zjy_work/ISIC2018/train/Images \
  --real_masks /data/zjy_work/ISIC2018/train/Masks \
  --gen_images /data/zjy_work/BGDiff/Ours/results/exp_mask2img_stage1_2/images \
  --gen_masks /data/zjy_work/BGDiff/Ours/results/exp_mask2img_stage1_2/masks \
  --device cuda \
  --out_json /data/zjy_work/BGDiff/Ours/data_txt/gen_quality_metrics_stage1-2.json \
  | tee $LOG_DIR/eval_stage1_2.log

echo "Done."
