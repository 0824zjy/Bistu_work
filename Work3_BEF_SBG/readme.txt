文件一：低标注 split 生成脚本

python /data/zjy_work/Work3_BEF_SBG/splits/make_low_label_splits.py \
  --image_dir /data/zjy_work/ISIC2018/train/Images \
  --mask_dir /data/zjy_work/ISIC2018/train/Masks \
  --out_dir /data/zjy_work/Work3_BEF_SBG/splits \
  --ratios 0.05,0.10,0.20,1.0 \
  --n_folds 5 \
  --seed 0

文件二：5-fold teacher 训练启动脚本

chmod +x /data/zjy_work/Work3_BEF_SBG/teacher/train_oof_teachers.sh

CUDA_VISIBLE_DEVICES=0 \
RATIO_TAG=5p \
bash /data/zjy_work/Work3_BEF_SBG/teacher/train_oof_teachers.sh
跑 10% teacher 时：
CUDA_VISIBLE_DEVICES=0 \
RATIO_TAG=10p \
bash /data/zjy_work/Work3_BEF_SBG/teacher/train_oof_teachers.sh

文件三：OOF teacher 预测脚本

python /data/zjy_work/Work3_BEF_SBG/teacher/infer_oof_predictions.py \
  --bgdnet_root /data/zjy_work/BGDNet \
  --test_data_path /data/zjy_work/ISIC2018/train \
  --split_dir /data/zjy_work/Work3_BEF_SBG/splits \
  --ratio_tag 5p \
  --n_folds 5 \
  --ckpt_root /data/zjy_work/Work3_BEF_SBG/teacher/checkpoints/ISIC2018_5p \
  --out_dir /data/zjy_work/Work3_BEF_SBG/results/oof_teacher/ISIC2018_5p \
  --testsize 352 \
  --device cuda:0

文件四：边界误差反馈构建脚本

python /data/zjy_work/Work3_BEF_SBG/feedback/build_boundary_feedback.py \
  --image_dir /data/zjy_work/ISIC2018/train/Images \
  --mask_dir /data/zjy_work/ISIC2018/train/Masks \
  --pred_mask_dir /data/zjy_work/Work3_BEF_SBG/results/oof_teacher/ISIC2018_5p/pred_masks \
  --pred_boundary_dir /data/zjy_work/Work3_BEF_SBG/results/oof_teacher/ISIC2018_5p/pred_boundaries \
  --list_txt /data/zjy_work/Work3_BEF_SBG/splits/low_5p_all.txt \
  --out_dir /data/zjy_work/Work3_BEF_SBG/results/boundary_feedback/ISIC2018_5p \
  --radius 12 \
  --tau0 4.0 \
  --kernel 3 \
  --gamma 2.0 \
  --lambda_u 0.5

文件五：边界误差反馈可视化脚本

默认可视化 low_5p_all.txt 前 12 张：

python /data/zjy_work/Work3_BEF_SBG/feedback/visualize_feedback.py \
  --image_dir /data/zjy_work/ISIC2018/train/Images \
  --mask_dir /data/zjy_work/ISIC2018/train/Masks \
  --pred_mask_dir /data/zjy_work/Work3_BEF_SBG/results/oof_teacher/ISIC2018_5p/pred_masks \
  --feedback_dir /data/zjy_work/Work3_BEF_SBG/results/boundary_feedback/ISIC2018_5p \
  --list_txt /data/zjy_work/Work3_BEF_SBG/splits/low_5p_all.txt \
  --out_dir /data/zjy_work/Work3_BEF_SBG/results/visual_feedback \
  --max_samples 12 \
  --tile_size 224


指定样本可视化：

python /data/zjy_work/Work3_BEF_SBG/feedback/visualize_feedback.py \
  --image_dir /data/zjy_work/ISIC2018/train/Images \
  --mask_dir /data/zjy_work/ISIC2018/train/Masks \
  --pred_mask_dir /data/zjy_work/Work3_BEF_SBG/results/oof_teacher/ISIC2018_5p/pred_masks \
  --feedback_dir /data/zjy_work/Work3_BEF_SBG/results/boundary_feedback/ISIC2018_5p \
  --list_txt /data/zjy_work/Work3_BEF_SBG/splits/low_5p_all.txt \
  --out_dir /data/zjy_work/Work3_BEF_SBG/results/visual_feedback \
  --samples ISIC_0000000,ISIC_0000001 \
  --tile_size 224