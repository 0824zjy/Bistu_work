下面是完整从头跑 Work3 BEF-SBG 的推荐顺序。
conda rename -n BGDiff SBGDiff 
Step 01：生成低标注划分
RATIO_TAG=5p SEED=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/01_make_low_label_splits.sh


查看：

tail -f /data/zjy_work/Work3_BEF_SBG/logs/splits/make_low_label_splits_seed0_*.log

Step 02：训练 5-fold OOF teacher
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/02_train_oof_teachers.sh


查看：

tail -f /data/zjy_work/Work3_BEF_SBG/logs/teacher/train_oof_teachers_5p_*.log

Step 03：OOF teacher 预测
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/03_infer_oof_predictions.sh

Step 04：构建边界误差反馈图
RATIO_TAG=5p SEED=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/04_build_boundary_feedback.sh

Step 05：可视化反馈图，可选
RATIO_TAG=5p SEED=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/05_visualize_feedback.sh

Step 06：生成 BEF diffusion prompt json
RATIO_TAG=5p SEED=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/06_make_bef_prompt_json.sh


检查：

wc -l /data/zjy_work/Work3_BEF_SBG/feedback/prompt_bef_train_5p.json

Step 07：训练扩散 Stage1
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/07_train_diffusion_stage1.sh


查看：

tail -f /data/zjy_work/Work3_BEF_SBG/logs/diffusion/stage1_bef_sbg_5p/train_stage1_bef_sbg_5p_*.log

Step 08：训练扩散 Stage2
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/08_train_diffusion_stage2.sh


查看：

tail -f /data/zjy_work/Work3_BEF_SBG/logs/diffusion/stage2_bef_sbg_decoder_5p/train_stage2_bef_sbg_5p_*.log

Step 09：推理生成 BEF-SBG 增强样本
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 N_SAMPLES=2 DDIM_STEPS=70 CFG=9.0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/09_infer_diffusion_bef_sbg.sh


输出：

/data/zjy_work/Work3_BEF_SBG/results/generated_bef_sbg/5p/
├── images/
├── masks/
├── boundary_prior/
├── difficulty/
└── boundary_hard/

Step 10：对生成样本评分
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
QUALITY_MODEL_PTH=/data/zjy_work/Work3_BEF_SBG/teacher/checkpoints/ISIC2018_5p/fold0/BGDNet-best.pth \
bash /data/zjy_work/Work3_BEF_SBG/scripts/10_score_generated_samples.sh


输出：

/data/zjy_work/Work3_BEF_SBG/results/generated_bef_sbg_scores_5p.csv
/data/zjy_work/Work3_BEF_SBG/results/generated_bef_sbg_accepted_5p.jsonl

Step 11：构建加权 BGDNet 训练集
RATIO_TAG=5p SEED=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/11_make_weighted_train_json.sh


输出：

/data/zjy_work/Work3_BEF_SBG/results/weighted_train_5p_bef.jsonl

Step 12：训练最终 BGDNet-BEF
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/12_train_final_bgdnet_bef.sh


输出：

/data/zjy_work/Work3_BEF_SBG/results/final_bgdnet_bef_5p/
├── BGDNet-BEF-last.pth
├── BGDNet-BEF-best.pth
└── train.log

Step 13：测试最终 BGDNet-BEF
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/13_test_final_bgdnet_bef.sh


输出：

/data/zjy_work/Work3_BEF_SBG/results/eval_final_bef_5p/
├── masks/
├── boundaries/
├── per_image_metrics.csv
└── summary_metrics.csv


查看最终指标：

cat /data/zjy_work/Work3_BEF_SBG/results/eval_final_bef_5p/summary_metrics.csv

四、消融实验怎么启动
1. 跑 10% 低标注

不需要改脚本，直接覆盖 RATIO_TAG：

RATIO_TAG=10p SEED=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/01_make_low_label_splits.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/02_train_oof_teachers.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/03_infer_oof_predictions.sh
RATIO_TAG=10p SEED=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/04_build_boundary_feedback.sh
RATIO_TAG=10p SEED=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/06_make_bef_prompt_json.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/07_train_diffusion_stage1.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/08_train_diffusion_stage2.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/09_infer_diffusion_bef_sbg.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/10_score_generated_samples.sh
RATIO_TAG=10p SEED=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/11_make_weighted_train_json.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/12_train_final_bgdnet_bef.sh
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 bash /data/zjy_work/Work3_BEF_SBG/scripts/13_test_final_bgdnet_bef.sh

2. 跑 20% 低标注
RATIO_TAG=20p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/02_train_oof_teachers.sh


其他步骤同理。

3. 改生成样本数
RATIO_TAG=5p N_SAMPLES=5 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/09_infer_diffusion_bef_sbg.sh

4. 改评分阈值
RATIO_TAG=5p CONS_THRESHOLD=0.80 BETA_HARD=0.5 CUDA_VISIBLE_DEVICES=0 \
bash /data/zjy_work/Work3_BEF_SBG/scripts/10_score_generated_samples.sh

五、状态查看与终止
查看所有任务状态
RATIO_TAG=5p \
bash /data/zjy_work/Work3_BEF_SBG/scripts/show_status.sh

终止某一步

例如终止扩散 Stage1：

bash /data/zjy_work/Work3_BEF_SBG/scripts/kill_by_pid.sh \
  07_train_diffusion_stage1_5p


终止最终 BGDNet 训练：

bash /data/zjy_work/Work3_BEF_SBG/scripts/kill_by_pid.sh \
  12_train_final_bgdnet_bef_5p


也可以直接：

kill $(cat /data/zjy_work/Work3_BEF_SBG/logs/pids/12_train_final_bgdnet_bef_5p.pid)
