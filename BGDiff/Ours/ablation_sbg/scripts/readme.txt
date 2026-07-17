Stage 1 启动命令

Mask-only
VARIANT=maskonly DATA_SPLIT=real5 GPU_ID=2 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/01_stage1_train_variant.sh

seed1:[OK] Stage1 ablation training started.
     variant: maskonly
     split:   real5
     gpu:     0
     logs:    /data/zjy_work/BGDiff/Ours/ablation_sbg/logs/abl_seed1_maskonly_real5_stage1

 VARIANT=hard DATA_SPLIT=real5 GPU_ID=3 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/01_stage1_train_variant.sh
[OK] Stage1 ablation training started.
     variant: hard
     split:   real5
     gpu:     1
     logs:    /data/zjy_work/BGDiff/Ours/ablation_sbg/logs/abl_seed1_hard_real5_stage1

 VARIANT=softonly DATA_SPLIT=real5 GPU_ID=2  bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/01_stage1_train_variant.sh
[OK] Stage1 ablation training started.
     variant: softonly
     split:   real5
     gpu:     2
     logs:    /data/zjy_work/BGDiff/Ours/ablation_sbg/logs/abl_seed1_softonly_real5_stage1

VARIANT=progressive DATA_SPLIT=real5 GPU_ID=0 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/01_stage1_train_variant.sh

Full SBG-Diff Stage 1，Real-5
VARIANT=full_sbg DATA_SPLIT=real5 GPU_ID=2 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/01_stage1_train_variant.sh

Full SBG-Diff Stage 1，Real-100(直接使用/data/zjy_work/BGDiff/Ours/scripts)
VARIANT=full_sbg DATA_SPLIT=real100 GPU_ID=3 \
bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/01_stage1_train_variant.sh



Stage 2 启动命令
Decoder modulation 消融

这个对应表 2 的：

+ Decoder modulation


它从 progressive 的 Stage 1 继续训练。

VARIANT=decoder_mod DATA_SPLIT=real5 GPU_ID=0 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/02_stage2_train_variant.sh

Full SBG-Diff Stage 2，Real-5
VARIANT=full_sbg DATA_SPLIT=real5 GPU_ID=3 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/02_stage2_train_variant.sh

Full SBG-Diff Stage 2，Real-100
VARIANT=full_sbg DATA_SPLIT=real100 GPU_ID=3 \
bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/02_stage2_train_variant.sh

推理启动命令
Mask-only
VARIANT=maskonly DATA_SPLIT=real5 GPU_ID=3 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh

Hard-boundary
VARIANT=hard DATA_SPLIT=real5 GPU_ID=2 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh

Soft prior only
VARIANT=softonly DATA_SPLIT=real5 GPU_ID=1 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh

Progressive
VARIANT=progressive DATA_SPLIT=real5 GPU_ID=1 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh

Decoder modulation
VARIANT=decoder_mod DATA_SPLIT=real5 GPU_ID=0 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh

Full SBG-Diff，Real-5
VARIANT=full_sbg DATA_SPLIT=real5 GPU_ID=3 bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh

Full SBG-Diff，Real-100
VARIANT=full_sbg DATA_SPLIT=real100 GPU_ID=3 \
bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/03_infer_variant.sh



bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/04_collect_real_images.sh


运行示例：

VARIANT=maskonly DATA_SPLIT=real5 \
bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/05_fid_kid_variant.sh

VARIANT=hard DATA_SPLIT=real5 \
bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/05_fid_kid_variant.sh

VARIANT=full_sbg DATA_SPLIT=real5 \
bash /data/zjy_work/BGDiff/Ours/ablation_sbg/scripts/05_fid_kid_variant.sh
