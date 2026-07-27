# Work: Boundary-Guided Dermoscopic Image Synthesis and Lesion Segmentation

本仓库用于皮肤镜图像合成与病灶分割实验，围绕低标注场景下的医学图像数据增强、边界先验建模和下游分割性能提升展开。项目主要包含三部分：

1. **SBG-Diff**：Soft Boundary Guided Diffusion，用于基于病灶 mask 生成皮肤镜图像。
2. **BGDNet**：Boundary-Guided Dual-Backbone Network，用于皮肤病灶分割。
3. **BEF-SBG**：Boundary Error Feedback + Soft Boundary Guidance 流水线，用于结合 teacher 反馈、扩散生成、样本筛选和最终分割训练。

本项目主要面向 ISIC2018、PH2 等皮肤镜图像数据集，支持真实数据训练、合成数据增强训练、低标注划分实验和端到端 BEF-SBG 实验流程。

---

## 1. 项目简介

皮肤病灶分割依赖大量精确像素级标注，但皮肤镜图像中的病灶边界通常存在模糊、低对比度、颜色渐变和局部不确定等问题。因此，本项目围绕两个核心问题展开：

* 如何在低标注条件下生成与真实病灶 mask 空间一致、且边界过渡自然的皮肤镜图像；
* 如何在分割模型中更有效地利用边界信息，提升弱边界和不规则病灶区域的分割性能。

对应地，本仓库实现的主要代码：

### SBG-Diff

SBG-Diff 将病灶 mask 作为主要空间条件，并从 mask 中构建软边界先验。与直接使用硬边界不同，SBG-Diff 将边界视为弱的、不确定的局部先验，通过以下方式逐步注入扩散模型：

* soft boundary prior 构建；
* timestep-aware ControlNet hint fusion；
* late decoder boundary modulation；
* low-noise tolerance-band regularization；
* 两阶段训练策略。

该模块主要用于固定 mask 条件下的皮肤镜图像生成和低标注数据增强。

### BGDNet

BGDNet 是下游皮肤病灶分割网络。模型采用双 backbone 编码器结构，结合 Swin Transformer 的全局建模能力和 TransXNet 的局部结构建模能力，并通过边界引导的跨分支注意力进行特征融合。

主要组件包括：

* Swin Transformer branch；
* TransXNet branch；
* Boundary Extraction Module，BEM；
* Boundary-Guided Cross-Attention，BGCA；
* Multi-Scale Deformable Decoder，MSDC；
* lesion mask 与 auxiliary boundary 的联合监督。

该模块可用于真实数据训练，也可用于真实数据 + SBG-Diff 合成数据的增强训练。

### BEF-SBG

BEF-SBG 是完整实验流水线，在 SBG-Diff 与 BGDNet 的基础上进一步引入边界误差反馈。整体流程包括：

1. 构建低标注划分；
2. 训练 K-fold OOF teacher；
3. 使用 teacher 预测训练集中的病灶 mask 和边界；
4. 根据预测误差生成 boundary error feedback；
5. 将反馈信息用于扩散生成；
6. 对合成样本进行质量评分；
7. 构建加权训练集；
8. 训练最终 BGDNet-BEF 分割模型；
9. 在测试集上评估最终结果。

---

## 2. 目录结构

```text
Work/
├── BGDNet/
│   ├── models/                 # BGDNet、UNet、TransXNet 等网络结构
│   ├── pretrained_pth/          # 分割模型预训练权重
│   ├── scripts/                 # BGDNet 训练、测试和数据检查脚本
│   ├── utils/                   # 数据加载、增强、预处理和训练工具
│   ├── BGDiff_train.py          # 分割训练入口
│   ├── BGDiff_test.py           # 分割测试入口
│   ├── train_ISIC2018.py        # ISIC2018 训练脚本
│   ├── test_ISIC2018.py         # ISIC2018 测试脚本
│   ├── train_ISIC2016.py        # ISIC2016 训练脚本
│   ├── test_ISIC2016.py         # ISIC2016 测试脚本
│   └── losses.py                # 分割损失函数
│
├── SBG_Diff/
│   ├── Baseline/                # baseline 相关代码
│   ├── Ours/                    # SBG-Diff 主要实验代码
│   │   ├── ablation_sbg/         # SBG-Diff 消融实验
│   │   ├── data_txt/             # 数据列表文件
│   │   ├── logs/                 # 训练和推理日志
│   │   ├── models/               # 初始化权重和合并权重
│   │   └── scripts/              # 训练、合并、推理、评估脚本
│   ├── cldm/                    # ControlNet / ControlLDM 相关实现
│   ├── data/                    # prompt json 文件
│   ├── data_txt/                # 训练、测试和混合比例 txt 文件
│   ├── ldm/                     # Latent Diffusion 基础模块
│   ├── models/                  # 扩散模型配置文件
│   ├── config.py                # 简单运行配置
│   ├── make_prompt_json.py      # 生成 prompt json
│   ├── tool_add_control.py      # 生成 ControlNet 初始化权重
│   └── tool_merge_control.py    # 合并 ControlNet 权重
│
├── BEF_SBG/
│   ├── diffusion/               # BEF-SBG 扩散训练和推理代码
│   ├── feedback/                # 边界误差反馈构建与可视化
│   ├── scoring/                 # 合成样本评分与筛选
│   ├── scripts/                 # 00-13 端到端实验脚本
│   ├── segmentation/            # 最终分割训练与测试
│   ├── splits/                  # 低标注划分与 K-fold 划分
│   └── infer_oof_predictions.py # OOF teacher 预测脚本
│
├── requirements_BGDNet.txt      # BGDNet 环境依赖
├── requirements_SBGDIff.txt     # SBG-Diff 环境依赖
├── LICENSE
├── PROJECT_FILE_ANALYSIS.md     # 主要路径文件详解
└── README.md
```

---

## 3. 环境配置

建议分别为分割模型和扩散模型创建独立环境，SBG-Diff环境也可进行分割模型训练（不建议统一使用，版本存在冲突）。

### 3.1 BGDNet 环境

```bash
conda create -n bgdnet python=3.8 -y
conda activate bgdnet

pip install -r requirements_BGDNet.txt
```

### 3.2 SBG-Diff 环境

```bash
conda create -n sbgdiff python=3.8 -y
conda activate sbgdiff

pip install -r requirements_SBGDIff.txt
```

如使用 Stable Diffusion v1.5 或 ControlNet 初始化权重，请将所需权重文件放入：

```text
SBG_Diff/Ours/models/
```

示例文件包括：

```text
v1-5-pruned-emaonly.safetensors
control_sd15_init.pth
merged_*.pth
```

---

## 4. 数据准备

本项目默认使用 ISIC2018 和 PH2 数据集。建议整理为如下结构：

```text
datasets/
├── ISIC2018/
│   ├── train/
│   │   ├── Images/
│   │   └── Masks/
│   └── test/
│       ├── Images/
│       └── Masks/
│
└── PH2/
    ├── train/
    │   ├── Images/
    │   └── Masks/
    └── test/
        ├── Images/
        └── Masks/
```

其中：

* `Images/` 存放皮肤镜原图；
* `Masks/` 存放对应二值病灶 mask；
* mask 文件应与 image 文件在命名上保持可匹配关系；
* 训练和测试数据列表可存放在 `SBG_Diff/data_txt/` 或 `BEF_SBG/splits/` 中。

---

## 5. SBG-Diff 使用说明

SBG-Diff 用于从病灶 mask 生成皮肤镜图像。它以 mask 作为主条件，并从 mask 中在线构建 soft boundary prior，用于弱边界引导的扩散生成。

### 5.1 生成 prompt json

以 PH2 为例：

```bash
python SBG_Diff/make_prompt_json.py \
  --img_dir /path/to/PH2/train/Images \
  --mask_dir /path/to/PH2/train/Masks \
  --out SBG_Diff/data/prompt_PH2_train.json \
  --prompt "dermoscopy image"

python SBG_Diff/make_prompt_json.py \
  --img_dir /path/to/PH2/test/Images \
  --mask_dir /path/to/PH2/test/Masks \
  --out SBG_Diff/data/prompt_PH2_test.json \
  --prompt "dermoscopy image"
```

### 5.2 初始化 ControlNet 权重

该步骤通常只需执行一次：

```bash
cd SBG_Diff

python tool_add_control.py \
  Ours/models/v1-5-pruned-emaonly.safetensors \
  Ours/models/control_sd15_init.pth
```

### 5.3 训练 SBG-Diff

进入项目目录：

```bash
cd SBG_Diff
conda activate sbgdiff
```

只训练 mask-conditioned ControlNet baseline：

```bash
bash Ours/scripts/CN-only.sh
```

训练 SBG-Diff stage 1：

```bash
bash Ours/scripts/stage1_sbp_pg.sh
```

训练 SBG-Diff stage 2：

```bash
bash Ours/scripts/stage2_sbp_pg.sh
```

### 5.4 合并推理权重

```bash
bash Ours/scripts/merge.sh
```

合并后权重通常保存在：

```text
SBG_Diff/Ours/models/
```

### 5.5 推理生成图像

```bash
bash Ours/scripts/infer_sbp_pg.sh
```

推理时只需要输入病灶 mask，soft boundary prior 会由程序根据 mask 自动生成。

### 5.6 生成质量评估

```bash
bash Ours/scripts/eval.sh
```

常用生成质量指标包括：

* FID；
* KID；
* downstream segmentation Dice；
* downstream segmentation IoU。

---

## 6. BGDNet 使用说明

BGDNet 用于训练和测试皮肤病灶分割模型。它既可以只使用真实数据训练，也可以使用真实数据与 SBG-Diff 合成数据联合训练。

### 6.1 数据加载检查

```bash
cd BGDNet
conda activate bgdnet

bash scripts/01_check_load.sh
```

### 6.2 只使用真实数据训练

```bash
bash scripts/02_train_real.sh
```

### 6.3 使用 CN-only 合成数据训练

```bash
bash scripts/02_train_cn_only.sh
```

### 6.4 使用 SBG-Diff stage1-2 合成数据训练

```bash
bash scripts/02_train_stage1-2.sh
```

### 6.5 测试分割模型

```bash
bash scripts/03_test.sh
```

训练输出通常保存在：

```text
BGDNet/model_out/
```

测试结果通常包括：

* predicted masks；
* predicted boundaries；
* per-image metrics；
* summary metrics。

---

## 7. BEF-SBG 完整流程

BEF-SBG 是用于在低标注场景下结合 teacher 边界误差反馈、软边界扩散生成和加权分割训练。

所有主流程脚本位于：

```text
BEF_SBG/scripts/
```

公共路径、GPU、输出目录等配置建议优先在以下文件中修改：

```text
BEF_SBG/scripts/00_common.sh
```

### 7.1 Step 01：生成低标注划分

```bash
RATIO_TAG=5p SEED=0 \
bash BEF_SBG/scripts/01_make_low_label_splits.sh
```

查看日志：

```bash
tail -f BEF_SBG/logs/splits/make_low_label_splits_seed0_*.log
```

### 7.2 Step 02：训练 5-fold OOF teacher

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/02_train_oof_teachers.sh
```

### 7.3 Step 03：OOF teacher 预测

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/03_infer_oof_predictions.sh
```

### 7.4 Step 04：构建边界误差反馈图

```bash
RATIO_TAG=5p SEED=0 \
bash BEF_SBG/scripts/04_build_boundary_feedback.sh
```

该步骤会根据 teacher 预测结果与真实 mask 的差异构建边界误差反馈，用于后续 BEF-SBG 扩散生成。

### 7.5 Step 05：可视化反馈图，可选

```bash
RATIO_TAG=5p SEED=0 \
bash BEF_SBG/scripts/05_visualize_feedback.sh
```

### 7.6 Step 06：生成 BEF diffusion prompt json

```bash
RATIO_TAG=5p SEED=0 \
bash BEF_SBG/scripts/06_make_bef_prompt_json.sh
```

检查生成结果：

```bash
wc -l BEF_SBG/feedback/prompt_bef_train_5p.json
```

### 7.7 Step 07：训练扩散 Stage 1

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/07_train_diffusion_stage1.sh
```

### 7.8 Step 08：训练扩散 Stage 2

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/08_train_diffusion_stage2.sh
```

### 7.9 Step 09：推理生成 BEF-SBG 增强样本

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 N_SAMPLES=2 DDIM_STEPS=70 CFG=9.0 \
bash BEF_SBG/scripts/09_infer_diffusion_bef_sbg.sh
```

输出示例：

```text
BEF_SBG/results/generated_bef_sbg/5p/
├── images/
├── masks/
├── boundary_prior/
├── difficulty/
└── boundary_hard/
```

### 7.10 Step 10：对生成样本评分

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
QUALITY_MODEL_PTH=BEF_SBG/teacher/checkpoints/ISIC2018_5p/fold0/BGDNet-best.pth \
bash BEF_SBG/scripts/10_score_generated_samples.sh
```

输出示例：

```text
BEF_SBG/results/generated_bef_sbg_scores_5p.csv
BEF_SBG/results/generated_bef_sbg_accepted_5p.jsonl
```

### 7.11 Step 11：构建加权 BGDNet 训练集

```bash
RATIO_TAG=5p SEED=0 \
bash BEF_SBG/scripts/11_make_weighted_train_json.sh
```

输出示例：

```text
BEF_SBG/results/weighted_train_5p_bef.jsonl
```

### 7.12 Step 12：训练最终 BGDNet-BEF

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/12_train_final_bgdnet_bef.sh
```

输出示例：

```text
BEF_SBG/results/final_bgdnet_bef_5p/
├── BGDNet-BEF-last.pth
├── BGDNet-BEF-best.pth
└── train.log
```

### 7.13 Step 13：测试最终 BGDNet-BEF

```bash
RATIO_TAG=5p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/13_test_final_bgdnet_bef.sh
```

输出示例：

```text
BEF_SBG/results/eval_final_bef_5p/
├── masks/
├── boundaries/
├── per_image_metrics.csv
└── summary_metrics.csv
```

查看最终指标：

```bash
cat BEF_SBG/results/eval_final_bef_5p/summary_metrics.csv
```

---

## 8. 常用实验设置

### 8.1 切换低标注比例

默认示例使用 `5p`，如需运行 10% 或 20% 低标注实验，只需修改 `RATIO_TAG`：

```bash
RATIO_TAG=10p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/02_train_oof_teachers.sh
```

或：

```bash
RATIO_TAG=20p SEED=0 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/02_train_oof_teachers.sh
```

### 8.2 修改生成样本数

```bash
RATIO_TAG=5p N_SAMPLES=5 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/09_infer_diffusion_bef_sbg.sh
```

### 8.3 修改评分阈值

```bash
RATIO_TAG=5p CONS_THRESHOLD=0.80 BETA_HARD=0.5 CUDA_VISIBLE_DEVICES=0 \
bash BEF_SBG/scripts/10_score_generated_samples.sh
```

### 8.4 查看任务状态

```bash
RATIO_TAG=5p \
bash BEF_SBG/scripts/show_status.sh
```

### 8.5 终止任务

例如终止扩散 Stage 1：

```bash
bash BEF_SBG/scripts/kill_by_pid.sh 07_train_diffusion_stage1_5p
```

例如终止最终 BGDNet 训练：

```bash
bash BEF_SBG/scripts/kill_by_pid.sh 12_train_final_bgdnet_bef_5p
```

也可以直接使用 pid 文件终止：

```bash
kill $(cat BEF_SBG/logs/pids/12_train_final_bgdnet_bef_5p.pid)
```

---

## 9. 结果说明

### 9.1 SBG-Diff 相关结果

论文中，SBG-Diff 在 ISIC2018 低标注固定 mask 生成协议下表现优于 mask-only ControlNet 和 hard-boundary 条件扩散变体。其核心结论是：对于皮肤镜图像合成，边界信息不应被视为强制硬边缘，而应作为弱的、不确定的、逐步注入的局部先验。

推荐关注以下指标：

* 下游 U-Net 或 BGDNet 的 Dice；
* 下游 U-Net 或 BGDNet 的 IoU；
* FID；
* KID；
* 合成图像边界区域视觉质量；
* 合成样本对下游分割训练的增益。

### 9.2 BGDNet 相关结果

论文中，BGDNet 在 ISIC2018、ISIC2016 和 PH2 上验证了边界引导双 backbone 分割框架的有效性。模型通过 Swin Transformer 和 TransXNet 建模全局上下文与局部结构，并利用 BEM 和 BGCA 进行边界感知特征融合。

推荐关注以下指标：

* IoU；
* Dice / DSC；
* Sensitivity；
* Accuracy；
* boundary prediction quality；
* weak-margin lesion cases 的分割效果。

### 9.3 BEF-SBG 相关结果

BEF-SBG 主要用于验证边界误差反馈是否能够进一步提升生成样本对分割训练的帮助。建议比较以下设置：

* real-only；
* real + CN-only synthetic；
* real + SBG-Diff synthetic；
* real + BEF-SBG synthetic；
* 不同低标注比例，例如 5p、10p、20p；
* 不同生成样本数，例如 `N_SAMPLES=2`、`N_SAMPLES=5`；
* 不同样本筛选阈值。

---

## 10. 引用

如果本代码或方法对你的研究有帮助，请引用相关工作：

```bibtex
@article{sbgdiff,
  title   = {Soft Boundary Guidance for Mask-Conditioned Dermoscopic Diffusion Synthesis},
  author  = {Zhang, Jiayu and Huang, Xiaoming},
  journal = {...},
  year    = {...}
}

@inproceedings{bgdnet,
  title     = {BGDNet: Boundary-Guided Dual-Backbone Encoder for Dermoscopic Lesion Segmentation},
  author    = {Zhang, Jiayu and Huang, Xiaoming},
  booktitle = {...},
  year      = {...}
}
```

---

## 11. License

This project is released under the Apache-2.0 License.

---

## 12. Notes

1. 请根据自己的服务器路径修改各脚本中的数据集路径、权重路径和输出路径。
2. 训练扩散模型前，请确认 Stable Diffusion / ControlNet 初始化权重已经准备完成。
3. 低标注实验中，生成图像应只使用训练集 mask，避免测试集信息泄漏。
4. 分割训练前建议先运行数据加载检查脚本，确认 image、mask 和 txt 文件可以正确匹配。
5. BEF-SBG 流程较长，建议逐步运行并检查每一步日志和输出文件。
