# 项目文件内容分析文档

文档重点分析源码、脚本、配置、说明和数据清单文件。

未展开逐条分析的数据包括：数据集图片、模型权重、训练日志、TensorBoard 事件、checkpoint、推理图片结果、缓存目录和训练输出目录。

## 总体概括

当前项目由三部分组成：

| 模块 | 概括 |
| --- | --- |
| `code/SBG-Diff` | 扩散生成主项目，基于 Stable Diffusion / ControlNet 从 mask 生成皮肤镜图像，并实现边界先验、SBP-PG、两阶段训练、推理和评估。 |
| `code/BGDNet` | 下游分割模型项目，包含 BGDNet 网络、数据加载、训练、测试和真实/合成混合训练脚本。 |
| `code/BEF_SBG` | BEF-SBG 流水线项目，串联低标注划分、OOF teacher、边界反馈、BEF prompt、扩散生成、样本评分、加权分割训练和最终评估。 |

## 根目录文件

| 文件 | 内容概括 |
| --- | --- |
| `README.md` | 根目录项目说明，概括项目用途、技术栈、目录结构、启动脚本、依赖、配置、风险命令和待确认事项。 |

## `code/SBG-Diff` 核心文件

| 文件 | 内容概括 |
| --- | --- |
| `readme.md` | 记录从 prompt 生成、ControlNet 初始化、扩散训练、合并权重、推理、抽样、混合比例生成到 BGDNet 分割训练的整体实验流程。原文件中文编码显示存在异常。 |
| `config.py` | 简单运行配置文件，目前只设置 `save_memory = True`。 |
| `share.py` | 调用 `cldm.hack.hack_everything()`，启用 ControlNet/LDM 运行时 monkey patch 与注意力优化。 |
| `tutorial_train.py` | 扩散训练入口；读取环境变量或参数，加载 prompt 数据、ControlLDM 模型、Lightning Trainer、DeepSpeed/DDP 策略和 checkpoint 保存逻辑。 |
| `tutorial_inference.py` | 扩散推理入口；加载合并权重和 prompt JSON，按 mask/边界先验采样生成图像与 mask 对应输出。 |
| `tutorial_dataset.py` | 扩散训练数据集；读取 prompt JSON，加载图像、mask、文本和边界条件，并生成训练张量。 |
| `tutorial_dataset_sample.py` | 扩散推理数据集；读取待生成样本，构造 mask 控制、边界控制和文本条件。 |
| `tool_add_control.py` | 将 Stable Diffusion 权重转换为 ControlNet 初始化权重，复制 encoder 相关参数并保存新 `.pth`。 |
| `tool_merge_control.py` | 合并 Stable Diffusion 基础权重与训练后的 ControlNet / decoder 权重，生成推理用合并模型。 |
| `make_prompt_json.py` | 扫描 image/mask 目录并生成扩散训练或推理用 prompt JSON。 |
| `generate_paired_txt.py` | 从图像目录生成成对样本 txt 列表，用于后续训练/评估。 |
| `extract_isic2018.py` | 从 ISIC2018 训练集按比例抽取低标注 split，并写出 train/test txt。 |
| `Build_data.py` | 构建真实与合成数据组合列表，支持按比例抽取 synthetic stems。 |
| `overly_mask.py` | 将 mask 叠加到图像上生成可视化 overlay。文件名疑似应为 `overlay_mask.py`。 |

## `code/SBG-Diff/Ours` 文件

| 文件 | 内容概括 |
| --- | --- |
| `Ours/eval_quality.py` | 对真实图像与生成图像进行质量评估；构造 masked folder，计算 FID/KID 等指标并输出 JSON。 |
| `Ours/make_soft_boundary_assets.py` | 从二值 mask 生成 hard boundary、soft boundary prior、可视化增强图和 overlay 资产。 |
| `Ours/make_framework_maps.py` | 为论文框架图生成 mask、soft boundary、scale/shift map、decoder feature 等中间可视化图。 |
| `Ours/make_framework_assets_v2.py` | 生成论文框架图所需的训练 mask、生成图、边界热图、噪声、特征栈、测试 overlay 和指标条。 |
| `Ours/scripts/init.sh` | 调用 `tool_add_control.py`，由 SD 1.5 safetensors 生成 region-boundary ControlNet 初始化权重。 |
| `Ours/scripts/CN-only.sh` | 启动 CN-only 扩散训练；只训练 ControlNet，不训练 UNet decoder，默认 1 GPU 后台运行。 |
| `Ours/scripts/stage1_sbp_pg.sh` | 启动 SBP-PG stage1；训练结构控制和边界先验，冻结 SD 主干。 |
| `Ours/scripts/stage2_sbp_pg.sh` | 启动 SBP-PG stage2；从 stage1 checkpoint 继续训练 decoder 相关部分并启用边界调制。 |
| `Ours/scripts/merge.sh` | 后台执行 `tool_merge_control.py`，将训练权重合并为推理权重。 |
| `Ours/scripts/infer.sh` | 使用合并权重和 `prompt_train.json` 进行 region-boundary 推理，生成训练用合成数据。 |
| `Ours/scripts/infer_sbp_pg.sh` | 使用 SBP-PG 配置推理，启用 soft boundary prior、progressive guidance 和 decoder modulation。 |
| `Ours/scripts/eval.sh` | 调用 `eval_quality.py` 评估 stage1-2 生成结果；脚本中 `$LOG_DIR` 当前定义被注释，运行前需修正。 |
| `Ours/scripts/asset.sh` | 生成论文框架图素材；依赖手动配置训练样本、测试样本、预测 mask 和指标值。 |

## `code/SBG-Diff/Baseline` 文件

| 文件 | 内容概括 |
| --- | --- |
| `Baseline/dataloader_seg_baseline.py` | 分割 baseline 数据加载器；支持真实/生成数据配对、txt 列表解析、mask 命名兼容和 train/test dataset。 |
| `Baseline/seg_baselines.py` | 分割 baseline 模型集合；包含 UNet、NNUNet2D 和 DeepLabV3-ResNet50 构建逻辑。 |
| `Baseline/train_seg_baseline.py` | 分割 baseline 训练入口；实现 Dice+BCE 损失、学习率调整、优化器创建、训练与验证。 |
| `Baseline/test_seg_baseline.py` | 分割 baseline 测试入口；加载 checkpoint，计算 Dice/IoU/HD95 等指标并可保存预测 mask。 |
| `Baseline/script/train_unet.sh` | 使用真实 ISIC 训练 UNet baseline 的后台脚本。 |
| `Baseline/script/train_unet_real_gen.sh` | 使用真实+生成数据训练 UNet baseline 的后台脚本。 |
| `Baseline/script/train_ph2.sh` | 针对 PH2 数据训练 UNet baseline 的后台脚本。 |
| `Baseline/script/test_unet.sh` | 对 ISIC UNet baseline 进行测试评估的后台脚本。 |
| `Baseline/script/testph.sh` | 对 PH2 baseline 结果进行测试评估的后台脚本。 |

## `code/SBG-Diff/cldm` 文件

| 文件 | 内容概括 |
| --- | --- |
| `cldm/cldm.py` | 当前 ControlLDM / ControlNet 核心实现；集成 mask control、boundary control、soft boundary prior、progressive guidance、decoder modulation、训练冻结策略和损失逻辑。 |
| `cldm/dhi.py` | DHI 特征提取模块；包含残差块、PatchMerging、FeatureExtractor 和 BoundarySpatialModulator。 |
| `cldm/ddim_hacked.py` | 修改版 DDIM sampler；支持 ControlNet 条件采样、encode/decode 和 ddim step。 |
| `cldm/hack.py` | 运行时 patch；关闭 transformers 冗余日志，启用 sliced attention，替换 CLIP forward 和注意力 forward。 |
| `cldm/logger.py` | Lightning 图像日志回调；保存验证图像和固定文件名可视化结果。 |
| `cldm/model.py` | 模型构建与权重加载工具；从 YAML 创建模型，加载 state_dict，比较权重差异。 |
| `cldm/or/cldm_or.py` | 原始 ControlNet/ControlLDM 版本，未集成后续边界增强。 |
| `cldm/or/cldm_3.16.py` | 早期实验版本，包含基础 ControlNet 训练、冻结策略和 p_losses。 |
| `cldm/or/cldm_3.20.py` | 加入区域权重图、Sobel 边界、加权损失等任务感知逻辑的版本。 |
| `cldm/or/cldm_4.3.py` | 加入形态复杂度估计、自适应一致性权重等改动的版本。 |
| `cldm/or/cldm_4.30.py` | 边界调制和更完整前向拆分版本，包含 boundary modulator、control/diffusion checkpointed forward。 |
| `cldm/or/dhi_3.20.py` | 旧版 DHI 特征提取器。 |
| `cldm/or/dhi_4.30.py` | 带 BoundarySpatialModulator 的新版 DHI 实验版本。 |

## `code/SBG-Diff/ldm` 文件

| 文件 | 内容概括 |
| --- | --- |
| `ldm/util.py` | LDM 通用工具；包含配置实例化、文本转图像日志、参数统计、AdamW EMA 优化器封装。 |
| `ldm/data/__init__.py` | LDM 数据模块包初始化文件。 |
| `ldm/data/util.py` | MiDaS 条件数据转换工具，提供 tensor/numpy 转换和深度条件预处理。 |
| `ldm/models/autoencoder.py` | AutoencoderKL 和 IdentityFirstStage 实现，用作扩散模型 first stage。 |
| `ldm/models/diffusion/ddpm.py` | DDPM 核心实现；包含 schedule、q/p 采样、EMA、训练/验证和 loss。 |
| `ldm/models/diffusion/ddim.py` | 标准 DDIM sampler 实现。 |
| `ldm/models/diffusion/plms.py` | PLMS sampler 实现，用于扩散快速采样。 |
| `ldm/models/diffusion/sampling_util.py` | 采样辅助函数，如维度补齐和阈值归一化。 |
| `ldm/models/diffusion/dpm_solver/__init__.py` | DPM-Solver 包初始化。 |
| `ldm/models/diffusion/dpm_solver/dpm_solver.py` | DPM-Solver 核心数值求解器和噪声调度实现。 |
| `ldm/models/diffusion/dpm_solver/sampler.py` | DPM-Solver sampler 封装。 |
| `ldm/modules/attention.py` | CrossAttention、SpatialSelfAttention、Transformer block 等注意力模块。 |
| `ldm/modules/ema.py` | Lightning EMA 参数维护工具。 |
| `ldm/modules/distributions/__init__.py` | distributions 包初始化。 |
| `ldm/modules/distributions/distributions.py` | Dirac 和 DiagonalGaussianDistribution，实现 VAE latent 分布计算。 |
| `ldm/modules/diffusionmodules/__init__.py` | diffusionmodules 包初始化。 |
| `ldm/modules/diffusionmodules/util.py` | beta schedule、DDIM timestep、checkpoint、卷积/归一化等扩散工具。 |
| `ldm/modules/diffusionmodules/model.py` | U-Net 基础模块；包含 ResnetBlock、Attention block、上下采样等。 |
| `ldm/modules/diffusionmodules/openaimodel.py` | OpenAI diffusion U-Net 结构组件，包含 ResBlock、AttentionPool、Timestep 模块。 |
| `ldm/modules/diffusionmodules/upscaling.py` | 低分辨率条件和图像拼接上采样辅助模块。 |
| `ldm/modules/encoders/__init__.py` | encoders 包初始化。 |
| `ldm/modules/encoders/modules.py` | 文本/类别 encoder；包含 FrozenCLIPEmbedder、FrozenT5Embedder 和 IdentityEncoder。 |
| `ldm/modules/image_degradation/__init__.py` | image_degradation 包初始化。 |
| `ldm/modules/image_degradation/bsrgan.py` | BSRGAN 图像退化实现，包含 blur、resize、noise、JPEG 等退化操作。 |
| `ldm/modules/image_degradation/bsrgan_light.py` | 轻量版 BSRGAN 退化实现。 |
| `ldm/modules/image_degradation/utils_image.py` | 图像读写、转换、patch 切分、目录创建等工具。 |
| `ldm/modules/midas/__init__.py` | MiDaS 模块包初始化。 |
| `ldm/modules/midas/api.py` | MiDaS 模型加载和推理封装。 |
| `ldm/modules/midas/utils.py` | MiDaS 图像、深度图和 PFM 读写工具。 |
| `ldm/modules/midas/midas/__init__.py` | MiDaS 内部包初始化。 |
| `ldm/modules/midas/midas/base_model.py` | MiDaS 基础模型类和权重加载方法。 |
| `ldm/modules/midas/midas/blocks.py` | MiDaS encoder/backbone/fusion block 构造函数。 |
| `ldm/modules/midas/midas/dpt_depth.py` | DPT depth 模型定义。 |
| `ldm/modules/midas/midas/midas_net.py` | 标准 MiDaS 网络定义。 |
| `ldm/modules/midas/midas/midas_net_custom.py` | 小型 MiDaS 网络和模型融合函数。 |
| `ldm/modules/midas/midas/transforms.py` | MiDaS 输入 resize、normalize、prepare transforms。 |
| `ldm/modules/midas/midas/vit.py` | ViT backbone 适配、readout 操作和 hook 逻辑。 |

## `code/SBG-Diff/models` 配置文件

| 文件 | 内容概括 |
| --- | --- |
| `models/cldm_v15.yaml` | 当前 ControlLDM 配置；定义 ControlNet、ControlledUnetModel、AutoencoderKL、FrozenCLIPEmbedder、boundary key 和 decoder modulation 参数。 |
| `models/cldm_v15_3.20.yaml` | 旧实验版本的 ControlLDM 配置。 |
| `models/cldm_v21.yaml` | Stable Diffusion v2.1 相关 ControlLDM 配置。 |

## `code/SBG-Diff/or` 文件

| 文件 | 内容概括 |
| --- | --- |
| `or/README.md` | 原始项目说明；记录 conda 环境创建、训练、采样和 ControlNet 权重合并方式。 |
| `or/environment.yaml` | conda 环境文件；包含 Python 3.10、PyTorch、CUDA、Lightning、Gradio、Transformers、OpenCLIP、Timm 等依赖。 |
| `or/tool_add_control.py` | 原始 ControlNet 初始化权重生成脚本。 |
| `or/tool_merge_control.py` | 原始 ControlNet 权重合并脚本。 |
| `or/tool_transfer_control.py` | 权重迁移脚本，用于在不同 ControlNet 结构间转移可复用参数。 |
| `or/tutorial_dataset.py` | 原始训练 dataset。 |
| `or/tutorial_dataset_3.20.py` | 3.20 版本训练 dataset。 |
| `or/tutorial_dataset_sample.py` | 原始推理 dataset。 |
| `or/tutorial_dataset_sample_3.20.py` | 3.20 版本推理 dataset。 |
| `or/tutorial_dataset_test.py` | 测试集推理 dataset。 |
| `or/tutorial_train.py` | 原始扩散训练入口。 |
| `or/tutorial_train_4.30.py` | 4.30 版本扩散训练入口，支持更多边界/decoder 配置。 |
| `or/tutorial_inference.py` | 原始扩散推理入口。 |
| `or/tutorial_inference_3.20.py` | 3.20 版本推理入口。 |
| `or/tutorial_inference_4.30.py` | 4.30 版本推理入口，包含 tensor 图像保存辅助逻辑。 |
| `or/tutorial_inference_4_3.py` | 4.3 版本推理入口。 |
| `or/stage1_cn.sh` | 旧版 stage1 ControlNet 训练脚本。 |
| `or/stage1_cn_3_18.sh` | task-aware 早期 stage1 脚本。 |
| `or/stage1_cn_morph_adapt_aug.sh` | 启用形态自适应一致性的 stage1 脚本。 |
| `or/stage1_cn_region_boundary.sh` | 使用 region-boundary 初始化权重的 stage1 脚本。 |
| `or/stage2_decoder.sh` | 旧版 decoder stage2 训练脚本。 |
| `or/stage2_decoder_3_18.sh` | task-aware decoder stage2 训练脚本。 |
| `or/stage2_decoder_morph_adapt_aug.sh` | 启用形态自适应和在线增强的 decoder stage2 脚本。 |
| `or/stage2_decoder_region_boundary.sh` | region-boundary decoder stage2 训练脚本。 |

## `code/SBG-Diff/Ours/ablation_sbg` 文件

| 文件 | 内容概括 |
| --- | --- |
| `Ours/ablation_sbg/scripts/readme.txt` | 消融实验脚本说明，描述各步骤的执行顺序。 |
| `Ours/ablation_sbg/scripts/00_make_real5_split.sh` | 从原始 prompt JSON 中抽取 5% real split，生成消融用 prompt。 |
| `Ours/ablation_sbg/scripts/01_stage1_train_variant.sh` | 消融 stage1 训练脚本；按 variant 配置训练不同边界条件策略。 |
| `Ours/ablation_sbg/scripts/02_stage2_train_variant.sh` | 消融 stage2 decoder 训练脚本。 |
| `Ours/ablation_sbg/scripts/03_infer_variant.sh` | 对指定消融 variant 做推理生成。 |
| `Ours/ablation_sbg/scripts/04_collect_real_images.sh` | 收集真实训练图像作为生成质量评估参照。 |
| `Ours/ablation_sbg/scripts/05_eval_generation_metrics.sh` | 计算生成质量指标，如 FID/KID/CLIP-I/LPIPS 等。 |
| `Ours/ablation_sbg/scripts/06_boundary_metrics_variant.sh` | 计算生成图与边界自然性相关指标。 |
| `Ours/ablation_sbg/scripts/07_eval_all_given_results.sh` | 一键运行给定结果的全部消融评估。 |
| `Ours/ablation_sbg/code/configs/cldm_v15_ablation.yaml` | 消融专用 ControlLDM 配置，target 指向 `ablation_cldm`。 |
| `Ours/ablation_sbg/code/train_ablation.py` | 消融训练入口；读取 variant 配置，打印配置块，支持关闭 checkpointing。 |
| `Ours/ablation_sbg/code/inference_ablation.py` | 消融推理入口；加载消融模型并保存多样本生成结果。 |
| `Ours/ablation_sbg/code/ablation_cldm/__init__.py` | 消融版 cldm 包初始化。 |
| `Ours/ablation_sbg/code/ablation_cldm/cldm.py` | 消融版 ControlLDM；实现不同 boundary condition mode、progressive weight、mask/soft boundary 策略。 |
| `Ours/ablation_sbg/code/ablation_cldm/ddim_hacked.py` | 消融版 DDIM sampler。 |
| `Ours/ablation_sbg/code/ablation_cldm/dhi.py` | 消融版 DHI 和 BoundarySpatialModulator。 |
| `Ours/ablation_sbg/code/ablation_cldm/logger.py` | 消融实验图像 logger。 |
| `Ours/ablation_sbg/code/ablation_cldm/model.py` | 消融模型构建和权重加载工具。 |
| `Ours/ablation_sbg/eval/collect_real_images.py` | 从 prompt 数据中复制真实图像到评估输入目录。 |
| `Ours/ablation_sbg/eval/eval_boundary_naturalness.py` | 计算边界自然性指标；包含 Sobel edge、hard boundary、soft prior 构造。 |
| `Ours/ablation_sbg/eval/eval_generation_metrics.py` | 通用生成质量评估；匹配真实/生成图像并计算 FID/KID/CLIP/CMMD/LPIPS/MOS proxy。 |
| `Ours/ablation_sbg/eval/eval_generation_metrics_isic.py` | ISIC 专用生成质量评估脚本，逻辑与通用版本接近但路径/命名更贴近 ISIC。 |

## `code/SBG-Diff/Ours/ablation_sbg_1779496819259` 文件

该目录是 `ablation_sbg` 的时间戳副本，代码结构和用途基本重复。

| 文件 | 内容概括 |
| --- | --- |
| `Ours/ablation_sbg_1779496819259/scripts/readme.txt` | 时间戳副本中的消融脚本说明。 |
| `Ours/ablation_sbg_1779496819259/scripts/00_make_real5_split.sh` | 生成 5% real prompt split 的副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/01_stage1_train_variant.sh` | 消融 stage1 训练副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/02_stage2_train_variant.sh` | 消融 stage2 训练副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/03_infer_variant.sh` | 消融推理副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/04_collect_real_images.sh` | 收集真实图像副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/05_eval_generation_metrics.sh` | 生成质量评估副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/06_boundary_metrics_variant.sh` | 边界自然性评估副本脚本。 |
| `Ours/ablation_sbg_1779496819259/scripts/07_eval_all_given_results.sh` | 一键评估副本脚本。 |
| `Ours/ablation_sbg_1779496819259/code/configs/cldm_v15_ablation.yaml` | 消融配置副本。 |
| `Ours/ablation_sbg_1779496819259/code/train_ablation.py` | 消融训练入口副本。 |
| `Ours/ablation_sbg_1779496819259/code/inference_ablation.py` | 消融推理入口副本。 |
| `Ours/ablation_sbg_1779496819259/code/ablation_cldm/__init__.py` | 消融 cldm 包初始化副本。 |
| `Ours/ablation_sbg_1779496819259/code/ablation_cldm/cldm.py` | 消融 ControlLDM 副本。 |
| `Ours/ablation_sbg_1779496819259/code/ablation_cldm/ddim_hacked.py` | 消融 DDIM sampler 副本。 |
| `Ours/ablation_sbg_1779496819259/code/ablation_cldm/dhi.py` | 消融 DHI 副本。 |
| `Ours/ablation_sbg_1779496819259/code/ablation_cldm/logger.py` | 消融 logger 副本。 |
| `Ours/ablation_sbg_1779496819259/code/ablation_cldm/model.py` | 消融模型工具副本。 |
| `Ours/ablation_sbg_1779496819259/eval/collect_real_images.py` | 真实图像收集副本。 |
| `Ours/ablation_sbg_1779496819259/eval/eval_boundary_naturalness.py` | 边界自然性评估副本。 |
| `Ours/ablation_sbg_1779496819259/eval/eval_generation_metrics.py` | 生成质量评估副本。 |
| `Ours/ablation_sbg_1779496819259/eval/eval_generation_metrics_isic.py` | ISIC 生成质量评估副本。 |

## `code/SBG-Diff` 数据清单与 JSON 文件

| 文件 | 内容概括 |
| --- | --- |
| `data/prompt_train.json` | ISIC 训练集扩散 prompt 清单。 |
| `data/prompt_train_or.json` | 原始训练 prompt 清单，供抽样或消融使用。 |
| `data/prompt_train_5p_seed1.json` | seed1 的 5% 低标注训练 prompt 清单。 |
| `data/prompt_train_5p_seed2.json` | seed2 的 5% 低标注训练 prompt 清单。 |
| `data/prompt_test.json` | ISIC 测试或评估用 prompt 清单。 |
| `data/prompt_PH2_train.json` | PH2 训练 prompt 清单。 |
| `data/prompt_PH2_train_5p_seed0.json` | PH2 5% 低标注训练 prompt 清单。 |
| `data/prompt_PH2_test.json` | PH2 测试 prompt 清单。 |
| `Ours/data_txt/gen_quality_metrics.json` | 生成质量评估指标 JSON。 |
| `Ours/data_txt/gen_quality_metrics_cn_only.json` | CN-only 生成结果质量指标 JSON。 |
| `Ours/data_txt/gen_quality_metrics_stage1-2.json` | stage1-2 生成结果质量指标 JSON。 |
| `data_txt/generate_mix_ratio_txts.py` | 根据真实列表和合成列表生成不同 real:gen 比例训练 txt。 |
| `data_txt/test.txt` | ISIC 测试样本 ID 列表。 |
| `data_txt/test_PH2.txt` | PH2 测试样本 ID 列表。 |
| `data_txt/train_all.txt` | 全量训练样本 ID 列表。 |
| `data_txt/train_5p.txt` | 5% 真实训练样本 ID 列表。 |
| `data_txt/train_5p_seed0.txt` | seed0 的 5% 真实训练列表。 |
| `data_txt/train_5p_seed1.txt` | seed1 的 5% 真实训练列表。 |
| `data_txt/train_10p.txt` | 10% 真实训练样本列表。 |
| `data_txt/train_20p.txt` | 20% 真实训练样本列表。 |
| `data_txt/train_exp_cn_only.txt` | CN-only 生成样本列表。 |
| `data_txt/train_exp_stage1_2.txt` | stage1-2 生成样本列表。 |
| `data_txt/train_Full_exp_stage1_2.txt` | stage1-2 全量生成或扩展训练列表。 |
| `data_txt/train_Full_sbg_real5_syn15_seed1.txt` | 真实 5% + SBG 合成 15 组的训练列表。 |
| `data_txt/train_full_seed1.txt` | seed1 全量/完整配置训练列表。 |
| `data_txt/train_full_sbg_real5_syn5_seed1.txt` | 真实 5% + SBG 合成 5 组训练列表。 |
| `data_txt/train_hard_seed1.txt` | hard boundary variant 训练列表。 |
| `data_txt/train_hard_real5_syn5_seed1.txt` | hard variant 真实 5% + 合成 5 组训练列表。 |
| `data_txt/train_hard_real5_syn10.txt` | hard variant 真实 5% + 合成 10 组训练列表。 |
| `data_txt/train_hard_real5_syn15_seed1.txt` | hard variant 真实 5% + 合成 15 组训练列表。 |
| `data_txt/train_maskonly_seed1.txt` | mask-only variant 训练列表。 |
| `data_txt/train_maskonly_real5_syn5_seed1.txt` | mask-only 真实 5% + 合成 5 组训练列表。 |
| `data_txt/train_maskonly_real5_syn10.txt` | mask-only 真实 5% + 合成 10 组训练列表。 |
| `data_txt/train_maskonly_real5_syn15_seed1.txt` | mask-only 真实 5% + 合成 15 组训练列表。 |
| `data_txt/train_progressive_seed1.txt` | progressive boundary guidance variant 训练列表。 |
| `data_txt/train_progressive_exp_stage1_2.txt` | progressive stage1-2 生成列表。 |
| `data_txt/train_progressive_real5_syn5_seed1.txt` | progressive 真实 5% + 合成 5 组训练列表。 |
| `data_txt/train_progressive_real5_syn10.txt` | progressive 真实 5% + 合成 10 组训练列表。 |
| `data_txt/train_progressive_real5_syn15_seed1.txt` | progressive 真实 5% + 合成 15 组训练列表。 |
| `data_txt/train_softonly_seed1.txt` | soft boundary only variant 训练列表。 |
| `data_txt/train_softonly_real5_syn5_seed1.txt` | soft-only 真实 5% + 合成 5 组训练列表。 |
| `data_txt/train_softonly_real5_syn10.txt` | soft-only 真实 5% + 合成 10 组训练列表。 |
| `data_txt/train_softonly_real5_syn15_seed1.txt` | soft-only 真实 5% + 合成 15 组训练列表。 |
| `data_txt/train_decoder_seed1.txt` | decoder variant 训练列表。 |
| `data_txt/train_decoder_real5_syn15_seed1.txt` | decoder 真实 5% + 合成 15 组训练列表。 |
| `data_txt/train_decoder_mod_exp_stage1_2.txt` | decoder modulation stage1-2 生成列表。 |
| `data_txt/train_decoder_mod_real5_syn5_seed1.txt` | decoder modulation 真实 5% + 合成 5 组训练列表。 |
| `data_txt/train_decoder_mod_real5_syn10.txt` | decoder modulation 真实 5% + 合成 10 组训练列表。 |
| `data_txt/train_PH2_5p.txt` | PH2 5% 训练列表。 |
| `data_txt/train_PH2_full_sbg_real5_syn15_seed0.txt` | PH2 真实 5% + full SBG 合成 15 组训练列表。 |
| `data_txt/train_PH2_hard_real5_syn15_seed0.txt` | PH2 hard variant 真实 5% + 合成 15 组训练列表。 |
| `data_txt/mix_ratio/*.txt` | 按 `real1_gen1`、`real1_gen2`、`real2_gen1`、`real3_gen1` 等比例生成的真实/合成混合训练列表，覆盖 5%、10%、20% 与 `cn_only`、`stage1_2` 两类生成源。 |

## `code/BGDNet` 文件

| 文件 | 内容概括 |
| --- | --- |
| `BGDNet.md` | 空文件，未写入说明内容。 |
| `BGDiff_train.py` | BGDNet 针对真实/生成混合数据的训练入口；实现 BCE+Dice 风格联合 mask/boundary loss、验证 Dice 和 checkpoint 保存。 |
| `BGDiff_test.py` | BGDNet 测试入口；加载 checkpoint，输出预测 mask，并保存 per-image 和 summary 指标 CSV。 |
| `train.py` | 通用 BGDNet 训练入口，使用真实训练集和验证集。 |
| `test.py` | 通用测试脚本，计算 Dice、IoU、Sensitivity、Precision、F1、Specificity、Accuracy、MCC 等指标。 |
| `train_ISIC2018.py` | ISIC2018 专用训练脚本。 |
| `test_ISIC2018.py` | ISIC2018 专用测试脚本，包含 ROC 与混淆矩阵指标。 |
| `train_ISIC2016.py` | ISIC2016 专用训练脚本。 |
| `test_ISIC2016.py` | ISIC2016 专用测试脚本。 |
| `test_con.py` | 对比测试脚本，包含 ROC 曲线和分割指标计算。 |
| `train_test.py` | 单文件实验集合，定义多种 UNet/AttentionUNet/ERUNet 等网络和训练测试逻辑。 |
| `split_dataset.py` | 数据集划分脚本。 |
| `dataset_split.py` | 数据集 split 生成或辅助脚本。 |
| `losses.py` | 分割损失函数集合；包含 HybridLoss、structure_loss、boundary_aware_loss 和 feature_consistency。 |
| `models/BGDNet.py` | BGDNet 主网络结构；包含空间注意力、特征融合、可变形卷积、多尺度 decoder 和边界输出相关模块。 |
| `models/UNet.py` | 标准 UNet 网络定义。 |
| `models/MSD.py` | 多尺度可变形卷积 decoder 相关网络模块。 |
| `models/MSDUnet.py` | 基于 MSD 结构的 UNet 变体。 |
| `models/LMSA.py` | 局部/多尺度注意力增强模块集合。 |
| `models/SwinBlock.py` | Swin Transformer block、window attention、patch merging 等组件。 |
| `models/transxnet.py` | TransXNet backbone 实现，包含 PatchEmbed、HybridTokenMixer、DynamicConv 等。 |
| `models/backbone/transxnet.py` | TransXNet backbone 副本或细分版本。 |
| `utils/dataloader.py` | 通用 polyp/lesion 数据加载器，输出图像、mask 和 Sobel boundary。 |
| `utils/dataloader_BGDiff.py` | 面向 BGDiff 合成数据的 dataloader；支持 txt 列表、真实/paired 命名规则和缺失检查。 |
| `utils/dataloader_enhanced.py` | 带动态增强的 dataloader，包含颜色扰动、空间扰动和噪声增强。 |
| `utils/dataset_ACDC.py` | ACDC 医学数据集 loader 和随机旋转/翻转增强。 |
| `utils/dataset_synapse.py` | Synapse 数据集 loader 和随机增强。 |
| `utils/format_conversion.py` | 图像格式转换和数据 split 辅助函数。 |
| `utils/joint_transforms.py` | 图像与 mask 同步变换，如 crop、scale、flip。 |
| `utils/misc.py` | 训练辅助工具；包含权重初始化、loss、评估、AverageMeter、PolyLR、Deformable conv 封装。 |
| `utils/transforms.py` | 单图像/Mask transforms，如随机翻转、反归一化、mask tensor 化、高斯模糊。 |
| `utils/utils.py` | 常用训练/评估工具；包含梯度裁剪、学习率调整、DiceLoss、volume 测试函数等。 |
| `utils/preprocess_synapse_data.py` | Synapse 数据预处理脚本。 |
| `utils/preprocess_synapse_data_3d.py` | Synapse 3D 数据预处理脚本。 |
| `utils/lesion/helpers.py` | lesion 指标与文件工具，包括 DSC、IoU、precision、recall。 |
| `utils/lesion/lesion_dataset.py` | lesion dataset 类。 |
| `utils/lesion/make_dataset.py` | lesion 数据文件列表生成工具。 |
| `scripts/01_check_load.sh` | 检查真实训练、合成训练和测试 txt 是否能正确解析为 image/mask pairs。 |
| `scripts/02_train_real.sh` | 仅使用真实数据训练 BGDNet 的后台脚本。 |
| `scripts/02_train_cn_only.sh` | 使用真实数据和 CN-only 合成数据训练 BGDNet 的后台脚本。 |
| `scripts/02_train_stage1-2.sh` | 使用真实数据和 stage1-2 合成数据训练 BGDNet 的后台脚本；`--train_list2` 行存在续行符前缺空格风险。 |
| `scripts/02_train_2gpu.sh` | 2 GPU 训练 BGDNet 的后台脚本。 |
| `scripts/03_test.sh` | 加载指定 BGDNet checkpoint 并输出测试指标的后台脚本。 |

## `code/BEF_SBG` 文件

| 文件 | 内容概括 |
| --- | --- |
| `readme.txt` | BEF-SBG 早期流程说明，记录 low-label split、OOF teacher、边界反馈和可视化命令。原文件中文编码显示存在异常。 |
| `train_oof_teachers.sh` | 旧版 5-fold OOF teacher 训练脚本，循环调用 BGDNet `BGDiff_train.py` 训练各 fold。 |
| `infer_oof_predictions.py` | OOF teacher 推理脚本；加载每折 teacher checkpoint，对训练集 fold val 生成 pred masks/boundaries 和指标。 |
| `splits/make_low_label_splits.py` | 根据图像和 mask 目录生成 5%、10%、20%、100% 低标注 K-fold split。 |
| `feedback/build_boundary_feedback.py` | 根据 GT mask、teacher pred mask/boundary 构建边界误差反馈、hard boundary、adaptive soft prior 和 difficulty。 |
| `feedback/make_bef_prompt_json.py` | 将图像、mask、adaptive prior、difficulty 合并成 BEF 扩散训练 prompt JSON。 |
| `feedback/visualize_feedback.py` | 生成反馈可视化拼图，展示图像、GT、预测和边界反馈。 |
| `feedback/prompt_bef_train_5p.json` | 5% BEF 扩散训练 prompt 清单。 |
| `diffusion/00_make_bef_prompt_json_5p.sh` | 生成 5% BEF prompt JSON 的后台脚本。 |
| `diffusion/stage1_bef_sbg.sh` | BEF-SBG 扩散 stage1 训练脚本，使用 external boundary prior。 |
| `diffusion/stage2_bef_sbg.sh` | BEF-SBG 扩散 stage2 decoder 训练脚本。 |
| `diffusion/infer_bef_sbg.sh` | BEF-SBG 扩散推理脚本，生成 BEF-SBG 合成样本。 |
| `scoring/score_generated_samples.py` | 使用 BGDNet quality model 对生成样本打分，计算一致性、边界质量、熵等，并输出 CSV/JSONL。 |
| `scoring/make_weighted_seg_dataset.py` | 将真实样本和被接受的生成样本合并成带权重的 segmentation 训练 JSONL。 |
| `scoring/scripts/score_generated_5p.sh` | 5% 生成样本打分后台脚本。 |
| `scoring/scripts/make_weighted_train_5p.sh` | 5% 加权训练 JSONL 生成脚本。 |
| `segmentation/dataloader_BEF.py` | 加权 BEF 分割训练 dataloader；读取 JSONL，加载图像、mask、boundary 和 sample weight。 |
| `segmentation/train_BGDNet_BEF.py` | 最终 BGDNet-BEF 训练入口；实现 weighted joint loss、训练循环、验证和 checkpoint 保存。 |
| `segmentation/test_BGDNet_BEF.py` | 最终 BGDNet-BEF 测试入口；计算分割指标、boundary F1/IoU、HD95、ASSD 并保存预测。 |
| `segmentation/scripts/train_final_bef_bgdnet.sh` | 简化版最终 BGDNet-BEF 训练后台脚本。 |
| `segmentation/scripts/test_final_bef_bgdnet.sh` | 简化版最终 BGDNet-BEF 测试后台脚本。 |

## `code/BEF_SBG/scripts` 主流水线

| 文件 | 内容概括 |
| --- | --- |
| `scripts/00_common.sh` | BEF-SBG 公共配置；定义 WORK_ROOT、BGDNet/BGDiff 路径、数据路径、比例、fold、输出目录、日志目录和 PID 目录。 |
| `scripts/01_make_low_label_splits.sh` | 后台生成低标注 K-fold split。 |
| `scripts/02_train_oof_teachers.sh` | 后台训练 OOF teacher；支持 batchsize、epoch、fold range、cuDNN 开关和 CUDA allocator 配置。 |
| `scripts/03_infer_oof_predictions.sh` | 后台运行 OOF teacher 推理，输出训练集 OOF 预测 mask/boundary。 |
| `scripts/04_build_boundary_feedback.sh` | 后台构建边界误差反馈和 adaptive boundary prior。 |
| `scripts/05_visualize_feedback.sh` | 后台生成边界反馈可视化图。 |
| `scripts/06_make_bef_prompt_json.sh` | 后台生成 BEF 扩散训练 prompt JSON。 |
| `scripts/07_train_diffusion_stage1.sh` | 后台训练 BEF-SBG 扩散 stage1，使用 external boundary prior 和稳定性环境变量。 |
| `scripts/08_train_diffusion_stage2.sh` | 后台训练 BEF-SBG 扩散 stage2 decoder。 |
| `scripts/09_infer_diffusion_bef_sbg.sh` | 后台执行 BEF-SBG 扩散推理，生成合成样本。 |
| `scripts/10_score_generated_samples.sh` | 后台对生成样本进行质量评分，默认使用 fold0 teacher 作为 quality model。 |
| `scripts/11_make_weighted_train_json.sh` | 后台构建最终加权分割训练 JSONL。 |
| `scripts/12_train_final_bgdnet_bef.sh` | 后台训练最终 BGDNet-BEF 分割模型。 |
| `scripts/13_test_final_bgdnet_bef.sh` | 后台测试最终 BGDNet-BEF 模型。 |
| `scripts/show_status.sh` | 读取 pid 文件并展示当前流水线运行状态和关键输出路径。 |
| `scripts/kill_by_pid.sh` | 根据 pid 文件或步骤关键词终止后台进程。 |
| `scripts/readme.txt` | 脚本目录说明文件。 |

## `code/BEF_SBG/splits` 数据清单

| 文件类型 | 内容概括 |
| --- | --- |
| `splits/ISIC2018_seed0/low_5p_*.txt` | seed0 下 5% 低标注数据的 all/fold train/fold val 列表。 |
| `splits/ISIC2018_seed0/low_10p_*.txt` | seed0 下 10% 低标注数据的 all/fold train/fold val 列表。 |
| `splits/ISIC2018_seed0/low_20p_*.txt` | seed0 下 20% 低标注数据的 all/fold train/fold val 列表。 |
| `splits/ISIC2018_seed0/low_100p_*.txt` | seed0 下全量数据的 all/fold train/fold val 列表。 |

## 主要风险

| 项目 | 说明 |
| --- | --- |
| 路径不一致 | 脚本使用了大量硬编码 `/data/zjy_work/...`。运行前需要确认 对应路径映射。 |
| 目录命名不一致 | 实际目录为 `SBG-Diff`、`BEF_SBG`，脚本常用 `BGDiff` 即对应 `SBG-Diff`、`Work3_BEF_SBG` 即对应 `BEF_SBG`。 |
| 后台任务 | 大量脚本使用 `nohup ... &`，会持续占用 GPU 并写日志、checkpoint 和结果。 |
| 权重覆盖 | `tool_add_control.py`、`tool_merge_control.py`、训练脚本可能生成或覆盖 `.pth/.ckpt` 文件。 |

