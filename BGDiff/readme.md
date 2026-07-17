# 完整启动流程（训练 → 合并 → 推理 → 迁移 → 组 txt → 分割训练 → 生成质量评估）
---

## Step 1：生成 prompt_train.json / prompt_test.json

（以 ISIC 为例）

```bash
python /data/zjy_work/BGDiff/make_prompt_json.py \
  --img_dir /data/zjy_work/PH2/train/Images \
  --mask_dir /data/zjy_work/PH2/train/Masks \
  --out /data/zjy_work/BGDiff/data/prompt_PH2_train.json \
  --prompt "dermoscopy image"

python /data/zjy_work/BGDiff/make_prompt_json.py \
  --img_dir /data/zjy_work/PH2/test/Images \
  --mask_dir /data/zjy_work/PH2/test/Masks \
  --out /data/zjy_work/BGDiff/data/prompt_PH2_test.json \
  --prompt "dermoscopy image"
```

---

## Step 2：生成 ControlNet 初始化权重（只做一次）

```bash
cd /data/zjy_work/BGDiff
python /data/zjy_work/BGDiff/tool_add_control.py \
  /data/zjy_work/BGDiff/Ours/models/v1-5-pruned-emaonly.safetensors \
  /data/zjy_work/BGDiff/Ours/models/control_sd15_init.pth
```

---

## Step 3：训练生成模型（CN-only 或 stage1-2）

```bash
source activate BGDiff
bash /data/zjy_work/BGDiff/Ours/scripts/CN-only.sh

bash /data/zjy_work/BGDiff/Ours/scripts/stage1_cn.sh
bash /data/zjy_work/BGDiff/Ours/scripts/stage2_decoder.sh

```
---

## Step 4：合并成推理权重（merged_*.pth）



* `/data/zjy_work/BGDiff/Ours/models/merged_cn_only_model.pth`
* `/data/zjy_work/BGDiff/Ours/models/merged_cn_decoder_model.pth`

（合并脚本保持原样即可。）

---

## Step 5：推理生成数据（**训练用合成必须用 prompt_train.json**）

### 5.1 训练用合成数据（用于分割训练）

#### stage1-2 生成训练合成集


```bash
bash /data/zjy_work/BGDiff/Ours/scripts/infer.sh
```

#### CN-only 生成训练合成集

同样脚本，只换：

```bash
export CKPT_PATH="/data/zjy_work/BGDiff/Ours/models/merged_cn_only_model.pth"
export OUT_DIR="/data/zjy_work/ISIC2018/syn_from_trainmask_cn_only"
```

---

### 5.2 评估用合成数据（只评估生成质量，不进分割训练）

```bash
bash /data/zjy_work/BGDiff/Ours/scripts/eval.sh
```

---

#  Step 6：抽取真实样本5 10 20%（只做一次）

```bash
python /data/zjy_work/ISIC2018/extract_isic2018.py --img_dir /data/zjy_work/ISIC2018/train/Images --mask_dir /data/zjy_work/ISIC2018/train/Masks --out_dir /data/zjy_work/ISIC2018/splits --seed 2026 --pcts 5 10 20 --sort_output
```

### 7.2 用真实 split 抽取“对应比例”的合成样本

例如： 对于真实数据5%这个数量的样本，按照1:1,1:2,2:1,3:1在cn_only中抽取对应样本存放到/data/zjy_work/data_txt/mix_ratio路径下
```bash
python /data/zjy_work/data_txt/generate_mix_ratio_txts.py \
  --real_txt /data/zjy_work/data_txt/train_5p.txt \
  --gen_txt  /data/zjy_work/data_txt/train_exp_cn_only.txt \
  --tag cn_only \
  --out_dir /data/zjy_work/data_txt/mix_ratio \
  --ratios 1:1,1:2,2:1,3:1 \
  --seed 2026 \
  --shuffle
```
对于真实数据5%这个数量的样本，按照1:1,1:2,2:1,3:1在stage1_2中抽取对应样本存放到/data/zjy_work/data_txt/mix_ratio路径下
```bash
python /data/zjy_work/data_txt/generate_mix_ratio_txts.py \
  --real_txt /data/zjy_work/data_txt/train_5p.txt \
  --gen_txt  /data/zjy_work/data_txt/train_exp_stage1_2.txt \
  --tag stage1_2 \
  --out_dir /data/zjy_work/data_txt/mix_ratio \
  --ratios 1:1,1:2,2:1,3:1 \
  --seed 2026 \
  --shuffle
```

### 7.3 你最终用于分割训练的 txt（示例）

以 5% -cn_only为例：

* 真实：`/data/zjy_work/data_txt/train_5p.txt`
* 合成：`/data/zjy_work/data_txt/mix_ratio/train_5p__cn_only__real1_gen1__nreal130_ngen130.txt`
* 测试：`/data/zjy_work/data_txt/test.txt`

---
# 做一次“加载数量自检”
```bash
bash /data/zjy_work/BGDNet/scripts/01_check_load.sh
```
---
#  Step 8：训练分割模型

```bash
bash /data/zjy_work/BGDNet/scripts/02_train.sh
```
只使用真实数据
```bash
#!/usr/bin/env bash
set -e

cd /data/zjy_work/BGDNet

export CUDA_VISIBLE_DEVICES=1

SAVE_DIR=/data/zjy_work/BGDNet/model_out/ISIC2018_real_only/5p
mkdir -p ${SAVE_DIR}

nohup python BGDiff_train.py \
  --epoch 200 \
  --batchsize 4 \
  --img_size 352 \
  --n_gpu 1 \
  --train_path1 /data/zjy_work/ISIC2018/train/ \
  --train_list1 /data/zjy_work/data_txt/train_5p.txt \
  --test_path /data/zjy_work/ISIC2018/test/ \
  --test_list /data/zjy_work/data_txt/test.txt \
  --train_save ${SAVE_DIR} \
  > ${SAVE_DIR}/nohup_train_real_only_5p.log 2>&1 &

echo "Training started. Log: ${SAVE_DIR}/nohup_train_real_only_5p.log"
```

---