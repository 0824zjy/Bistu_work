path_sd15 = '/data/zjy_work/BGDiff/Ours/models/v1-5-pruned-emaonly.safetensors'
path_sd15_with_control = '/data/zjy_work/BGDiff/Ours/logs/exp_mask2img_stage2_sbp_pg_decoder/checkpoints/best.ckpt'
path_output = '/data/zjy_work/BGDiff/Ours/logs/exp_mask2img/merged_stage1_2_model.pth'

import os
import torch
import safetensors.torch

assert os.path.exists(path_sd15), 'Input path_sd15 does not exists!'
assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exists!'
assert os.path.exists(os.path.dirname(path_output)), 'Output folder not exists!'

def safe_load_state_dict(ckpt_path, map_location='cpu'):
    """安全加载 .safetensors 或 .ckpt 权重，自动处理 Lightning 格式"""
    if ckpt_path.endswith('.safetensors'):
        # safetensors 返回 dict，需遍历转换设备（不强制转 dtype）
        state_dict = safetensors.torch.load_file(ckpt_path, device=map_location)
        return state_dict
    else:
        # 处理 .ckpt：兼容 Lightning 格式（含 state_dict）和普通 state_dict
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']  # Lightning checkpoint
            elif 'model' in checkpoint:
                return checkpoint['model']       # 其他框架格式
        return checkpoint  # 已是纯 state_dict

# 安全加载两个模型
sd15_state_dict = safe_load_state_dict(path_sd15)
sd15_with_control_state_dict = safe_load_state_dict(path_sd15_with_control)

# 合并：以原始SD为基底，用训练权重覆盖/新增
final_state_dict = sd15_state_dict.copy()
for key, value in sd15_with_control_state_dict.items():
    if key in final_state_dict and final_state_dict[key].shape != value.shape:
        print(f"⚠️  Shape mismatch for {key}: {final_state_dict[key].shape} vs {value.shape} (skipped)")
        continue
    final_state_dict[key] = value
    print(f"✓ Merged: {key}")

# 保存纯净推理权重（保持原始 dtype，避免精度损失）
torch.save(final_state_dict, path_output)
print(f'\nMerged model saved successfully to:\n{path_output}')
print(f'   Total parameters: {len(final_state_dict)} keys')