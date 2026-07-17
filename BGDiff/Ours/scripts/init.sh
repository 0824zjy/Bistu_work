#4,14
python /data/zjy_work/BGDiff/tool_add_control.py \
  /data/zjy_work/BGDiff/Ours/models/v1-5-pruned-emaonly.safetensors \
  /data/zjy_work/BGDiff/Ours/models/control_sd15_region_boundary_init.pth


# (BGDiff) root@de5b9aa5f049:/data/zjy_work/BGDiff/Ours/scripts# bash init.sh 
# /opt/conda/envs/BGDiff/lib/python3.10/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
#   torch.utils._pytree._register_pytree_node(
# /opt/conda/envs/BGDiff/lib/python3.10/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
#   torch.utils._pytree._register_pytree_node(
# /opt/conda/envs/BGDiff/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
#   @torch.library.impl_abstract("xformers_flash::flash_fwd")
# /opt/conda/envs/BGDiff/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
#   @torch.library.impl_abstract("xformers_flash::flash_bwd")
# logging improved.
# Enabled sliced_attention.
# /opt/conda/envs/BGDiff/lib/python3.10/site-packages/lightning_fabric/__init__.py:40: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
# ControlLDM: Running in eps-prediction mode
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# DiffusionWrapper has 859.64 M params.
# making attention of type 'vanilla-xformers' with 512 in_channels
# building MemoryEfficientAttnBlock with 512 in_channels...
# Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
# making attention of type 'vanilla-xformers' with 512 in_channels
# building MemoryEfficientAttnBlock with 512 in_channels...
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
# Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
# Loaded model config from [/data/zjy_work/BGDiff/models/cldm_v15.yaml]
# These weights are newly added: logvar
# These weights are newly added: model.diffusion_model.boundary_modulators.0.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.0.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.0.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.0.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.0.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.0.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.1.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.1.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.1.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.1.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.1.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.1.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.2.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.2.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.2.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.2.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.2.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.2.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.3.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.3.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.3.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.3.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.3.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.3.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.4.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.4.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.4.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.4.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.4.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.4.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.5.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.5.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.5.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.5.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.5.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.5.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.6.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.6.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.6.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.6.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.6.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.6.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.7.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.7.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.7.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.7.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.7.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.7.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.8.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.8.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.8.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.8.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.8.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.8.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.9.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.9.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.9.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.9.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.9.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.9.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.10.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.10.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.10.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.10.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.10.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.10.net.4.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.11.net.0.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.11.net.0.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.11.net.2.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.11.net.2.bias
# These weights are newly added: model.diffusion_model.boundary_modulators.11.net.4.weight
# These weights are newly added: model.diffusion_model.boundary_modulators.11.net.4.bias
# These weights are newly added: control_model.boundary_alpha
# These weights are newly added: control_model.zero_convs.0.0.weight
# These weights are newly added: control_model.zero_convs.0.0.bias
# These weights are newly added: control_model.zero_convs.1.0.weight
# These weights are newly added: control_model.zero_convs.1.0.bias
# These weights are newly added: control_model.zero_convs.2.0.weight
# These weights are newly added: control_model.zero_convs.2.0.bias
# These weights are newly added: control_model.zero_convs.3.0.weight
# These weights are newly added: control_model.zero_convs.3.0.bias
# These weights are newly added: control_model.zero_convs.4.0.weight
# These weights are newly added: control_model.zero_convs.4.0.bias
# These weights are newly added: control_model.zero_convs.5.0.weight
# These weights are newly added: control_model.zero_convs.5.0.bias
# These weights are newly added: control_model.zero_convs.6.0.weight
# These weights are newly added: control_model.zero_convs.6.0.bias
# These weights are newly added: control_model.zero_convs.7.0.weight
# These weights are newly added: control_model.zero_convs.7.0.bias
# These weights are newly added: control_model.zero_convs.8.0.weight
# These weights are newly added: control_model.zero_convs.8.0.bias
# These weights are newly added: control_model.zero_convs.9.0.weight
# These weights are newly added: control_model.zero_convs.9.0.bias
# These weights are newly added: control_model.zero_convs.10.0.weight
# These weights are newly added: control_model.zero_convs.10.0.bias
# These weights are newly added: control_model.zero_convs.11.0.weight
# These weights are newly added: control_model.zero_convs.11.0.bias
# These weights are newly added: control_model.input_hint_block.0.initial_conv.weight
# These weights are newly added: control_model.input_hint_block.0.initial_conv.bias
# These weights are newly added: control_model.input_hint_block.0.layer1.conv1.weight
# These weights are newly added: control_model.input_hint_block.0.layer1.conv1.bias
# These weights are newly added: control_model.input_hint_block.0.layer1.conv2.weight
# These weights are newly added: control_model.input_hint_block.0.layer1.conv2.bias
# These weights are newly added: control_model.input_hint_block.0.conv_before_res2.weight
# These weights are newly added: control_model.input_hint_block.0.conv_before_res2.bias
# These weights are newly added: control_model.input_hint_block.0.layer2.conv1.weight
# These weights are newly added: control_model.input_hint_block.0.layer2.conv1.bias
# These weights are newly added: control_model.input_hint_block.0.layer2.conv2.weight
# These weights are newly added: control_model.input_hint_block.0.layer2.conv2.bias
# These weights are newly added: control_model.input_hint_block.0.patch_merge1.norm.weight
# These weights are newly added: control_model.input_hint_block.0.patch_merge1.norm.bias
# These weights are newly added: control_model.input_hint_block.0.patch_merge1.reduction.weight
# These weights are newly added: control_model.input_hint_block.0.patch_merge1.reduction.bias
# These weights are newly added: control_model.input_hint_block.0.conv_before_res3.weight
# These weights are newly added: control_model.input_hint_block.0.conv_before_res3.bias
# These weights are newly added: control_model.input_hint_block.0.layer3.conv1.weight
# These weights are newly added: control_model.input_hint_block.0.layer3.conv1.bias
# These weights are newly added: control_model.input_hint_block.0.layer3.conv2.weight
# These weights are newly added: control_model.input_hint_block.0.layer3.conv2.bias
# These weights are newly added: control_model.input_hint_block.0.patch_merge2.norm.weight
# These weights are newly added: control_model.input_hint_block.0.patch_merge2.norm.bias
# These weights are newly added: control_model.input_hint_block.0.patch_merge2.reduction.weight
# These weights are newly added: control_model.input_hint_block.0.patch_merge2.reduction.bias
# These weights are newly added: control_model.input_hint_block.0.conv_before_res4.weight
# These weights are newly added: control_model.input_hint_block.0.conv_before_res4.bias
# These weights are newly added: control_model.input_hint_block.0.layer4.conv1.weight
# These weights are newly added: control_model.input_hint_block.0.layer4.conv1.bias
# These weights are newly added: control_model.input_hint_block.0.layer4.conv2.weight
# These weights are newly added: control_model.input_hint_block.0.layer4.conv2.bias
# These weights are newly added: control_model.input_hint_block.0.patch_merge3.norm.weight
# These weights are newly added: control_model.input_hint_block.0.patch_merge3.norm.bias
# These weights are newly added: control_model.input_hint_block.0.patch_merge3.reduction.weight
# These weights are newly added: control_model.input_hint_block.0.patch_merge3.reduction.bias
# These weights are newly added: control_model.input_hint_block.0.conv_before_res5.weight
# These weights are newly added: control_model.input_hint_block.0.conv_before_res5.bias
# These weights are newly added: control_model.input_hint_block.0.layer5.conv1.weight
# These weights are newly added: control_model.input_hint_block.0.layer5.conv1.bias
# These weights are newly added: control_model.input_hint_block.0.layer5.conv2.weight
# These weights are newly added: control_model.input_hint_block.0.layer5.conv2.bias
# These weights are newly added: control_model.input_hint_block.1.weight
# These weights are newly added: control_model.input_hint_block.1.bias
# These weights are newly added: control_model.boundary_hint_block.0.initial_conv.weight
# These weights are newly added: control_model.boundary_hint_block.0.initial_conv.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer1.conv1.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer1.conv1.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer1.conv2.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer1.conv2.bias
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res2.weight
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res2.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer2.conv1.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer2.conv1.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer2.conv2.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer2.conv2.bias
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge1.norm.weight
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge1.norm.bias
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge1.reduction.weight
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge1.reduction.bias
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res3.weight
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res3.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer3.conv1.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer3.conv1.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer3.conv2.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer3.conv2.bias
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge2.norm.weight
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge2.norm.bias
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge2.reduction.weight
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge2.reduction.bias
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res4.weight
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res4.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer4.conv1.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer4.conv1.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer4.conv2.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer4.conv2.bias
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge3.norm.weight
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge3.norm.bias
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge3.reduction.weight
# These weights are newly added: control_model.boundary_hint_block.0.patch_merge3.reduction.bias
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res5.weight
# These weights are newly added: control_model.boundary_hint_block.0.conv_before_res5.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer5.conv1.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer5.conv1.bias
# These weights are newly added: control_model.boundary_hint_block.0.layer5.conv2.weight
# These weights are newly added: control_model.boundary_hint_block.0.layer5.conv2.bias
# These weights are newly added: control_model.boundary_hint_block.1.weight
# These weights are newly added: control_model.boundary_hint_block.1.bias
# These weights are newly added: control_model.middle_block_out.0.weight
# These weights are newly added: control_model.middle_block_out.0.bias
# /Done.