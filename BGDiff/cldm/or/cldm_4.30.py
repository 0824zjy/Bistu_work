
import os

import math
import random
import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler

from cldm.dhi import FeatureExtractor, BoundarySpatialModulator


class ControlledUnetModel(UNetModel):
    def __init__(
        self,
        *args,
        boundary_modulation=True,
        boundary_mod_hidden=32,
        boundary_mod_start_ratio=0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.boundary_modulation = boundary_modulation
        self.boundary_modulators = nn.ModuleList(
            [BoundarySpatialModulator(in_channels=1, hidden_channels=boundary_mod_hidden) for _ in range(len(self.output_blocks))]
        )
        self.boundary_mod_start = int(len(self.output_blocks) * boundary_mod_start_ratio)

    def _apply_boundary_modulation(self, h, boundary_map, block_idx):
        if (not self.boundary_modulation) or (boundary_map is None):
            return h
        if block_idx < self.boundary_mod_start:
            return h

        if boundary_map.shape[1] > 1:
            boundary_map = boundary_map.mean(dim=1, keepdim=True)

        boundary_resized = F.interpolate(
            boundary_map, size=h.shape[-2:], mode="bilinear", align_corners=False
        )
        gamma, beta = self.boundary_modulators[block_idx](boundary_resized)
        h = h * (1.0 + gamma) + beta
        return h

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        boundary_map=None,
        **kwargs
    ):
        hs = []

        param_dtype = next(self.time_embed.parameters()).dtype
        param_device = x.device

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(device=param_device, dtype=param_dtype)
        emb = self.time_embed(t_emb)

        with torch.no_grad():
            h = x.to(dtype=param_dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h = h + control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)

            h = module(h, emb, context)
            h = self._apply_boundary_modulation(h, boundary_map=boundary_map, block_idx=i)

        out_param_dtype = None
        for m in self.out:
            if hasattr(m, "weight") and m.weight is not None:
                out_param_dtype = m.weight.dtype
                break
        if out_param_dtype is None:
            out_param_dtype = param_dtype

        h = h.to(dtype=out_param_dtype)
        return self.out(h)


class ControlNet(nn.Module):
    """
    Region-Boundary Dual-Branch ControlNet.

    - region branch keeps the old name `input_hint_block` for easier partial loading
    - boundary branch is newly added
    - both are fused before entering the shared ControlNet trunk
    """
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            boundary_channels=3,
            num_res_blocks=2,
            attention_resolutions=(4, 2, 1),
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'You forgot to include context_dim.'

        if context_dim is not None:
            assert use_spatial_transformer, 'You forgot to enable spatial transformer.'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("num_res_blocks must be int or list matching channel_mult length")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # ---- region branch (old path, keeps parameter name for partial loading) ----
        self.input_hint_block = TimestepEmbedSequential(
            FeatureExtractor(hint_channels),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # ---- boundary branch (new path) ----
        self.boundary_hint_block = TimestepEmbedSequential(
            FeatureExtractor(boundary_channels),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # learnable branch fusion, initialized conservatively
        self.boundary_alpha = nn.Parameter(torch.tensor(0.0))

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels,
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, boundary_hint=None, **kwargs):
        param_dtype = next(self.time_embed.parameters()).dtype
        param_device = x.device

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(device=param_device, dtype=param_dtype)
        emb = self.time_embed(t_emb)

        x = x.to(dtype=param_dtype)
        hint = hint.to(dtype=param_dtype)

        region_guided_hint = self.input_hint_block(hint, emb, context)

        boundary_guided_hint = None
        if boundary_hint is not None:
            boundary_hint = boundary_hint.to(dtype=param_dtype)
            boundary_guided_hint = self.boundary_hint_block(boundary_hint, emb, context)

        guided_hint = region_guided_hint
        if boundary_guided_hint is not None:
            guided_hint = guided_hint + torch.tanh(self.boundary_alpha) * boundary_guided_hint

        outs = []
        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h = h + guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, boundary_key="boundary", only_mid_control=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.boundary_key = boundary_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # ===== Morphology-Adaptive Consistency =====
        self.enable_adaptive_consistency = os.environ.get("ENABLE_ADAPTIVE_CONSISTENCY", "1").strip() == "1"
        self.consistency_w_min = float(os.environ.get("CONSISTENCY_W_MIN", "0.20"))
        self.consistency_w_max = float(os.environ.get("CONSISTENCY_W_MAX", "1.00"))
        self.consistency_t_gate = int(os.environ.get("CONSISTENCY_T_GATE", "300"))

        # ===== Morphology-Balanced Online-Aug =====
        self.enable_online_aug = os.environ.get("ENABLE_ONLINE_AUG", "1").strip() == "1"
        self.lambda_aug = float(os.environ.get("LAMBDA_AUG", "0.50"))
        self.aug_prob_min = float(os.environ.get("AUG_PROB_MIN", "0.15"))
        self.aug_prob_max = float(os.environ.get("AUG_PROB_MAX", "0.60"))
        self.aug_easy_bias = float(os.environ.get("AUG_EASY_BIAS", "1.0"))
        self.boundary_dilate_kernel = int(os.environ.get("BOUNDARY_DILATE_KERNEL", "3"))

        dm = self.model.diffusion_model

        for p in dm.parameters():
            p.requires_grad = False

        if not getattr(self, "sd_locked", False):
            for p in dm.output_blocks.parameters():
                p.requires_grad = True
            for p in dm.out.parameters():
                p.requires_grad = True

        for p in self.control_model.parameters():
            p.requires_grad = True

    # ============================================================
    # I/O helpers
    # ============================================================

    def _extract_boundary_from_mask(self, mask):
        """
        mask: B,C,H,W in [0,1]
        return: boundary tensor, same spatial size, repeated to original channel count
        """
        repeat_channels = mask.shape[1]
        mask_single = self._to_single_channel_mask(mask)

        k = self.boundary_dilate_kernel
        if k % 2 == 0:
            k += 1

        dil = F.max_pool2d(mask_single, kernel_size=k, stride=1, padding=k // 2)
        ero = 1.0 - F.max_pool2d(1.0 - mask_single, kernel_size=k, stride=1, padding=k // 2)
        boundary = (dil - ero).clamp(0.0, 1.0)

        if repeat_channels > 1:
            boundary = boundary.repeat(1, repeat_channels, 1, 1)
        return boundary

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # ---- region mask ----
        control_mask = batch[self.control_key]
        if bs is not None:
            control_mask = control_mask[:bs]
        control_mask = control_mask.to(self.device)
        control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
        control_mask = control_mask.to(memory_format=torch.contiguous_format)

        # ---- boundary ----
        if self.boundary_key in batch:
            control_boundary = batch[self.boundary_key]
            if bs is not None:
                control_boundary = control_boundary[:bs]
            control_boundary = control_boundary.to(self.device)
            control_boundary = einops.rearrange(control_boundary, 'b h w c -> b c h w')
            control_boundary = control_boundary.to(memory_format=torch.contiguous_format)
        else:
            control_boundary = self._extract_boundary_from_mask(control_mask)

        # ---- image ----
        control_image = (batch["jpg"] + 1.0) / 2.0
        if bs is not None:
            control_image = control_image[:bs]
        control_image = control_image.to(self.device)
        control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
        control_image = control_image.to(memory_format=torch.contiguous_format)

        return x, dict(
            c_crossattn=[c],
            c_concat_mask=[control_mask],
            c_concat_boundary=[control_boundary],
            c_concat_image=[control_image]
        )

    def apply_model(self, x_noisy, t, cond, *args, no_grad_model=False, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        boundary_tensor = None
        if 'c_concat_boundary' in cond and cond['c_concat_boundary'] is not None:
            boundary_tensor = torch.cat(cond['c_concat_boundary'], 1)
        elif 'c_concat' in cond and cond['c_concat'] is not None:
            boundary_tensor = self._extract_boundary_from_mask(torch.cat(cond['c_concat'], 1))

        boundary_map = None
        if boundary_tensor is not None:
            boundary_map = self._to_single_channel_mask(boundary_tensor)

        def _run_control(hint_tensor, boundary_hint_tensor):
            return self.control_model(
                x=x_noisy,
                hint=hint_tensor,
                boundary_hint=boundary_hint_tensor,
                timesteps=t,
                context=cond_txt
            )

        def _run_diffusion(control):
            return diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
                boundary_map=boundary_map
            )

        # 关键：reference / teacher 分支可整段 no_grad，避免无效建图
        with torch.no_grad() if no_grad_model else torch.enable_grad():
            if cond.get('c_concat', None) is None:
                eps = _run_diffusion(control=None)

            else:
                if 'c_concat_image' not in cond:
                    hint_region = torch.cat(cond['c_concat'], 1)
                    control = _run_control(hint_region, boundary_tensor)
                    control = [c * scale for c, scale in zip(control, self.control_scales)]

                    eps = _run_diffusion(control)

                else:
                    hint_image = torch.cat(cond['c_concat_image'], 1)
                    hint_mask = torch.cat(cond['c_concat'], 1)

                    # student 分支：保留梯度
                    control_image = _run_control(hint_image, boundary_tensor)
                    control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]

                    # reference 分支：只作为融合锚点，不需要反传，必须整段 no_grad
                    with torch.no_grad():
                        control_mask = _run_control(hint_mask, boundary_tensor)
                        control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]

                    if hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "max_steps"):
                        w_img = float(self.global_step) / float(max(1, self.trainer.max_steps))
                    else:
                        w_img = 1.0
                    w_mask = 1.0

                    control = [
                        (w_mask * cm) + (w_img * ci)
                        for cm, ci in zip(control_mask, control_image)
                    ]

                    eps = _run_diffusion(control)

        return eps

    def _apply_model_checkpointed(self, x_noisy, t, cond):
        """
        对额外学生分支使用 activation checkpoint，减少前向保存的激活显存。
        不改变目标函数，只是在 backward 时重算该分支 forward。
        """
        c_crossattn = cond["c_crossattn"][0]
        c_concat = cond["c_concat"][0]

        h, w = c_concat.shape[-2], c_concat.shape[-1]
        b = c_concat.shape[0]

        if "c_concat_boundary" in cond and cond["c_concat_boundary"] is not None:
            c_boundary = cond["c_concat_boundary"][0]
        else:
            c_boundary = c_concat.new_zeros((b, 0, h, w))

        if "c_concat_image" in cond and cond["c_concat_image"] is not None:
            c_image = cond["c_concat_image"][0]
        else:
            c_image = c_concat.new_zeros((b, 0, h, w))

        def _forward(x_noisy_, t_, c_crossattn_, c_concat_, c_boundary_, c_image_):
            cond_ = {
                "c_crossattn": [c_crossattn_],
                "c_concat": [c_concat_],
            }
            if c_boundary_.shape[1] > 0:
                cond_["c_concat_boundary"] = [c_boundary_]
            if c_image_.shape[1] > 0:
                cond_["c_concat_image"] = [c_image_]

            return self.apply_model(
                x_noisy_,
                t_,
                cond_,
                no_grad_model=False
            )

        return checkpoint(
            _forward,
            x_noisy,
            t,
            c_crossattn,
            c_concat,
            c_boundary,
            c_image,
            use_reentrant=False
        )

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=50,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        use_image_control=False,
        sampler="ddim",
        dpm_steps=20,
        hybrid_split=0.5,
        **kwargs
    ):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat_mask = c["c_concat_mask"][0][:N]
        c_cat_boundary = c["c_concat_boundary"][0][:N]
        c_cat_image = c["c_concat_image"][0][:N]
        c_txt = c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        log["control_mask"] = c_cat_mask * 2.0 - 1.0
        log["control_boundary"] = c_cat_boundary * 2.0 - 1.0
        log["control_image"] = c_cat_image * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for tt in range(self.num_timesteps):
                if tt % self.log_every_t == 0 or tt == self.num_timesteps - 1:
                    t = repeat(torch.tensor([tt]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            if use_image_control:
                cond_infer = {
                    "c_concat": [c_cat_mask],
                    "c_concat_boundary": [c_cat_boundary],
                    "c_concat_image": [c_cat_image],
                    "c_crossattn": [c_txt]
                }
            else:
                cond_infer = {
                    "c_concat": [c_cat_mask],
                    "c_concat_boundary": [c_cat_boundary],
                    "c_crossattn": [c_txt]
                }

            samples, z_denoise_row = self.sample_log(
                cond=cond_infer,
                batch_size=N,
                sampler=sampler,
                ddim_steps=ddim_steps,
                dpm_steps=dpm_steps,
                hybrid_split=hybrid_split,
                eta=ddim_eta,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if plot_denoise_rows and z_denoise_row is not None:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat_mask
            uc_boundary = c_cat_boundary

            uc_full = {
                "c_concat": [uc_cat],
                "c_concat_boundary": [uc_boundary],
                "c_crossattn": [uc_cross]
            }

            if use_image_control:
                cond_infer = {
                    "c_concat": [c_cat_mask],
                    "c_concat_boundary": [c_cat_boundary],
                    "c_concat_image": [c_cat_image],
                    "c_crossattn": [c_txt]
                }
            else:
                cond_infer = {
                    "c_concat": [c_cat_mask],
                    "c_concat_boundary": [c_cat_boundary],
                    "c_crossattn": [c_txt]
                }

            samples_cfg, _ = self.sample_log(
                cond=cond_infer,
                batch_size=N,
                sampler=sampler,
                ddim_steps=ddim_steps,
                dpm_steps=dpm_steps,
                hybrid_split=hybrid_split,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        sampler="ddim",
        ddim_steps=50,
        dpm_steps=20,
        hybrid_split=0.5,
        eta=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        **kwargs
    ):
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)

        if sampler == "ddim":
            ddim_sampler = DDIMSampler(self)
            samples, intermediates = ddim_sampler.sample(
                S=ddim_steps,
                batch_size=batch_size,
                shape=shape,
                conditioning=cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs
            )
            return samples, intermediates

        if sampler == "dpm":
            dpm_sampler = DPMSolverSampler(self)
            samples, _ = dpm_sampler.sample(
                S=dpm_steps,
                batch_size=batch_size,
                shape=shape,
                conditioning=cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs
            )
            return samples, None

        if sampler == "hybrid":
            total = int(ddim_steps)
            dpm_part = max(1, int(total * float(hybrid_split)))
            ddim_part = max(1, total - dpm_part)

            dpm_sampler = DPMSolverSampler(self)
            x_mid, _ = dpm_sampler.sample(
                S=dpm_part,
                batch_size=batch_size,
                shape=shape,
                conditioning=cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs
            )

            ddim_sampler = DDIMSampler(self)
            x_final, intermediates = ddim_sampler.sample(
                S=ddim_part,
                batch_size=batch_size,
                shape=shape,
                conditioning=cond,
                verbose=False,
                eta=eta,
                x_T=x_mid,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs
            )
            return x_final, intermediates

        raise ValueError(f"Unknown sampler={sampler}")

    def _apply_freeze_policy(self):
        train_unet_decoder = os.environ.get("TRAIN_UNET_DECODER", "0").strip() == "1"
        train_controlnet = os.environ.get("TRAIN_CONTROLNET", "1").strip() == "1"

        if hasattr(self, "first_stage_model") and self.first_stage_model is not None:
            self.first_stage_model.eval()
            for p in self.first_stage_model.parameters():
                p.requires_grad = False

        if hasattr(self, "cond_stage_model") and self.cond_stage_model is not None:
            self.cond_stage_model.eval()
            for p in self.cond_stage_model.parameters():
                p.requires_grad = False

        dm = self.model.diffusion_model
        for p in dm.parameters():
            p.requires_grad = False

        if train_unet_decoder and (not getattr(self, "sd_locked", False)):
            for p in dm.output_blocks.parameters():
                p.requires_grad = True
            for p in dm.out.parameters():
                p.requires_grad = True

        for p in self.control_model.parameters():
            p.requires_grad = train_controlnet

        print(
            f"[FreezePolicy] TRAIN_UNET_DECODER={1 if train_unet_decoder else 0}, "
            f"TRAIN_CONTROLNET={1 if train_controlnet else 0}, "
            f"sd_locked={getattr(self,'sd_locked',False)}"
        )

    def configure_optimizers(self):
        lr = float(self.learning_rate)
        self._apply_freeze_policy()

        train_unet_decoder = os.environ.get("TRAIN_UNET_DECODER", "0").strip() == "1"
        train_controlnet = os.environ.get("TRAIN_CONTROLNET", "1").strip() == "1"
        offload = os.environ.get("OFFLOAD_OPTIMIZER", "0").strip() == "1"

        params = []

        if train_controlnet:
            params += list(self.control_model.parameters())

        if train_unet_decoder and (not getattr(self, "sd_locked", False)):
            dm = self.model.diffusion_model
            params += list(dm.output_blocks.parameters())
            params += list(dm.out.parameters())

        params = [p for p in params if p.requires_grad]
        print(f"[Optimizer] trainable_params={sum(p.numel() for p in params)/1e6:.1f}M")

        if offload:
            try:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
            except Exception as e:
                raise ImportError(
                    "OFFLOAD_OPTIMIZER=1 requires deepspeed.ops.adam.DeepSpeedCPUAdam."
                ) from e

            opt = DeepSpeedCPUAdam(
                params, lr=lr,
                betas=(0.9, 0.999), eps=1e-8,
                weight_decay=1e-2, adamw_mode=True,
            )
            print("[Optimizer] Using DeepSpeedCPUAdam (adamw_mode=True)")
        else:
            opt = torch.optim.AdamW(
                params, lr=lr,
                betas=(0.9, 0.999), eps=1e-8,
                weight_decay=1e-2,
            )
            print("[Optimizer] Using torch.optim.AdamW")

        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    # ============================================================
    # morphology helpers
    # ============================================================

    def _to_single_channel_mask(self, mask):
        if mask.shape[1] == 1:
            out = mask
        else:
            out = mask.mean(dim=1, keepdim=True)
        return out.clamp(0.0, 1.0)

    def _sobel_mag(self, x):
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]],
            device=x.device, dtype=x.dtype
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]],
            device=x.device, dtype=x.dtype
        ).view(1, 1, 3, 3)

        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        return mag

    def _estimate_morph_complexity(self, mask):
        mask = self._to_single_channel_mask(mask)
        mask_bin = (mask > 0.5).float()

        area = mask_bin.sum(dim=(1, 2, 3)) + 1e-6
        perimeter = self._sobel_mag(mask_bin)
        perimeter = (perimeter > 0.05).float().sum(dim=(1, 2, 3)) + 1e-6

        compactness = (perimeter ** 2) / (4.0 * math.pi * area + 1e-6)
        area_ratio = area / float(mask_bin.shape[-1] * mask_bin.shape[-2])

        smallness = 1.0 - torch.clamp(area_ratio / 0.25, 0.0, 1.0)
        comp_norm = torch.clamp((compactness - 1.0) / 8.0, 0.0, 1.0)

        score = 0.6 * comp_norm + 0.4 * smallness
        return torch.clamp(score, 0.0, 1.0)

    def _adaptive_consistency_weight(self, mask, t):
        B = mask.shape[0]
        device = mask.device
        dtype = mask.dtype

        morph = self._estimate_morph_complexity(mask).to(device=device, dtype=dtype)

        if hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "max_steps"):
            train_prog = float(self.global_step) / float(max(1, self.trainer.max_steps))
        else:
            train_prog = 1.0
        train_prog = torch.full((B,), train_prog, device=device, dtype=dtype)

        t_gate = (t <= self.consistency_t_gate).float().to(device=device, dtype=dtype)
        w = self.consistency_w_min + (self.consistency_w_max - self.consistency_w_min) * morph * train_prog * t_gate
        return w

    def _random_shift(self, mask, max_shift=8):
        B, C, H, W = mask.shape
        out = []
        for i in range(B):
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            out.append(torch.roll(mask[i:i+1], shifts=(dy, dx), dims=(2, 3)))
        return torch.cat(out, dim=0)

    def _random_scale_pad(self, mask, scale_min=0.90, scale_max=1.10):
        B, C, H, W = mask.shape
        outs = []
        for i in range(B):
            s = random.uniform(scale_min, scale_max)
            nh = max(8, int(round(H * s)))
            nw = max(8, int(round(W * s)))
            m = F.interpolate(mask[i:i+1], size=(nh, nw), mode="bilinear", align_corners=False)
            if nh >= H and nw >= W:
                y0 = (nh - H) // 2
                x0 = (nw - W) // 2
                m = m[:, :, y0:y0+H, x0:x0+W]
            else:
                pad_t = (H - nh) // 2
                pad_b = H - nh - pad_t
                pad_l = (W - nw) // 2
                pad_r = W - nw - pad_l
                m = F.pad(m, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0)
            outs.append(m)
        return torch.cat(outs, dim=0)

    def _morph_op(self, mask, op="dilate", k=3):
        if k % 2 == 0:
            k += 1
        if op == "dilate":
            return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
        elif op == "erode":
            return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=k // 2)
        else:
            return mask

    def _boundary_jitter(self, mask):
        edge = self._sobel_mag(mask)
        edge = (edge > 0.05).float()
        noise = torch.randn_like(mask) * 0.08
        jitter = torch.clamp(mask + edge * noise, 0.0, 1.0)
        return jitter

    def _morphology_balanced_online_aug(self, mask):
        mask = self._to_single_channel_mask(mask)
        complexity = self._estimate_morph_complexity(mask)

        outs = []
        for i in range(mask.shape[0]):
            m = mask[i:i+1]
            c = float(complexity[i].item())

            p_aug = self.aug_prob_min + (self.aug_prob_max - self.aug_prob_min) * max(0.0, 1.0 - self.aug_easy_bias * c)
            p_aug = max(self.aug_prob_min, min(self.aug_prob_max, p_aug))

            if random.random() > p_aug:
                outs.append(m)
                continue

            op_id = random.choice(["dilate", "erode", "shift", "scale", "jitter"])

            if op_id == "dilate":
                k = 3 if c > 0.5 else 5
                m2 = self._morph_op(m, op="dilate", k=k)
            elif op_id == "erode":
                k = 3 if c > 0.5 else 5
                m2 = self._morph_op(m, op="erode", k=k)
            elif op_id == "shift":
                m2 = self._random_shift(m, max_shift=6 if c > 0.5 else 10)
            elif op_id == "scale":
                m2 = self._random_scale_pad(m, scale_min=0.92, scale_max=1.08)
            else:
                m2 = self._boundary_jitter(m)

            m2 = torch.clamp(m2, 0.0, 1.0)
            if mask.shape[1] > 1:
                m2 = m2.repeat(1, mask.shape[1], 1, 1)
            outs.append(m2)

        out = torch.cat(outs, dim=0)
        return out

    def p_losses(self, x_start, cond, t, noise=None):
        cond_mask = {
            "c_crossattn": [cond["c_crossattn"][0]],
            "c_concat": [cond["c_concat_mask"][0]],
            "c_concat_boundary": [cond["c_concat_boundary"][0]],
        }

        cond_image = {
            "c_crossattn": [cond["c_crossattn"][0]],
            "c_concat": [cond["c_concat_mask"][0]],
            "c_concat_boundary": [cond["c_concat_boundary"][0]],
            "c_concat_image": [cond["c_concat_image"][0]],
        }

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 主 mask 分支：保留正常图
        model_output_mask = self.apply_model(x_noisy, t, cond_mask, no_grad_model=False)

        master_dtype = model_output_mask.dtype
        master_device = x_start.device

        weights_ones = torch.ones_like(t, device=master_device, dtype=master_dtype)
        weights_thre = torch.where(
            t <= 200,
            torch.ones_like(t, device=master_device, dtype=master_dtype),
            torch.zeros_like(t, device=master_device, dtype=master_dtype),
        )

        weights_mask = 1.0 * weights_ones
        weights_image = 1.0 * weights_ones
        weights_mask_regularization = 1.0 * weights_thre

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        target = target.to(device=master_device, dtype=master_dtype)
        cond_mask_tensor = cond["c_concat_mask"][0].to(device=master_device, dtype=master_dtype)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        # 是否会进入 mask regularization（提前判定，方便尽早切断 image 图）
        run_mask_reg = (
            hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "max_steps")
            and (self.global_step > (self.trainer.max_steps * 1 / 3))
            and bool(weights_mask_regularization.any().item())
        )

        loss_simple_mask = self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
        loss_simple_mask = weights_mask * loss_simple_mask
        print(f"loss_simple_mask: {loss_simple_mask.mean()}")

        loss_simple = loss_simple_mask

        model_output_image = None
        recon_output_image_detached = None

        if bool(weights_image.all().item()):
            # 额外 image 分支：用 checkpoint，降低激活驻留
            model_output_image = self._apply_model_checkpointed(x_noisy, t, cond_image)

            loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
            loss_simple_image = loss_simple_image.to(dtype=master_dtype)

            print(f"loss_simple_image: {loss_simple_image.mean()}")
            loss_simple = loss_simple + weights_image * loss_simple_image
            loss_dict.update({f"{prefix}/loss_simple_image": loss_simple_image.mean()})

            # 提前为后续 mask_reg 生成 detached target，避免 image 图继续拖到第 4 个 forward
            if run_mask_reg:
                with torch.no_grad():
                    recon_output_image_detached = self.predict_start_from_noise(
                        x_noisy,
                        t=t,
                        noise=model_output_image.detach()
                    ).to(dtype=master_dtype)

        if model_output_image is not None:
            loss_cons = self.get_loss(
                model_output_mask, model_output_image.detach(), mean=False
            ).mean([1, 2, 3]).to(dtype=master_dtype)

            if self.enable_adaptive_consistency:
                wc = self._adaptive_consistency_weight(cond_mask_tensor, t).to(dtype=master_dtype)
            else:
                wc = torch.ones_like(loss_cons, dtype=master_dtype, device=master_device)

            loss_cons = wc * loss_cons
            print(f"loss_adaptive_consistency: {loss_cons.mean()}")

            loss_simple = loss_simple + loss_cons
            loss_dict.update({f"{prefix}/loss_adaptive_consistency": loss_cons.mean()})
            loss_dict.update({f"{prefix}/wc_mean": wc.mean()})

            # 到这里 image 分支只剩标量 loss 参与总 loss，
            # 后续 mask_reg 不再需要它的原始图了
            del model_output_image

        # ===== 先做 mask regularization，再做 online aug，避免在 mask_reg 峰值时叠加 aug 图 =====
        if run_mask_reg and (recon_output_image_detached is not None):
            noise_image_2_mask = torch.randn_like(recon_output_image_detached)
            x_noisy_mask_recon = self.q_sample(
                x_start=recon_output_image_detached,
                t=t,
                noise=noise_image_2_mask
            )

            # 这仍然是学生分支，需要梯度；但用 checkpoint 降激活显存
            model_output_mask_xt = self._apply_model_checkpointed(x_noisy_mask_recon.detach(), t, cond_mask)

            loss_mask_reg = self.get_loss(
                model_output_mask_xt, noise_image_2_mask, mean=False
            ).mean([1, 2, 3]).to(dtype=master_dtype)

            loss_mask_reg = self.lambda_aug * 0.5 * loss_mask_reg
            print(f"loss_mask_regularization: {loss_mask_reg.mean()}")

            loss_simple = loss_simple + weights_mask_regularization * loss_mask_reg
            loss_dict.update({f"{prefix}/loss_mask_regularization": loss_mask_reg.mean()})

            del model_output_mask_xt
            del x_noisy_mask_recon
            del noise_image_2_mask
            del recon_output_image_detached

        if self.enable_online_aug:
            aug_mask = self._morphology_balanced_online_aug(cond_mask_tensor)
            if cond_mask_tensor.shape[1] > 1 and aug_mask.shape[1] == 1:
                aug_mask = aug_mask.repeat(1, cond_mask_tensor.shape[1], 1, 1)

            aug_boundary = self._extract_boundary_from_mask(aug_mask)

            cond_aug = {
                "c_crossattn": [cond["c_crossattn"][0]],
                "c_concat": [aug_mask],
                "c_concat_boundary": [aug_boundary],
            }

            # online aug 分支也用 checkpoint
            model_output_aug = self._apply_model_checkpointed(x_noisy, t, cond_aug)

            loss_aug = self.get_loss(model_output_aug, target, mean=False).mean([1, 2, 3]).to(dtype=master_dtype)
            loss_aug = self.lambda_aug * loss_aug

            print(f"loss_online_aug: {loss_aug.mean()}")
            loss_simple = loss_simple + loss_aug
            loss_dict.update({f"{prefix}/loss_online_aug": loss_aug.mean()})

            del model_output_aug

        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(device=master_device, dtype=loss_simple.dtype)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = (loss.mean() * float(self.l_simple_weight)).to(dtype=master_dtype)

        loss_vlb = loss_simple
        lvlb_w = self.lvlb_weights[t].to(device=master_device, dtype=loss_vlb.dtype)
        loss_vlb = (lvlb_w * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})

        loss = loss + float(self.original_elbo_weight) * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        loss = loss.to(dtype=master_dtype)
        return loss, loss_dict
