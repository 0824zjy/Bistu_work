import os
import copy
import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler

from cldm.dhi import FeatureExtractor


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []

        # 以参数 dtype 为准（兼容 DeepSpeed FP16 强制 cast）
        param_dtype = next(self.time_embed.parameters()).dtype
        param_device = x.device

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(device=param_device, dtype=param_dtype)
        emb = self.time_embed(t_emb)

        # encoder（no grad）
        with torch.no_grad():
            h = x.to(dtype=param_dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h = h + control.pop()

        for module in self.output_blocks:
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        # 输出层 dtype 对齐
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
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
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
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint

        # NOTE: keep this, but DON'T rely on it in forward (DeepSpeed may override)
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
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            FeatureExtractor(hint_channels),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
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
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
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
                        )
                        if resblock_updown
                        else Downsample(
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
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
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
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        # 以 time_embed 参数 dtype 为准（DeepSpeed cast 后这里才真实）
        param_dtype = next(self.time_embed.parameters()).dtype
        param_device = x.device

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(device=param_device, dtype=param_dtype)
        emb = self.time_embed(t_emb)

        # hint / x 同 dtype，避免 conv/linear float vs half
        x = x.to(dtype=param_dtype)
        hint = hint.to(dtype=param_dtype)

        guided_hint = self.input_hint_block(hint, emb, context)

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

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        # ============================================================
        # Task-aware loss hyper-parameters
        # ------------------------------------------------------------
        # foreground / boundary / background 权重
        # 边界 > 前景 > 背景
        # ============================================================
        self.loss_fg_weight = float(os.environ.get("LOSS_FG_WEIGHT", "2.0"))
        self.loss_bd_weight = float(os.environ.get("LOSS_BD_WEIGHT", "4.0"))
        self.loss_bg_weight = float(os.environ.get("LOSS_BG_WEIGHT", "1.0"))

        # 边界一致性损失
        self.lambda_boundary = float(os.environ.get("LAMBDA_BOUNDARY", "0.5"))

        # 前景感知的 mask->image consistency
        self.lambda_mask2image = float(os.environ.get("LAMBDA_MASK2IMAGE", "1.0"))

        # mask regularization 系数
        self.lambda_mask_reg = float(os.environ.get("LAMBDA_MASK_REG", "1.0"))

        # 边界带宽度
        self.boundary_dilate_kernel = int(os.environ.get("BOUNDARY_DILATE_KERNEL", "5"))

        # ============================================================
        # DeepSpeed ZeRO2 兼容
        # ============================================================
        dm = self.model.diffusion_model  # ControlledUnetModel

        # 1) 先全冻结 diffusion UNet
        for p in dm.parameters():
            p.requires_grad = False

        # 2) 若允许训练 decoder，则只解冻 output_blocks/out
        if not getattr(self, "sd_locked", False):
            for p in dm.output_blocks.parameters():
                p.requires_grad = True
            for p in dm.out.parameters():
                p.requires_grad = True

        # 3) ControlNet 确保为可训练
        for p in self.control_model.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control_mask = batch[self.control_key]
        if bs is not None:
            control_mask = control_mask[:bs]
        control_mask = control_mask.to(self.device)
        control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
        control_mask = control_mask.to(memory_format=torch.contiguous_format)

        control_image = (batch["jpg"] + 1.0) / 2.0
        if bs is not None:
            control_image = control_image[:bs]
        control_image = control_image.to(self.device)
        control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
        control_image = control_image.to(memory_format=torch.contiguous_format)

        return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        train_controlnet = any(p.requires_grad for p in self.control_model.parameters())

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt,
                control=None, only_mid_control=self.only_mid_control
            )
        else:
            # mask-only control (default)
            if 'c_concat_image' not in cond:
                if train_controlnet:
                    control = self.control_model(
                        x=x_noisy,
                        hint=torch.cat(cond['c_concat'], 1),
                        timesteps=t,
                        context=cond_txt
                    )
                else:
                    with torch.no_grad():
                        control = self.control_model(
                            x=x_noisy,
                            hint=torch.cat(cond['c_concat'], 1),
                            timesteps=t,
                            context=cond_txt
                        )
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                eps = diffusion_model(
                    x=x_noisy, timesteps=t, context=cond_txt,
                    control=control, only_mid_control=self.only_mid_control
                )
            else:
                # mask + image control (image optional enhancement)
                # image control: WITH grad
                if train_controlnet:
                    control_image = self.control_model(
                        x=x_noisy,
                        hint=torch.cat(cond['c_concat_image'], 1),
                        timesteps=t,
                        context=cond_txt
                    )
                else:
                    with torch.no_grad():
                        control_image = self.control_model(
                            x=x_noisy,
                            hint=torch.cat(cond['c_concat_image'], 1),
                            timesteps=t,
                            context=cond_txt
                        )
                control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]

                # mask control: compute under grad, but detach outputs to block gradients
                if train_controlnet:
                    control_mask = self.control_model(
                        x=x_noisy,
                        hint=torch.cat(cond['c_concat'], 1),
                        timesteps=t,
                        context=cond_txt
                    )
                else:
                    with torch.no_grad():
                        control_mask = self.control_model(
                            x=x_noisy,
                            hint=torch.cat(cond['c_concat'], 1),
                            timesteps=t,
                            context=cond_txt
                        )
                control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]
                control_mask = [c.detach() for c in control_mask]

                # weights schedule: mask fixed 1.0, image ramp up with global_step
                if hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "max_steps"):
                    w_img = float(self.global_step) / float(max(1, self.trainer.max_steps))
                else:
                    w_img = 1.0
                w_mask = 1.0

                control = [
                    (w_mask * cm.detach()) + (w_img * ci)
                    for cm, ci in zip(control_mask, control_image)
                ]

                eps = diffusion_model(
                    x=x_noisy, timesteps=t, context=cond_txt,
                    control=control, only_mid_control=self.only_mid_control
                )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   use_image_control=False,
                   sampler="ddim",
                   dpm_steps=20,
                   hybrid_split=0.5,
                   **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat_mask, c_cat_image, c = c["c_concat_mask"][0][:N], c["c_concat_image"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["control_mask"] = c_cat_mask * 2.0 - 1.0
        log["control_image"] = c_cat_image * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
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
                cond_infer = {"c_concat": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]}
            else:
                cond_infer = {"c_concat": [c_cat_mask], "c_crossattn": [c]}

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
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

            if use_image_control:
                pass

            if use_image_control:
                cond_infer = {"c_concat": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]}
            else:
                cond_infer = {"c_concat": [c_cat_mask], "c_crossattn": [c]}

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
        """
        cond: dict, 至少包含:
              - "c_concat": [mask]
              - "c_crossattn": [text]
              可选:
              - "c_concat_image": [image]

        sampler:
          - "ddim": 直接 DDIM 采样
          - "dpm": 直接 DPM-Solver 采样
          - "hybrid": 先 DPM，再 DDIM refine
        """
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
        """
        冻结策略（支持 Stage2：只训 UNet decoder，冻结 ControlNet）
        环境变量：
        - TRAIN_UNET_DECODER=1/0
        - TRAIN_CONTROLNET=1/0
        - sd_locked=True/False
        """
        train_unet_decoder = os.environ.get("TRAIN_UNET_DECODER", "0").strip() == "1"
        train_controlnet = os.environ.get("TRAIN_CONTROLNET", "1").strip() == "1"

        # 1) 冻结 VAE & 文本编码器
        if hasattr(self, "first_stage_model") and self.first_stage_model is not None:
            self.first_stage_model.eval()
            for p in self.first_stage_model.parameters():
                p.requires_grad = False

        if hasattr(self, "cond_stage_model") and self.cond_stage_model is not None:
            self.cond_stage_model.eval()
            for p in self.cond_stage_model.parameters():
                p.requires_grad = False

        # 2) 冻结 diffusion UNet 全部参数（先全冻结）
        dm = self.model.diffusion_model
        for p in dm.parameters():
            p.requires_grad = False

        # 3) 是否训练 UNet decoder(output_blocks/out)
        if train_unet_decoder and (not getattr(self, "sd_locked", False)):
            for p in dm.output_blocks.parameters():
                p.requires_grad = True
            for p in dm.out.parameters():
                p.requires_grad = True

        # 4) ControlNet 是否训练
        for p in self.control_model.parameters():
            p.requires_grad = train_controlnet

        print(
            f"[FreezePolicy] TRAIN_UNET_DECODER={1 if train_unet_decoder else 0}, "
            f"TRAIN_CONTROLNET={1 if train_controlnet else 0}, "
            f"sd_locked={getattr(self,'sd_locked',False)}"
        )

    def configure_optimizers(self):
        """
        Stage2（decoder-only）推荐配置：
        - TRAIN_UNET_DECODER=1
        - SD_LOCKED=0
        - TRAIN_CONTROLNET=0
        """
        lr = float(self.learning_rate)

        self._apply_freeze_policy()

        train_unet_decoder = os.environ.get("TRAIN_UNET_DECODER", "0").strip() == "1"
        train_controlnet = os.environ.get("TRAIN_CONTROLNET", "1").strip() == "1"
        offload = os.environ.get("OFFLOAD_OPTIMIZER", "0").strip() == "1"

        params = []

        # 1) ControlNet
        if train_controlnet:
            params += list(self.control_model.parameters())

        # 2) UNet decoder
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
                params,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-2,
                adamw_mode=True,
            )
            print("[Optimizer] Using DeepSpeedCPUAdam (adamw_mode=True) because OFFLOAD_OPTIMIZER=1")
        else:
            opt = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-2,
            )
            print("[Optimizer] Using torch.optim.AdamW (OFFLOAD_OPTIMIZER=0)")

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
    # helper functions for task-aware loss
    # ============================================================

    def _to_single_channel_mask(self, mask):
        """
        mask: [B,C,H,W], 通常是 3 通道 binary hint
        return: [B,1,H,W] in [0,1]
        """
        if mask.shape[1] == 1:
            out = mask
        else:
            out = mask.mean(dim=1, keepdim=True)
        return out.clamp(0.0, 1.0)

    def _sobel_mag(self, x):
        """
        Sobel 梯度幅值
        x: [B,C,H,W] or [B,1,H,W]
        return: [B,1,H,W]
        """
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

        B = mag.shape[0]
        mag_flat = mag.view(B, -1)
        mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        mag = mag / (mag_max + 1e-6)
        return mag

    def _build_region_weight_map(self, mask, out_h, out_w,
                                 fg_weight=None, bd_weight=None, bg_weight=None):
        """
        从输入 mask 构建区域权重图（latent 尺度）：
        - foreground 权重高
        - boundary 权重最高
        - background 权重低

        mask: [B,C,H,W] or [B,1,H,W], image-space
        return:
            weight_map:   [B,1,out_h,out_w]
            fg_map:       [B,1,out_h,out_w]
            boundary_map: [B,1,out_h,out_w]
        """
        fg_weight = self.loss_fg_weight if fg_weight is None else fg_weight
        bd_weight = self.loss_bd_weight if bd_weight is None else bd_weight
        bg_weight = self.loss_bg_weight if bg_weight is None else bg_weight

        mask_1c = self._to_single_channel_mask(mask)
        fg_map = F.interpolate(mask_1c, size=(out_h, out_w), mode="bilinear", align_corners=False)
        fg_map = fg_map.clamp(0.0, 1.0)

        boundary_map = self._sobel_mag(fg_map)

        k = max(1, int(self.boundary_dilate_kernel))
        if k % 2 == 0:
            k += 1
        boundary_map = F.max_pool2d(boundary_map, kernel_size=k, stride=1, padding=k // 2)
        boundary_map = boundary_map.clamp(0.0, 1.0)

        weight_map = bg_weight * torch.ones_like(fg_map)
        weight_map = weight_map + (fg_weight - bg_weight) * fg_map
        weight_map = weight_map + (bd_weight - bg_weight) * boundary_map

        return weight_map, fg_map, boundary_map

    def _weighted_reduce(self, loss_map, weight_map):
        """
        pixel-wise weighted average
        loss_map:   [B,C,H,W]
        weight_map: [B,1,H,W]
        return: [B]
        """
        if weight_map.shape[-2:] != loss_map.shape[-2:]:
            weight_map = F.interpolate(weight_map, size=loss_map.shape[-2:], mode="bilinear", align_corners=False)

        weighted = loss_map * weight_map
        num = weighted.sum(dim=(1, 2, 3))
        den = weight_map.sum(dim=(1, 2, 3)) * loss_map.shape[1] + 1e-6
        return num / den

    def _boundary_consistency_loss(self, pred_x0, target_x0, boundary_weight):
        """
        边界一致性损失：
        pred_x0 / target_x0 做 Sobel
        只在 lesion boundary band 上约束
        """
        pred_edge = self._sobel_mag(pred_x0)
        target_edge = self._sobel_mag(target_x0)
        edge_diff = torch.abs(pred_edge - target_edge)
        return self._weighted_reduce(edge_diff, boundary_weight)

    def p_losses(self, x_start, cond, t, noise=None):
        """
        Task-aware diffusion loss for better downstream segmentation:
        1) 边界一致性损失 L_boundary
        2) 前景/边界区域重加权损失
        3) 前景感知的 Noise Consistency Loss

        同时保留你原有的：
        - mask branch denoising
        - image branch denoising
        - mask -> image distillation
        - mask regularization
        """

        # --- build cond dicts ---
        cond_mask = {
            "c_crossattn": [cond["c_crossattn"][0]],
            "c_concat": [cond["c_concat_mask"][0]],
        }

        cond_image = {
            "c_crossattn": [cond["c_crossattn"][0]],
            "c_concat": [cond["c_concat_mask"][0]],
            "c_concat_image": [cond["c_concat_image"][0]],
        }

        # --- noise / q_sample ---
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # --- forward (mask branch first) ---
        model_output_mask = self.apply_model(x_noisy, t, cond_mask)

        # master dtype/device
        master_dtype = model_output_mask.dtype
        master_device = x_start.device

        # --- cond mask -> region weights on latent resolution ---
        cond_mask_tensor = cond["c_concat_mask"][0].to(device=master_device, dtype=master_dtype)
        _, _, out_h, out_w = model_output_mask.shape

        region_weight_map, fg_map, boundary_map = self._build_region_weight_map(
            cond_mask_tensor,
            out_h=out_h,
            out_w=out_w,
            fg_weight=self.loss_fg_weight,
            bd_weight=self.loss_bd_weight,
            bg_weight=self.loss_bg_weight,
        )

        # --- scalar scheduling weights ---
        weights_ones = torch.ones_like(t, device=master_device, dtype=master_dtype)
        weights_thre = torch.where(
            t <= 200,
            torch.ones_like(t, device=master_device, dtype=master_dtype),
            torch.zeros_like(t, device=master_device, dtype=master_dtype),
        )

        weights_mask = 1.0 * weights_ones
        weights_image = 1.0 * weights_ones
        weights_mask_2_image = 1.0 * weights_ones
        weights_mask_regularization = 1.0 * weights_thre

        # --- targets ---
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        target = target.to(device=master_device, dtype=master_dtype)
        x_start_master = x_start.to(device=master_device, dtype=master_dtype)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        # ============================================================
        # loss 0: mask denoising loss (foreground / boundary weighted)
        # ============================================================
        loss_map_mask = self.get_loss(model_output_mask, target, mean=False).to(dtype=master_dtype)
        loss_simple_mask = self._weighted_reduce(loss_map_mask, region_weight_map)
        loss_simple = weights_mask * loss_simple_mask

        print(f"loss_simple_mask: {loss_simple_mask.mean()}")

        # ============================================================
        # loss 1: image branch denoising loss (foreground / boundary weighted)
        # ============================================================
        model_output_image = None
        if bool(weights_image.all().item()):
            model_output_image = self.apply_model(x_noisy, t, cond_image)

            loss_map_image = self.get_loss(model_output_image, target, mean=False).to(dtype=master_dtype)
            loss_simple_image = self._weighted_reduce(loss_map_image, region_weight_map)

            print(f"loss_simple_image: {loss_simple_image.mean()}")

            loss_simple = loss_simple + weights_image * loss_simple_image
            loss_dict.update({f"{prefix}/loss_simple_image": loss_simple_image.mean()})

        # ============================================================
        # loss 2: foreground-aware noise consistency / distillation
        #         mask branch -> image branch
        # ============================================================
        if model_output_image is not None and bool(weights_mask_2_image.all().item()):
            loss_map_mask_2_image = self.get_loss(
                model_output_mask, model_output_image.detach(), mean=False
            ).to(dtype=master_dtype)

            loss_simple_mask_2_image = self._weighted_reduce(loss_map_mask_2_image, region_weight_map)
            loss_simple_mask_2_image = self.lambda_mask2image * loss_simple_mask_2_image

            print(f"loss_simple_mask_2_image: {loss_simple_mask_2_image.mean()}")

            loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image
            loss_dict.update({f"{prefix}/loss_mask2image": loss_simple_mask_2_image.mean()})

        # ============================================================
        # loss 3: mask regularization (late stage)
        #         也使用前景/边界重加权
        # ============================================================
        if (
            hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "max_steps")
            and (self.global_step > (self.trainer.max_steps * 1 / 3))
            and bool(weights_mask_regularization.any().item())
            and model_output_image is not None
        ):
            recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
            recon_output_image = recon_output_image.to(dtype=master_dtype)

            noise_image_2_mask = torch.randn_like(recon_output_image)
            x_noisy_mask_recon = self.q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)

            model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask)

            loss_map_mask_regularization = self.get_loss(
                model_output_mask_xt, noise_image_2_mask, mean=False
            ).to(dtype=master_dtype)

            loss_simple_mask_regularization = self._weighted_reduce(
                loss_map_mask_regularization, region_weight_map
            )
            loss_simple_mask_regularization = self.lambda_mask_reg * loss_simple_mask_regularization

            print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")

            loss_simple = loss_simple + weights_mask_regularization * loss_simple_mask_regularization
            loss_dict.update({f"{prefix}/loss_mask_regularization": loss_simple_mask_regularization.mean()})

        # ============================================================
        # loss 4: boundary consistency loss
        #         mask branch 重建 x0 与 target/teacher 的边界响应对齐
        # ============================================================
        recon_output_mask = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_mask)
        recon_output_mask = recon_output_mask.to(dtype=master_dtype)

        if model_output_image is not None:
            recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
            recon_output_image = recon_output_image.to(dtype=master_dtype)

            # teacher anchor: 真实 latent + image branch latent 的平均
            boundary_target = 0.5 * x_start_master + 0.5 * recon_output_image.detach()
        else:
            boundary_target = x_start_master.detach()

        loss_boundary = self._boundary_consistency_loss(
            pred_x0=recon_output_mask,
            target_x0=boundary_target,
            boundary_weight=boundary_map
        ).to(dtype=master_dtype)

        loss_boundary = self.lambda_boundary * loss_boundary

        print(f"loss_boundary: {loss_boundary.mean()}")

        loss_simple = loss_simple + loss_boundary

        # ============================================================
        # logging
        # ============================================================
        loss_dict.update({
            f"{prefix}/loss_simple": loss_simple.mean(),
            f"{prefix}/loss_boundary": loss_boundary.mean(),
        })

        # --- logvar_t MUST match dtype of loss_simple ---
        logvar_t = self.logvar[t].to(device=master_device, dtype=loss_simple.dtype)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        # l_simple_weight is python float; keep result in master dtype
        loss = (loss.mean() * float(self.l_simple_weight)).to(dtype=master_dtype)

        # --- vlb term ---
        loss_vlb = loss_simple
        lvlb_w = self.lvlb_weights[t].to(device=master_device, dtype=loss_vlb.dtype)
        loss_vlb = (lvlb_w * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})

        loss = loss + float(self.original_elbo_weight) * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        # ultimate guard
        loss = loss.to(dtype=master_dtype)

        return loss, loss_dict
