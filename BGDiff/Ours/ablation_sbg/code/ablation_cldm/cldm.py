#5.5
import os
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)

from ablation_cldm.dhi import FeatureExtractor, BoundarySpatialModulator


# ============================================================
# Environment helpers
# ============================================================

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).lower() in ["1", "true", "yes", "y", "on"]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _as_single_channel(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def _repeat_to_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x is None:
        return None
    if x.shape[1] == channels:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, channels, 1, 1)
    return x[:, :1].repeat(1, channels, 1, 1)


def _maybe_hwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if x.ndim == 4 and x.shape[-1] in [1, 3, 4]:
        x = x.permute(0, 3, 1, 2).contiguous()
    return x


def _run_hint_block(block, x, emb=None, context=None):
    """
    Compatible runner for both:

    1. TimestepEmbedSequential:
        block(x, emb, context)

    2. nn.Sequential:
        block(x)

    Your old checkpoint uses:
        input_hint_block.0.initial_conv.*
        input_hint_block.1.weight

    This corresponds to:
        nn.Sequential(FeatureExtractor(...), zero_conv)
    """
    try:
        return block(x, emb, context)
    except TypeError:
        return block(x)


# ============================================================
# Controlled UNet
# ============================================================

class ControlledUnetModel(UNetModel):
    """
    UNet backbone with optional ControlNet features.

    New SBP-PG behavior:
        boundary is not a hard condition.
        boundary enters only as soft prior through weak late-stage decoder modulation:

            h = h * (1 + alpha_t * boundary_mod_scale * gamma)
                + alpha_t * boundary_mod_scale * beta

    For old checkpoint compatibility, this class keeps:

        model.diffusion_model.boundary_modulators.0.net.*
        ...
        model.diffusion_model.boundary_modulators.11.net.*
    """

    def __init__(
        self,
        *args,
        boundary_modulation=True,
        boundary_mod_hidden=32,
        boundary_mod_start_ratio=0.75,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.enable_boundary_modulation = _env_bool(
            "ENABLE_BOUNDARY_MODULATION",
            bool(boundary_modulation),
        )
        self.boundary_mod_scale = _env_float("BOUNDARY_MOD_SCALE", 0.10)
        self.boundary_mod_start_ratio = _env_float(
            "BOUNDARY_MOD_START_RATIO",
            float(boundary_mod_start_ratio),
        )

        # Keep legacy checkpoint key structure:
        #   boundary_modulators.0.net.*
        #   ...
        #   boundary_modulators.11.net.*
        self.boundary_modulators = nn.ModuleList(
            [
                BoundarySpatialModulator(
                    in_channels=1,
                    hidden_channels=int(boundary_mod_hidden),
                    dim=2,
                )
                for _ in range(len(self.output_blocks))
            ]
        )

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        boundary_map=None,
        boundary_weight=None,
        boundary_mod_scale=None,
        boundary_mod_start_ratio=None,
        enable_boundary_modulation=None,
        **kwargs,
    ):
        hs = []

        t_emb = timestep_embedding(
            timesteps,
            self.model_channels,
            repeat_only=False,
        )
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        if control is not None:
            h = h + control.pop()

        if enable_boundary_modulation is None:
            enable_boundary_modulation = self.enable_boundary_modulation

        if boundary_mod_scale is None:
            boundary_mod_scale = self.boundary_mod_scale

        if boundary_mod_start_ratio is None:
            boundary_mod_start_ratio = self.boundary_mod_start_ratio

        n_output_blocks = len(self.output_blocks)
        mod_start_idx = int(float(boundary_mod_start_ratio) * n_output_blocks)
        mod_start_idx = max(0, min(mod_start_idx, n_output_blocks - 1))

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)

            h = module(h, emb, context)

            # ------------------------------------------------------------
            # Weak Boundary-aware Decoder Modulation
            # ------------------------------------------------------------
            if (
                enable_boundary_modulation
                and boundary_map is not None
                and boundary_weight is not None
                and hasattr(self, "boundary_modulators")
                and i < len(self.boundary_modulators)
                and i >= mod_start_idx
                and float(boundary_mod_scale) > 0.0
            ):
                bm = _as_single_channel(boundary_map)
                bm = F.interpolate(
                    bm,
                    size=h.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                bm = bm.to(device=h.device, dtype=h.dtype)

                gamma, beta = self.boundary_modulators[i](bm)
                gamma = gamma.to(device=h.device, dtype=h.dtype)
                beta = beta.to(device=h.device, dtype=h.dtype)

                bw = boundary_weight.to(device=h.device, dtype=h.dtype)
                scale = bw * float(boundary_mod_scale)

                # Weak modulation only. Boundary is soft prior, not hard control.
                h = h * (1.0 + scale * gamma) + scale * beta

        h = h.type(x.dtype)
        return self.out(h)


# ============================================================
# ControlNet
# ============================================================

class ControlNet(nn.Module):
    """
    ControlNet with mask as main condition and soft boundary prior as weak
    progressive auxiliary condition.

    Old checkpoint compatibility:

        control_model.input_hint_block.0.initial_conv.*
        control_model.input_hint_block.1.*

        control_model.boundary_hint_block.0.initial_conv.*
        control_model.boundary_hint_block.1.*
    """

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
        boundary_channels=3,
        **kwargs,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, "context_dim must be provided when using SpatialTransformer."

        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels must be set."

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels must be set."

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        self.boundary_channels = boundary_channels
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_checkpoint = use_checkpoint
        self.channel_mult = channel_mult

        self.boundary_branch_scale = _env_float("BOUNDARY_BRANCH_SCALE", 0.10)

        # Learnable boundary alpha, initialized as zero to avoid strong boundary control.
        self.boundary_alpha = nn.Parameter(
    torch.tensor(_env_float("BOUNDARY_ALPHA_INIT", 1.0))
)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # ------------------------------------------------------------
        # Main mask / region branch
        # Compatible with old checkpoint key names.
        # ------------------------------------------------------------
        self.input_hint_block = nn.Sequential(
            FeatureExtractor(hint_channels=hint_channels, dim=dims),
            zero_module(
                conv_nd(
                    dims,
                    256,
                    model_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
        )

        # ------------------------------------------------------------
        # Soft boundary prior branch
        # Compatible with old checkpoint key names.
        # Input should be soft boundary prior, not hard binary boundary.
        # ------------------------------------------------------------
        self.boundary_hint_block = nn.Sequential(
            FeatureExtractor(hint_channels=boundary_channels, dim=dims),
            zero_module(
                conv_nd(
                    dims,
                    256,
                    model_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
        )

        # ------------------------------------------------------------
        # Optional image-assisted branch
        # New branch; old checkpoints will not contain these weights.
        # strict=False will allow loading.
        # ------------------------------------------------------------
        self.image_hint_block = nn.Sequential(
            FeatureExtractor(hint_channels=3, dim=dims),
            zero_module(
                conv_nd(
                    dims,
                    256,
                    model_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList(
            [self.make_zero_conv(model_channels)]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(num_res_blocks):
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

                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim[0] if isinstance(context_dim, list) else context_dim,
                                disable_self_attn=False,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
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
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
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
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim[0] if isinstance(context_dim, list) else context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            )
            if use_spatial_transformer
            else AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
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
            zero_module(conv_nd(2, channels, channels, 1, padding=0))
        )

    def forward(
        self,
        x,
        hint,
        timesteps,
        context,
        boundary_hint=None,
        boundary_weight=None,
        boundary_branch_scale=None,
        image_hint=None,
        **kwargs,
    ):
        t_emb = timestep_embedding(
            timesteps,
            self.model_channels,
            repeat_only=False,
        )
        emb = self.time_embed(t_emb)

        # Main mask / region branch.
        region_guided_hint = _run_hint_block(
            self.input_hint_block,
            hint,
            emb,
            context,
        )
        guided_hint = region_guided_hint

        # Weak soft boundary prior branch.
        if boundary_hint is not None:
            boundary_hint = _repeat_to_channels(boundary_hint, self.boundary_channels)
            boundary_hint = boundary_hint.to(device=hint.device, dtype=hint.dtype)

            boundary_guided_hint = _run_hint_block(
                self.boundary_hint_block,
                boundary_hint,
                emb,
                context,
            )

            if boundary_weight is None:
                b = x.shape[0]
                boundary_weight = torch.zeros(
                    (b, 1, 1, 1),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                boundary_weight = boundary_weight.to(device=x.device, dtype=x.dtype)

            if boundary_branch_scale is None:
                boundary_branch_scale = self.boundary_branch_scale

            guided_hint = guided_hint + (
                boundary_weight
                * float(boundary_branch_scale)
                * torch.tanh(self.boundary_alpha)
                * boundary_guided_hint
            )

        # Optional image-assisted branch.
        if image_hint is not None:
            image_hint = image_hint.to(device=hint.device, dtype=hint.dtype)
            if image_hint.shape[1] != 3:
                image_hint = _repeat_to_channels(image_hint, 3)

            image_guided_hint = _run_hint_block(
                self.image_hint_block,
                image_hint,
                emb,
                context,
            )

            image_branch_scale = _env_float("IMAGE_BRANCH_SCALE", 0.10)
            guided_hint = guided_hint + float(image_branch_scale) * image_guided_hint

        outs = []
        h = x.type(self.dtype)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)

                # Safety alignment:
                # guided_hint comes from mask / soft boundary encoder.
                # It should match latent h spatial size, but when yaml image_size
                # differs from actual dataset resolution, it may mismatch.
                if guided_hint.shape[-2:] != h.shape[-2:]:
                    guided_hint = F.interpolate(
                        guided_hint,
                        size=h.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                guided_hint = guided_hint.to(device=h.device, dtype=h.dtype)

                h = h + guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)

            outs.append(zero_conv(h, emb, context))


        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


# ============================================================
# ControlLDM
# ============================================================

class ControlLDM(LatentDiffusion):
    """
    Latent Diffusion with mask-conditioned ControlNet and soft boundary prior.

    Main path:
        mask / region condition.

    Boundary path:
        soft boundary prior only.
        Used through:
            1. weak progressive ControlNet fusion
            2. weak late decoder modulation
            3. optional tolerance-band loss
    """

    def __init__(
        self,
        control_stage_config,
        control_key,
        boundary_key=None,
        only_mid_control=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.boundary_key = boundary_key
        self.only_mid_control = only_mid_control

        self.control_scales = [1.0] * 13

        self.enable_soft_boundary_prior = _env_bool("ENABLE_SOFT_BOUNDARY_PRIOR", True)
        self.boundary_prior_tau = _env_float("BOUNDARY_PRIOR_TAU", 4.0)
        self.boundary_prior_radius = _env_int("BOUNDARY_PRIOR_RADIUS", 12)
        self.boundary_dilate_kernel = _env_int("BOUNDARY_DILATE_KERNEL", 3)
        self.boundary_condition_mode = os.environ.get(
    "BOUNDARY_CONDITION_MODE",
    "soft_from_mask"
).lower()

        self.enable_progressive_boundary_guidance = _env_bool(
            "ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE",
            True,
        )
        self.boundary_guidance_max = _env_float("BOUNDARY_GUIDANCE_MAX", 0.15)
        self.boundary_guidance_start_ratio = _env_float(
            "BOUNDARY_GUIDANCE_START_RATIO",
            0.35,
        )
        self.boundary_guidance_temperature = _env_float(
            "BOUNDARY_GUIDANCE_TEMPERATURE",
            0.05,
        )

        self.enable_boundary_modulation = _env_bool("ENABLE_BOUNDARY_MODULATION", True)
        self.boundary_mod_scale = _env_float("BOUNDARY_MOD_SCALE", 0.10)
        self.boundary_mod_start_ratio = _env_float("BOUNDARY_MOD_START_RATIO", 0.75)

        self.boundary_branch_scale = _env_float("BOUNDARY_BRANCH_SCALE", 0.10)

        self.enable_tolerance_band_loss = _env_bool("ENABLE_TOLERANCE_BAND_LOSS", True)
        self.lambda_band = _env_float("LAMBDA_BAND", 0.02)
        self.boundary_band_t_gate = _env_int("BOUNDARY_BAND_T_GATE", 200)

    # ------------------------------------------------------------
    # Soft Boundary Prior
    # ------------------------------------------------------------

    def _extract_hard_boundary_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return None

        m = _as_single_channel(mask.float())
        m = torch.clamp(m, 0.0, 1.0)
        m_bin = (m > 0.5).float()

        k = max(3, int(self.boundary_dilate_kernel))
        if k % 2 == 0:
            k += 1

        pad = k // 2

        dilation = F.max_pool2d(m_bin, kernel_size=k, stride=1, padding=pad)
        erosion = 1.0 - F.max_pool2d(1.0 - m_bin, kernel_size=k, stride=1, padding=pad)

        hard_boundary = torch.clamp(dilation - erosion, 0.0, 1.0)
        return hard_boundary

    def _soft_boundary_prior_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return None

        b, _, h, w = mask.shape

        if not self.enable_soft_boundary_prior:
            return torch.zeros(
                (b, 1, h, w),
                device=mask.device,
                dtype=mask.dtype,
            )

        hard = self._extract_hard_boundary_from_mask(mask)
        hard = hard.float()

        radius = max(1, int(self.boundary_prior_radius))
        tau = max(1e-6, float(self.boundary_prior_tau))

        soft = hard.clone()
        prev = hard.clone()

        for r in range(1, radius + 1):
            k = 2 * r + 1
            dilated = F.max_pool2d(
                hard,
                kernel_size=k,
                stride=1,
                padding=r,
            )
            shell = torch.clamp(dilated - prev, 0.0, 1.0)

            weight = math.exp(-float(r) / tau)
            soft = torch.maximum(soft, shell * weight)
            prev = dilated

        soft = torch.clamp(soft, 0.0, 1.0)
        return soft.to(dtype=mask.dtype)
    
    def _make_boundary_condition(
        self,
        control_mask: torch.Tensor,
        batch_boundary: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Build boundary condition according to BOUNDARY_CONDITION_MODE.

        Supported modes:
            none
            soft_from_mask
            hard_from_mask
            soft_from_batch
            hard_from_batch
        """
        mode = getattr(self, "boundary_condition_mode", "soft_from_mask").lower()

        b, _, h, w = control_mask.shape

        if mode in ["none", "mask_only", "disable"]:
            return torch.zeros(
                (b, 1, h, w),
                device=control_mask.device,
                dtype=control_mask.dtype,
            )

        if mode == "hard_from_mask":
            hard = self._extract_hard_boundary_from_mask(control_mask)
            return torch.clamp(hard, 0.0, 1.0).to(dtype=control_mask.dtype)

        if mode == "soft_from_mask":
            soft = self._soft_boundary_prior_from_mask(control_mask)
            return torch.clamp(soft, 0.0, 1.0).to(dtype=control_mask.dtype)

        if mode == "hard_from_batch" and batch_boundary is not None:
            hard = _as_single_channel(batch_boundary.float())
            hard = torch.clamp(hard, 0.0, 1.0)
            return hard.to(device=control_mask.device, dtype=control_mask.dtype)

        if mode == "soft_from_batch" and batch_boundary is not None:
            soft = self._soft_boundary_prior_from_mask(batch_boundary)
            return torch.clamp(soft, 0.0, 1.0).to(dtype=control_mask.dtype)

        soft = self._soft_boundary_prior_from_mask(control_mask)
        return torch.clamp(soft, 0.0, 1.0).to(dtype=control_mask.dtype)

    def _boundary_progressive_weight(
        self,
        t: torch.Tensor,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        alpha_max = float(self.boundary_guidance_max)

        if alpha_max <= 0:
            return torch.zeros(
                (batch_size, 1, 1, 1),
                device=device,
                dtype=dtype,
            )

        if t is None:
            return torch.full(
                (batch_size, 1, 1, 1),
                alpha_max,
                device=device,
                dtype=dtype,
            )

        if t.ndim == 0:
            t = t[None].repeat(batch_size)

        t = t.to(device=device).float()
        num_timesteps = float(getattr(self, "num_timesteps", 1000))
        t_norm = torch.clamp(t / max(num_timesteps - 1.0, 1.0), 0.0, 1.0)

        if self.enable_progressive_boundary_guidance:
            start = float(self.boundary_guidance_start_ratio)
            temp = max(1e-6, float(self.boundary_guidance_temperature))
            alpha = alpha_max * torch.sigmoid((start - t_norm) / temp)
        else:
            alpha = torch.full_like(t_norm, alpha_max)

        return alpha.view(batch_size, 1, 1, 1).to(device=device, dtype=dtype)

    def _sobel_edge_response(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] > 1:
            gray = img.mean(dim=1, keepdim=True)
        else:
            gray = img

        gray = (gray + 1.0) / 2.0
        gray = torch.clamp(gray, 0.0, 1.0)

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            device=img.device,
            dtype=img.dtype,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            device=img.device,
            dtype=img.dtype,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)

        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)

        denom = edge.flatten(1).amax(dim=1).view(-1, 1, 1, 1).detach()
        edge = edge / (denom + 1e-6)
        edge = torch.clamp(edge, 0.0, 1.0)

        return edge

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        if bs is not None:
            x = x[:bs]
            c = c[:bs]

        control_mask = batch[self.control_key]

        if bs is not None:
            control_mask = control_mask[:bs]

        control_mask = control_mask.to(self.device)
        control_mask = _maybe_hwc_to_bchw(control_mask)
        control_mask = control_mask.to(memory_format=torch.contiguous_format).float()
        control_mask = torch.clamp(control_mask, 0.0, 1.0)

        batch_boundary = None

        if "boundary_prior" in batch:
            batch_boundary = batch["boundary_prior"]

            if bs is not None:
                batch_boundary = batch_boundary[:bs]

            batch_boundary = batch_boundary.to(self.device)
            batch_boundary = _maybe_hwc_to_bchw(batch_boundary)
            batch_boundary = batch_boundary.to(memory_format=torch.contiguous_format).float()
            batch_boundary = torch.clamp(batch_boundary, 0.0, 1.0)

        elif self.boundary_key is not None and self.boundary_key in batch:
            batch_boundary = batch[self.boundary_key]

            if bs is not None:
                batch_boundary = batch_boundary[:bs]

            batch_boundary = batch_boundary.to(self.device)
            batch_boundary = _maybe_hwc_to_bchw(batch_boundary)
            batch_boundary = batch_boundary.to(memory_format=torch.contiguous_format).float()
            batch_boundary = torch.clamp(batch_boundary, 0.0, 1.0)

        boundary_prior = self._make_boundary_condition(
            control_mask=control_mask,
            batch_boundary=batch_boundary,
        )

        boundary_prior = _as_single_channel(boundary_prior)
        boundary_prior = torch.clamp(boundary_prior, 0.0, 1.0)


        cond = dict(
            c_crossattn=[c],
            c_concat=[control_mask],
            c_concat_boundary_prior=[boundary_prior],
            c_concat_boundary=[boundary_prior],
        )

        if "control_image" in batch:
            control_image = batch["control_image"]

            if bs is not None:
                control_image = control_image[:bs]

            control_image = control_image.to(self.device)
            control_image = _maybe_hwc_to_bchw(control_image)
            control_image = control_image.to(memory_format=torch.contiguous_format).float()
            control_image = torch.clamp(control_image, 0.0, 1.0)

            cond["c_concat_image"] = [control_image]

        return x, cond

    # ------------------------------------------------------------
    # Apply model
    # ------------------------------------------------------------

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict), "ControlLDM expects cond to be a dict."

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], dim=1)

        c_concat = cond.get("c_concat", None)
        control_mask = None

        if c_concat is not None and len(c_concat) > 0 and c_concat[0] is not None:
            control_mask = torch.cat(c_concat, dim=1)

        boundary_prior = None

        if "c_concat_boundary_prior" in cond and cond["c_concat_boundary_prior"] is not None:
            boundary_prior = torch.cat(cond["c_concat_boundary_prior"], dim=1)
        elif "c_concat_boundary" in cond and cond["c_concat_boundary"] is not None:
            boundary_prior = torch.cat(cond["c_concat_boundary"], dim=1)

        if boundary_prior is None and control_mask is not None:
            boundary_prior = self._soft_boundary_prior_from_mask(control_mask)

        if boundary_prior is not None:
            boundary_prior = _as_single_channel(boundary_prior)
            boundary_prior = torch.clamp(boundary_prior.float(), 0.0, 1.0)

        b = x_noisy.shape[0]
        boundary_weight = self._boundary_progressive_weight(
            t=t,
            batch_size=b,
            device=x_noisy.device,
            dtype=x_noisy.dtype,
        )

        control_image = None
        if "c_concat_image" in cond and cond["c_concat_image"] is not None:
            control_image = torch.cat(cond["c_concat_image"], dim=1)

        if control_mask is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
                boundary_map=boundary_prior,
                boundary_weight=boundary_weight,
                boundary_mod_scale=self.boundary_mod_scale,
                boundary_mod_start_ratio=self.boundary_mod_start_ratio,
                enable_boundary_modulation=self.enable_boundary_modulation,
            )
            return eps

        control = self.control_model(
            x=x_noisy,
            hint=control_mask,
            timesteps=t,
            context=cond_txt,
            boundary_hint=boundary_prior,
            boundary_weight=boundary_weight,
            boundary_branch_scale=self.boundary_branch_scale,
            image_hint=control_image,
        )

        control = [
            c * scale
            for c, scale in zip(control, self.control_scales)
        ]

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=control,
            only_mid_control=self.only_mid_control,
            boundary_map=boundary_prior,
            boundary_weight=boundary_weight,
            boundary_mod_scale=self.boundary_mod_scale,
            boundary_mod_start_ratio=self.boundary_mod_start_ratio,
            enable_boundary_modulation=self.enable_boundary_modulation,
        )

        return eps

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    def p_losses(self, x_start, cond, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise

        x_noisy = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        )

        model_output = self.apply_model(
            x_noisy,
            t,
            cond,
        )

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(self.parameterization)

        loss_simple = self.get_loss(
            model_output,
            target,
            mean=False,
        ).mean([1, 2, 3])

        loss_dict.update({
            f"{prefix}/loss_simple": loss_simple.mean(),
        })

        # ------------------------------------------------------------
        # Tolerance-band boundary consistency loss
        # Weak regularization only. Not hard boundary loss.
        # ------------------------------------------------------------
        if (
            self.enable_tolerance_band_loss
            and float(self.lambda_band) > 0.0
            and isinstance(cond, dict)
        ):
            boundary_prior = None

            if "c_concat_boundary_prior" in cond and cond["c_concat_boundary_prior"] is not None:
                boundary_prior = torch.cat(cond["c_concat_boundary_prior"], dim=1)
            elif "c_concat_boundary" in cond and cond["c_concat_boundary"] is not None:
                boundary_prior = torch.cat(cond["c_concat_boundary"], dim=1)
            elif "c_concat" in cond and cond["c_concat"] is not None:
                mask = torch.cat(cond["c_concat"], dim=1)
                boundary_prior = self._soft_boundary_prior_from_mask(mask)

            if boundary_prior is not None:
                boundary_prior = _as_single_channel(boundary_prior)
                boundary_prior = torch.clamp(boundary_prior, 0.0, 1.0)

                low_noise_mask = (t <= int(self.boundary_band_t_gate)).float()

                if low_noise_mask.sum() > 0:
                    if self.parameterization == "eps":
                        pred_x0 = self.predict_start_from_noise(
                            x_t=x_noisy,
                            t=t,
                            noise=model_output,
                        )
                    elif self.parameterization == "x0":
                        pred_x0 = model_output
                    elif self.parameterization == "v":
                        pred_x0 = self.predict_start_from_z_and_v(
                            x_noisy,
                            t,
                            model_output,
                        )
                    else:
                        pred_x0 = None

                    if pred_x0 is not None:
                        pred_img = self.decode_first_stage(pred_x0)
                        edge_response = self._sobel_edge_response(pred_img)

                        prior_img = F.interpolate(
                            boundary_prior,
                            size=edge_response.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        prior_img = torch.clamp(prior_img, 0.0, 1.0)

                        per_sample_band = (
                            edge_response * (1.0 - prior_img)
                        ).mean(dim=[1, 2, 3])

                        loss_band = (
                            per_sample_band * low_noise_mask
                        ).sum() / (low_noise_mask.sum() + 1e-6)

                        loss_simple = loss_simple + float(self.lambda_band) * loss_band

                        loss_dict.update({
                            f"{prefix}/loss_tolerance_band": loss_band.detach(),
                        })

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({
                f"{prefix}/loss_gamma": loss.mean(),
                "logvar": self.logvar.data.mean(),
            })

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(
            model_output,
            target,
            mean=False,
        ).mean(dim=(1, 2, 3))

        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()

        loss_dict.update({
            f"{prefix}/loss_vlb": loss_vlb,
        })

        loss = loss + self.original_elbo_weight * loss_vlb

        loss_dict.update({
            f"{prefix}/loss": loss,
        })

        return loss, loss_dict

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    def configure_optimizers(self):
        lr = self.learning_rate

        params = []

        train_controlnet = os.environ.get("TRAIN_CONTROLNET", "1") == "1"
        train_unet_decoder = os.environ.get("TRAIN_UNET_DECODER", "0") == "1"

        if train_controlnet:
            params += list(self.control_model.parameters())

        if (not self.sd_locked) or train_unet_decoder:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

            if hasattr(self.model.diffusion_model, "boundary_modulators"):
                params += list(self.model.diffusion_model.boundary_modulators.parameters())

            if hasattr(self.model.diffusion_model, "boundary_modulator"):
                params += list(self.model.diffusion_model.boundary_modulator.parameters())

        if len(params) == 0:
            raise RuntimeError(
                "No trainable parameters selected. "
                "Check TRAIN_CONTROLNET / TRAIN_UNET_DECODER / SD_LOCKED."
            )

        opt = torch.optim.AdamW(params, lr=lr)
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

    # ------------------------------------------------------------
    # Sampling / logging
    # ------------------------------------------------------------
    def _infer_latent_shape_from_cond(self, cond):
        """
        Infer latent sampling shape from spatial condition.

        For SD1.x VAE, image/mask resolution is usually 8x latent resolution.
        Example:
            mask:   384 x 384
            latent:  48 x 48

        This avoids mismatch between:
            sampled latent shape = 64x64
            hint encoder output  = 48x48
        """
        if isinstance(cond, dict) and "c_concat" in cond and cond["c_concat"] is not None:
            hint = cond["c_concat"][0]

            if isinstance(hint, torch.Tensor) and hint.ndim == 4:
                h, w = hint.shape[-2:]

                latent_h = max(1, h // 8)
                latent_w = max(1, w // 8)

                return (
                    self.channels,
                    latent_h,
                    latent_w,
                )

        return (
            self.channels,
            self.image_size,
            self.image_size,
        )

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        **kwargs,
    ):
        if ddim:
            from ablation_cldm.ddim_hacked import DDIMSampler

            ddim_sampler = DDIMSampler(self)

            # Important:
            # Infer latent shape from mask condition.
            # For 384x384 mask, latent should be 48x48, not yaml image_size=64.
            shape = self._infer_latent_shape_from_cond(cond)

            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                **kwargs,
            )
        else:
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                **kwargs,
            )

        return samples, intermediates

    @torch.no_grad()
    def _get_empty_text_conditioning(self, batch_size: int):
        """
        Get unconditional cross-attention condition using empty prompts.

        The base LatentDiffusion.get_unconditional_conditioning() in this codebase
        is not implemented, so we use get_learned_conditioning([""] * batch_size)
        instead.
        """
        uc = self.get_learned_conditioning([""] * batch_size)

        if isinstance(uc, torch.Tensor):
            uc = uc.to(self.device)

        return uc

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=True,
        ddim_steps=50,
        ddim_eta=0.0,
        use_image_control=False,
        unconditional_guidance_scale=9.0,
        sampler="ddim",
        dpm_steps=20,
        hybrid_split=0.5,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = {}

        x, c = self.get_input(batch, self.first_stage_key, bs=N)

        N = min(x.shape[0], N)
        x = x.to(self.device)

        log["reconstruction"] = self.decode_first_stage(x)
        log["control_mask"] = _repeat_to_channels(c["c_concat"][0][:N], 3) * 2.0 - 1.0

        boundary_prior = c.get("c_concat_boundary_prior", None)

        if boundary_prior is not None:
            boundary_prior_img = boundary_prior[0][:N]
            log["control_boundary_prior"] = _repeat_to_channels(boundary_prior_img, 3) * 2.0 - 1.0

            hard_boundary = self._extract_hard_boundary_from_mask(c["c_concat"][0][:N])
            log["control_boundary_hard"] = _repeat_to_channels(hard_boundary, 3) * 2.0 - 1.0

        if use_image_control and "c_concat_image" in c:
            log["control_image"] = _repeat_to_channels(c["c_concat_image"][0][:N], 3) * 2.0 - 1.0

        if not sample:
            return log

        cond_infer = {
            "c_concat": [c["c_concat"][0][:N]],
            "c_concat_boundary_prior": [c["c_concat_boundary_prior"][0][:N]],
            "c_concat_boundary": [c["c_concat_boundary_prior"][0][:N]],
            "c_crossattn": [c["c_crossattn"][0][:N]],
        }

        if use_image_control and "c_concat_image" in c:
            cond_infer["c_concat_image"] = [c["c_concat_image"][0][:N]]

        uc_cross = self._get_empty_text_conditioning(N)


        uc_full = {
            "c_concat": [c["c_concat"][0][:N]],
            "c_concat_boundary_prior": [c["c_concat_boundary_prior"][0][:N]],
            "c_concat_boundary": [c["c_concat_boundary_prior"][0][:N]],
            "c_crossattn": [uc_cross],
        }

        if use_image_control and "c_concat_image" in c:
            uc_full["c_concat_image"] = [c["c_concat_image"][0][:N]]

        samples, _ = self.sample_log(
            cond=cond_infer,
            batch_size=N,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc_full,
        )

        x_samples = self.decode_first_stage(samples)
        log["samples"] = x_samples

        key = f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"
        log[key] = x_samples

        return log
