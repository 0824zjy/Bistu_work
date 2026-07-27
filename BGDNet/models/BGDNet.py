# /data/zjy_work/BGDNet/models/BGDNet.py
# BGCA+BEM=BGDNet

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import torchvision
from mmcv.cnn import build_activation_layer
from timm.models.layers import trunc_normal_

from models.backbone.transxnet import transxnet_xxs
from .SwinBlock import SwinTransformer


# ============================================================
# Pretrained loading utilities
# ============================================================

def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).lower() in ["1", "true", "yes", "y", "on"]


def _extract_state_dict(raw):
    if not isinstance(raw, dict):
        return raw

    for key in ["state_dict", "model", "net", "module"]:
        if key in raw and isinstance(raw[key], dict):
            return raw[key]

    return raw


def _clean_key(k: str) -> str:
    prefixes = [
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "swin.",
        "xnet.",
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                changed = True

    return k


def remap_swin_timm_to_bgdnet_keys(k: str) -> str:
    """
    Remap timm/official Swin keys to this project's SwinBlock keys.

    Common checkpoint:
        layers.0.xxx
        layers.1.xxx
        layers.2.xxx
        layers.3.xxx

    Current SwinBlock may use:
        layers1.0.xxx
        layers2.0.xxx
        layers3.0.xxx
        layers4.0.xxx
    """
    if k.startswith("layers."):
        parts = k.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            stage_idx = int(parts[1]) + 1
            return f"layers{stage_idx}.0.{parts[2]}"

    return k


def load_pretrained_with_report(
    module: nn.Module,
    path: str,
    name: str = "backbone",
    strict_zero: bool = False,
    key_remap=None,
):
    if path is None or path == "":
        msg = f"[{name} pretrained] empty path, skip loading."
        print(msg)
        if strict_zero:
            raise RuntimeError(msg)
        return 0, len(module.state_dict())

    if not os.path.isfile(path):
        msg = f"[{name} pretrained] file not found: {path}"
        print(msg)
        if strict_zero:
            raise RuntimeError(msg)
        return 0, len(module.state_dict())

    print(f"[{name} pretrained] loading from: {path}")

    raw = torch.load(path, map_location="cpu")
    raw = _extract_state_dict(raw)

    if not isinstance(raw, dict):
        msg = f"[{name} pretrained] checkpoint is not a valid state_dict."
        print(msg)
        if strict_zero:
            raise RuntimeError(msg)
        return 0, len(module.state_dict())

    cleaned = {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            continue

        nk = _clean_key(k)

        if key_remap is not None:
            nk = key_remap(nk)

        cleaned[nk] = v

    model_dict = module.state_dict()

    matched = {}
    shape_mismatch = []
    unexpected = []

    for k, v in cleaned.items():
        if k not in model_dict:
            unexpected.append(k)
            continue

        if v.shape != model_dict[k].shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(model_dict[k].shape)))
            continue

        matched[k] = v

    matched_num = len(matched)
    total_num = len(model_dict)

    print(
        f"[{name} pretrained] matched "
        f"{matched_num}/{total_num} tensors "
        f"({matched_num / max(total_num, 1):.2%})"
    )

    print(
        f"[{name} pretrained] checkpoint tensors={len(cleaned)}, "
        f"unexpected={len(unexpected)}, "
        f"shape_mismatch={len(shape_mismatch)}"
    )

    if len(shape_mismatch) > 0:
        print(f"[{name} pretrained] first 10 shape mismatches:")
        for item in shape_mismatch[:10]:
            k, ckpt_shape, model_shape = item
            print(f"  {k}: checkpoint{ckpt_shape} != model{model_shape}")

    if matched_num == 0:
        msg = f"[{name} pretrained] WARNING: no tensors were matched."
        print(msg)
        if strict_zero:
            raise RuntimeError(msg)
        return 0, total_num

    model_dict.update(matched)
    module.load_state_dict(model_dict, strict=True)

    print(f"[{name} pretrained] loaded successfully.")

    return matched_num, total_num


# =================== 已有模块：SpatialAttention、FU、DeformConv、MSDC等 ===================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * residual


class FU(nn.Module):
    def __init__(
        self,
        in_channels,
        groups,
        kernel_size=(3, 3),
        padding=1,
        stride=1,
        dilation=1,
        bias=True,
    ):
        super(FU, self).__init__()
        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=True,
        )
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x, skip):
        offsets = self.offset_net(x)
        out = self.deform_conv(skip, offsets)
        return out


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        groups,
        kernel_size=(3, 3),
        padding=1,
        stride=1,
        dilation=1,
        bias=True,
    ):
        super(DeformConv, self).__init__()
        self.offset_net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=True,
        )
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class MultiScaleDeformConv_3x3(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleDeformConv_3x3, self).__init__()
        self.sub_channel = in_channels // 4
        groups = self.sub_channel

        self.deform_conv1 = nn.Conv2d(
            self.sub_channel,
            groups,
            kernel_size=(1, 1),
        )

        self.deform_conv3 = DeformConv(
            self.sub_channel,
            groups,
            kernel_size=(3, 3),
            padding=1,
            dilation=1,
        )

        self.deform_conv5 = DeformConv(
            self.sub_channel,
            groups,
            kernel_size=(3, 3),
            padding=2,
            dilation=2,
        )

        self.deform_conv7 = DeformConv(
            self.sub_channel,
            groups,
            kernel_size=(3, 3),
            padding=3,
            dilation=3,
        )

    def forward(self, x):
        c1, c2, c3, c4 = torch.chunk(x, 4, dim=1)

        out1 = self.deform_conv1(c1)
        out3 = self.deform_conv3(c2)
        out5 = self.deform_conv5(c3)
        out7 = self.deform_conv7(c4)

        out = torch.cat([out1, out3, out5, out7], dim=1)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, 1, 1, 1) * init_value,
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            groups=x.shape[1],
        )
        return x


class MSDCDecoder_3x3_LS_up(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=dict(type="GELU"),
        drop=0,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Conv2d(
                in_features,
                hidden_features,
                kernel_size=1,
                bias=False,
            ),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )

        self.dwconv = MultiScaleDeformConv_3x3(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)

        self.fc2 = nn.Sequential(
            nn.Conv2d(
                hidden_features,
                out_features,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_features),
        )

        self.drop = nn.Dropout(drop)
        self.norm1 = nn.BatchNorm2d(in_features)
        self.up_conv = up_layer(out_features, out_features)

        self.layer_scale = (
            LayerScale(out_features, layer_scale_init_value)
            if layer_scale_init_value is not None
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = (
                m.kernel_size[0]
                * m.kernel_size[1]
                * m.out_channels
                // m.groups
            )
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_up, x_skip):
        x = torch.cat([x_up, x_skip], dim=1)
        x = self.norm1(x)

        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        x = self.up_conv(x)
        x = self.layer_scale(x) + x

        return x


class up_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_layer, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


# =================== 新增模块：BEM（边界特征提取） ===================

class BEM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

        self.compress = nn.Sequential(
            nn.Conv2d(
                2 * in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        kx = self.kx.repeat(C, 1, 1, 1)
        ky = self.ky.repeat(C, 1, 1, 1)

        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)

        feat = torch.cat([gx, gy], dim=1)
        out = self.compress(feat)

        return out


# =================== 新增模块：BGCA（边界引导交叉注意力） ===================

class BGCA(nn.Module):
    def __init__(self, dim, qkv_dim=None):
        super().__init__()

        d = qkv_dim or dim

        self.qx = nn.Conv2d(dim, d, 1, bias=False)
        self.ky = nn.Conv2d(dim, d, 1, bias=False)
        self.vy = nn.Conv2d(dim, d, 1, bias=False)

        self.qy = nn.Conv2d(dim, d, 1, bias=False)
        self.kx = nn.Conv2d(dim, d, 1, bias=False)
        self.vx = nn.Conv2d(dim, d, 1, bias=False)

        self.proj_x = nn.Conv2d(d, dim, 1, bias=False)
        self.proj_y = nn.Conv2d(d, dim, 1, bias=False)

        self.bn_x = nn.BatchNorm2d(dim)
        self.bn_y = nn.BatchNorm2d(dim)

    def _attend(self, q, k, v):
        B, C, H, W = q.shape
        scale = C ** -0.5

        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)

        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)

        v = v.view(B, C, H * W).permute(0, 2, 1)
        out = torch.bmm(attn, v)

        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out

    def forward(self, X, Y, Bx, By):
        qx = self.qx(X + Bx)
        ky = self.ky(Y + By)
        vy = self.vy(Y)

        out_x = self._attend(qx, ky, vy)
        out_x = self.proj_x(out_x)
        Xp = self.bn_x(X + out_x)

        qy = self.qy(Y + By)
        kx = self.kx(X + Bx)
        vx = self.vx(X)

        out_y = self._attend(qy, kx, vx)
        out_y = self.proj_y(out_y)
        Yp = self.bn_y(Y + out_y)

        return Xp, Yp


# =================== 新模型：BGDNet ===================

class BGDNet(nn.Module):
    """
    Standard BGDNet only.

    Swin configuration:
        embed_dim = 96
        patch_size = 4
        feature channels = [96, 192, 384, 768]

    TransXNet feature channels:
        [64, 128, 320, 512]
    """

    def __init__(self, num_classes=1):
        super().__init__()

        self.arch_name = "standard_swin96_patch4"

        # ----------------------------------------------------
        # Only allow the new standard architecture.
        # ----------------------------------------------------
        requested_variant = os.environ.get(
            "BGDNET_SWIN_VARIANT",
            "standard",
        ).strip().lower()

        allowed_variants = {
            "standard",
            "new",
            "patch4",
            "standard_swin96_patch4",
        }

        if requested_variant not in allowed_variants:
            raise ValueError(
                "This BGDNet.py only supports the new standard "
                "architecture: embed_dim=96, patch_size=4. "
                f"Received BGDNET_SWIN_VARIANT={requested_variant!r}"
            )

        strict_pretrain = _env_flag(
            "BGDNET_STRICT_PRETRAIN",
            "0",
        )

        load_backbone_pretrain = _env_flag(
            "BGDNET_LOAD_BACKBONE_PRETRAIN",
            "1",
        )

        # ====================================================
        # 1. Backbone configuration
        # ====================================================
        feature_size = 96
        patch_size = (4, 4)
        window_size = (7, 7)
        depths = (2, 2, 2, 2)
        num_heads = (3, 6, 12, 24)

        # SwinBlock returns five feature maps in this project.
        # Standard BGDNet uses the first four.
        self.swin_feature_indices = (0, 1, 2, 3)

        swin_dims = [96, 192, 384, 768]
        xnet_dims = [64, 128, 320, 512]
        fuse_dims = [64, 128, 320, 512]

        self.swin_dims = tuple(swin_dims)
        self.xnet_dims = tuple(xnet_dims)
        self.fuse_dims = tuple(fuse_dims)

        print(
            "[BGDNet Config] "
            f"arch={self.arch_name}, "
            f"embed_dim={feature_size}, "
            f"patch_size={patch_size}, "
            f"swin_dims={swin_dims}, "
            f"xnet_dims={xnet_dims}"
        )

        # ====================================================
        # 2. Create backbone modules
        #
        # Important:
        # Modules must be created before loading pretrained
        # weights.
        # ====================================================
        self.swin = SwinTransformer(
            in_chans=3,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=2,
            downsample="merging",
        )

        self.xnet = transxnet_xxs()

        # ====================================================
        # 3. Load backbone pretrained weights
        # ====================================================
        swin_pretrained_path = os.environ.get(
            "BGDNET_SWIN_PRETRAIN",
            "/data/zjy_work/BGDNet/pretrained_pth/"
            "swin_tiny_patch4_window7_224_22k.pth",
        )

        transx_pretrained_path = os.environ.get(
            "BGDNET_TRANSX_PRETRAIN",
            "/data/zjy_work/BGDNet/pretrained_pth/"
            "transxnet/transx-s.pth.tar",
        )

        if load_backbone_pretrain:
            load_pretrained_with_report(
                self.swin,
                swin_pretrained_path,
                name="Swin",
                strict_zero=strict_pretrain,
                key_remap=remap_swin_timm_to_bgdnet_keys,
            )

            load_pretrained_with_report(
                self.xnet,
                transx_pretrained_path,
                name="TransXNet",
                strict_zero=strict_pretrain,
            )
        else:
            print(
                "[BGDNet pretrained] backbone pretrained loading "
                "disabled by BGDNET_LOAD_BACKBONE_PRETRAIN=0"
            )

        # ====================================================
        # 4. Backbone channel projections
        # ====================================================
        self.swin_proj = nn.ModuleList([
            nn.Conv2d(
                in_channels=sd,
                out_channels=fd,
                kernel_size=1,
                bias=False,
            )
            for sd, fd in zip(swin_dims, fuse_dims)
        ])

        self.xnet_proj = nn.ModuleList([
            nn.Conv2d(
                in_channels=xd,
                out_channels=fd,
                kernel_size=1,
                bias=False,
            )
            for xd, fd in zip(xnet_dims, fuse_dims)
        ])

        # ====================================================
        # 5. Boundary Enhancement Modules
        # ====================================================
        self.bem_s = nn.ModuleList([
            BEM(fd) for fd in fuse_dims
        ])

        self.bem_t = nn.ModuleList([
            BEM(fd) for fd in fuse_dims
        ])

        # ====================================================
        # 6. Boundary-Guided Cross Attention
        # ====================================================
        self.bgca = nn.ModuleList([
            BGCA(fd) for fd in fuse_dims
        ])

        # ====================================================
        # 7. Decoder
        # ====================================================
        dec_dims = [fd * 2 for fd in fuse_dims]

        # dec_dims:
        # [128, 256, 640, 1024]
        self.dec_dims = tuple(dec_dims)

        self.up_last = up_layer(
            dec_dims[3],
            dec_dims[2],
        )

        self.msecoder_3 = MSDCDecoder_3x3_LS_up(
            in_features=dec_dims[2] * 2,
            hidden_features=dec_dims[2] * 2,
            out_features=dec_dims[1],
        )

        self.msecoder_2 = MSDCDecoder_3x3_LS_up(
            in_features=dec_dims[1] * 2,
            hidden_features=dec_dims[1] * 2,
            out_features=dec_dims[0],
        )

        self.msecoder_1 = MSDCDecoder_3x3_LS_up(
            in_features=dec_dims[0] * 2,
            hidden_features=dec_dims[0] * 2,
            out_features=dec_dims[0],
        )

        self.FU_3 = FU(
            dec_dims[2],
            dec_dims[2],
            kernel_size=(7, 7),
            padding=6,
            dilation=2,
        )

        self.FU_2 = FU(
            dec_dims[1],
            dec_dims[1],
            kernel_size=(5, 5),
            padding=4,
            dilation=2,
        )

        self.FU_1 = FU(
            dec_dims[0],
            dec_dims[0],
            kernel_size=(3, 3),
            padding=2,
            dilation=2,
        )

        self.FU_E = SpatialAttention()

        # ====================================================
        # 8. Output heads
        # ====================================================
        self.mask_head = nn.Conv2d(
            dec_dims[0],
            num_classes,
            kernel_size=1,
            bias=False,
        )

        self.bound_head = nn.Conv2d(
            dec_dims[0],
            1,
            kernel_size=1,
            bias=False,
        )

        # ====================================================
        # 9. Optional signed-distance task head
        #
        # Keep this disabled by default so the old two-head teacher
        # checkpoints remain strictly loadable in Steps 02/03/10.
        # Enable only for the final CHFS model with:
        #     BGDNET_ENABLE_DISTANCE_HEAD=1
        # ====================================================
        self.enable_distance_head = _env_flag(
            "BGDNET_ENABLE_DISTANCE_HEAD",
            "0",
        )
        self.distance_refine_scale = float(
            os.environ.get("BGDNET_DISTANCE_REFINE_SCALE", "0.10")
        )

        if self.enable_distance_head:
            distance_hidden = dec_dims[0]
            self.distance_feature = nn.Sequential(
                nn.Conv2d(
                    dec_dims[0],
                    distance_hidden,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, distance_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    distance_hidden,
                    distance_hidden,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, distance_hidden),
                nn.SiLU(inplace=True),
            )

            self.distance_head = nn.Conv2d(
                distance_hidden,
                1,
                kernel_size=1,
                bias=True,
            )

            # Zero-initialized cross-task refiners keep the initial behavior
            # identical to the original mask/boundary heads and are learned
            # only when useful.
            self.distance_to_mask = nn.Conv2d(
                distance_hidden,
                num_classes,
                kernel_size=1,
                bias=False,
            )
            self.distance_to_boundary = nn.Conv2d(
                distance_hidden,
                1,
                kernel_size=1,
                bias=False,
            )
            nn.init.zeros_(self.distance_to_mask.weight)
            nn.init.zeros_(self.distance_to_boundary.weight)

    @staticmethod
    def _validate_feature_list(
        features,
        expected_channels,
        backbone_name,
    ):
        """
        Validate backbone outputs and return the first four
        NCHW feature maps.
        """
        if not isinstance(features, (list, tuple)):
            raise RuntimeError(
                f"{backbone_name} output must be list or tuple, "
                f"got {type(features)}"
            )

        if len(features) < 4:
            raise RuntimeError(
                f"{backbone_name} returned {len(features)} "
                "feature maps; at least four are required."
            )

        selected = list(features[:4])

        for index, (feature, expected_channel) in enumerate(
            zip(selected, expected_channels)
        ):
            if not torch.is_tensor(feature):
                raise RuntimeError(
                    f"{backbone_name} feature[{index}] is not "
                    f"a tensor: {type(feature)}"
                )

            if feature.ndim != 4:
                raise RuntimeError(
                    f"{backbone_name} feature[{index}] must be "
                    f"NCHW 4D, got shape={tuple(feature.shape)}"
                )

            actual_channel = feature.shape[1]

            if actual_channel != expected_channel:
                raise RuntimeError(
                    f"{backbone_name} feature[{index}] channel "
                    f"mismatch: expected={expected_channel}, "
                    f"actual={actual_channel}, "
                    f"shape={tuple(feature.shape)}"
                )

        return selected

    def forward(self, x):
        input_size = x.shape[2:]

        if x.ndim != 4:
            raise ValueError(
                f"BGDNet input must be NCHW 4D, "
                f"got shape={tuple(x.shape)}"
            )

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.size(1) != 3:
            raise ValueError(
                f"BGDNet expects 1 or 3 input channels, "
                f"got {x.size(1)}"
            )

        # ====================================================
        # 1. Swin features
        # ====================================================
        sw_feats = self.swin(x)

        if not isinstance(sw_feats, (list, tuple)):
            raise RuntimeError(
                "Swin output must be list or tuple."
            )

        max_swin_index = max(self.swin_feature_indices)

        if len(sw_feats) <= max_swin_index:
            raise RuntimeError(
                f"Swin returned {len(sw_feats)} features, "
                f"but indices={self.swin_feature_indices} "
                "are required."
            )

        sw_feats = [
            sw_feats[index]
            for index in self.swin_feature_indices
        ]

        s0, s1, s2, s3 = self._validate_feature_list(
            sw_feats,
            self.swin_dims,
            "Swin",
        )

        # Project Swin channels:
        # [96,192,384,768] -> [64,128,320,512]
        s0 = self.swin_proj[0](s0)
        s1 = self.swin_proj[1](s1)
        s2 = self.swin_proj[2](s2)
        s3 = self.swin_proj[3](s3)

        # ====================================================
        # 2. TransXNet features
        # ====================================================
        tx_feats = self.xnet(x)

        t0, t1, t2, t3 = self._validate_feature_list(
            tx_feats,
            self.xnet_dims,
            "TransXNet",
        )

        # Project TransXNet channels:
        # [64,128,320,512] -> [64,128,320,512]
        t0 = self.xnet_proj[0](t0)
        t1 = self.xnet_proj[1](t1)
        t2 = self.xnet_proj[2](t2)
        t3 = self.xnet_proj[3](t3)

        # ====================================================
        # 3. Spatial alignment
        # ====================================================
        if t0.shape[2:] != s0.shape[2:]:
            t0 = F.interpolate(
                t0,
                size=s0.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if t1.shape[2:] != s1.shape[2:]:
            t1 = F.interpolate(
                t1,
                size=s1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if t2.shape[2:] != s2.shape[2:]:
            t2 = F.interpolate(
                t2,
                size=s2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        if t3.shape[2:] != s3.shape[2:]:
            t3 = F.interpolate(
                t3,
                size=s3.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # ====================================================
        # 4. Boundary enhancement
        # ====================================================
        bs0 = self.bem_s[0](s0)
        bs1 = self.bem_s[1](s1)
        bs2 = self.bem_s[2](s2)
        bs3 = self.bem_s[3](s3)

        bt0 = self.bem_t[0](t0)
        bt1 = self.bem_t[1](t1)
        bt2 = self.bem_t[2](t2)
        bt3 = self.bem_t[3](t3)

        # ====================================================
        # 5. Boundary-guided cross attention
        # ====================================================
        s0p, t0p = self.bgca[0](
            s0,
            t0,
            bs0,
            bt0,
        )

        s1p, t1p = self.bgca[1](
            s1,
            t1,
            bs1,
            bt1,
        )

        s2p, t2p = self.bgca[2](
            s2,
            t2,
            bs2,
            bt2,
        )

        s3p, t3p = self.bgca[3](
            s3,
            t3,
            bs3,
            bt3,
        )

        # ====================================================
        # 6. Dual-backbone fusion
        # ====================================================
        x0 = torch.cat([s0p, t0p], dim=1)
        x1 = torch.cat([s1p, t1p], dim=1)
        x2 = torch.cat([s2p, t2p], dim=1)
        x3 = torch.cat([s3p, t3p], dim=1)

        # ====================================================
        # 7. Decoder
        # ====================================================
        x3_up = self.up_last(x3)

        if x3_up.shape[2:] != x2.shape[2:]:
            x3_up = F.interpolate(
                x3_up,
                size=x2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        x2_aln = self.FU_3(
            x3_up,
            x2,
        )
        x2_aln = self.FU_E(x2_aln)

        d3 = self.msecoder_3(
            x3_up,
            x2_aln,
        )

        if d3.shape[2:] != x1.shape[2:]:
            d3 = F.interpolate(
                d3,
                size=x1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        x1_aln = self.FU_2(
            d3,
            x1,
        )
        x1_aln = self.FU_E(x1_aln)

        d2 = self.msecoder_2(
            d3,
            x1_aln,
        )

        if d2.shape[2:] != x0.shape[2:]:
            d2 = F.interpolate(
                d2,
                size=x0.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        x0_aln = self.FU_1(
            d2,
            x0,
        )
        x0_aln = self.FU_E(x0_aln)

        d1 = self.msecoder_1(
            d2,
            x0_aln,
        )

        # Restore original input resolution.
        d0 = F.interpolate(
            d1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # ====================================================
        # 8. Prediction heads
        # ====================================================
        mask_logits = self.mask_head(d0)
        boundary_logits = self.bound_head(d0)

        if self.enable_distance_head:
            distance_feature = self.distance_feature(d0)
            distance_logits = self.distance_head(distance_feature)

            refine_scale = float(self.distance_refine_scale)
            if refine_scale > 0.0:
                mask_logits = mask_logits + refine_scale * self.distance_to_mask(
                    distance_feature
                )
                boundary_logits = boundary_logits + refine_scale * self.distance_to_boundary(
                    distance_feature
                )

            return mask_logits, boundary_logits, distance_logits

        return mask_logits, boundary_logits

