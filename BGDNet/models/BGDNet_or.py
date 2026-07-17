#BGCA+BEM=BGDNet
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

# =================== 已有模块：SpatialAttention、FU、DeformConv、MSDC等 ===================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
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
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(FU, self).__init__()
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x, skip):
        offsets = self.offset_net(x)
        out = self.deform_conv(skip, offsets)
        return out

class DeformConv(nn.Module):
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)
    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class MultiScaleDeformConv_3x3(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleDeformConv_3x3, self).__init__()
        self.sub_channel = in_channels // 4
        groups = self.sub_channel
        self.deform_conv1 = nn.Conv2d(self.sub_channel, groups, kernel_size=(1, 1))
        self.deform_conv3 = DeformConv(self.sub_channel, groups, kernel_size=(3, 3), padding=1, dilation=1)
        self.deform_conv5 = DeformConv(self.sub_channel, groups, kernel_size=(3, 3), padding=2, dilation=2)
        self.deform_conv7 = DeformConv(self.sub_channel, groups, kernel_size=(3, 3), padding=3, dilation=3)

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
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class MSDCDecoder_3x3_LS_up(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_cfg=dict(type='GELU'), drop=0, layer_scale_init_value=1e-5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDeformConv_3x3(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.BatchNorm2d(in_features)
        self.up_conv = up_layer(out_features, out_features)
        self.layer_scale = LayerScale(out_features, layer_scale_init_value) if layer_scale_init_value is not None else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.up(x)

# =================== 新增模块：BEM（边界特征提取） ===================

class BEM(nn.Module):
    """
    Boundary Extraction Module:
    - 使用固定Sobel核, 对每个通道做深度可分离卷积(不训练)
    - 将 Gx/Gy 拼接 -> 1x1 Conv + BN + ReLU 压缩回 in_channels
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # 注册固定Sobel核为buffer
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        # 压缩层
        self.compress = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 深度分组卷积: 每通道同一Sobel核
        kx = self.kx.repeat(C, 1, 1, 1)  # [C,1,3,3]
        ky = self.ky.repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        feat = torch.cat([gx, gy], dim=1)        # [B,2C,H,W]
        out = self.compress(feat)                # [B,C,H,W]
        return out

# =================== 新增模块：BGCA（边界引导交叉注意力） ===================

class BGCA(nn.Module):
    """
    Boundary-Guided Cross Attention
    输入: X, Y 及其边界增强 Bx, By (与X/Y同shape)
    输出: X', Y'（互相增强后）
    """
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
        scale = (C ** -0.5)
        q = q.view(B, C, H*W).permute(0, 2, 1)             # [B,HW,C]
        k = k.view(B, C, H*W)                              # [B,C,HW]
        attn = torch.bmm(q, k) * scale                     # [B,HW,HW]
        attn = F.softmax(attn, dim=-1)
        v = v.view(B, C, H*W).permute(0, 2, 1)             # [B,HW,C]
        out = torch.bmm(attn, v)                           # [B,HW,C]
        out = out.permute(0, 2, 1).view(B, C, H, W)        # [B,C,H,W]
        return out

    def forward(self, X, Y, Bx, By):
        # X' <- (X+Bx) 与 (Y+By) 交互
        qx = self.qx(X + Bx)
        ky = self.ky(Y + By)
        vy = self.vy(Y)
        out_x = self._attend(qx, ky, vy)
        out_x = self.proj_x(out_x)
        Xp = self.bn_x(X + out_x)

        # Y' <- (Y+By) 与 (X+Bx) 交互
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
    双主干(Swin+TransXNet) + BEM + BGCA + MSDC + 双头输出(M掩膜, B边界)
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # ---------- Backbones ----------
        # Swin
        feature_size = 24
        patch_size = (2, 2)
        window_size = (7, 7)
        depths = (2, 2, 2, 2)
        num_heads = (3, 6, 12, 24)
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
        # 加载Swin预训练
        try:
            path = '/data/zjy_work/BGDNet/pretrained_pth/swin_tiny_patch4_window7_224_22k.pth'
            sd = torch.load(path, map_location='cpu')
            model_dict = self.swin.state_dict()
            state_dict = {k: v for k, v in sd.items() if k in model_dict}
            model_dict.update(state_dict)
            self.swin.load_state_dict(model_dict, strict=False)
        except Exception as e:
            print(f'[BGDNet] Warn: load Swin pretrained failed: {e}')

        # TransXNet（局部纹理）
        self.xnet = transxnet_xxs()
        try:
            path = '/data/zjy_work/BGDNet/pretrained_pth/transxnet/transx-s.pth.tar'
            sd = torch.load(path, map_location='cpu')
            model_dict = self.xnet.state_dict()
            state_dict = {k: v for k, v in sd.items() if k in model_dict}
            model_dict.update(state_dict)
            self.xnet.load_state_dict(model_dict, strict=False)
        except Exception as e:
            print(f'[BGDNet] Warn: load TransXNet pretrained failed: {e}')

        # Backbone 输出通道（与原工程保持一致）
        swin_dims = [48, 96, 192, 384]
        xnet_dims = [64, 128, 320, 512]
        fuse_dims = [64, 128, 320, 512]          # 统一到较强维度用于解码
        # 将Swin通道投影到 fuse_dims, 便于与xnet拼接/对齐
        self.swin_proj = nn.ModuleList([
            nn.Conv2d(sd, fd, 1, bias=False) for sd, fd in zip(swin_dims, fuse_dims)
        ])
        self.xnet_proj = nn.ModuleList([
            nn.Conv2d(xd, fd, 1, bias=False) for xd, fd in zip(xnet_dims, fuse_dims)
        ])

        # ---------- BEM ----------
        self.bem_s = nn.ModuleList([BEM(fd) for fd in fuse_dims])  # 对齐后做BEM
        self.bem_t = nn.ModuleList([BEM(fd) for fd in fuse_dims])

        # ---------- BGCA ----------
        self.bgca = nn.ModuleList([BGCA(fd) for fd in fuse_dims])

        # ---------- Decoder (MSDC) ----------
        # 拼接后通道翻倍 -> [2*fd]
        dec_dims = [fd*2 for fd in fuse_dims]
        self.up_last = up_layer(dec_dims[3], dec_dims[2])   # 4->3
        self.msecoder_3 = MSDCDecoder_3x3_LS_up(in_features=dec_dims[2]*2, hidden_features=dec_dims[2]*2,
                                                out_features=dec_dims[1]) # 输出通道 256
        self.msecoder_2 = MSDCDecoder_3x3_LS_up(in_features=dec_dims[1]*2, hidden_features=dec_dims[1]*2,
                                                out_features=dec_dims[0]) # 输出通道 128
        self.msecoder_1 = MSDCDecoder_3x3_LS_up(in_features=dec_dims[0]*2, hidden_features=dec_dims[0]*2,
                                                out_features=dec_dims[0]) # 输出通道 128

        # 对齐/引导模块
        self.FU_3 = FU(dec_dims[2], dec_dims[2], kernel_size=(7, 7), padding=6, dilation=2)
        self.FU_2 = FU(dec_dims[1], dec_dims[1], kernel_size=(5, 5), padding=4, dilation=2)
        self.FU_1 = FU(dec_dims[0], dec_dims[0], kernel_size=(3, 3), padding=2, dilation=2)
        self.FU_E = SpatialAttention()# self.up_top = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        # ---------- 双头输出 ----------352×352
        self.mask_head = nn.Conv2d(dec_dims[0], num_classes, kernel_size=1, bias=False)  # M
        self.bound_head = nn.Conv2d(dec_dims[0], 1,           kernel_size=1, bias=False)  # B

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Swin: 返回 [x, s0, s1, s2, s3]
        _, s0, s1, s2, s3 = self.swin(x)
        # TransXNet: 返回四层
        t0, t1, t2, t3 = self.xnet(x)

        # 通道对齐到 fuse_dims
        s0, s1, s2, s3 = self.swin_proj[0](s0), self.swin_proj[1](s1), self.swin_proj[2](s2), self.swin_proj[3](s3)
        t0, t1, t2, t3 = self.xnet_proj[0](t0), self.xnet_proj[1](t1), self.xnet_proj[2](t2), self.xnet_proj[3](t3)

        # BEM (边界先验增强)
        bs0, bs1, bs2, bs3 = self.bem_s[0](s0), self.bem_s[1](s1), self.bem_s[2](s2), self.bem_s[3](s3)
        bt0, bt1, bt2, bt3 = self.bem_t[0](t0), self.bem_t[1](t1), self.bem_t[2](t2), self.bem_t[3](t3)

        # BGCA（跨分支交互，显式注入边界）
        s0p, t0p = self.bgca[0](s0, t0, bs0, bt0)
        s1p, t1p = self.bgca[1](s1, t1, bs1, bt1)
        s2p, t2p = self.bgca[2](s2, t2, bs2, bt2)
        s3p, t3p = self.bgca[3](s3, t3, bs3, bt3)

        # 融合（concat）
        x0 = torch.cat([s0p, t0p], dim=1) # (B,128,176,176)
        x1 = torch.cat([s1p, t1p], dim=1) # (B,256,88,88)
        x2 = torch.cat([s2p, t2p], dim=1) # (B,640,44,44)
        x3 = torch.cat([s3p, t3p], dim=1) # (B,1024,22,22)

        # 解码
        x3_up = self.up_last(x3)                    # 4->3(B,640,44,44)
        x2_aln = self.FU_E(self.FU_3(x3_up, x2))    # 对齐+引导
        d3 = self.msecoder_3(x3_up, x2_aln)         # (B,256,88,88)

        x1_aln = self.FU_E(self.FU_2(d3, x1))        # (B,256,88,88)
        d2 = self.msecoder_2(d3, x1_aln)             # (B,128,176,176)

        x0_aln = self.FU_E(self.FU_1(d2, x0))       # (B,128,176,176)
        d1 = self.msecoder_1(d2, x0_aln)            # (B,128,352,352)
        d0 = d1                  # (B,128,352,352)
        M = self.mask_head(d0)   # logits (B,num_classes,352,352)
        B = self.bound_head(d0)  # logits (B,1,352,352)
        return M, B
