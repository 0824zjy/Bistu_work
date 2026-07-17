#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

-------------------------------------------------------------------------------
- Trains and evaluates binary lesion segmentation models on ISIC2018.
- Computes: Dice, IoU, Sensitivity (TPR), Specificity (TNR), Accuracy.

NOTE
----
Usage examples
--------------
Train UNet (default paths):
CUDA_VISIBLE_DEVICES=1 nohup python train.py  /zjy_work/MSDUNet-main/model_out/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train_test.py --mode train --model TransUNetLite > /zjy_work/MSDUNet-main/model_out/train_TransUNetLite.log 2>&1 &

Test + save masks:
CUDA_VISIBLE_DEVICES=0 python train_test.py  --mode test --model BRAUNetPP  --checkpoint /zjy_work/MSDUNet-main/model_out/BRAUNetPP_best.pt --save_dir /zjy_work/MSDUNet-main/model_out/BRAUNetPP
"""

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------------
# Minimal built-in reference segmentation models
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)

class ERUBlock(nn.Module):
    """ERU (Extended/Enhanced Residual Unit) style block: residual + squeeze-excitation."""
    def __init__(self, ch, se_ratio=16):
        super().__init__()
        self.res = ResidualBlock(ch)
        self.se_fc1 = nn.Conv2d(ch, max(1, ch//se_ratio), 1)
        self.se_fc2 = nn.Conv2d(max(1, ch//se_ratio), ch, 1)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        y = self.res(x)
        w = F.adaptive_avg_pool2d(y, 1)
        w = self.act(self.se_fc1(w))
        w = self.sig(self.se_fc2(w))
        return y * w + y

class ERUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        ch = features[0]
        for f in features:
            self.downs.append(nn.Sequential(ERUBlock(ch), nn.Conv2d(ch, f, 1) if ch!=f else nn.Identity()))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ERUBlock(features[-1]), ERUBlock(features[-1]))
        rev = list(reversed(features))
        ch = features[-1]
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, 2))
            self.ups.append(nn.Sequential(ERUBlock(f+f), nn.Conv2d(f+f, f, 1)))
            ch = f
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)
    def forward(self, x):
        x0 = self.stem(x)
        skips = []
        x = x0
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return self.head(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev = list(reversed(features))
        ch = features[-1]*2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch, f))
            ch = f
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1)
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return self.head(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        attn = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(attn)
        return x * psi

class AttentionUNet(UNet):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__(in_channels, out_channels, features)
        rev = list(reversed(features))
        self.gates = nn.ModuleList()
        for f in rev:
            self.gates.append(AttentionGate(F_g=f, F_l=f, F_int=max(1, f//2)))
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        gate_idx = 0
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]
            gated = self.gates[gate_idx](x, skip)
            gate_idx += 1
            if x.shape[-1] != gated.shape[-1] or x.shape[-2] != gated.shape[-2]:
                x = F.interpolate(x, size=gated.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([gated, x], dim=1)
            x = self.ups[idx+1](x)
        return self.head(x)

# ---- FAT-Net (Lite faithful approx) ----
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=96, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, (H,W)

class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class WindowSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*4)
    def forward(self, x):
        h = self.norm(x)
        y,_ = self.attn(h,h,h, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm(x))
        return x

class FATNetLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=64, embed_dim=96, depth=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.BatchNorm2d(base_ch*4), nn.ReLU(inplace=True),
        )
        self.pe = PatchEmbed(in_ch=in_channels, embed_dim=embed_dim, patch=4)
        self.tr = nn.ModuleList([WindowSelfAttention(embed_dim) for _ in range(depth)])
        self.fuse = nn.Conv2d(base_ch*4 + embed_dim, base_ch*4, 1)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.head = nn.Conv2d(base_ch, out_channels, 1)
    def forward(self, x):
        e = self.enc(x)
        tokens, (H,W) = self.pe(x)
        for blk in self.tr:
            tokens = blk(tokens)
        t = tokens.transpose(1,2).reshape(x.size(0), -1, H, W)
        if t.shape[-2:] != e.shape[-2:]:
            t = F.interpolate(t, size=e.shape[-2:], mode='bilinear', align_corners=False)
        f = self.fuse(torch.cat([e, t], dim=1))
        y = self.up2(f)
        y = self.up1(y)
        return self.head(y)

# ---- MedT (Lite faithful approx with axial attentions) ----
class AxialAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.qkv_h = nn.Linear(dim, dim*3)
        self.qkv_w = nn.Linear(dim, dim*3)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def _attn(self, x, qkv):
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q * self.scale
        attn = Q @ K.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn @ V
    def forward(self, x):
        x = self.norm(x)
        B, N, C = x.shape
        S = int(N**0.5)
        h = x.view(B, S, S, C)

        # H轴注意
        h_perm = h.permute(0, 2, 1, 3).contiguous().view(B*S, S, C)
        qkv = self.qkv_h(h_perm)
        y_h = self._attn(h_perm, qkv).view(B, S, S, C).permute(0, 2, 1, 3).contiguous()   # ← 加 contiguous()

        # W轴注意
        w_perm = h.contiguous().view(B*S, S, C)
        qkvw = self.qkv_w(w_perm)
        y_w = self._attn(w_perm, qkvw).view(B, S, S, C).contiguous()                      # ← 加 contiguous()

        y = y_h + y_w
        y = y.reshape(B, S*S, C)   # ← 用 reshape 替代 view（或 y.contiguous().view(B, S*S, C)）
        return self.proj(y)

class MedTLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, embed_dim=96, depth=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 7, padding=3, stride=2), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([AxialAttention(embed_dim) for _ in range(depth)])
        self.conv_local = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, 2, 2)
        self.up2 = nn.ConvTranspose2d(embed_dim//2, embed_dim//4, 2, 2)
        self.head = nn.Conv2d(embed_dim//4, out_channels, 1)
    def forward(self, x):
        z = self.stem(x)
        B, C, H, W = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = tokens + blk(tokens)
        t = tokens.transpose(1, 2).reshape(B, C, H, W)
        t = t + self.conv_local(t)
        y = self.up1(t)
        y = self.up2(y)
        return self.head(y)

# ---- nnU-Net (2D, faithful blocks: Conv + InstanceNorm + LeakyReLU, no deep supervision by default) ----
class ConvINLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class NnUNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            ConvINLReLU(in_ch, out_ch),
            ConvINLReLU(out_ch, out_ch)
        )
    def forward(self, x):
        return self.op(x)

class nnUNet2D(nn.Module):
    """
    A 2D nnU-Net-like architecture (Isensee et al.),
    using Conv+InstanceNorm+LeakyReLU blocks, 5 stages encoder/decoder, feature cap at 320.
    Deep supervision is disabled by default for compatibility with this trainer.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32, depth=5, max_features=320):
        super().__init__()
        feats = []
        f = base_features
        for d in range(depth):
            feats.append(min(max_features, f))
            f *= 2
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        for f in feats:
            self.enc_blocks.append(NnUNetBlock(ch_in, f))
            self.pools.append(nn.AvgPool2d(2))
            ch_in = f
        # Bottleneck
        bott_ch = min(max_features, feats[-1]*2)
        self.bottleneck = NnUNetBlock(feats[-1], bott_ch)
        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = bott_ch
        for f in reversed(feats):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, 2, 2))
            self.dec_blocks.append(NnUNetBlock(in_ch=f+f, out_ch=f))
            ch = f
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x):
        skips = []
        h = x
        for blk, pool in zip(self.enc_blocks, self.pools):
            h = blk(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for up, blk, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([skip, h], dim=1)
            h = blk(h)
        return self.head(h)

# ---- BATFormer (Lite: Boundary-aware dual-branch with transformer refinement) ----
class SobelEdge(nn.Module):
    """Simple learnable edge extractor initialized with Sobel-like kernels."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
            ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
            w = torch.zeros(2, in_ch, 3, 3)
            for c in range(in_ch):
                w[0, c] = kx
                w[1, c] = ky
            self.conv.weight.copy_(w)
    def forward(self, x):
        g = self.conv(x)               # B,2,H,W
        mag = torch.norm(g, dim=1, keepdim=True)  # B,1,H,W
        return mag

class TinyTransformerBlock(nn.Module):
    """Lightweight transformer block over flattened HW tokens."""
    def __init__(self, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim*mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
    def forward(self, x):  # x: B,C,H,W -> tokens inside
        B,C,H,W = x.shape
        t = x.flatten(2).transpose(1,2)          # B,HW,C
        y,_ = self.attn(self.norm1(t), self.norm1(t), self.norm1(t), need_weights=False)
        t = t + y
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1,2).reshape(B,C,H,W)

class BoundaryAttentionGate(nn.Module):
    """Use boundary map to gate decoder features."""
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(1, ch, 1, bias=False), nn.Sigmoid())
    def forward(self, feat, boundary):
        gate = self.proj(boundary)
        return feat * (1.0 + gate)  # residual gating

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BATFormerLite(nn.Module):
    """
    Region-Decoder + Boundary-Branch + Transformer refinement + Boundary Attention.
    Encoder: 简洁的3层下采样CNN；Boundary分支在浅层提边并用Transformer聚合上下文；
    融合：在解码各级用边界注意力引导。
    """
    def __init__(self, in_channels=3, out_channels=1, base_ch=64, trans_dim=128, trans_blocks=2):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(ConvBNReLU(in_channels, base_ch), ConvBNReLU(base_ch, base_ch))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ConvBNReLU(base_ch, base_ch*2), ConvBNReLU(base_ch*2, base_ch*2))
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(ConvBNReLU(base_ch*2, base_ch*4), ConvBNReLU(base_ch*4, base_ch*4))
        # Boundary branch on shallow features
        self.edge = SobelEdge(in_ch=base_ch)      # from e1 output
        self.edge_proj = nn.Conv2d(1, trans_dim, 1)
        self.trans = nn.Sequential(*[TinyTransformerBlock(trans_dim, heads=4) for _ in range(trans_blocks)])
        # Decoder with boundary attention at each stage
        self.d3_up = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.d3    = nn.Sequential(ConvBNReLU(base_ch*4, base_ch*2), ConvBNReLU(base_ch*2, base_ch*2))
        self.gate3 = BoundaryAttentionGate(base_ch*2)

        self.d2_up = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.d2    = nn.Sequential(ConvBNReLU(base_ch*2, base_ch), ConvBNReLU(base_ch, base_ch))
        self.gate2 = BoundaryAttentionGate(base_ch)

        self.head  = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)          # B,C,H,W
        e2 = self.e2(self.p1(e1))# B,2C,H/2,W/2
        e3 = self.e3(self.p2(e2))# B,4C,H/4,W/4
        # Boundary branch (on e1)
        # bmap = self.edge(e1)                         # B,1,H,W
        # btok = self.edge_proj(bmap)                  # B,T,H,W
        # btok = self.trans(btok)                      # transformer-refined boundary context
        bmap = self.edge(e1)                         # B,1,H,W
        btok = self.edge_proj(bmap)                  # B,T,H,W
        # ↓↓↓ 关键：先把特征图降到更小的空间再做 Transformer，最后再插值回来
        ds = 8   # 可改成 4/8；8 更省显存
        # 若尺寸不是 ds 的倍数，先 pad 一下，避免池化尺寸对不齐
        pad_h = (ds - btok.shape[-2] % ds) % ds
        pad_w = (ds - btok.shape[-1] % ds) % ds
        if pad_h or pad_w:
            btok = F.pad(btok, (0, pad_w, 0, pad_h))

        btok_small = F.avg_pool2d(btok, kernel_size=ds, stride=ds)      # B,T,H/ds,W/ds
        btok_small = self.trans(btok_small)                              # 在小分辨率上跑 MHA
        btok = F.interpolate(btok_small, size=bmap.shape[-2:], mode='bilinear', align_corners=False)
        # Decoder stage 3
        y  = self.d3_up(e3)                          # -> H/2
        if y.shape[-2:] != e2.shape[-2:]:
            y = F.interpolate(y, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        y  = torch.cat([e2, y], dim=1)
        y  = self.d3(y)
        # Gate with boundary (downsample bmap to this scale)
        b2 = F.interpolate(bmap, size=y.shape[-2:], mode='bilinear', align_corners=False)
        y  = self.gate3(y, b2)

        # Decoder stage 2
        y  = self.d2_up(y)                           # -> H
        if y.shape[-2:] != e1.shape[-2:]:
            y = F.interpolate(y, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        y  = torch.cat([e1, y], dim=1)
        y  = self.d2(y)
        b1 = F.interpolate(btok, size=y.shape[-2:], mode='bilinear', align_corners=False)
        # 将 transformer-refined boundary 特征的均值投影成注意力
        b1m = b1.mean(dim=1, keepdim=True)
        y   = self.gate2(y, b1m)

        return self.head(y)

# ---- HSH-UNet (Lite: Hybrid-Selective-skip U-Net with SK-style gating on skips) ----
class SKUnit(nn.Module):
    """Selective-Kernel-like gate: fuse two conv paths (3x3 & 5x5) with channel-wise selection."""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(ch, ch, 5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(ch, max(1, ch//reduction), 1)
        self.fc2 = nn.Conv2d(max(1, ch//reduction), ch*2, 1)
        self.softmax = nn.Softmax(dim=1)  # over 2 branches
    def forward(self, x):
        a = self.relu(self.bn(self.conv3(x)))
        b = self.relu(self.bn(self.conv5(x)))
        u = a + b
        s = F.adaptive_avg_pool2d(u, 1)
        z = self.fc2(F.relu(self.fc1(s)))
        a_w, b_w = torch.chunk(z, 2, dim=1)
        a_w = torch.sigmoid(a_w)
        b_w = torch.sigmoid(b_w)
        out = a * a_w + b * b_w
        return out

class HSHUNet(nn.Module):
    """
    Hybrid-Selective-skip U-Net:
    - 编解码骨架同 U-Net；
    - 在每级 skip 进入解码前做 SK 选择（模拟 HSH 的“混合/层次跳连选择”思想）；
    - 解码采用标准双卷积。
    """
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.skgates = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(nn.Sequential(
                ConvBNReLU(ch, f), ConvBNReLU(f, f)
            ))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBNReLU(features[-1], features[-1]*2),
                                        ConvBNReLU(features[-1]*2, features[-1]*2))
        rev = list(reversed(features))
        ch = features[-1]*2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, 2))
            # SK gate on skip (f channels)
            self.skgates.append(SKUnit(f))
            # decoder conv after concat
            self.ups.append(nn.Sequential(
                ConvBNReLU(f+f, f), ConvBNReLU(f, f)
            ))
            ch = f
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x):
        skips = []
        h = x
        for down in self.downs:
            h = down(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        skips = skips[::-1]
        gate_idx = 0
        for i in range(0, len(self.ups), 2):
            h = self.ups[i](h)
            skip = skips[i//2]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            skip = self.skgates[gate_idx](skip)  # selective hybrid on skip
            gate_idx += 1
            h = torch.cat([skip, h], dim=1)
            h = self.ups[i+1](h)
        return self.head(h)

# ---- Common helpers for Vision Transformers ----
class PatchEmbedOverlap(nn.Module):
    """Overlapping patch embedding (for MISSFormer-like encoders)."""
    def __init__(self, in_ch=3, embed_dim=64, k=7, s=4, p=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=k, stride=s, padding=p)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)                     # B,C,H',W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # B,HW,C
        return self.norm(x), (H, W)

class Mlp(nn.Module):
    def __init__(self, dim, hidden_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# ---- Swin-UNet (Lite, window attention + patch merging/expand, U-Net skips) ----
def window_partition(x, win):
    # x: B,C,H,W -> (B*nW, C, win, win)
    B, C, H, W = x.shape
    x = x.view(B, C, H//win, win, W//win, win).permute(0,2,4,1,3,5)  # B, nH, nW, C, win, win
    return x.reshape(-1, C, win, win)

def window_reverse(windows, win, H, W, B):
    # windows: (B*nW, C, win, win) -> B,C,H,W
    nH = H // win; nW = W // win
    x = windows.view(B, nH, nW, -1, win, win).permute(0,3,1,4,2,5).contiguous()
    return x.view(B, -1, H, W)

class WindowMSA(nn.Module):
    """Window-based MSA (简化版；不实现shift，仅基本窗口注意)"""
    def __init__(self, dim, heads=4, win=7, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.win = win
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)
    def forward(self, x, H, W):     # x: B,HW,C
        B, N, C = x.shape
        feat = x.transpose(1,2).reshape(B, C, H, W)
        # pad to multiples of win
        padH = (self.win - H % self.win) % self.win
        padW = (self.win - W % self.win) % self.win
        if padH or padW:
            feat = F.pad(feat, (0,padW,0,padH))
        Hp, Wp = feat.shape[-2:]
        # partition
        windows = window_partition(feat, self.win)           # Bw, C, win, win
        Bw = windows.shape[0]
        tokens = windows.flatten(2).transpose(1,2)           # Bw, win*win, C
        # MSA
        y,_ = self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens), need_weights=False)
        y = tokens + y
        y = y + self.drop(self.proj(self.norm(y)))
        # reverse
        y = y.transpose(1,2).reshape(Bw, C, self.win, self.win)
        feat = window_reverse(y, self.win, Hp, Wp, B)
        feat = feat[:, :, :H, :W]
        return feat

class SwinBlock(nn.Module):
    def __init__(self, dim, heads=4, win=7, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.msa = WindowMSA(dim, heads=heads, win=win)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_ratio=mlp_ratio, drop=drop)
    def forward(self, x, H, W):      # x: B,HW,C
        B,N,C = x.shape
        # window attention (in image space)
        feat = x.transpose(1,2).reshape(B, C, H, W)
        feat = self.msa(x, H, W) + feat
        # feed-forward
        tokens = feat.flatten(2).transpose(1,2)
        tokens = tokens + self.mlp(self.norm(tokens))
        return tokens

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(dim*4, dim*2)
        self.norm = nn.LayerNorm(dim*4)
    def forward(self, x, H, W):  # x: B,HW,C
        B,N,C = x.shape
        feat = x.transpose(1,2).reshape(B, C, H, W)
        # 2x downsample by concatenating 2x2 neighbors
        feat = torch.stack([
            feat[:,:,0::2,0::2],
            feat[:,:,1::2,0::2],
            feat[:,:,0::2,1::2],
            feat[:,:,1::2,1::2]
        ], dim=2)   # B,C,4,H/2,W/2
        feat = feat.permute(0,3,4,1,2).contiguous().view(B, (H//2)*(W//2), C*4)
        feat = self.reduction(self.norm(feat))
        return feat, H//2, W//2

class PatchExpand(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out*4)
        self.conv = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, H, W):  # x: B,HW,Cin
        B,N,C = x.shape
        x = self.proj(x)                        # B,HW,4*Cout
        Cout = x.shape[-1] // 4
        x = x.transpose(1,2).reshape(B, Cout*4, H, W)
        x = F.pixel_shuffle(x, 2)               # B, Cout, H*2, W*2
        x = self.act(self.bn(self.conv(x)))
        H, W = H*2, W*2
        return x, H, W

class SwinUNetLite(nn.Module):
    """
    Swin-UNet: Hierarchical Swin blocks + U-shaped skip connections.
    Stages: C->2C->4C->8C (down), then up with PatchExpand & skip concat.
    """
    def __init__(self, in_channels=3, out_channels=1, base_dim=64, heads=(2,4,8,8), depths=(2,2,2,2), win=7):
        super().__init__()
        self.patch_embed = PatchEmbedOverlap(in_ch=in_channels, embed_dim=base_dim, k=7, s=4, p=3)
        self.stage1 = nn.ModuleList([SwinBlock(base_dim, heads[0], win) for _ in range(depths[0])])
        self.down1 = PatchMerging(base_dim)

        self.stage2 = nn.ModuleList([SwinBlock(base_dim*2, heads[1], win) for _ in range(depths[1])])
        self.down2 = PatchMerging(base_dim*2)

        self.stage3 = nn.ModuleList([SwinBlock(base_dim*4, heads[2], win) for _ in range(depths[2])])
        self.down3 = PatchMerging(base_dim*4)

        self.stage4 = nn.ModuleList([SwinBlock(base_dim*8, heads[3], win) for _ in range(depths[3])])

        # decoder
        self.up3 = PatchExpand(base_dim*8, base_dim*4)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_dim*8, base_dim*4, 3, padding=1), nn.BatchNorm2d(base_dim*4), nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*4, base_dim*4, 3, padding=1), nn.BatchNorm2d(base_dim*4), nn.ReLU(inplace=True)
        )

        self.up2 = PatchExpand(base_dim*4, base_dim*2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*2, 3, padding=1), nn.BatchNorm2d(base_dim*2), nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*2, base_dim*2, 3, padding=1), nn.BatchNorm2d(base_dim*2), nn.ReLU(inplace=True)
        )

        self.up1 = PatchExpand(base_dim*2, base_dim)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, 3, padding=1), nn.BatchNorm2d(base_dim), nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1), nn.BatchNorm2d(base_dim), nn.ReLU(inplace=True)
        )

        self.head = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x):
        # stage 1
        t1, (H1,W1) = self.patch_embed(x)            # B,HW1,C1
        B = t1.shape[0]
        for blk in self.stage1: t1 = blk(t1, H1, W1)
        f1 = t1.transpose(1,2).reshape(B, -1, H1, W1)

        # stage 2
        t2, H2, W2 = self.down1(t1, H1, W1)
        for blk in self.stage2: t2 = blk(t2, H2, W2)
        f2 = t2.transpose(1,2).reshape(B, -1, H2, W2)

        # stage 3
        t3, H3, W3 = self.down2(t2, H2, W2)
        for blk in self.stage3: t3 = blk(t3, H3, W3)
        f3 = t3.transpose(1,2).reshape(B, -1, H3, W3)

        # stage 4 (bottleneck)
        t4, H4, W4 = self.down3(t3, H3, W3)
        for blk in self.stage4: t4 = blk(t4, H4, W4)
        f4 = t4.transpose(1,2).reshape(B, -1, H4, W4)

        # up 3
        y3, H3u, W3u = self.up3(t4, H4, W4)          # B,C3,H3,W3
        if y3.shape[-2:] != f3.shape[-2:]:
            y3 = F.interpolate(y3, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        y3 = torch.cat([f3, y3], dim=1)
        y3 = self.dec3(y3)

        # up 2
        y2_tokens = y3.flatten(2).transpose(1,2)     # 把feature转回tokens以复用Expand
        y2, H2u, W2u = self.up2(y2_tokens, H3, W3)
        if y2.shape[-2:] != f2.shape[-2:]:
            y2 = F.interpolate(y2, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([f2, y2], dim=1)
        y2 = self.dec2(y2)

        # up 1
        y1_tokens = y2.flatten(2).transpose(1,2)
        y1, H1u, W1u = self.up1(y1_tokens, H2, W2)
        if y1.shape[-2:] != f1.shape[-2:]:
            y1 = F.interpolate(y1, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([f1, y1], dim=1)
        y1 = self.dec1(y1)

        return self.head(y1)

# ---- MISSFormer (Lite: PVT-style encoder with Spatial-Reduction Attention + Mix-FFN, FPN decoder) ----
class SRAttention(nn.Module):
    """Spatial-Reduction Attention: 减少KV分辨率以降算力 (PVT/MISS核心思想之一)."""
    def __init__(self, dim, heads=4, sr_ratio=4, drop=0.):
        super().__init__()
        self.heads = heads
        self.sr = sr_ratio
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x, H, W):   # x: B,HW,C
        B,N,C = x.shape
        q = self.q(self.norm(x)).view(B,N,self.heads,C//self.heads)
        q = q.transpose(1,2)                          # B,h,N,Ch
        # 下采样KV的空间分辨率
        feat = x.transpose(1,2).reshape(B, C, H, W)
        if self.sr > 1:
            feat = F.avg_pool2d(feat, kernel_size=self.sr, stride=self.sr)
        Hs, Ws = feat.shape[-2:]
        kv_in = feat.flatten(2).transpose(1,2)        # B,Ns,C
        kv = self.kv(self.norm(kv_in)).view(B, -1, 2, self.heads, C//self.heads).permute(2,0,3,1,4) # 2,B,h,Ns,Ch
        k, v = kv[0], kv[1]                           # B,h,Ns,Ch
        attn = (q * self.scale) @ k.transpose(-2,-1)  # B,h,N,Ns
        attn = attn.softmax(dim=-1)
        y = attn @ v                                  # B,h,N,Ch
        y = y.transpose(1,2).reshape(B,N,C)
        y = self.drop(self.proj(y))
        return x + y

class EncoderBlockMISS(nn.Module):
    def __init__(self, dim, heads, sr_ratio, mlp_ratio=4.0, drop=0.):
        super().__init__()
        self.attn = SRAttention(dim, heads=heads, sr_ratio=sr_ratio, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_ratio=mlp_ratio, drop=drop)
    def forward(self, x, H, W):
        x = self.attn(x, H, W)
        x = x + self.mlp(self.norm(x))
        return x

class PatchMergingConv(nn.Module):
    """Conv downsample to next stage."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x, H, W):   # x: B,HW,Cin
        B,N,C = x.shape
        feat = x.transpose(1,2).reshape(B, C, H, W)
        feat = self.proj(feat)
        B,C2,H2,W2 = feat.shape
        t = feat.flatten(2).transpose(1,2)
        return self.norm(t), H2, W2

class FPNFuse(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_c):
        super().__init__()
        self.l1 = nn.Conv2d(c1, out_c, 1)
        self.l2 = nn.Conv2d(c2, out_c, 1)
        self.l3 = nn.Conv2d(c3, out_c, 1)
        self.l4 = nn.Conv2d(c4, out_c, 1)
        self.out = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, f1, f2, f3, f4):
        s = f1.shape[-2:]
        f2u = F.interpolate(f2, size=s, mode='bilinear', align_corners=False)
        f3u = F.interpolate(f3, size=s, mode='bilinear', align_corners=False)
        f4u = F.interpolate(f4, size=s, mode='bilinear', align_corners=False)
        x = torch.cat([self.l1(f1), self.l2(f2u), self.l3(f3u), self.l4(f4u)], dim=1)
        return self.out(x)

class MISSFormerLite(nn.Module):
    """
    MISSFormer-style pyramid: Overlap Patch Embedding + SRA 编码，
    多尺度特征 FPN 融合 + 上采样输出。
    """
    def __init__(self, in_channels=3, out_channels=1,
                 dims=(64, 128, 256, 512),
                 heads=(2, 4, 8, 8),
                 srs=(8, 4, 2, 1),
                 depths=(2, 2, 2, 2)):
        super().__init__()
        # Stage 1
        self.pe1 = PatchEmbedOverlap(in_ch=in_channels, embed_dim=dims[0], k=7, s=4, p=3)
        self.blk1 = nn.ModuleList([EncoderBlockMISS(dims[0], heads[0], srs[0]) for _ in range(depths[0])])
        self.ds1 = PatchMergingConv(dims[0], dims[1])
        # Stage 2
        self.blk2 = nn.ModuleList([EncoderBlockMISS(dims[1], heads[1], srs[1]) for _ in range(depths[1])])
        self.ds2 = PatchMergingConv(dims[1], dims[2])
        # Stage 3
        self.blk3 = nn.ModuleList([EncoderBlockMISS(dims[2], heads[2], srs[2]) for _ in range(depths[2])])
        self.ds3 = PatchMergingConv(dims[2], dims[3])
        # Stage 4
        self.blk4 = nn.ModuleList([EncoderBlockMISS(dims[3], heads[3], srs[3]) for _ in range(depths[3])])

        # Convert tokens back to feature maps for FPN
        self.to2d1 = lambda t,H,W: t.transpose(1,2).reshape(t.size(0), -1, H, W)
        self.to2d  = self.to2d1

        # FPN-like fuse
        self.fuse = FPNFuse(dims[0], dims[1], dims[2], dims[3], out_c=dims[0])

        # Up head to full scale (x4 because initial stride=4)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0]//2, 2, 2),
            nn.BatchNorm2d(dims[0]//2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dims[0]//2, dims[0]//4, 2, 2),
            nn.BatchNorm2d(dims[0]//4), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dims[0]//4, out_channels, 1)

    def forward(self, x):
        # Stage 1
        t1, (H1,W1) = self.pe1(x)
        for blk in self.blk1: t1 = blk(t1, H1, W1)
        f1 = self.to2d1(t1, H1, W1)
        # Stage 2
        t2, H2, W2 = self.ds1(t1, H1, W1)
        for blk in self.blk2: t2 = blk(t2, H2, W2)
        f2 = self.to2d(t2, H2, W2)
        # Stage 3
        t3, H3, W3 = self.ds2(t2, H2, W2)
        for blk in self.blk3: t3 = blk(t3, H3, W3)
        f3 = self.to2d(t3, H3, W3)
        # Stage 4
        t4, H4, W4 = self.ds3(t3, H3, W3)
        for blk in self.blk4: t4 = blk(t4, H4, W4)
        f4 = self.to2d(t4, H4, W4)

        # FPN融合（上采样到 stage1 尺度后拼接）
        fused = self.fuse(f1, f2, f3, f4)
        y = self.up(fused)
        return self.head(y)

# ---- Shared small utils (reuse if needed) ----
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act=='relu' else nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch//r), 1)
        self.fc2 = nn.Conv2d(max(1, ch//r), ch, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class TinyMHSA(nn.Module):
    """简化 MHSA：tokens(B,N,C) 上的多头注意 + FFN。"""
    def __init__(self, dim, heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=drop)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim*mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
    def forward(self, t):  # B,N,C
        y,_ = self.attn(self.n(t), self.n(t), self.n(t), need_weights=False)
        t = t + y
        t = t + self.mlp(self.n2(t))
        return t

def fmap_to_tokens(x):  # B,C,H,W -> B,HW,C
    B,C,H,W = x.shape
    return x.flatten(2).transpose(1,2), (H,W)

def tokens_to_fmap(t, shape):  # B,HW,C -> B,C,H,W
    B,N,C = t.shape
    H,W = shape
    return t.transpose(1,2).reshape(B, C, H, W)

# ======================================================================
# BRAU-Net++ (Lite): U形混合 CNN-Transformer。各stage: Conv堆叠 + MHSA 融合。
# ======================================================================

class BRAUBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.conv = nn.Sequential(ConvBNAct(ch, ch), ConvBNAct(ch, ch))
        self.se   = SE(ch)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.mhsa = TinyMHSA(ch, heads=heads)
        # ↓ 注意力前的空间下采样倍率（4 或 8 均可；8 更省显存）
        self.ds   = 8

    def forward(self, x):
        # 基础卷积 + SE
        h = self.se(self.conv(x))  # B,C,H,W

        # ↓ 在更小的空间分辨率上做 MHSA，避免 N^2 爆炸
        if self.ds > 1:
            pad_h = (self.ds - h.shape[-2] % self.ds) % self.ds
            pad_w = (self.ds - h.shape[-1] % self.ds) % self.ds
            if pad_h or pad_w:
                h_pad = F.pad(h, (0, pad_w, 0, pad_h))
            else:
                h_pad = h
            h_small = F.avg_pool2d(h_pad, kernel_size=self.ds, stride=self.ds)  # B,C,H/ds,W/ds
        else:
            h_small = h

        # 在小分辨率上做 MHSA
        t, hw = fmap_to_tokens(h_small)      # B, (H/ds * W/ds), C
        t = self.mhsa(t)
        h_att_small = tokens_to_fmap(t, hw)  # B, C, H/ds, W/ds

        # 插值回原尺度
        h_att = F.interpolate(h_att_small, size=h.shape[-2:], mode='bilinear', align_corners=False)

        # 残差融合 + 投影
        return self.proj(h + h_att)

class BRAUNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64, heads=(2,4,8,8)):
        super().__init__()
        C1, C2, C3, C4 = base, base*2, base*4, base*8
        # Encoder
        self.e1 = nn.Sequential(ConvBNAct(in_channels, C1), BRAUBlock(C1, heads[0]))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ConvBNAct(C1, C2, s=2, p=1), BRAUBlock(C2, heads[1]))
        self.e3 = nn.Sequential(ConvBNAct(C2, C3, s=2, p=1), BRAUBlock(C3, heads[2]))
        self.e4 = nn.Sequential(ConvBNAct(C3, C4, s=2, p=1), BRAUBlock(C4, heads[3]))
        # Bottleneck
        self.bott = BRAUBlock(C4, heads[3])
        # Decoder
        self.up3 = nn.ConvTranspose2d(C4, C3, 2, 2)
        self.d3  = nn.Sequential(ConvBNAct(C3 + C3, C3), BRAUBlock(C3, heads[2]))
        self.up2 = nn.ConvTranspose2d(C3, C2, 2, 2)
        self.d2  = nn.Sequential(ConvBNAct(C2 + C2, C2), BRAUBlock(C2, heads[1]))
        self.up1 = nn.ConvTranspose2d(C2, C1, 2, 2)
        self.d1  = nn.Sequential(ConvBNAct(C1 + C1, C1), BRAUBlock(C1, heads[0]))
        self.head = nn.Conv2d(C1, out_channels, 1)

    def forward(self, x):
        # 编码
        e1 = self.e1(x)              # H
        e2 = self.e2(self.p1(e1))    # H/4  （先 MaxPool /2，再 Conv s=2 约等 /2）
        e3 = self.e3(e2)             # H/8
        e4 = self.e4(e3)             # H/16
        h  = self.bott(e4)           # H/16

        # 解码 3：H/16 → H/8
        y3 = self.up3(h)
        if y3.shape[-2:] != e3.shape[-2:]:
            y3 = F.interpolate(y3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        y3 = torch.cat([e3, y3], dim=1)
        y3 = self.d3(y3)

        # 解码 2：H/8 → H/4
        y2 = self.up2(y3)
        if y2.shape[-2:] != e2.shape[-2:]:
            y2 = F.interpolate(y2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([e2, y2], dim=1)
        y2 = self.d2(y2)

        # 解码 1：H/4 → H/2，然后**对齐到 e1 的 H×W** 再拼接
        y1 = self.up1(y2)  # -> H/2
        if y1.shape[-2:] != e1.shape[-2:]:
            y1 = F.interpolate(y1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([e1, y1], dim=1)
        y1 = self.d1(y1)

        return self.head(y1)

# ======================================================================
# HiFormer (Lite): CNN分支 + Swin样式分支；在跳连处用 DLF 双层融合。
# ======================================================================
class DLF(nn.Module):
    """Double-Level Fusion: 通道级+空间级融合（简化）：1x1对齐 + 逐元素/拼接 -> 3x3整合"""
    def __init__(self, c_cnn, c_tr, out_c):
        super().__init__()
        self.cc = nn.Conv2d(c_cnn, out_c, 1)
        self.ct = nn.Conv2d(c_tr,  out_c, 1)
        self.mix = nn.Sequential(ConvBNAct(out_c*2, out_c), ConvBNAct(out_c, out_c))
    def forward(self, f_cnn, f_tr):
        # 假设两个特征已在同一分辨率
        c1 = self.cc(f_cnn); c2 = self.ct(f_tr)
        return self.mix(torch.cat([c1, c2], 1))

# ---------------------------------------------------------
# TinySwinStage：加入空间降采样注意力（ds=8）以避免 N^2 显存爆炸
# ---------------------------------------------------------
class TinySwinStage(nn.Module):
    """轻量 'Swin 风格' stage：把 fmap 降采样后转 tokens 做 MHSA，再插回原尺度"""
    def __init__(self, c, depth=2, heads=4, win=7, ds=8):
        super().__init__()
        self.blocks = nn.ModuleList([TinyMHSA(c, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(c)
        # ↓ 注意力前的空间降采样倍率（4/8；8 更省显存，4 稍更精细）
        self.ds = ds

    def forward(self, x):  # x: B,C,H,W
        h = x
        if self.ds > 1:
            pad_h = (self.ds - h.shape[-2] % self.ds) % self.ds
            pad_w = (self.ds - h.shape[-1] % self.ds) % self.ds
            if pad_h or pad_w:
                h_pad = F.pad(h, (0, pad_w, 0, pad_h))
            else:
                h_pad = h
            h_small = F.avg_pool2d(h_pad, kernel_size=self.ds, stride=self.ds)  # B,C,H/ds,W/ds
        else:
            h_small = h

        t, hw = fmap_to_tokens(h_small)  # B,(H/ds·W/ds),C
        for blk in self.blocks:
            t = blk(t)                   # MHSA 在小分辨率上执行
        y_small = tokens_to_fmap(self.norm(t), hw)  # B,C,H/ds,W/ds
        y = F.interpolate(y_small, size=h.shape[-2:], mode='bilinear', align_corners=False)
        return y

# ---------------------------------------------------------
# HiFormerLite：在每次拼接前做尺寸对齐，避免尺寸不匹配
# ---------------------------------------------------------
class HiFormerLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64, win=7):
        super().__init__()
        C1,C2,C3,C4 = base, base*2, base*4, base*8
        # CNN encoder
        self.c1 = nn.Sequential(ConvBNAct(in_channels, C1), ConvBNAct(C1, C1))
        self.p1 = nn.MaxPool2d(2)
        self.c2 = nn.Sequential(ConvBNAct(C1, C2, s=2, p=1), ConvBNAct(C2, C2))
        self.c3 = nn.Sequential(ConvBNAct(C2, C3, s=2, p=1), ConvBNAct(C3, C3))
        self.c4 = nn.Sequential(ConvBNAct(C3, C4, s=2, p=1), ConvBNAct(C4, C4))
        # Transformer-like encoder (降采样注意力版)
        self.t1 = TinySwinStage(C1, depth=2, heads=2, win=win, ds=8)
        self.t2 = TinySwinStage(C2, depth=2, heads=4, win=win, ds=8)
        self.t3 = TinySwinStage(C3, depth=2, heads=8, win=win, ds=8)
        self.t4 = TinySwinStage(C4, depth=2, heads=8, win=win, ds=8)
        # DLF fusion at skips
        self.f1 = DLF(C1, C1, C1)
        self.f2 = DLF(C2, C2, C2)
        self.f3 = DLF(C3, C3, C3)
        # decoder
        self.up3 = nn.ConvTranspose2d(C4, C3, 2, 2)
        self.d3  = nn.Sequential(ConvBNAct(C3+C3, C3), ConvBNAct(C3, C3))
        self.up2 = nn.ConvTranspose2d(C3, C2, 2, 2)
        self.d2  = nn.Sequential(ConvBNAct(C2+C2, C2), ConvBNAct(C2, C2))
        self.up1 = nn.ConvTranspose2d(C2, C1, 2, 2)
        self.d1  = nn.Sequential(ConvBNAct(C1+C1, C1), ConvBNAct(C1, C1))
        self.head = nn.Conv2d(C1, out_channels, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        # transformer-like 分支（同尺度）
        t1 = self.t1(c1)
        t2 = self.t2(c2)
        t3 = self.t3(c3)
        _  = self.t4(c4)  # 仅用于全局语义（本 lite 版未直接用，可按需扩展）

        # DLF 融合后的 skip
        s3 = self.f3(c3, t3)

        # 解码 3：H/16→H/8
        y3 = self.up3(c4)
        if y3.shape[-2:] != s3.shape[-2:]:
            y3 = F.interpolate(y3, size=s3.shape[-2:], mode='bilinear', align_corners=False)
        y3 = torch.cat([s3, y3], 1)
        y3 = self.d3(y3)

        # 解码 2：H/8→H/4
        s2 = self.f2(c2, t2)
        y2 = self.up2(y3)
        if y2.shape[-2:] != s2.shape[-2:]:
            y2 = F.interpolate(y2, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([s2, y2], 1)
        y2 = self.d2(y2)

        # 解码 1：H/4→H/2，并对齐到 c1 的 H×W
        s1 = self.f1(c1, t1)
        y1 = self.up1(y2)
        if y1.shape[-2:] != s1.shape[-2:]:
            y1 = F.interpolate(y1, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([s1, y1], 1)
        y1 = self.d1(y1)

        return self.head(y1)

# ======================================================================
# H2Former (Lite): 层次混合块（CNN + MSCA + Token MHSA），金字塔式编码与U型解码。
# ======================================================================
class MSCA(nn.Module):
    """Multi-Scale Channel Attention（简化）：3x3+5x5分支 + SE"""
    def __init__(self, ch):
        super().__init__()
        self.b3 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.b5 = nn.Conv2d(ch, ch, 5, padding=2, groups=ch)
        self.p  = nn.Conv2d(ch*2, ch, 1)
        self.se = SE(ch, r=8)
    def forward(self, x):
        a = self.b3(x); b = self.b5(x)
        u = self.p(torch.cat([a,b],1))
        return self.se(F.relu(u, inplace=True))

# ======================================================================
# H2Block：加入空间降采样注意力（ds=8），从根上抑制 MHSA 的 N^2 显存爆炸
# ======================================================================
class H2Block(nn.Module):
    """Hybrid block: CNN conv -> MSCA -> token MHSA (on downsampled fmap) -> fuse"""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.conv = nn.Sequential(ConvBNAct(ch, ch), ConvBNAct(ch, ch))
        self.msca = MSCA(ch)
        self.mhsa = TinyMHSA(ch, heads=heads)
        self.prj  = nn.Conv2d(ch, ch, 1)
        # ↓ 在做 MHSA 前的空间降采样倍率（4 或 8；8 更省显存）
        self.ds   = 8

    def forward(self, x):
        # 局部表征
        h = self.conv(x)
        h = self.msca(h)                # B,C,H,W

        # ↓ 在更小分辨率上执行 MHSA，避免 N^2 爆炸
        if self.ds > 1:
            pad_h = (self.ds - h.shape[-2] % self.ds) % self.ds
            pad_w = (self.ds - h.shape[-1] % self.ds) % self.ds
            if pad_h or pad_w:
                h_pad = F.pad(h, (0, pad_w, 0, pad_h))
            else:
                h_pad = h
            h_small = F.avg_pool2d(h_pad, kernel_size=self.ds, stride=self.ds)   # B,C,H/ds,W/ds
        else:
            h_small = h

        t, hw = fmap_to_tokens(h_small)  # B,(H/ds·W/ds),C
        t = self.mhsa(t)                 # MHSA 在小分辨率上
        h2_small = tokens_to_fmap(t, hw) # B,C,H/ds,W/ds

        # 回到原尺度并融合
        h2 = F.interpolate(h2_small, size=h.shape[-2:], mode='bilinear', align_corners=False)
        return self.prj(h + h2)

# ======================================================================
# H2FormerLite.forward：在每次拼接前做尺寸对齐，避免 256/128 类尺寸不匹配
# ======================================================================
class H2FormerLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64, heads=(2,4,8,8)):
        super().__init__()
        C1,C2,C3,C4 = base, base*2, base*4, base*8
        self.e1 = nn.Sequential(ConvBNAct(in_channels, C1), H2Block(C1, heads[0]))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ConvBNAct(C1, C2, s=2, p=1), H2Block(C2, heads[1]))
        self.e3 = nn.Sequential(ConvBNAct(C2, C3, s=2, p=1), H2Block(C3, heads[2]))
        self.e4 = nn.Sequential(ConvBNAct(C3, C4, s=2, p=1), H2Block(C4, heads[3]))
        self.bott = H2Block(C4, heads[3])

        self.up3 = nn.ConvTranspose2d(C4, C3, 2, 2)
        self.d3  = nn.Sequential(ConvBNAct(C3+C3, C3), H2Block(C3, heads[2]))
        self.up2 = nn.ConvTranspose2d(C3, C2, 2, 2)
        self.d2  = nn.Sequential(ConvBNAct(C2+C2, C2), H2Block(C2, heads[1]))
        self.up1 = nn.ConvTranspose2d(C2, C1, 2, 2)
        self.d1  = nn.Sequential(ConvBNAct(C1+C1, C1), H2Block(C1, heads[0]))
        self.head = nn.Conv2d(C1, out_channels, 1)

    def forward(self, x):
        e1 = self.e1(x)                # H
        e2 = self.e2(self.p1(e1))      # H/4（先 /2 再 /2）
        e3 = self.e3(e2)               # H/8
        e4 = self.e4(e3)               # H/16
        h  = self.bott(e4)             # H/16

        # stage 3: H/16 → H/8
        y3 = self.up3(h)
        if y3.shape[-2:] != e3.shape[-2:]:
            y3 = F.interpolate(y3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        y3 = torch.cat([e3, y3], dim=1)
        y3 = self.d3(y3)

        # stage 2: H/8 → H/4
        y2 = self.up2(y3)
        if y2.shape[-2:] != e2.shape[-2:]:
            y2 = F.interpolate(y2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([e2, y2], dim=1)
        y2 = self.d2(y2)

        # stage 1: H/4 → H/2，并对齐到 e1 的 H×W
        y1 = self.up1(y2)  # -> H/2
        if y1.shape[-2:] != e1.shape[-2:]:
            y1 = F.interpolate(y1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([e1, y1], dim=1)
        y1 = self.d1(y1)

        return self.head(y1)
# ======================================================================
# TransUNet (Lite): CNN编码 -> ViT编码器 -> U型解码 + 跳连。
# ======================================================================
class ViTEncoder(nn.Module):
    def __init__(self, in_dim, depth=6, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([TinyMHSA(in_dim, heads=heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(in_dim)
    def forward(self, t):  # B,N,C
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)

class TransUNetLite(nn.Module):
    """
    CNN backbone 提供多尺度特征；将中高层特征打包为 tokens 送入 ViT（global context），
    解码阶段与浅层CNN跳连融合。
    """
    def __init__(self, in_channels=3, out_channels=1, base=64, vit_depth=8, vit_heads=8):
        super().__init__()
        C1,C2,C3 = base, base*2, base*4
        # CNN encoder
        self.c1 = nn.Sequential(ConvBNAct(in_channels, C1), ConvBNAct(C1, C1))          # H
        self.p1 = nn.MaxPool2d(2)
        self.c2 = nn.Sequential(ConvBNAct(C1, C2, s=2, p=1), ConvBNAct(C2, C2))         # H/2
        self.c3 = nn.Sequential(ConvBNAct(C2, C3, s=2, p=1), ConvBNAct(C3, C3))         # H/4
        # ViT encoder on stage-3 feature
        self.to_tokens = lambda f: fmap_to_tokens(f)[0]
        self.vit = ViTEncoder(C3, depth=vit_depth, heads=vit_heads)
        # Decoder
        self.up2 = nn.ConvTranspose2d(C3, C2, 2, 2)
        self.d2  = nn.Sequential(ConvBNAct(C2+C2, C2), ConvBNAct(C2, C2))
        self.up1 = nn.ConvTranspose2d(C2, C1, 2, 2)
        self.d1  = nn.Sequential(ConvBNAct(C1+C1, C1), ConvBNAct(C1, C1))
        self.head = nn.Conv2d(C1, out_channels, 1)
    def forward(self, x):
        f1 = self.c1(x)                   # H
        f2 = self.c2(self.p1(f1))         # H/4 （先 /2 再 /2）
        f3 = self.c3(f2)                  # H/8

        t3, hw = fmap_to_tokens(f3)
        t3 = self.vit(t3)
        f3g = tokens_to_fmap(t3, hw)      # H/8

        y2 = self.up2(f3g)                # -> H/4
        if y2.shape[-2:] != f2.shape[-2:]:
            y2 = F.interpolate(y2, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([f2, y2], 1)
        y2 = self.d2(y2)

        y1 = self.up1(y2)                 # -> H/2
        # 🔧 关键修复：对齐到 f1 的 H×W
        if y1.shape[-2:] != f1.shape[-2:]:
            y1 = F.interpolate(y1, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([f1, y1], 1)
        y1 = self.d1(y1)

        return self.head(y1)

# ===========================
# Helpers (共用的小模块)
# ===========================
class ConvBNGELU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = ConvBNGELU(in_ch, in_ch, k, s, p, g=in_ch)
        self.pw = ConvBNGELU(in_ch, out_ch, 1, 1, 0)
    def forward(self, x): return self.pw(self.dw(x))

# ===========================
# 1) TransFuse (Lite)
# ===========================
class BiFusionBlock(nn.Module):
    """简化 BiFusion：通道对齐 + 跨分支注意 + 融合"""
    def __init__(self, c_cnn, c_tr, out_c):
        super().__init__()
        self.cnn_proj = nn.Conv2d(c_cnn, out_c, 1)
        self.tr_proj  = nn.Conv2d(c_tr,  out_c, 1)
        self.gate_c = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_c, out_c, 1), nn.Sigmoid())
        self.mix = nn.Sequential(ConvBNGELU(out_c*2, out_c), ConvBNGELU(out_c, out_c))
    def forward(self, f_cnn, f_tr):
        f1 = self.cnn_proj(f_cnn)
        f2 = self.tr_proj(f_tr)
        g  = self.gate_c(f2)                 # 用 transformer 的全局语义引导
        f1 = f1 * (1. + g)
        return self.mix(torch.cat([f1, f2], 1))

# ===========================
# TinyViTStage（显存友好版）
# 关键：在做 MHSA 前，先把 fmap 按 ds=8 降采样；MHSA 完成后再插值回原尺寸。
# ===========================
class TinyViTStage(nn.Module):
    """轻量 ViT 编码（在降采样后的空间上做 MHSA，避免 N^2 爆炸）"""
    def __init__(self, dim, depth=2, heads=4, ds=8):
        super().__init__()
        self.blocks = nn.ModuleList([TinyMHSA(dim, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # 注意力前的空间降采样倍率（4/8；8 更省显存，4 更精细）
        self.ds = ds

    def forward(self, fmap):
        # fmap: B, C, H, W
        h = fmap
        if self.ds > 1:
            pad_h = (self.ds - h.shape[-2] % self.ds) % self.ds
            pad_w = (self.ds - h.shape[-1] % self.ds) % self.ds
            if pad_h or pad_w:
                h_pad = F.pad(h, (0, pad_w, 0, pad_h))
            else:
                h_pad = h
            h_small = F.avg_pool2d(h_pad, kernel_size=self.ds, stride=self.ds)  # B,C,H/ds,W/ds
        else:
            h_small = h

        # 在小分辨率上做 MHSA
        t, hw = fmap_to_tokens(h_small)   # B, (H/ds * W/ds), C
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        y_small = tokens_to_fmap(t, hw)   # B, C, H/ds, W/ds

        # 插值回原尺度
        y = F.interpolate(y_small, size=h.shape[-2:], mode='bilinear', align_corners=False)
        return y

class TransFuseLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()
        C1,C2,C3 = base, base*2, base*4
        # CNN 分支
        self.c1 = nn.Sequential(ConvBNGELU(in_channels, C1), ConvBNGELU(C1, C1))
        self.p1 = nn.MaxPool2d(2)
        self.c2 = nn.Sequential(ConvBNGELU(C1, C2, s=2, p=1), ConvBNGELU(C2, C2))
        self.c3 = nn.Sequential(ConvBNGELU(C2, C3, s=2, p=1), ConvBNGELU(C3, C3))
        # Transformer 分支（直接对各尺度 fmap 做 tokens 编码）
        self.t1 = TinyViTStage(C1, depth=2, heads=2)
        self.t2 = TinyViTStage(C2, depth=2, heads=4)
        self.t3 = TinyViTStage(C3, depth=4, heads=8)
        # 多级 Bi-Fusion
        self.f3 = BiFusionBlock(C3, C3, C3)
        self.f2 = BiFusionBlock(C2, C2, C2)
        self.f1 = BiFusionBlock(C1, C1, C1)
        # 解码
        self.up2 = nn.ConvTranspose2d(C3, C2, 2, 2)
        self.d2  = nn.Sequential(ConvBNGELU(C2+C2, C2), ConvBNGELU(C2, C2))
        self.up1 = nn.ConvTranspose2d(C2, C1, 2, 2)
        self.d1  = nn.Sequential(ConvBNGELU(C1+C1, C1), ConvBNGELU(C1, C1))
        self.head = nn.Conv2d(C1, out_channels, 1)
    def forward(self, x):
        c1 = self.c1(x)                   # H
        c2 = self.c2(self.p1(c1))         # H/2
        c3 = self.c3(c2)                  # H/4
        t1 = self.t1(c1); t2 = self.t2(c2); t3 = self.t3(c3)  # 同尺度 token 编码结果
        s3 = self.f3(c3, t3)
        y2 = self.up2(s3)
        if y2.shape[-2:] != c2.shape[-2:]: y2 = F.interpolate(y2, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        s2 = self.f2(c2, t2)
        y2 = self.d2(torch.cat([s2, y2], 1))
        y1 = self.up1(y2)
        if y1.shape[-2:] != c1.shape[-2:]: y1 = F.interpolate(y1, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        s1 = self.f1(c1, t1)
        y1 = self.d1(torch.cat([s1, y1], 1))
        return self.head(y1)

# ===========================
# 2) PVT_GCASCAD (Lite)
# ===========================
class GlobalContext(nn.Module):
    """全局上下文：GAP 后通道重标定"""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(1,c//r), 1); self.fc2 = nn.Conv2d(max(1,c//r), c, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w)); w = torch.sigmoid(self.fc2(w))
        return x * (1. + w)

class CascadeDecoder(nn.Module):
    """级联解码：自顶向下，每一级做上采样与融合，同时注入全局上下文校正。"""
    def __init__(self, c1, c2, c3, c4, out_c):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(c4, c3, 2, 2); self.f4 = nn.Sequential(ConvBNGELU(c3+c3, c3), ConvBNGELU(c3, c3))
        self.up3 = nn.ConvTranspose2d(c3, c2, 2, 2); self.f3 = nn.Sequential(ConvBNGELU(c2+c2, c2), ConvBNGELU(c2, c2))
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, 2); self.f2 = nn.Sequential(ConvBNGELU(c1+c1, c1), ConvBNGELU(c1, c1))
        self.out = nn.Conv2d(c1, out_c, 1)
        self.g3 = GlobalContext(c3); self.g2 = GlobalContext(c2); self.g1 = GlobalContext(c1)
    def forward(self, f1, f2, f3, f4):
        y = self.up4(f4); 
        if y.shape[-2:] != f3.shape[-2:]: y = F.interpolate(y, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        y = self.f4(torch.cat([self.g3(f3), y], 1))
        y = self.up3(y); 
        if y.shape[-2:] != f2.shape[-2:]: y = F.interpolate(y, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        y = self.f3(torch.cat([self.g2(f2), y], 1))
        y = self.up2(y);
        if y.shape[-2:] != f1.shape[-2:]: y = F.interpolate(y, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        y = self.f2(torch.cat([self.g1(f1), y], 1))
        return self.out(y)

class PVT_GCASCAD_Lite(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dims=(64,128,256,512), heads=(2,4,8,8), srs=(8,4,2,1), depths=(2,2,2,2)):
        super().__init__()
        # 编码：Overlap Patch + SRA（同 MISSFormerLite）
        self.pe1 = PatchEmbedOverlap(in_ch=in_channels, embed_dim=dims[0], k=7, s=4, p=3)
        self.blk1 = nn.ModuleList([EncoderBlockMISS(dims[0], heads[0], srs[0]) for _ in range(depths[0])])
        self.ds1 = PatchMergingConv(dims[0], dims[1])
        self.blk2 = nn.ModuleList([EncoderBlockMISS(dims[1], heads[1], srs[1]) for _ in range(depths[1])])
        self.ds2 = PatchMergingConv(dims[1], dims[2])
        self.blk3 = nn.ModuleList([EncoderBlockMISS(dims[2], heads[2], srs[2]) for _ in range(depths[2])])
        self.ds3 = PatchMergingConv(dims[2], dims[3])
        self.blk4 = nn.ModuleList([EncoderBlockMISS(dims[3], heads[3], srs[3]) for _ in range(depths[3])])

        self.to2d = lambda t,H,W: t.transpose(1,2).reshape(t.size(0), -1, H, W)
        self.dec = CascadeDecoder(dims[0], dims[1], dims[2], dims[3], out_channels)

    def forward(self, x):
        t1,(H1,W1)= self.pe1(x)
        for b in self.blk1: t1 = b(t1, H1, W1); f1 = self.to2d(t1, H1, W1)
        t2,H2,W2 = self.ds1(t1, H1, W1)
        for b in self.blk2: t2 = b(t2, H2, W2); f2 = self.to2d(t2, H2, W2)
        t3,H3,W3 = self.ds2(t2, H2, W2)
        for b in self.blk3: t3 = b(t3, H3, W3); f3 = self.to2d(t3, H3, W3)
        t4,H4,W4 = self.ds3(t3, H3, W3)
        for b in self.blk4: t4 = b(t4, H4, W4); f4 = self.to2d(t4, H4, W4)
        return self.dec(f1, f2, f3, f4)

# ===========================
# 3) D-LKA (Lite)
# ===========================
class DLKA(nn.Module):
    """Dilated Large Kernel Attention：分解为 (dw k×1 + 1×k) + dilated dw，再 1×1 聚合"""
    def __init__(self, ch, k=21, d=3):
        super().__init__()
        p = (k-1)//2
        self.dw1 = nn.Conv2d(ch, ch, (k,1), padding=(p,0), groups=ch, bias=False)
        self.dw2 = nn.Conv2d(ch, ch, (1,k), padding=(0,p), groups=ch, bias=False)
        self.dwd = nn.Conv2d(ch, ch, 3, padding=d, dilation=d, groups=ch, bias=False)
        self.pw  = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(ch)
        self.act = nn.GELU()
    def forward(self, x):
        u = self.dw1(x); u = self.dw2(u); u = self.dwd(u)
        u = self.pw(u)
        return self.act(self.bn(u + x))

class DLKABlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(ConvBNGELU(ch, ch), ConvBNGELU(ch, ch))
        self.attn = DLKA(ch)
        self.fuse = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        h = self.conv(x)
        h = self.attn(h)
        return self.fuse(h)

class DLKA_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()
        C1,C2,C3,C4 = base, base*2, base*4, base*8
        self.e1 = nn.Sequential(ConvBNGELU(in_channels, C1), DLKABlock(C1))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(ConvBNGELU(C1, C2, s=2, p=1), DLKABlock(C2))
        self.e3 = nn.Sequential(ConvBNGELU(C2, C3, s=2, p=1), DLKABlock(C3))
        self.e4 = nn.Sequential(ConvBNGELU(C3, C4, s=2, p=1), DLKABlock(C4))
        self.bott = DLKABlock(C4)
        self.up3 = nn.ConvTranspose2d(C4, C3, 2, 2); self.d3 = nn.Sequential(ConvBNGELU(C3+C3, C3), DLKABlock(C3))
        self.up2 = nn.ConvTranspose2d(C3, C2, 2, 2); self.d2 = nn.Sequential(ConvBNGELU(C2+C2, C2), DLKABlock(C2))
        self.up1 = nn.ConvTranspose2d(C2, C1, 2, 2); self.d1 = nn.Sequential(ConvBNGELU(C1+C1, C1), DLKABlock(C1))
        self.head = nn.Conv2d(C1, out_channels, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        h  = self.bott(e4)

        y3 = self.up3(h)
        if y3.shape[-2:] != e3.shape[-2:]:
            y3 = F.interpolate(y3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        y3 = torch.cat([e3, y3], 1)
        y3 = self.d3(y3)

        y2 = self.up2(y3)
        if y2.shape[-2:] != e2.shape[-2:]:
            y2 = F.interpolate(y2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        y2 = torch.cat([e2, y2], 1)
        y2 = self.d2(y2)

        y1 = self.up1(y2)
        # 🔧 关键修复：与 e1 对齐到同一分辨率
        if y1.shape[-2:] != e1.shape[-2:]:
            y1 = F.interpolate(y1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        y1 = torch.cat([e1, y1], 1)
        y1 = self.d1(y1)

        return self.head(y1)

# ===========================
# 4) SUnet (Lite)
# ===========================
class ChannelAttn(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1,ch//r), 1)
        self.fc2 = nn.Conv2d(max(1,ch//r), ch, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class SpatialAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv,_ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, maxv], 1)
        m = torch.sigmoid(self.conv(a))
        return x * m

class SCAGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttn(ch)
        self.sa = SpatialAttn(ch)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return self.proj(x)

class SUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.downs = nn.ModuleList(); self.ups = nn.ModuleList(); self.gates = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(nn.Sequential(ConvBNGELU(ch, f), ConvBNGELU(f, f)))
            self.gates.append(SCAGate(f))
            ch = f
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBNGELU(features[-1], features[-1]*2),
                                        ConvBNGELU(features[-1]*2, features[-1]*2))
        rev = list(reversed(features)); ch = features[-1]*2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, 2))
            self.ups.append(nn.Sequential(ConvBNGELU(f+f, f), ConvBNGELU(f, f)))
            ch = f
        self.head = nn.Conv2d(ch, out_channels, 1)

        # 🔧 关键：让 gates 顺序与 skips[::-1] 对齐（512,256,128,64）
        self.gates = nn.ModuleList(list(self.gates)[::-1])

    def forward(self, x):
        skips=[]; h=x
        for d in self.downs:
            h = d(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            h = self.ups[i](h)
            s = self.gates[i//2](skips[i//2])  # 顺序一一对应
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([s, h], 1)
            h = self.ups[i+1](h)
        return self.head(h)


# -----------------------------
# Registry of built-in models
# -----------------------------
BUILTIN_MODELS = {
    'UNet': UNet,
    'AttentionUNet': AttentionUNet,
    'ERUNet': ERUNet,
    'FATNetLite': FATNetLite,
    'MedTLite': MedTLite,
    'nnUNet2D': nnUNet2D,
    'BATFormerLite': BATFormerLite,
    'HSHUNet': HSHUNet,
    'SwinUNetLite': SwinUNetLite,
    'MISSFormerLite': MISSFormerLite,
    'BRAUNetPP': BRAUNetPP,
    'HiFormerLite': HiFormerLite,
    'H2FormerLite': H2FormerLite,
    'TransUNetLite': TransUNetLite,
    'TransFuseLite': TransFuseLite,
    'PVT_GCASCAD_Lite': PVT_GCASCAD_Lite,
    'DLKA_UNet': DLKA_UNet,
    'SUnet': SUnet,
}

# -----------------------------
# Dataset
# -----------------------------
class ISIC2018Dataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 mask_dir: Optional[str] = None,
                 img_size: int = 256,
                 is_train: bool = True):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.img_size = img_size
        exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(exts)]
        self.img_paths.sort()
        self.mask_paths = []
        for p in self.img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            candidates = []
            if mask_dir and os.path.isdir(mask_dir):
                for suf in ['_segmentation.png', '_segmentation.jpg', '.png', '.jpg']:
                    cand = os.path.join(mask_dir, stem + suf)
                    if os.path.exists(cand):
                        candidates.append(cand)
            else:
                for suf in ['_segmentation.png', '_segmentation.jpg', '_mask.png', '_mask.jpg']:
                    cand = os.path.join(os.path.dirname(p), stem + suf)
                    if os.path.exists(cand):
                        candidates.append(cand)
            self.mask_paths.append(candidates[0] if candidates else None)
        self.tf_img_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.tf_img_eval = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.tf_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
        ])
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        mask_path = self.mask_paths[idx]
        if self.is_train:
            img_t = self.tf_img_train(img)
        else:
            img_t = self.tf_img_eval(img)
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = self.tf_mask(mask)
            mask = np.array(mask)
            mask = (mask > 127).astype(np.float32)
            mask_t = torch.from_numpy(mask)[None, ...]
        else:
            mask_t = torch.zeros((1, img_t.shape[-2], img_t.shape[-1]), dtype=torch.float32)
        return img_t, mask_t, img_path, (mask_path or '')

# -----------------------------
# Metrics
# -----------------------------
@dataclass
class Metrics:
    dice: float
    iou: float
    sensitivity: float
    specificity: float
    accuracy: float

def compute_confusion(pred: torch.Tensor, target: torch.Tensor) -> Tuple[int,int,int,int]:
    pred = pred.int(); target = target.int()
    tp = int(((pred == 1) & (target == 1)).sum().item())
    tn = int(((pred == 0) & (target == 0)).sum().item())
    fp = int(((pred == 1) & (target == 0)).sum().item())
    fn = int(((pred == 0) & (target == 1)).sum().item())
    return tp, tn, fp, fn

def metrics_from_confusion(tp:int, tn:int, fp:int, fn:int) -> Metrics:
    eps = 1e-7
    dice = (2*tp) / (2*tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return Metrics(dice, iou, sensitivity, specificity, accuracy)

# -----------------------------
# Utils
# -----------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def import_from_path(dotted: str):
    mod_name, cls_name = dotted.rsplit('.', 1)
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)

def make_model(name: str, in_channels: int, out_channels: int, model_kwargs: Dict[str, Any]):
    """Factory: build model by name or dotted path."""
    if name in BUILTIN_MODELS:
        cls = BUILTIN_MODELS[name]
        return cls(in_channels=in_channels, out_channels=out_channels, **model_kwargs)
    elif '.' in name:
        cls = import_from_path(name)
        return cls(in_channels=in_channels, out_channels=out_channels, **model_kwargs)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Use one of {list(BUILTIN_MODELS.keys())} or a dotted class path."
        )

# -----------------------------
# Train / Eval loops (supervised segmentation)
# -----------------------------
def train_epoch(model, loader, opt, device, bce_weight=0.5):
    model.train()
    total_loss = 0.0
    for imgs, masks, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(imgs)

        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        logits = logits.squeeze(1)
        masks_ = masks.squeeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, masks_)
        probs = torch.sigmoid(logits)
        inter = (probs*masks_).sum(dim=(1,2))
        dice = (2*inter + 1) / (probs.sum(dim=(1,2)) + masks_.sum(dim=(1,2)) + 1)
        dice_loss = 1 - dice.mean()
        loss = bce_weight*bce + (1-bce_weight)*dice_loss
        loss.backward()
        opt.step()
        total_loss += float(loss.item())*imgs.size(0)
    return total_loss/len(loader.dataset)

def eval_epoch(model, loader, device, save_masks_dir: Optional[str]=None, threshold: float=0.5) -> Tuple[Metrics, List[Tuple[str, Metrics]]]:
    model.eval()
    all_tp=all_tn=all_fp=all_fn=0
    per_image = []
    os.makedirs(save_masks_dir, exist_ok=True) if save_masks_dir else None
    with torch.no_grad():
        for imgs, masks, img_paths, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            for i in range(imgs.size(0)):
                pred = preds[i,0].cpu()
                tgt = masks[i,0].cpu()
                tp, tn, fp, fn = compute_confusion(pred, tgt)
                m = metrics_from_confusion(tp, tn, fp, fn)
                per_image.append((img_paths[i], m))
                all_tp += tp; all_tn += tn; all_fp += fp; all_fn += fn
                if save_masks_dir:
                    pm = (pred.numpy()*255).astype(np.uint8)
                    out_name = os.path.splitext(os.path.basename(img_paths[i]))[0] + '_pred.png'
                    Image.fromarray(pm).save(os.path.join(save_masks_dir, out_name))
    agg = metrics_from_confusion(all_tp, all_tn, all_fp, all_fn)
    return agg, per_image

# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, choices=['train','test'], required=True)
    # DATA
    p.add_argument('--train_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2_Training_Input/images/', help='train dataset root')
    p.add_argument('--train_mask_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2_Training_Input/masks/', help='train masks root (optional; auto-detect if empty)')
    p.add_argument('--test_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2/test/images/', help='val/test dataset root')
    p.add_argument('--test_mask_path', type=str, default='/zjy_work/ISIC2018/ISIC2018_Task1-2/test/masks/', help='test masks root (optional; auto-detect if empty)')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=8)
    # MODEL
    p.add_argument('--model', type=str, default='UNet', help="Model name: e.g., UNet, AttentionUNet, nnUNet2D, SwinUNetLite, etc.")
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--out_channels', type=int, default=1)
    p.add_argument('--model-kwargs', type=str, default='{}', help="JSON or Python dict string for extra model kwargs")
    # OPT
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--bce_weight', type=float, default=0.5)
    # IO
    p.add_argument('--save_dir', type=str, default='/zjy_work/MSDUNet-main/model_out')
    p.add_argument('--checkpoint', type=str, default='')
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--threshold', type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    try:
        mk = json.loads(args.model_kwargs)
    except Exception:
        mk = ast.literal_eval(args.model_kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device, "CUDA available:", torch.cuda.is_available())

    model = make_model(args.model, args.in_channels, args.out_channels, mk).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | Params: {n_params/1e6:.3f}M")

    if args.mode == 'train':
        ds = ISIC2018Dataset(args.train_path, args.train_mask_path if args.train_mask_path else None, args.img_size, is_train=True)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_ds = ISIC2018Dataset(args.test_path, args.test_mask_path if args.test_mask_path else None, args.img_size, is_train=False)
        val_dl = DataLoader(val_ds, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=args.num_workers)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_dice = -1.0
        best_path = os.path.join(args.save_dir, f"{args.model}_best.pt")
        
        for epoch in range(1, args.epochs+1):
            loss = train_epoch(model, dl, opt, device, bce_weight=args.bce_weight)
            agg, _ = eval_epoch(model, val_dl, device, save_masks_dir=None, threshold=args.threshold)
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | DICE {agg.dice:.4f} IoU {agg.iou:.4f} "
                  f"Sen {agg.sensitivity:.4f} Spe {agg.specificity:.4f} Acc {agg.accuracy:.4f}")

            # 只保存最优模型
            if agg.dice > best_dice:
                best_dice = agg.dice
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'params': n_params}, best_path)
                print(f"🔥 New best model saved at epoch {epoch} with DICE {best_dice:.4f} -> {best_path}")


    elif args.mode == 'test':
        assert args.checkpoint and os.path.isfile(args.checkpoint), 'Please provide a valid --checkpoint for test mode.'
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        ds = ISIC2018Dataset(args.test_path, args.test_mask_path if args.test_mask_path else None, args.img_size, is_train=False)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        save_masks_dir = os.path.join(args.save_dir, 'pred_masks')
        agg, per_img = eval_epoch(model, dl, device, save_masks_dir=save_masks_dir, threshold=args.threshold)
        summary_path = os.path.join(args.save_dir, 'test_metrics.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Params: {n_params}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write("Aggregate Metrics\n")
            f.write(f"Dice: {agg.dice:.6f}\nIoU: {agg.iou:.6f}\nSensitivity: {agg.sensitivity:.6f}\nSpecificity: {agg.specificity:.6f}\nAccuracy: {agg.accuracy:.6f}\n")
            f.write("\nPer-image Metrics (path, Dice, IoU, Sen, Spe, Acc)\n")
            for path, m in per_img:
                f.write(f"{path}\t{m.dice:.6f}\t{m.iou:.6f}\t{m.sensitivity:.6f}\t{m.specificity:.6f}\t{m.accuracy:.6f}\n")
        print(f"Saved masks to: {save_masks_dir}")
        print(f"Wrote metrics to: {summary_path}")

if __name__ == '__main__':
    main()
