from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ConvNeXt teacher requires a torchvision version that provides convnext_tiny."
    ) from exc


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 3),
            ConvBNAct(out_channels, out_channels, 3),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ConvNeXtFeatureEncoder(nn.Module):
    """Expose the four spatial stages of torchvision ConvNeXt-Tiny."""

    channels: Tuple[int, int, int, int] = (96, 192, 384, 768)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.features = backbone.features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (1, 3, 5, 7):
                outputs.append(x)
        if len(outputs) != 4:
            raise RuntimeError(f"Expected four ConvNeXt features, received {len(outputs)}")
        return outputs


class ConvNeXtUNetTeacher(nn.Module):
    """A structurally different CNN teacher with mask and boundary heads."""

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_backbone: bool = True,
        decoder_channels: Tuple[int, int, int, int] = (384, 192, 96, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ConvNeXtFeatureEncoder(pretrained=pretrained_backbone)
        c1, c2, c3, c4 = self.encoder.channels
        d4, d3, d2, d1 = decoder_channels

        self.top = ResidualConvBlock(c4, d4)
        self.decode3 = DecoderBlock(d4, c3, d3)
        self.decode2 = DecoderBlock(d3, c2, d2)
        self.decode1 = DecoderBlock(d2, c1, d1)

        self.refine = nn.Sequential(
            ResidualConvBlock(d1, d1),
            nn.Dropout2d(dropout),
        )
        self.mask_head = nn.Conv2d(d1, num_classes, kernel_size=1)
        self.boundary_head = nn.Sequential(
            ResidualConvBlock(d1, d1),
            nn.Conv2d(d1, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        input_size = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        x = self.top(f4)
        x = self.decode3(x, f3)
        x = self.decode2(x, f2)
        x = self.decode1(x, f1)
        x = self.refine(x)
        mask = self.mask_head(x)
        boundary = self.boundary_head(x)
        mask = F.interpolate(mask, size=input_size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary, size=input_size, mode="bilinear", align_corners=False)
        return mask, boundary
