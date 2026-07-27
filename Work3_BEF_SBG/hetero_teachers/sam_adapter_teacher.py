from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SAM adapter teacher requires the official segment-anything package. "
        "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from exc


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
        )


class ResidualEmbeddingAdapter(nn.Module):
    """A lightweight domain adapter applied to SAM image embeddings."""

    def __init__(self, channels: int = 256, bottleneck: int = 64, scale_init: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(16, bottleneck), num_channels=bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.scale) * self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x)


class PromptFreeSAMDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c1),
            ConvNormAct(c1, c1),
        )
        self.up1 = UpsampleBlock(c1, c2)
        self.up2 = UpsampleBlock(c2, c3)
        self.up3 = UpsampleBlock(c3, c4)
        self.up4 = UpsampleBlock(c4, c4)
        self.mask_head = nn.Conv2d(c4, num_classes, kernel_size=1)
        self.boundary_head = nn.Sequential(
            ConvNormAct(c4, c4),
            nn.Conv2d(c4, 1, kernel_size=1),
        )

    def forward(self, embedding: torch.Tensor, output_size: Tuple[int, int]):
        x = self.stem(embedding)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        mask = self.mask_head(x)
        boundary = self.boundary_head(x)
        if mask.shape[-2:] != output_size:
            mask = F.interpolate(mask, size=output_size, mode="bilinear", align_corners=False)
            boundary = F.interpolate(boundary, size=output_size, mode="bilinear", align_corners=False)
        return mask, boundary


class PromptFreeSAMAdapterTeacher(nn.Module):
    """
    Automatic lesion segmentation teacher based on the official SAM/MedSAM ViT-B encoder.

    The standard prompt encoder and SAM mask decoder are not used. The image encoder is
    followed by a trainable residual domain adapter and a dense prompt-free decoder.
    By default the image encoder is frozen, making one-GPU training practical.
    """

    def __init__(
        self,
        sam_type: str = "vit_b",
        base_checkpoint: Optional[str] = None,
        num_classes: int = 1,
        freeze_image_encoder: bool = True,
        unfreeze_last_blocks: int = 0,
        adapter_bottleneck: int = 64,
    ) -> None:
        super().__init__()
        if sam_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM model type: {sam_type}")
        self.sam_type = sam_type
        sam = sam_model_registry[sam_type](checkpoint=base_checkpoint)
        self.image_encoder = sam.image_encoder
        self.image_size = int(self.image_encoder.img_size)
        self.register_buffer("pixel_mean", sam.pixel_mean.detach().clone(), persistent=True)
        self.register_buffer("pixel_std", sam.pixel_std.detach().clone(), persistent=True)
        del sam
        self.adapter = ResidualEmbeddingAdapter(
            channels=256,
            bottleneck=int(adapter_bottleneck),
        )
        self.decoder = PromptFreeSAMDecoder(in_channels=256, num_classes=num_classes)
        self.configure_encoder_trainability(
            freeze_image_encoder=freeze_image_encoder,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )

    def configure_encoder_trainability(
        self,
        freeze_image_encoder: bool,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)

        if not freeze_image_encoder:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad_(True)
            return

        unfreeze_last_blocks = max(0, int(unfreeze_last_blocks))
        if unfreeze_last_blocks > 0:
            blocks = self.image_encoder.blocks
            for block in blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.image_encoder.neck.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen SAM modules must stay in eval mode; adapter/decoder remain trainable.
        if not any(parameter.requires_grad for parameter in self.image_encoder.parameters()):
            self.image_encoder.eval()
        return self

    def forward(self, image_0_255: torch.Tensor):
        if image_0_255.ndim != 4 or image_0_255.shape[1] != 3:
            raise ValueError("SAM teacher expects BCHW RGB input.")
        output_size = image_0_255.shape[-2:]
        if output_size != (self.image_size, self.image_size):
            image_0_255 = F.interpolate(
                image_0_255,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        preprocessed = (image_0_255 - self.pixel_mean) / self.pixel_std
        pad_h = self.image_size - preprocessed.shape[-2]
        pad_w = self.image_size - preprocessed.shape[-1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Input is larger than SAM image encoder size.")
        preprocessed = F.pad(preprocessed, (0, pad_w, 0, pad_h))
        embedding = self.image_encoder(preprocessed)
        embedding = self.adapter(embedding)
        return self.decoder(embedding, output_size=output_size)
