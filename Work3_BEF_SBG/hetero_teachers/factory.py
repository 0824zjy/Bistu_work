from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .cnn_teacher import ConvNeXtUNetTeacher
from .common import TeacherMetadata, extract_state_dict, read_checkpoint_metadata


def _import_bgdnet(bgdnet_root: str):
    if bgdnet_root not in sys.path:
        sys.path.insert(0, bgdnet_root)
    os.environ["BGDNET_ENABLE_DISTANCE_HEAD"] = "0"
    from models.BGDNet import BGDNet  # type: ignore

    return BGDNet


def build_teacher_for_training(
    teacher_type: str,
    pretrained: bool = True,
    sam_type: str = "vit_b",
    sam_base_checkpoint: Optional[str] = None,
    sam_freeze_encoder: bool = True,
    sam_unfreeze_last_blocks: int = 0,
    sam_adapter_bottleneck: int = 64,
    bgdnet_root: str = "/data/zjy_work/BGDNet",
) -> Tuple[nn.Module, TeacherMetadata]:
    teacher_type = teacher_type.lower()
    if teacher_type == "cnn":
        model = ConvNeXtUNetTeacher(pretrained_backbone=pretrained)
        metadata = TeacherMetadata(
            teacher_type="cnn",
            architecture="convnext_tiny_unet",
            image_size=352,
            extra={"pretrained_backbone": bool(pretrained)},
        )
        return model, metadata

    if teacher_type in {"sam", "sam_adapter", "medsam_adapter"}:
        if sam_base_checkpoint is None or not os.path.exists(sam_base_checkpoint):
            raise FileNotFoundError(
                "Training the SAM adapter requires --sam_base_checkpoint pointing to "
                "an official SAM ViT-B or MedSAM ViT-B checkpoint."
            )
        from .sam_adapter_teacher import PromptFreeSAMAdapterTeacher
        model = PromptFreeSAMAdapterTeacher(
            sam_type=sam_type,
            base_checkpoint=sam_base_checkpoint,
            freeze_image_encoder=sam_freeze_encoder,
            unfreeze_last_blocks=sam_unfreeze_last_blocks,
            adapter_bottleneck=sam_adapter_bottleneck,
        )
        metadata = TeacherMetadata(
            teacher_type="sam_adapter",
            architecture=f"prompt_free_{sam_type}_adapter",
            image_size=model.image_size,
            base_checkpoint=sam_base_checkpoint,
            extra={
                "sam_type": sam_type,
                "freeze_image_encoder": bool(sam_freeze_encoder),
                "unfreeze_last_blocks": int(sam_unfreeze_last_blocks),
                "adapter_bottleneck": int(sam_adapter_bottleneck),
            },
        )
        return model, metadata

    if teacher_type == "bgdnet":
        BGDNet = _import_bgdnet(bgdnet_root)
        model = BGDNet(num_classes=1)
        metadata = TeacherMetadata(
            teacher_type="bgdnet",
            architecture="BGDNet_standard",
            image_size=352,
            extra={"bgdnet_root": bgdnet_root},
        )
        return model, metadata

    raise ValueError(f"Unsupported teacher_type: {teacher_type}")


def build_teacher_from_checkpoint(
    checkpoint_path: str,
    teacher_type: Optional[str] = None,
    device: Any = "cpu",
    bgdnet_root: str = "/data/zjy_work/BGDNet",
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("metadata"), dict):
        metadata = dict(checkpoint["metadata"])

    resolved_type = (teacher_type or metadata.get("teacher_type") or "").lower()
    if not resolved_type:
        # Existing BGDNet checkpoints do not contain metadata.
        resolved_type = "bgdnet"

    if resolved_type == "cnn":
        model = ConvNeXtUNetTeacher(pretrained_backbone=False)
    elif resolved_type in {"sam", "sam_adapter", "medsam_adapter"}:
        from .sam_adapter_teacher import PromptFreeSAMAdapterTeacher
        extra = metadata.get("extra") or {}
        sam_type = str(extra.get("sam_type", "vit_b"))
        adapter_bottleneck = int(extra.get("adapter_bottleneck", 64))
        model = PromptFreeSAMAdapterTeacher(
            sam_type=sam_type,
            base_checkpoint=None,
            freeze_image_encoder=True,
            unfreeze_last_blocks=0,
            adapter_bottleneck=adapter_bottleneck,
        )
        resolved_type = "sam_adapter"
    elif resolved_type == "bgdnet":
        BGDNet = _import_bgdnet(bgdnet_root)
        model = BGDNet(num_classes=1)
    else:
        raise ValueError(f"Unsupported teacher_type in checkpoint: {resolved_type}")

    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(
            "[WARN] non-strict load:",
            "missing=", len(incompatible.missing_keys),
            "unexpected=", len(incompatible.unexpected_keys),
        )
    model = model.to(device).eval()

    metadata.setdefault("teacher_type", resolved_type)
    if "image_size" not in metadata:
        metadata["image_size"] = 1024 if resolved_type == "sam_adapter" else 352
    return model, metadata


def teacher_expects_sam_input(metadata: Dict[str, Any]) -> bool:
    return str(metadata.get("teacher_type", "")).lower() in {
        "sam",
        "sam_adapter",
        "medsam_adapter",
    }
