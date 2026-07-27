from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = IMG_EXTS


@dataclass
class TeacherMetadata:
    teacher_type: str
    architecture: str
    image_size: int
    num_classes: int = 1
    base_checkpoint: str = ""
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result["extra"] is None:
            result["extra"] = {}
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def finite_array(
    array: np.ndarray,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> np.ndarray:
    output = np.asarray(array, dtype=np.float32)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    if low is not None or high is not None:
        output = np.clip(
            output,
            -np.inf if low is None else float(low),
            np.inf if high is None else float(high),
        )
    return output.astype(np.float32)


def read_id_list(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            stem = os.path.splitext(os.path.basename(item))[0]
            if stem.endswith("_segmentation"):
                stem = stem[: -len("_segmentation")]
            ids.append(stem)
    return ids


def find_existing_file(root: str, stem: str, exts: Sequence[str]) -> Optional[str]:
    for extension in exts:
        path = os.path.join(root, stem + extension)
        if os.path.exists(path):
            return path
    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    path = find_existing_file(mask_dir, stem + "_segmentation", MASK_EXTS)
    if path is not None:
        return path
    return find_existing_file(mask_dir, stem, MASK_EXTS)


def list_images(root: str) -> List[str]:
    result: List[str] = []
    for name in sorted(os.listdir(root)):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            result.append(os.path.join(root, name))
    return result


def read_gray(path: str) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if array.size and float(array.max()) > 1.0:
        array /= 255.0
    return finite_array(array, 0.0, 1.0)


def save_gray(array: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = np.round(finite_array(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(path, output):
        raise RuntimeError(f"Unable to save image: {path}")


def build_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    binary = (mask >= 0.5).astype(np.uint8)
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return np.clip(dilated - eroded, 0, 1).astype(np.float32)


def build_soft_boundary(mask: np.ndarray, radius: int = 12, tau: float = 4.0) -> np.ndarray:
    hard = build_boundary(mask, kernel_size=3)
    soft = hard.copy()
    previous = hard.copy()
    tau = max(float(tau), 1e-6)
    for distance in range(1, int(radius) + 1):
        kernel = np.ones((2 * distance + 1, 2 * distance + 1), np.uint8)
        dilated = cv2.dilate(hard, kernel, iterations=1)
        shell = np.clip(dilated - previous, 0.0, 1.0)
        soft = np.maximum(soft, shell * math.exp(-distance / tau))
        previous = dilated
    return finite_array(soft, 0.0, 1.0)


def largest_component_fill(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    output = (labels == largest).astype(np.uint8)
    inverse = 1 - output
    flooded = inverse.copy()
    flood_mask = np.zeros((output.shape[0] + 2, output.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 0)
    return np.clip(output + flooded, 0, 1).astype(np.uint8)


def normalized_entropy(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    probability = np.clip(finite_array(probability, 0.0, 1.0), eps, 1.0 - eps)
    entropy = -probability * np.log(probability) - (1.0 - probability) * np.log(1.0 - probability)
    return finite_array(entropy / math.log(2.0), 0.0, 1.0)


def dice_score(probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    prediction = (probability >= threshold).float()
    target = (target >= 0.5).float()
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    if weight is None:
        weight = torch.ones_like(target)
    numerator = 2.0 * (probability * target * weight).sum(dim=(2, 3)) + 1.0
    denominator = ((probability + target) * weight).sum(dim=(2, 3)) + 1.0
    return (1.0 - numerator / denominator).mean()


def segmentation_boundary_loss(
    mask_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    mask_target: torch.Tensor,
    boundary_target: torch.Tensor,
    boundary_weight: float = 0.35,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target)
    mask_dice = soft_dice_loss(mask_logits, mask_target)
    boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
    boundary_dice = soft_dice_loss(boundary_logits, boundary_target)
    loss = mask_bce + mask_dice + float(boundary_weight) * (boundary_bce + boundary_dice)
    parts = {
        "mask_bce": float(mask_bce.detach()),
        "mask_dice": float(mask_dice.detach()),
        "boundary_bce": float(boundary_bce.detach()),
        "boundary_dice": float(boundary_dice.detach()),
    }
    return loss, parts


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def save_teacher_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_dice: float,
    metadata: TeacherMetadata,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "metadata": metadata.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def read_checkpoint_metadata(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("metadata"), dict):
        return dict(checkpoint["metadata"])
    return {}


class ISICTeacherDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        list_txt: str,
        image_size: int,
        augment: bool,
        sam_input: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = read_id_list(list_txt)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.sam_input = bool(sam_input)

        self.samples: List[Tuple[str, str, str]] = []
        for stem in self.ids:
            image_path = find_existing_file(image_dir, stem, IMG_EXTS)
            mask_path = find_mask(mask_dir, stem)
            if image_path is None or mask_path is None:
                continue
            self.samples.append((stem, image_path, mask_path))
        if not self.samples:
            raise RuntimeError(f"No valid image-mask pairs found from {list_txt}")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.2:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if random.random() < 0.5:
            angle = random.uniform(-25.0, 25.0)
            image = TF.rotate(image, angle, InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, InterpolationMode.NEAREST, fill=0)
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
            image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        stem, image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        if self.sam_input:
            image_tensor = image_tensor * 255.0
        else:
            image_tensor = TF.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        mask_tensor = (TF.to_tensor(mask) >= 0.5).float()
        boundary = build_boundary(mask_tensor.squeeze(0).numpy(), kernel_size=3)
        boundary_tensor = torch.from_numpy(boundary).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "stem": stem,
            "image_path": image_path,
            "mask_path": mask_path,
        }


class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], image_size: int, sam_input: bool = False) -> None:
        self.image_paths = list(image_paths)
        self.image_size = int(image_size)
        self.sam_input = bool(sam_input)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        resized = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized)
        if self.sam_input:
            tensor = tensor * 255.0
        else:
            tensor = TF.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        return {
            "image": tensor,
            "path": path,
            "stem": os.path.splitext(os.path.basename(path))[0],
            "original_size": torch.tensor([height, width], dtype=torch.int64),
        }


def collate_image_only(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "stem": [item["stem"] for item in batch],
        "original_size": [tuple(map(int, item["original_size"].tolist())) for item in batch],
    }


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
