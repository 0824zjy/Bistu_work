import os
import json
import random
import math
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def finite_float(x, default=0.0):
    try:
        y = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(y):
        return float(default)
    return float(y)


def sanitize_weight(weight, source="unknown"):
    if source == "real":
        return 1.0
    w = finite_float(weight, default=0.0)
    if w <= 0.0:
        return 0.0
    return float(max(0.0, min(w, 1.5)))


def build_morphological_boundary(mask_tensor: torch.Tensor, kernel_size: int = 3):
    """Build a binary morphology-gradient boundary from [1,H,W] mask."""
    if mask_tensor.dim() != 3 or mask_tensor.size(0) != 1:
        raise ValueError(f"mask_tensor must be [1,H,W], got {tuple(mask_tensor.shape)}")
    x = (mask_tensor >= 0.5).float().unsqueeze(0)
    dilated = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    eroded = -F.max_pool2d(
        -x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return (dilated - eroded).clamp(0.0, 1.0).squeeze(0)


def build_signed_distance_target(
    mask_tensor: torch.Tensor,
    max_distance: float = 20.0,
):
    """
    Build normalized signed distance target in [-1,1].

    Positive: lesion interior.
    Zero: boundary.
    Negative: background.
    """
    mask_np = (
        mask_tensor.squeeze(0).detach().cpu().numpy() >= 0.5
    ).astype(np.uint8)

    if mask_np.max() == 0:
        signed = -np.ones_like(mask_np, dtype=np.float32)
    elif mask_np.min() == 1:
        signed = np.ones_like(mask_np, dtype=np.float32)
    else:
        inside = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        outside = cv2.distanceTransform(1 - mask_np, cv2.DIST_L2, 5)
        signed = inside - outside
        max_distance = max(float(max_distance), 1.0)
        signed = np.clip(signed, -max_distance, max_distance) / max_distance

    return torch.from_numpy(signed.astype(np.float32)).unsqueeze(0)


def read_jsonl(path: str) -> List[Dict]:
    data_items = []
    bad_json = 0
    missing_file = 0
    bad_weight = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
            except Exception as e:
                bad_json += 1
                print(f"[WARN] skip bad json line={line_no}: {e}")
                continue

            image_path = item.get("image", "")
            mask_path = item.get("mask", "")
            source = item.get("source", "unknown")
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                missing_file += 1
                continue

            weight = sanitize_weight(item.get("weight", 1.0), source=source)
            if source != "real" and weight <= 0.0:
                bad_weight += 1
                continue

            item["weight"] = float(weight)
            item["source"] = source
            data_items.append(item)

    print("[read_jsonl]")
    print(f"  path         = {path}")
    print(f"  valid        = {len(data_items)}")
    print(f"  bad_json     = {bad_json}")
    print(f"  missing_file = {missing_file}")
    print(f"  bad_weight   = {bad_weight}")
    return data_items


class WeightedBEFDataset(data.Dataset):
    """
    Return:
      image:       [3,H,W]
      gt:          [1,H,W]
      boundary:    [1,H,W]
      distance:    [1,H,W], normalized signed distance in [-1,1]
      reliability: [1,H,W], teacher pixel confidence
      hardness:    [1,H,W], reliable hard-boundary map
      weight:      scalar tensor
      source:      string
    """

    def __init__(
        self,
        weighted_jsonl: str,
        trainsize: int = 352,
        augmentation: bool = False,
        distance_max_px: float = 20.0,
    ):
        self.weighted_jsonl = weighted_jsonl
        self.trainsize = int(trainsize)
        self.augmentation = bool(augmentation)
        self.distance_max_px = float(distance_max_px)
        self.items = read_jsonl(weighted_jsonl)
        if not self.items:
            raise RuntimeError(f"No training samples found in {weighted_jsonl}")

        self.size = len(self.items)
        self.real_count = sum(x.get("source", "") == "real" for x in self.items)
        self.synthetic_count = self.size - self.real_count

        print("[WeightedBEFDataset]")
        print(f"  jsonl           = {weighted_jsonl}")
        print(f"  total           = {self.size}")
        print(f"  real            = {self.real_count}")
        print(f"  synthetic       = {self.synthetic_count}")
        print(f"  augmentation    = {self.augmentation}")
        print(f"  distance_max_px = {self.distance_max_px}")

        if self.augmentation:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(
                    degrees=(0, 90),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
            self.mask_transform = transforms.Compose([
                transforms.RandomRotation(
                    degrees=(0, 90),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ])
            self.soft_map_transform = transforms.Compose([
                transforms.RandomRotation(
                    degrees=(0, 90),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ])
            self.soft_map_transform = transforms.Compose([
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.size

    @staticmethod
    def rgb_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    @staticmethod
    def gray_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")

    @staticmethod
    def constant_gray(value: int, size):
        return Image.new("L", size=size, color=int(value))

    @staticmethod
    def apply_with_seed(transform, image, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return transform(image)

    def __getitem__(self, index):
        item = self.items[index]
        source = item.get("source", "unknown")
        weight = sanitize_weight(item.get("weight", 1.0), source=source)

        image_pil = self.rgb_loader(item["image"])
        gt_pil = self.gray_loader(item["mask"])

        reliability_path = item.get("reliability", "")
        hardness_path = item.get("hardness", "")

        if reliability_path and os.path.exists(reliability_path):
            reliability_pil = self.gray_loader(reliability_path)
        else:
            reliability_pil = self.constant_gray(255, gt_pil.size)

        if hardness_path and os.path.exists(hardness_path):
            hardness_pil = self.gray_loader(hardness_path)
        else:
            hardness_pil = self.constant_gray(0, gt_pil.size)

        seed = int(np.random.randint(0, 2**31 - 1))
        image = self.apply_with_seed(self.img_transform, image_pil, seed)
        gt = self.apply_with_seed(self.mask_transform, gt_pil, seed)
        reliability = self.apply_with_seed(
            self.soft_map_transform,
            reliability_pil,
            seed,
        )
        hardness = self.apply_with_seed(
            self.soft_map_transform,
            hardness_pil,
            seed,
        )

        gt = (gt >= 0.5).float()
        reliability = reliability.clamp(0.0, 1.0)
        hardness = hardness.clamp(0.0, 1.0)

        with torch.no_grad():
            boundary = build_morphological_boundary(gt, kernel_size=3)
            distance = build_signed_distance_target(
                gt,
                max_distance=self.distance_max_px,
            )

        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        boundary = torch.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0)
        distance = torch.nan_to_num(distance, nan=0.0, posinf=1.0, neginf=-1.0)
        reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)
        hardness = torch.nan_to_num(hardness, nan=0.0, posinf=1.0, neginf=0.0)

        return (
            image,
            gt.clamp(0.0, 1.0),
            boundary.clamp(0.0, 1.0),
            distance.clamp(-1.0, 1.0),
            reliability.clamp(0.0, 1.0),
            hardness.clamp(0.0, 1.0),
            torch.tensor(weight, dtype=torch.float32),
            source,
        )


def get_weighted_loader(
    weighted_jsonl: str,
    batchsize: int,
    trainsize: int,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    augmentation: bool = False,
    drop_last: bool = True,
    distance_max_px: float = 20.0,
):
    dataset = WeightedBEFDataset(
        weighted_jsonl=weighted_jsonl,
        trainsize=trainsize,
        augmentation=augmentation,
        distance_max_px=distance_max_px,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
