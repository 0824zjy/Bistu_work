import os
import json
import random
import math
from typing import List, Dict

from PIL import Image

import numpy as np
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
    """
    real samples are forced to 1.0.
    synthetic samples with invalid weights become 0.0 and can be skipped later.
    """
    if source == "real":
        return 1.0

    w = finite_float(weight, default=0.0)

    if w <= 0.0:
        return 0.0

    return float(max(0.0, min(w, 1.5)))

def build_morphological_boundary(mask_tensor: torch.Tensor, kernel_size: int = 3):
    """
    Build a stable binary boundary from a binary mask.

    Args:
        mask_tensor: [1,H,W], value range {0,1}
        kernel_size: boundary width control, 3 is recommended first.

    Returns:
        boundary: [1,H,W], value range {0,1}
    """
    assert mask_tensor.dim() == 3 and mask_tensor.size(0) == 1

    mask_tensor = (mask_tensor >= 0.5).float()

    x = mask_tensor.unsqueeze(0)  # [1,1,H,W]

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

    boundary = (dilated - eroded).clamp(0.0, 1.0)

    return boundary.squeeze(0)  # [1,H,W]


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
    Weighted segmentation dataset for final BGDNet training.

    Each jsonl line:
    {
      "image": image_path,
      "mask": mask_path,
      "weight": 1.0,
      "source": "real" or "bef_sbg"
    }

    Return:
      image:    [3,H,W]
      gt:       [1,H,W]
      boundary: [1,H,W]
      weight:   scalar tensor
      source:   string
    """

    def __init__(
        self,
        weighted_jsonl: str,
        trainsize: int = 352,
        augmentation: bool = False,
    ):
        self.weighted_jsonl = weighted_jsonl
        self.trainsize = trainsize
        self.augmentation = augmentation

        self.items = read_jsonl(weighted_jsonl)

        if len(self.items) == 0:
            raise RuntimeError(f"No training samples found in {weighted_jsonl}")

        self.items = [
            x for x in self.items
            if os.path.exists(x.get("image", "")) and os.path.exists(x.get("mask", ""))
        ]

        if len(self.items) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {weighted_jsonl}")

        self.size = len(self.items)

        self.real_count = sum(1 for x in self.items if x.get("source", "") == "real")
        self.synthetic_count = sum(1 for x in self.items if x.get("source", "") != "real")

        print("[WeightedBEFDataset]")
        print(f"  jsonl           = {weighted_jsonl}")
        print(f"  total           = {self.size}")
        print(f"  real            = {self.real_count}")
        print(f"  synthetic       = {self.synthetic_count}")
        print(f"  augmentation    = {self.augmentation}")

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

            self.gt_transform = transforms.Compose([
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

            self.gt_transform = transforms.Compose([
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def __getitem__(self, index):
        item = self.items[index]

        image_path = item["image"]
        mask_path = item["mask"]

        source = item.get("source", "unknown")
        weight = sanitize_weight(item.get("weight", 1.0), source=source)


        image = self.rgb_loader(image_path)
        gt = self.binary_loader(mask_path)

        seed = np.random.randint(1998)

        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)

        # mask must be binary after nearest-neighbor resize
        gt = (gt >= 0.5).float()

        with torch.no_grad():
            boundary = build_morphological_boundary(gt, kernel_size=3)

        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        boundary = torch.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0)

        gt = torch.clamp(gt, 0.0, 1.0)
        boundary = torch.clamp(boundary, 0.0, 1.0)

        weight_tensor = torch.tensor(weight, dtype=torch.float32)

        return image, gt, boundary, weight_tensor, source

def get_weighted_loader(
    weighted_jsonl: str,
    batchsize: int,
    trainsize: int,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    augmentation: bool = False,
    drop_last: bool = True,
):
    dataset = WeightedBEFDataset(
        weighted_jsonl=weighted_jsonl,
        trainsize=trainsize,
        augmentation=augmentation,
    )

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return loader
