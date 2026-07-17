#work3
import os
import cv2
import json
import random
import math
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations


def build_boundary_from_mask(mask_gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    mask_gray: H x W, uint8, {0,255}
    return: H x W, uint8, {0,255}
    """
    mask_bin = (mask_gray > 127).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    boundary = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, kernel)
    boundary = (boundary > 0).astype(np.uint8) * 255
    return boundary


def build_soft_boundary_from_mask(
    mask_gray: np.ndarray,
    kernel_size: int = 3,
    radius: int = 12,
    tau: float = 4.0,
) -> np.ndarray:
    """
    Fallback soft boundary prior when jsonl does not provide boundary_prior.
    """
    hard = build_boundary_from_mask(mask_gray, kernel_size=kernel_size)
    hard_f = (hard > 127).astype(np.float32)

    soft = hard_f.copy()
    prev = hard_f.copy()

    for r in range(1, int(radius) + 1):
        k = 2 * r + 1
        kernel = np.ones((k, k), np.uint8)

        dilated = cv2.dilate(hard_f, kernel, iterations=1)
        shell = np.clip(dilated - prev, 0.0, 1.0)

        weight = math.exp(-float(r) / max(float(tau), 1e-6))
        soft = np.maximum(soft, shell * weight)

        prev = dilated

    soft = np.clip(soft, 0.0, 1.0)
    return (soft * 255.0).round().astype(np.uint8)


def read_gray_uint8(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L")).astype(np.uint8)


def gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Unsupported gray/rgb array shape: {arr.shape}")


class MyDataset(Dataset):
    """
    BEF-SBG training dataset.

    Each jsonl line:
    {
      "source": mask_path,
      "target": image_path,
      "prompt": "dermoscopic image",
      "boundary_prior": adaptive_prior_path,
      "difficulty": difficulty_path optional
    }

    Output is compatible with ControlLDM.get_input:
      jpg: HWC, float32, [-1,1]
      txt: prompt
      hint: HWC RGB, float32, [0,1]
      boundary: HWC RGB, float32, [0,1]
      boundary_prior: HWC RGB, float32, [0,1]
      difficulty: HWC RGB, float32, [0,1]
    """

    def __init__(
        self,
        prompt_json="./data/prompt_train.json",
        empty_prompt_prob=0.05,
        size=384,
        fallback_radius=12,
        fallback_tau=4.0,
    ):
        self.data = []
        self.prompt_json = prompt_json
        self.empty_prompt_prob = empty_prompt_prob
        self.size = size
        self.fallback_radius = fallback_radius
        self.fallback_tau = fallback_tau

        with open(self.prompt_json, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        if len(self.data) == 0:
            raise RuntimeError(f"No samples found in {self.prompt_json}")

        print(f"[BEF Dataset] json={self.prompt_json}, samples={len(self.data)}, size={self.size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item.get("prompt", "")

        if random.random() < self.empty_prompt_prob:
            prompt = ""

        # -------------------------
        # Mask / hint
        # -------------------------
        source_gray = read_gray_uint8(source_filename)
        source_bin = np.where(source_gray > 127, 255, 0).astype(np.uint8)

        # -------------------------
        # Hard boundary
        # -------------------------
        boundary_gray = build_boundary_from_mask(source_bin, kernel_size=3)

        # -------------------------
        # Adaptive boundary prior
        # -------------------------
        boundary_prior_path = item.get("boundary_prior", None)

        if boundary_prior_path is not None and os.path.exists(boundary_prior_path):
            boundary_prior_gray = read_gray_uint8(boundary_prior_path)
        else:
            boundary_prior_gray = build_soft_boundary_from_mask(
                source_bin,
                kernel_size=3,
                radius=self.fallback_radius,
                tau=self.fallback_tau,
            )

        # -------------------------
        # Difficulty map, optional
        # -------------------------
        difficulty_path = item.get("difficulty", None)
        if difficulty_path is not None and os.path.exists(difficulty_path):
            difficulty_gray = read_gray_uint8(difficulty_path)
        else:
            difficulty_gray = np.zeros_like(source_bin, dtype=np.uint8)

        # -------------------------
        # Target image
        # -------------------------
        if not os.path.exists(target_filename):
            raise FileNotFoundError(f"Missing target image: {target_filename}")

        target = np.array(Image.open(target_filename).convert("RGB")).astype(np.uint8)

        source = gray_to_rgb(source_bin).astype(np.uint8)
        boundary = gray_to_rgb(boundary_gray).astype(np.uint8)
        boundary_prior = gray_to_rgb(boundary_prior_gray).astype(np.uint8)
        difficulty = gray_to_rgb(difficulty_gray).astype(np.uint8)

        preprocess = self.transform(size=self.size)(
            image=target,
            mask=source,
            boundary=boundary,
            boundary_prior=boundary_prior,
            difficulty=difficulty,
        )

        target = preprocess["image"]
        source = preprocess["mask"]
        boundary = preprocess["boundary"]
        boundary_prior = preprocess["boundary_prior"]
        difficulty = preprocess["difficulty"]

        # Keep mask / hard boundary binary after resize.
        source = np.where(source > 127, 255, 0).astype(np.uint8)
        boundary = np.where(boundary > 127, 255, 0).astype(np.uint8)

        source = source.astype(np.float32) / 255.0
        boundary = boundary.astype(np.float32) / 255.0
        boundary_prior = boundary_prior.astype(np.float32) / 255.0
        difficulty = difficulty.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0

        return dict(
            jpg=target,
            txt=prompt,
            hint=source,
            boundary=boundary,
            boundary_prior=boundary_prior,
            difficulty=difficulty,
        )

    def transform(self, size=384):
        return albumentations.Compose(
            [
                albumentations.Resize(
                    height=size,
                    width=size,
                    interpolation=cv2.INTER_LINEAR,
                )
            ],
            additional_targets={
                "boundary": "mask",
                "boundary_prior": "mask",
                "difficulty": "mask",
            },
        )
