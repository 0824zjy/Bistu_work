import cv2
import json
import random
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


class MyDataset(Dataset):
    def __init__(self, prompt_json='./data/prompt_train.json', empty_prompt_prob=0.05, size=384):
        self.data = []
        self.prompt_json = prompt_json
        self.empty_prompt_prob = empty_prompt_prob
        self.size = size

        with open(self.prompt_json, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item.get('prompt', "")

        if random.random() < self.empty_prompt_prob:
            prompt = ""

        # ---- read mask ----
        source = Image.open(source_filename).convert('L')
        source_array = np.array(source)
        binary_array = np.where(source_array > 127, 255, 0).astype(np.uint8)

        # ---- build boundary from mask ----
        boundary_array = build_boundary_from_mask(binary_array, kernel_size=3)

        # convert to RGB for compatibility with current pipeline
        source_rgb = Image.fromarray(binary_array).convert('RGB')
        boundary_rgb = Image.fromarray(boundary_array).convert('RGB')

        # ---- read target image ----
        target = Image.open(target_filename).convert('RGB')

        source = np.array(source_rgb).astype(np.uint8)
        boundary = np.array(boundary_rgb).astype(np.uint8)
        target = np.array(target).astype(np.uint8)

        preprocess = self.transform(size=self.size)(
            image=target,
            mask=source,
            boundary=boundary
        )

        source = preprocess['mask']
        boundary = preprocess['boundary']
        target = preprocess['image']

        source = source.astype(np.float32) / 255.0       # [0,1]
        boundary = boundary.astype(np.float32) / 255.0   # [0,1]
        target = target.astype(np.float32) / 127.5 - 1.0 # [-1,1]

        return dict(
            jpg=target,
            txt=prompt,
            hint=source,
            boundary=boundary
        )

    def transform(self, size=384):
        transforms = albumentations.Compose(
            [
                albumentations.Resize(height=size, width=size)
            ],
            additional_targets={
                'boundary': 'mask'
            }
        )
        return transforms
