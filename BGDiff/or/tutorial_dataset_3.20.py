import cv2
import json
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations


class MyDataset(Dataset):
    def __init__(self, prompt_json='./data/prompt_train.json', empty_prompt_prob=0.05, size=384):
        """
        prompt_json: jsonl 文件路径，每行一个样本: {source, target, prompt}
        empty_prompt_prob: 训练时随机置空 prompt 的概率（CFG/鲁棒性）
        size: resize 尺寸
        """
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

        # 随机置空 prompt（可配置）
        if random.random() < self.empty_prompt_prob:
            prompt = ""

        # 读 mask：灰度 → 二值化 → RGB (3ch)
        source = Image.open(source_filename).convert('L')
        source_array = np.array(source)
        threshold = 127
        binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
        binary_image = Image.fromarray(binary_array)
        source = binary_image.convert('RGB')

        # 读 image：RGB
        target = Image.open(target_filename).convert('RGB')

        source = np.array(source).astype(np.uint8)
        target = np.array(target).astype(np.uint8)

        preprocess = self.transform(size=self.size)(image=target, mask=source)
        source, target = preprocess['mask'], preprocess['image']

        # Mask-Image Pair
        source = source.astype(np.float32) / 255.0          # [0,1]
        target = target.astype(np.float32) / 127.5 - 1.0    # [-1,1]

        return dict(jpg=target, txt=prompt, hint=source)

    def transform(self, size=384):
        transforms = albumentations.Compose([
            albumentations.Resize(height=size, width=size)
        ])
        return transforms
