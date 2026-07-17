import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torch.nn.functional as F


def _sobel_boundary_from_mask(mask_tensor, thresh: float = 0.0):
    """
    输入: mask_tensor [1,H,W], 值域[0,1]
    输出: boundary [1,H,W], 值域[0,1]
    说明: 使用固定Sobel核, 不参与训练; 可选阈值化(默认不阈值化, 交由BCE学习)
    """
    assert mask_tensor.dim() == 3 and mask_tensor.size(0) == 1
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], dtype=mask_tensor.dtype, device=mask_tensor.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], dtype=mask_tensor.dtype, device=mask_tensor.device).view(1, 1, 3, 3)
    gx = F.conv2d(mask_tensor.unsqueeze(0), kx, padding=1)
    gy = F.conv2d(mask_tensor.unsqueeze(0), ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0)  # [1,H,W]
    # 归一化到[0,1]
    mag = mag / (mag.max() + 1e-8)
    if thresh > 0:
        mag = (mag > thresh).float()
    return mag


class PolypDataset(data.Dataset):
    """
    dataloader for lesion/ polyp segmentation with boundary supervision
    返回: image, mask, boundary
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations is True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # 对image与gt应用相同随机变换的随机种子
        seed = np.random.randint(1998)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)  # [1,H,W], 值域[0,1]

        # 由变换后的gt生成boundary（确保空间对齐）
        with torch.no_grad():
            boundary = _sobel_boundary_from_mask(gt)  # [1,H,W] in [0,1]

        return image, gt, boundary

    def filter_files(self):
        assert len(self.images) == len(self.gts), "Image/GT count mismatch!"
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=2, pin_memory=True, augmentation=False, drop_last=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name  # 推理阶段评估分割主头，boundary可选保存

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
