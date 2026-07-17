import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

# -------------------------
# boundary from mask
# -------------------------
def _sobel_boundary_from_mask(mask_tensor, thresh: float = 0.0):
    """
    输入: mask_tensor [1,H,W], 值域[0,1]
    输出: boundary [1,H,W], 值域[0,1]
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
    mag = mag / (mag.max() + 1e-8)
    if thresh > 0:
        mag = (mag > thresh).float()
    return mag


# -------------------------
# helpers for TXT-driven loading
# -------------------------
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def read_id_list(list_txt: str) -> List[str]:
    ids = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids

def find_existing_file(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None

def build_pairs_from_txt(
    image_root: str,
    gt_root: str,
    list_txt: str,
    mode: str,
    mask_suffix: str = "_segmentation",
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    mode:
      - "isic":
          image stem == ID
          mask 先尝试 ID + mask_suffix（train 常见：_segmentation.png）
          若不存在，再尝试 ID（test 常见：同名 .jpg/.png）
      - "paired":
          image stem == ID, mask stem == ID (same name)
    """
    ids = read_id_list(list_txt)
    images, masks = [], []
    miss_img, miss_mask = 0, 0

    for _id in ids:
        # image
        img_path = find_existing_file(image_root, _id, IMG_EXTS)
        if img_path is None:
            miss_img += 1
            continue

        # mask
        if mode == "isic":
            # ✅ 先尝试 _segmentation
            mask_path = find_existing_file(gt_root, _id + mask_suffix, MASK_EXTS)
            # ✅ 再尝试同名（test）
            if mask_path is None:
                mask_path = find_existing_file(gt_root, _id, MASK_EXTS)

        elif mode == "paired":
            mask_path = find_existing_file(gt_root, _id, MASK_EXTS)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'isic' or 'paired'.")

        if mask_path is None:
            miss_mask += 1
            continue

        images.append(img_path)
        masks.append(mask_path)

    if verbose:
        print(f"[TXT LOAD] {list_txt}")
        print(f"  image_root: {image_root}")
        print(f"  gt_root   : {gt_root}")
        print(f"  mode      : {mode}")
        print(f"  ids={len(ids)}, pairs={len(images)}, miss_img={miss_img}, miss_mask={miss_mask}")

    return images, masks


# -------------------------
# Train Dataset
# -------------------------
class PolypDataset(data.Dataset):
    """
    支持多路径训练集（带边界监督）
    返回: image, mask, boundary

    如果提供 list_txts，则严格按 txt 列表加载（可复现、可控）。
    """
    def __init__(
        self,
        image_roots: List[str],
        gt_roots: List[str],
        trainsize: int,
        augmentations: bool,
        list_txts: Optional[List[Optional[str]]] = None,
        modes: Optional[List[str]] = None,
        mask_suffix: str = "_segmentation",
    ):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(f"数据增强开启状态: {self.augmentations}")

        assert len(image_roots) == len(gt_roots), "image_roots 与 gt_roots 长度必须一致"

        if list_txts is None:
            list_txts = [None] * len(image_roots)
        if modes is None:
            modes = ["paired"] * len(image_roots)

        assert len(list_txts) == len(image_roots), "list_txts 长度必须与 image_roots 一致"
        assert len(modes) == len(image_roots), "modes 长度必须与 image_roots 一致"

        self.images: List[str] = []
        self.gts: List[str] = []

        for img_root, gt_root, list_txt, mode in zip(image_roots, gt_roots, list_txts, modes):
            if list_txt is not None:
                imgs, gts = build_pairs_from_txt(
                    image_root=img_root,
                    gt_root=gt_root,
                    list_txt=list_txt,
                    mode=mode,
                    mask_suffix=mask_suffix,
                    verbose=True
                )
            else:
                # legacy: list all files
                img_files = [os.path.join(img_root, f) for f in os.listdir(img_root)
                             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp'))]
                gt_files = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp'))]
                img_files = sorted(img_files)
                gt_files = sorted(gt_files)
                imgs, gts = img_files, gt_files

                print(f"[DIR LOAD] img_root={img_root} -> {len(imgs)}")
                print(f"[DIR LOAD] gt_root ={gt_root} -> {len(gts)}")

            self.images.extend(imgs)
            self.gts.extend(gts)

        # 过滤尺寸不匹配
        self.filter_files()
        self.size = len(self.images)
        print(f"合并后数据集总数: {self.size}")

        if self.augmentations is True:
            print('使用数据增强: RandomRotation, RandomFlip')
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
            print('未使用数据增强')
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

        # 同步随机变换
        seed = np.random.randint(1998)
        random.seed(seed); torch.manual_seed(seed)
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed); torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)  # [1,H,W]

        with torch.no_grad():
            boundary = _sobel_boundary_from_mask(gt)

        return image, gt, boundary

    def filter_files(self):
        assert len(self.images) == len(self.gts), "图片/掩码数量不匹配！"
        images, gts = [], []
        bad = 0
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
            else:
                bad += 1
        self.images = images
        self.gts = gts
        print(f"过滤后有效数据数: {len(self.images)} (丢弃尺寸不匹配: {bad})")

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


def get_loader(
    image_roots: List[str],
    gt_roots: List[str],
    batchsize: int,
    trainsize: int,
    shuffle: bool = False,
    num_workers: int = 16,
    pin_memory: bool = True,
    augmentation: bool = False,
    drop_last: bool = False,
    list_txts: Optional[List[Optional[str]]] = None,
    modes: Optional[List[str]] = None,
    mask_suffix: str = "_segmentation",
):
    dataset = PolypDataset(
        image_roots=image_roots,
        gt_roots=gt_roots,
        trainsize=trainsize,
        augmentations=augmentation,
        list_txts=list_txts,
        modes=modes,
        mask_suffix=mask_suffix,
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
    return data_loader


# -------------------------
# Test/Val Dataset
# -------------------------
class test_dataset:
    """
    支持:
      - 原始目录遍历（不推荐）
      - 或者提供 list_txt + mode 来严格按 txt 加载（推荐）
    """
    def __init__(
        self,
        image_root: str,
        gt_root: str,
        testsize: int,
        list_txt: Optional[str] = None,
        mode: str = "isic",
        mask_suffix: str = "_segmentation",
    ):
        self.testsize = testsize
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        if list_txt is not None:
            imgs, gts = build_pairs_from_txt(
                image_root=image_root,
                gt_root=gt_root,
                list_txt=list_txt,
                mode=mode,
                mask_suffix=mask_suffix,
                verbose=True
            )
            self.images = imgs
            self.gts = gts
        else:
            self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp'))]
            self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp'))]
            self.images = sorted(self.images)
            self.gts = sorted(self.gts)

        self.size = len(self.images)
        self.index = 0
        print(f"[TEST DATASET] size={self.size}")

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = os.path.basename(self.images[self.index])
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
