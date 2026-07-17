import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    IMG_INTERP = InterpolationMode.BILINEAR
    GT_INTERP = InterpolationMode.NEAREST
except Exception:
    IMG_INTERP = Image.BILINEAR
    GT_INTERP = Image.NEAREST


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

SUPPORTED_MODES = ["isic", "ph2", "paired", "auto"]


def read_id_list(list_txt: str) -> List[str]:
    """
    读取样本 ID 列表。

    支持以下格式：
        IMD003
        IMD003.bmp
        /some/path/IMD003.bmp

    如果一行包含多个空格分隔字段，默认取第一个字段。
    """
    if not os.path.isfile(list_txt):
        raise FileNotFoundError(f"找不到 txt 文件: {list_txt}")

    ids = []

    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            # 默认取每行第一个字段
            sample_id = s.split()[0]
            ids.append(sample_id)

    return ids


def normalize_sample_id(sample_id: str) -> str:
    """
    将以下形式统一转换为无扩展名的 stem：

        IMD003
        IMD003.bmp
        /path/to/IMD003.bmp

    返回：
        IMD003
    """
    sample_id = sample_id.strip()
    sample_id = os.path.basename(sample_id)

    stem, ext = os.path.splitext(sample_id)

    if ext.lower() in IMG_EXTS or ext.lower() in MASK_EXTS:
        return stem

    return sample_id


def find_existing_file(
    dir_path: str,
    stem: str,
    exts: List[str],
) -> Optional[str]:
    """
    在目录中寻找 stem + extension。

    同时兼容：
        .bmp
        .BMP
        .png
        .PNG
        等大小写形式。
    """
    if not os.path.isdir(dir_path):
        return None

    # 先直接尝试，速度较快
    for ext in exts:
        candidates = [
            os.path.join(dir_path, stem + ext),
            os.path.join(dir_path, stem + ext.upper()),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

    # 再做一次大小写不敏感匹配
    target_names = set()

    for ext in exts:
        target_names.add((stem + ext).lower())

    for filename in os.listdir(dir_path):
        if filename.lower() in target_names:
            path = os.path.join(dir_path, filename)

            if os.path.isfile(path):
                return path

    return None


def get_mask_stems(
    sample_stem: str,
    mode: str,
    mask_suffix: str = "_segmentation",
) -> List[str]:
    """
    根据数据集模式返回可能的 mask stem。

    isic:
        ISIC_0000000_segmentation
        ISIC_0000000

    ph2:
        IMD003_lesion
        IMD003

    paired:
        sample_id

    auto:
        sample_id_segmentation
        sample_id_lesion
        sample_id
    """
    mode = mode.lower()

    if mode == "isic":
        return [
            sample_stem + mask_suffix,
            sample_stem,
        ]

    if mode == "ph2":
        return [
            sample_stem + "_lesion",
            sample_stem,
        ]

    if mode == "paired":
        return [
            sample_stem,
        ]

    if mode == "auto":
        return [
            sample_stem + mask_suffix,
            sample_stem + "_lesion",
            sample_stem,
        ]

    raise ValueError(
        f"Unknown mode: {mode}. "
        f"Supported modes: {SUPPORTED_MODES}"
    )


def resolve_pair(
    image_root: str,
    gt_root: str,
    sample_id: str,
    mode: str,
    mask_suffix: str = "_segmentation",
) -> Tuple[Optional[str], Optional[str]]:
    """
    根据一个样本 ID 查找对应图片和掩码。
    """
    sample_stem = normalize_sample_id(sample_id)

    image_path = find_existing_file(
        dir_path=image_root,
        stem=sample_stem,
        exts=IMG_EXTS,
    )

    if image_path is None:
        return None, None

    mask_path = None

    mask_stems = get_mask_stems(
        sample_stem=sample_stem,
        mode=mode,
        mask_suffix=mask_suffix,
    )

    for mask_stem in mask_stems:
        mask_path = find_existing_file(
            dir_path=gt_root,
            stem=mask_stem,
            exts=MASK_EXTS,
        )

        if mask_path is not None:
            break

    return image_path, mask_path


def build_pairs_from_txt(
    image_root: str,
    gt_root: str,
    list_txt: str,
    mode: str,
    mask_suffix: str = "_segmentation",
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    根据 txt 中的 ID 精确加载图片和掩码。

    mode:
        isic:
            image: ID.*
            mask : ID_segmentation.* 或 ID.*

        ph2:
            image: ID.*
            mask : ID_lesion.* 或 ID.*

        paired:
            image: ID.*
            mask : ID.*

        auto:
            自动尝试 ISIC、PH2 和同名规则
    """
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"找不到图片目录: {image_root}")

    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"找不到掩码目录: {gt_root}")

    mode = mode.lower()

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Supported modes: {SUPPORTED_MODES}"
        )

    ids = read_id_list(list_txt)

    images: List[str] = []
    masks: List[str] = []

    miss_img = 0
    miss_mask = 0

    missing_image_examples = []
    missing_mask_examples = []

    for sample_id in ids:
        sample_stem = normalize_sample_id(sample_id)

        image_path = find_existing_file(
            dir_path=image_root,
            stem=sample_stem,
            exts=IMG_EXTS,
        )

        if image_path is None:
            miss_img += 1

            if len(missing_image_examples) < 5:
                missing_image_examples.append(sample_stem)

            continue

        mask_path = None

        mask_stems = get_mask_stems(
            sample_stem=sample_stem,
            mode=mode,
            mask_suffix=mask_suffix,
        )

        for mask_stem in mask_stems:
            mask_path = find_existing_file(
                dir_path=gt_root,
                stem=mask_stem,
                exts=MASK_EXTS,
            )

            if mask_path is not None:
                break

        if mask_path is None:
            miss_mask += 1

            if len(missing_mask_examples) < 5:
                missing_mask_examples.append(
                    {
                        "id": sample_stem,
                        "tried": mask_stems,
                    }
                )

            continue

        images.append(image_path)
        masks.append(mask_path)

    if verbose:
        print(f"[TXT LOAD] {list_txt}")
        print(f"  image_root: {image_root}")
        print(f"  gt_root   : {gt_root}")
        print(f"  mode      : {mode}")
        print(
            f"  ids={len(ids)}, "
            f"pairs={len(images)}, "
            f"miss_img={miss_img}, "
            f"miss_mask={miss_mask}"
        )

        if missing_image_examples:
            print("  missing image examples:")

            for sample_stem in missing_image_examples:
                print(f"    {sample_stem}")

        if missing_mask_examples:
            print("  missing mask examples:")

            for item in missing_mask_examples:
                print(
                    f"    id={item['id']}, "
                    f"tried={item['tried']}"
                )

    return images, masks


def build_pairs_from_directory(
    image_root: str,
    gt_root: str,
    mode: str,
    mask_suffix: str = "_segmentation",
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    没有提供 txt 时，根据 image_root 中的图片逐一寻找对应 mask。

    不再使用“图片列表和 mask 列表分别排序后直接 zip”的方式，
    避免 PH2 的 _lesion 后缀造成错误配对。
    """
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"找不到图片目录: {image_root}")

    if not os.path.isdir(gt_root):
        raise FileNotFoundError(f"找不到掩码目录: {gt_root}")

    mode = mode.lower()

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Supported modes: {SUPPORTED_MODES}"
        )

    image_files = sorted([
        filename
        for filename in os.listdir(image_root)
        if filename.lower().endswith(tuple(IMG_EXTS))
    ])

    images: List[str] = []
    masks: List[str] = []

    miss_mask = 0
    missing_examples = []

    for filename in image_files:
        image_path = os.path.join(image_root, filename)
        sample_stem = os.path.splitext(filename)[0]

        mask_path = None

        mask_stems = get_mask_stems(
            sample_stem=sample_stem,
            mode=mode,
            mask_suffix=mask_suffix,
        )

        for mask_stem in mask_stems:
            mask_path = find_existing_file(
                dir_path=gt_root,
                stem=mask_stem,
                exts=MASK_EXTS,
            )

            if mask_path is not None:
                break

        if mask_path is None:
            miss_mask += 1

            if len(missing_examples) < 5:
                missing_examples.append(
                    {
                        "image": filename,
                        "tried": mask_stems,
                    }
                )

            continue

        images.append(image_path)
        masks.append(mask_path)

    if verbose:
        print(f"[DIR LOAD]")
        print(f"  image_root: {image_root}")
        print(f"  gt_root   : {gt_root}")
        print(f"  mode      : {mode}")
        print(
            f"  images={len(image_files)}, "
            f"pairs={len(images)}, "
            f"miss_mask={miss_mask}"
        )

        if missing_examples:
            print("  missing mask examples:")

            for item in missing_examples:
                print(
                    f"    image={item['image']}, "
                    f"tried={item['tried']}"
                )

    return images, masks


class SegTrainDataset(data.Dataset):
    """
    通用二分类分割训练 Dataset。

    返回：
        image: Tensor [3, H, W]
        mask : Tensor [1, H, W]

    支持：
        多数据源合并
        txt 精确加载
        isic / ph2 / paired / auto 匹配规则
    """

    def __init__(
        self,
        image_roots: List[str],
        gt_roots: List[str],
        trainsize: int,
        augmentation: bool = False,
        list_txts: Optional[List[Optional[str]]] = None,
        modes: Optional[List[str]] = None,
        mask_suffix: str = "_segmentation",
    ):
        self.trainsize = trainsize
        self.augmentation = augmentation

        if len(image_roots) != len(gt_roots):
            raise ValueError(
                "image_roots 与 gt_roots 长度必须一致，"
                f"得到 {len(image_roots)} 和 {len(gt_roots)}"
            )

        if list_txts is None:
            list_txts = [None] * len(image_roots)

        if modes is None:
            modes = ["paired"] * len(image_roots)

        if len(list_txts) != len(image_roots):
            raise ValueError(
                "list_txts 长度必须与 image_roots 一致"
            )

        if len(modes) != len(image_roots):
            raise ValueError(
                "modes 长度必须与 image_roots 一致"
            )

        self.images: List[str] = []
        self.gts: List[str] = []

        for img_root, gt_root, list_txt, mode in zip(
            image_roots,
            gt_roots,
            list_txts,
            modes,
        ):
            mode = mode.lower()

            if list_txt is not None and list_txt != "":
                imgs, gts = build_pairs_from_txt(
                    image_root=img_root,
                    gt_root=gt_root,
                    list_txt=list_txt,
                    mode=mode,
                    mask_suffix=mask_suffix,
                    verbose=True,
                )
            else:
                imgs, gts = build_pairs_from_directory(
                    image_root=img_root,
                    gt_root=gt_root,
                    mode=mode,
                    mask_suffix=mask_suffix,
                    verbose=True,
                )

            self.images.extend(imgs)
            self.gts.extend(gts)

        self.filter_files()
        self.size = len(self.images)

        print(f"[TRAIN DATASET] augmentation={self.augmentation}")
        print(f"[TRAIN DATASET] total size={self.size}")

        if self.size == 0:
            raise RuntimeError(
                "训练数据集大小为 0。请检查图片目录、掩码目录、"
                "txt 内容以及数据集 mode。"
            )

        if self.augmentation:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(
                    90,
                    interpolation=IMG_INTERP,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=IMG_INTERP,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])

            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(
                    90,
                    interpolation=GT_INTERP,
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=GT_INTERP,
                ),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=IMG_INTERP,
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
                    interpolation=GT_INTERP,
                ),
                transforms.ToTensor(),
            ])

    def filter_files(self):
        """
        过滤打不开的文件以及训练阶段尺寸不一致的图片-mask 对。
        """
        if len(self.images) != len(self.gts):
            raise RuntimeError(
                f"图片和 mask 数量不一致: "
                f"{len(self.images)} vs {len(self.gts)}"
            )

        valid_images: List[str] = []
        valid_gts: List[str] = []

        size_mismatch = 0
        open_error = 0

        for img_path, gt_path in zip(self.images, self.gts):
            try:
                with Image.open(img_path) as img:
                    img_size = img.size

                with Image.open(gt_path) as gt:
                    gt_size = gt.size

                if img_size == gt_size:
                    valid_images.append(img_path)
                    valid_gts.append(gt_path)
                else:
                    size_mismatch += 1

                    if size_mismatch <= 5:
                        print(
                            f"[SIZE MISMATCH] "
                            f"image={img_path}, size={img_size}; "
                            f"mask={gt_path}, size={gt_size}"
                        )

            except Exception as exc:
                open_error += 1

                if open_error <= 5:
                    print(
                        f"[OPEN ERROR] "
                        f"image={img_path}, mask={gt_path}, "
                        f"error={exc}"
                    )

        self.images = valid_images
        self.gts = valid_gts

        print(
            f"[FILTER] valid={len(self.images)}, "
            f"drop_size_mismatch={size_mismatch}, "
            f"drop_open_error={open_error}"
        )

    @staticmethod
    def rgb_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")

    @staticmethod
    def binary_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            image = Image.open(f)
            return image.convert("L")

    def __getitem__(self, index: int):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # 保证 image 和 gt 使用完全相同的随机增强参数
        seed = int(np.random.randint(0, 2**31 - 1))

        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)

        # 使用最近邻插值后再次二值化
        gt = (gt >= 0.5).float()

        return image, gt

    def __len__(self):
        return self.size


def get_loader(
    image_roots: List[str],
    gt_roots: List[str],
    batchsize: int,
    trainsize: int,
    shuffle: bool = True,
    num_workers: int = 16,
    pin_memory: bool = True,
    augmentation: bool = False,
    drop_last: bool = True,
    list_txts: Optional[List[Optional[str]]] = None,
    modes: Optional[List[str]] = None,
    mask_suffix: str = "_segmentation",
):
    dataset = SegTrainDataset(
        image_roots=image_roots,
        gt_roots=gt_roots,
        trainsize=trainsize,
        augmentation=augmentation,
        list_txts=list_txts,
        modes=modes,
        mask_suffix=mask_suffix,
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


class SegTestDataset:
    """
    测试 Dataset。

    处理逻辑：
        image resize 到 testsize
        gt 保持原始尺寸
        推理后再将预测 resize 回 gt 尺寸
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

        mode = mode.lower()

        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unknown mode: {mode}. "
                f"Supported modes: {SUPPORTED_MODES}"
            )

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.testsize, self.testsize),
                interpolation=IMG_INTERP,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        if list_txt is not None and list_txt != "":
            self.images, self.gts = build_pairs_from_txt(
                image_root=image_root,
                gt_root=gt_root,
                list_txt=list_txt,
                mode=mode,
                mask_suffix=mask_suffix,
                verbose=True,
            )
        else:
            self.images, self.gts = build_pairs_from_directory(
                image_root=image_root,
                gt_root=gt_root,
                mode=mode,
                mask_suffix=mask_suffix,
                verbose=True,
            )

        self.size = len(self.images)
        self.index = 0

        print(f"[TEST DATASET] size={self.size}")

        if self.size == 0:
            print(
                "[WARN] 测试数据集为空。请检查测试路径、"
                "test_list 和 test mode。"
            )

    def load_data(self):
        if self.index >= self.size:
            raise IndexError(
                f"测试数据已经读取完毕: index={self.index}, size={self.size}"
            )

        image_path = self.images[self.index]
        gt_path = self.gts[self.index]

        image = self.rgb_loader(image_path)
        image = self.transform(image).unsqueeze(0)

        # gt 保持原始空间尺寸
        gt = self.binary_loader(gt_path)

        name = os.path.basename(image_path)

        self.index += 1

        return image, gt, name

    @staticmethod
    def rgb_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")

    @staticmethod
    def binary_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            image = Image.open(f)
            return image.convert("L")
