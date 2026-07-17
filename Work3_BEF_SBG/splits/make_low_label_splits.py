import os
import argparse
import random
from typing import List, Optional


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_existing_file(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    """
    兼容 ISIC mask 命名：
      1. stem_segmentation.png
      2. stem.png / stem.jpg / ...
    """
    p = find_existing_file(mask_dir, stem + "_segmentation", MASK_EXTS)
    if p is not None:
        return p

    p = find_existing_file(mask_dir, stem, MASK_EXTS)
    if p is not None:
        return p

    return None


def collect_valid_stems(image_dir: str, mask_dir: str) -> List[str]:
    stems = []

    for fn in os.listdir(image_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMG_EXTS:
            continue

        stem = os.path.splitext(fn)[0]
        img_path = os.path.join(image_dir, fn)
        mask_path = find_mask(mask_dir, stem)

        if os.path.isfile(img_path) and mask_path is not None:
            stems.append(stem)

    stems = sorted(list(set(stems)))
    return stems


def parse_ratios(ratio_str: str) -> List[float]:
    ratios = []
    for x in ratio_str.split(","):
        x = x.strip()
        if not x:
            continue
        ratios.append(float(x))
    return ratios


def ratio_to_tag(ratio: float) -> str:
    if abs(ratio - 1.0) < 1e-8:
        return "100p"

    percent = ratio * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"{int(round(percent))}p"

    s = f"{percent:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") + "p"


def write_txt(path: str, stems: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in stems:
            f.write(s + "\n")


def make_kfold_splits(stems: List[str], n_folds: int):
    """
    简单稳定 K-fold：
      - 输入 stems 已经被随机打乱；
      - 第 fold 份作为 val；
      - 其余作为 train。
    """
    folds = [[] for _ in range(n_folds)]
    for i, s in enumerate(stems):
        folds[i % n_folds].append(s)

    split_items = []
    for fold in range(n_folds):
        val = folds[fold]
        train = []
        for j in range(n_folds):
            if j != fold:
                train.extend(folds[j])

        split_items.append((sorted(train), sorted(val)))

    return split_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/zjy_work/ISIC2018/train/Images",
        help="ISIC2018 train Images directory",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/data/zjy_work/ISIC2018/train/Masks",
        help="ISIC2018 train Masks directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/zjy_work/Work3_BEF_SBG/splits",
        help="Output split txt directory",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.05,0.10,0.20,1.0",
        help="Comma-separated low-label ratios",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for teacher OOF prediction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_stems = collect_valid_stems(args.image_dir, args.mask_dir)
    if len(all_stems) == 0:
        raise RuntimeError(
            f"No valid image-mask pairs found.\n"
            f"image_dir={args.image_dir}\n"
            f"mask_dir={args.mask_dir}"
        )

    print(f"[INFO] valid image-mask pairs: {len(all_stems)}")

    ratios = parse_ratios(args.ratios)

    for ratio in ratios:
        tag = ratio_to_tag(ratio)

        rng = random.Random(args.seed)

        shuffled = all_stems[:]
        rng.shuffle(shuffled)

        if ratio >= 1.0:
            selected = shuffled
        else:
            n_select = int(round(len(shuffled) * ratio))
            n_select = max(args.n_folds, n_select)
            n_select = min(len(shuffled), n_select)
            selected = shuffled[:n_select]

        selected_sorted = sorted(selected)

        all_txt = os.path.join(args.out_dir, f"low_{tag}_all.txt")
        write_txt(all_txt, selected_sorted)

        print(f"[SPLIT] ratio={ratio} tag={tag} selected={len(selected_sorted)}")
        print(f"        all: {all_txt}")

        fold_splits = make_kfold_splits(selected, args.n_folds)

        for fold, (train_stems, val_stems) in enumerate(fold_splits):
            train_txt = os.path.join(args.out_dir, f"low_{tag}_fold{fold}_train.txt")
            val_txt = os.path.join(args.out_dir, f"low_{tag}_fold{fold}_val.txt")

            write_txt(train_txt, train_stems)
            write_txt(val_txt, val_stems)

            print(
                f"        fold={fold}: "
                f"train={len(train_stems)} val={len(val_stems)}"
            )

    print("[DONE] low-label split generation finished.")


if __name__ == "__main__":
    main()
