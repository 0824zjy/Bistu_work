# python /data/zjy_work/BGDiff/Ours/eval_quality.py --real_images /data/zjy_work/ISIC2018/train/Images --real_masks  /data/zjy_work/ISIC2018/train/Masks --gen_images  /data/zjy_work/ISIC2018/exp_mask2img_stage1_2/images --gen_masks   /data/zjy_work/ISIC2018/exp_mask2img_stage1_2/masks    --device cuda    --out_json /data/zjy_work/BGDiff/Ours/results/gen_quality_metrics_stage1-2_three.json
import os
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(d: Path):
    """
    递归列出目录下所有图像文件
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not d.exists():
        return []
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def read_mask_as_binary(mask_path: Path) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    arr = np.array(m, dtype=np.uint8)
    # >0 视为前景
    return (arr > 0).astype(np.uint8)


def apply_mask(img_path: Path, mask_bin: np.ndarray, mode: str) -> Image.Image:
    """
    mode:
      - "fg": 保留mask内，外部置0
      - "bg": 保留mask外，内部置0
    """
    img = Image.open(img_path).convert("RGB")
    im = np.array(img, dtype=np.uint8)
    h, w = im.shape[:2]

    if mask_bin.shape != (h, w):
        # resize mask to image size (nearest)
        m = Image.fromarray((mask_bin * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
        mask_bin = (np.array(m) > 0).astype(np.uint8)

    if mode == "fg":
        keep = mask_bin[..., None]
    elif mode == "bg":
        keep = (1 - mask_bin)[..., None]
    else:
        raise ValueError("mode must be fg or bg")

    out = im * keep
    return Image.fromarray(out)


# -------------------------
# Mask matching (ISIC-friendly)
# -------------------------
MASK_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def find_mask(masks_dir: Path, img_stem: str) -> Path | None:
    """
    查找与 img_stem 对应的 mask，兼容：
      1) 同名:            img_stem + ext
      2) ISIC 常见命名:   img_stem + "_segmentation" + ext
      3) 前缀唯一匹配:    glob(img_stem + "*" + ext) 若唯一则返回
    """
    # 1) 完全同名
    for ext in MASK_EXTS:
        p = masks_dir / (img_stem + ext)
        if p.exists():
            return p

    # 2) *_segmentation
    for ext in MASK_EXTS:
        p = masks_dir / (img_stem + "_segmentation" + ext)
        if p.exists():
            return p

    # 3) 同前缀兜底（必须唯一，否则返回 None 避免误匹配）
    cands = []
    for ext in MASK_EXTS:
        cands.extend(list(masks_dir.rglob(img_stem + "*" + ext)))
    cands = sorted([p for p in cands if p.is_file()])

    if len(cands) == 1:
        return cands[0]

    return None


def build_masked_folder(images_dir: Path,
                        masks_dir: Path,
                        out_dir: Path,
                        mode: str,
                        limit: int = 0,
                        debug_miss: int = 10):
    """
    将 images_dir 中每张图按对应 mask 做 fg/bg masking 输出到 out_dir
    若 limit>0 则只处理前 limit 张（用于快速评测）
    """
    ensure_dir(out_dir)

    imgs = list_images(images_dir)
    if limit > 0:
        imgs = imgs[:limit]

    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    # 统计/调试
    miss_examples = []
    n_ok = 0

    for img_p in tqdm(imgs, desc=f"masking {images_dir.name} ({mode})"):
        mask_p = find_mask(masks_dir, img_p.stem)
        if mask_p is None:
            if len(miss_examples) < debug_miss:
                miss_examples.append(img_p.name)
            continue

        mask_bin = read_mask_as_binary(mask_p)
        out_img = apply_mask(img_p, mask_bin, mode=mode)
        out_img.save(out_dir / (img_p.stem + ".png"))
        n_ok += 1

    if n_ok == 0:
        # 更友好的报错信息
        total_masks = len(list_images(masks_dir))
        msg = (
            f"No masked images produced for {images_dir} with masks {masks_dir}.\n"
            f"Images found: {len(imgs)} | Masks found: {total_masks}\n"
            f"Example missing matches (up to {debug_miss}): {miss_examples}\n"
            f"Likely cause: naming mismatch (e.g., ISIC uses *_segmentation.png) or wrong masks_dir."
        )
        raise RuntimeError(msg)

    return n_ok


def prepare_resized_folder(src_dir: Path, dst_dir: Path, size: int = 299, limit: int = 0):
    """
    将 src_dir 中的图片统一 resize 到 (size, size) 并保存到 dst_dir（png）
    limit>0 表示只处理前 limit 张（按 list_images 排序）
    """
    ensure_dir(dst_dir)
    imgs = list_images(src_dir)
    if limit > 0:
        imgs = imgs[:limit]

    n_ok = 0
    for img_p in tqdm(imgs, desc=f"resizing {src_dir.name} -> {dst_dir.name} ({size}x{size})"):
        try:
            with Image.open(img_p) as im:
                im = im.convert("RGB")
                im_res = im.resize((size, size), resample=Image.LANCZOS)
                im_res.save(dst_dir / (img_p.stem + ".png"))
                n_ok += 1
        except Exception as e:
            print(f"skip {img_p}: {e}")
            continue

    if n_ok == 0:
        raise RuntimeError(f"No images saved when resizing {src_dir}")
    return n_ok


def calc_fid_kid(path_real: str, path_gen: str, device: str = "cuda", batch_size: int = None):
    try:
        from torch_fidelity import calculate_metrics
    except Exception as e:
        raise RuntimeError(
            "torch-fidelity not installed. Please run: pip install torch-fidelity\n"
            f"Original error: {e}"
        )

    kwargs = dict(
        input1=path_real,
        input2=path_gen,
        cuda=(device.startswith("cuda")),
        isc=False,
        fid=True,
        kid=True,
        verbose=False,
    )
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    metrics = calculate_metrics(**kwargs)
    return {
        "FID": float(metrics["frechet_inception_distance"]),
        "KID_mean": float(metrics["kernel_inception_distance_mean"]),
        "KID_std": float(metrics["kernel_inception_distance_std"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_images", required=True, help="真实图像目录（例如 ISIC test/Images）")
    ap.add_argument("--real_masks", required=False, default=None,
                    help="真实mask目录（如果要算 fg/bg FID 必须提供；ISIC为 test/Masks）")
    ap.add_argument("--gen_images", required=True, help="生成图像目录（你的输出/images）")
    ap.add_argument("--gen_masks", required=False, default=None,
                    help="生成mask目录（如果要算 fg/bg FID 必须提供；你的输出/masks）")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--work_dir", default="/data/zjy_work/data_txt/gen_eval_tmp",
                    help="中间masked图像输出目录")
    ap.add_argument("--limit", type=int, default=0, help="可选：只取前N张做快速评估，0=全量")
    ap.add_argument("--resize", type=int, default=299, help="将输入图像 resize 到 NxN（0=不resize，改用 batch_size=1）")
    ap.add_argument("--out_json", default="/data/zjy_work/data_txt/gen_quality_metrics.json")
    args = ap.parse_args()

    real_images = Path(args.real_images)
    gen_images = Path(args.gen_images)
    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)

    results = {}

    use_resize = args.resize > 0
    batch_size_fid = None if use_resize else 1

    # prepare resized folders for "full" images
    if use_resize:
        real_resized = work_dir / "real_resized"
        gen_resized = work_dir / "gen_resized"
        for p in [real_resized, gen_resized]:
            if p.exists():
                shutil.rmtree(p)
        prepare_resized_folder(real_images, real_resized, size=args.resize, limit=args.limit)
        prepare_resized_folder(gen_images, gen_resized, size=args.resize, limit=args.limit)
        real_for_fid = str(real_resized)
        gen_for_fid = str(gen_resized)
    else:
        real_for_fid = str(real_images)
        gen_for_fid = str(gen_images)

    # 1) 全图 FID/KID
    results["full"] = calc_fid_kid(real_for_fid, gen_for_fid, device=args.device, batch_size=batch_size_fid)

    # 2) fg/bg FID/KID（需要两边都有 masks）
    if args.real_masks is not None and args.gen_masks is not None:
        real_masks = Path(args.real_masks)
        gen_masks = Path(args.gen_masks)

        real_fg = work_dir / "real_fg"
        real_bg = work_dir / "real_bg"
        gen_fg = work_dir / "gen_fg"
        gen_bg = work_dir / "gen_bg"

        for p in [real_fg, real_bg, gen_fg, gen_bg]:
            if p.exists():
                shutil.rmtree(p)

        n1 = build_masked_folder(real_images, real_masks, real_fg, mode="fg", limit=args.limit)
        n2 = build_masked_folder(real_images, real_masks, real_bg, mode="bg", limit=args.limit)
        n3 = build_masked_folder(gen_images, gen_masks, gen_fg, mode="fg", limit=args.limit)
        n4 = build_masked_folder(gen_images, gen_masks, gen_bg, mode="bg", limit=args.limit)

        results["masked_info"] = {"real_fg": n1, "real_bg": n2, "gen_fg": n3, "gen_bg": n4}

        if use_resize:
            real_fg_resized = work_dir / "real_fg_resized"
            real_bg_resized = work_dir / "real_bg_resized"
            gen_fg_resized = work_dir / "gen_fg_resized"
            gen_bg_resized = work_dir / "gen_bg_resized"
            for p in [real_fg_resized, real_bg_resized, gen_fg_resized, gen_bg_resized]:
                if p.exists():
                    shutil.rmtree(p)

            prepare_resized_folder(real_fg, real_fg_resized, size=args.resize, limit=args.limit)
            prepare_resized_folder(real_bg, real_bg_resized, size=args.resize, limit=args.limit)
            prepare_resized_folder(gen_fg, gen_fg_resized, size=args.resize, limit=args.limit)
            prepare_resized_folder(gen_bg, gen_bg_resized, size=args.resize, limit=args.limit)

            results["fg"] = calc_fid_kid(str(real_fg_resized), str(gen_fg_resized), device=args.device, batch_size=batch_size_fid)
            results["bg"] = calc_fid_kid(str(real_bg_resized), str(gen_bg_resized), device=args.device, batch_size=batch_size_fid)
        else:
            results["fg"] = calc_fid_kid(str(real_fg), str(gen_fg), device=args.device, batch_size=batch_size_fid)
            results["bg"] = calc_fid_kid(str(real_bg), str(gen_bg), device=args.device, batch_size=batch_size_fid)
    else:
        results["fg_bg"] = "SKIPPED (need --real_masks and --gen_masks)"

    out_json = Path(args.out_json)
    ensure_dir(out_json.parent)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("[DONE] metrics saved to:", out_json)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
