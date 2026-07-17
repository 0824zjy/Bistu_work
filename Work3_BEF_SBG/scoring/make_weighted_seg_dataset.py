import os
import json
import math
import argparse
from typing import List, Optional


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def finite_float(x, default=None):
    try:
        y = float(x)
    except Exception:
        return default

    if not math.isfinite(y):
        return default

    return float(y)


def valid_weight(x):
    w = finite_float(x, default=None)
    if w is None:
        return None

    if w <= 0.0:
        return None

    return float(max(0.0, min(w, 1.5)))


def write_jsonl(f, item):
    """
    allow_nan=False prevents writing invalid JSON containing NaN.
    """
    f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")

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


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    p = find_existing_file(mask_dir, stem + "_segmentation", MASK_EXTS)
    if p is not None:
        return p

    p = find_existing_file(mask_dir, stem, MASK_EXTS)
    if p is not None:
        return p

    return None

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_image_dir", type=str, required=True)
    parser.add_argument("--real_mask_dir", type=str, required=True)
    parser.add_argument("--real_list_txt", type=str, required=True)
    parser.add_argument("--gen_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)

    # Control synthetic-to-real ratio.
    # For 5% low-label setting, synthetic samples should not dominate too much.
    # Recommended:
    #   1.0: synthetic <= real
    #   2.0: synthetic <= 2 * real
    #   -1 : no limit
    parser.add_argument("--synthetic_ratio", type=float, default=2.0)

    # If true, keep highest-weight synthetic samples first.
    parser.add_argument("--sort_gen_by_weight", action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    real_ids = read_id_list(args.real_list_txt)

    real_items = []
    real_missing_image = 0
    real_missing_mask = 0

    for stem in real_ids:
        image_path = find_existing_file(args.real_image_dir, stem, IMG_EXTS)
        mask_path = find_mask(args.real_mask_dir, stem)

        if image_path is None:
            real_missing_image += 1
            continue

        if mask_path is None:
            real_missing_mask += 1
            continue

        real_items.append({
            "image": image_path,
            "mask": mask_path,
            "weight": 1.0,
            "source": "real",
        })

    real_count = len(real_items)

    gen_items = []
    bad_gen_weight = 0
    missing_gen_file = 0
    bad_json = 0

    if os.path.exists(args.gen_jsonl):
        with open(args.gen_jsonl, "r", encoding="utf-8") as gf:
            for line_no, line in enumerate(gf, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except Exception as e:
                    bad_json += 1
                    print(f"[WARN] bad json line in gen_jsonl line={line_no}: {e}")
                    continue

                image_path = item.get("image", None)
                mask_path = item.get("mask", None)

                if image_path is None or mask_path is None:
                    missing_gen_file += 1
                    continue

                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    missing_gen_file += 1
                    continue

                weight = valid_weight(item.get("weight", 1.0))
                if weight is None:
                    bad_gen_weight += 1
                    continue

                gen_items.append({
                    "image": image_path,
                    "mask": mask_path,
                    "weight": float(weight),
                    "source": item.get("source", "bef_sbg"),
                })
    else:
        print(f"[WARN] gen_jsonl does not exist: {args.gen_jsonl}")

    # Deduplicate image-mask pairs.
    seen = set()
    gen_dedup = []
    duplicate_gen = 0

    for item in gen_items:
        key = (item["image"], item["mask"])
        if key in seen:
            duplicate_gen += 1
            continue
        seen.add(key)
        gen_dedup.append(item)

    gen_items = gen_dedup

    # Keep highest-weight generated samples first.
    if args.sort_gen_by_weight:
        gen_items = sorted(
            gen_items,
            key=lambda x: float(x.get("weight", 0.0)),
            reverse=True,
        )

    # Limit synthetic-to-real ratio.
    if float(args.synthetic_ratio) >= 0.0:
        max_gen = int(round(real_count * float(args.synthetic_ratio)))
        gen_items = gen_items[:max_gen]

    gen_count = len(gen_items)

    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for item in real_items:
            write_jsonl(out_f, item)

        for item in gen_items:
            write_jsonl(out_f, item)

    total_count = real_count + gen_count

    print("[DONE] weighted segmentation dataset jsonl generated.")
    print(f"  out_jsonl           = {args.out_jsonl}")
    print(f"  real_count          = {real_count}")
    print(f"  gen_count           = {gen_count}")
    print(f"  total_count         = {total_count}")
    print(f"  synthetic_ratio     = {args.synthetic_ratio}")
    print(f"  real_missing_image  = {real_missing_image}")
    print(f"  real_missing_mask   = {real_missing_mask}")
    print(f"  bad_json            = {bad_json}")
    print(f"  bad_gen_weight      = {bad_gen_weight}")
    print(f"  missing_gen_file    = {missing_gen_file}")
    print(f"  duplicate_gen       = {duplicate_gen}")
if __name__ == "__main__":
    main()
