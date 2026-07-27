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
    if w is None or w <= 0.0:
        return None
    return float(max(0.0, min(w, 1.5)))


def write_jsonl(f, item):
    f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def read_id_list(list_txt: str) -> List[str]:
    ids = []
    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
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
    return find_existing_file(mask_dir, stem, MASK_EXTS)


def optional_existing_path(item, key):
    path = item.get(key, "")
    if path and os.path.exists(path):
        return path
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_image_dir", type=str, required=True)
    parser.add_argument("--real_mask_dir", type=str, required=True)
    parser.add_argument("--real_list_txt", type=str, required=True)
    parser.add_argument("--gen_jsonl", type=str, required=True)
    parser.add_argument("--pseudo_jsonl", type=str, default="")
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--synthetic_ratio", type=float, default=2.0)
    parser.add_argument("--pseudo_ratio", type=float, default=2.0)
    parser.add_argument("--sort_gen_by_weight", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    real_items = []
    real_missing_image = 0
    real_missing_mask = 0
    for stem in read_id_list(args.real_list_txt):
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
            "reliability": "",
            "hardness": "",
            "boundary_prior": "",
            "weight": 1.0,
            "source": "real",
        })

    real_count = len(real_items)
    gen_items = []
    bad_gen_weight = 0
    missing_gen_file = 0
    missing_reliability = 0
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
                    print(f"[WARN] bad json line={line_no}: {e}")
                    continue

                image_path = item.get("image", "")
                mask_path = item.get("mask", "")
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    missing_gen_file += 1
                    continue

                weight = valid_weight(item.get("weight", 1.0))
                if weight is None:
                    bad_gen_weight += 1
                    continue

                reliability_path = optional_existing_path(item, "reliability")
                hardness_path = optional_existing_path(item, "hardness")
                prior_path = optional_existing_path(item, "boundary_prior")
                if not reliability_path:
                    missing_reliability += 1

                gen_items.append({
                    "image": image_path,
                    "mask": mask_path,
                    "reliability": reliability_path,
                    "hardness": hardness_path,
                    "boundary_prior": prior_path,
                    "weight": float(weight),
                    "source": item.get("source", "bef_sbg"),
                })
    else:
        print(f"[WARN] gen_jsonl does not exist: {args.gen_jsonl}")

    seen = set()
    dedup = []
    duplicate_gen = 0
    for item in gen_items:
        key = (item["image"], item["mask"])
        if key in seen:
            duplicate_gen += 1
            continue
        seen.add(key)
        dedup.append(item)
    gen_items = dedup

    if args.sort_gen_by_weight:
        gen_items = sorted(
            gen_items,
            key=lambda x: float(x.get("weight", 0.0)),
            reverse=True,
        )

    if float(args.synthetic_ratio) >= 0.0:
        max_gen = int(round(real_count * float(args.synthetic_ratio)))
        gen_items = gen_items[:max_gen]

    pseudo_items = []
    bad_pseudo = 0
    if args.pseudo_jsonl and os.path.exists(args.pseudo_jsonl):
        with open(args.pseudo_jsonl, "r", encoding="utf-8") as pf:
            for line_no, line in enumerate(pf, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    bad_pseudo += 1
                    continue
                image_path = item.get("image", "")
                mask_path = item.get("mask", "")
                weight = valid_weight(item.get("weight", 1.0))
                if (
                    not image_path
                    or not mask_path
                    or not os.path.exists(image_path)
                    or not os.path.exists(mask_path)
                    or weight is None
                ):
                    bad_pseudo += 1
                    continue
                pseudo_items.append({
                    "image": image_path,
                    "mask": mask_path,
                    "reliability": optional_existing_path(item, "reliability"),
                    "hardness": optional_existing_path(item, "hardness"),
                    "boundary_prior": optional_existing_path(item, "boundary_prior"),
                    "weight": float(weight),
                    "source": "pseudo_real",
                })

    pseudo_items = sorted(
        pseudo_items,
        key=lambda x: float(x.get("weight", 0.0)),
        reverse=True,
    )
    if float(args.pseudo_ratio) >= 0.0:
        max_pseudo = int(round(real_count * float(args.pseudo_ratio)))
        pseudo_items = pseudo_items[:max_pseudo]

    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for item in real_items:
            write_jsonl(out_f, item)
        for item in gen_items:
            write_jsonl(out_f, item)
        for item in pseudo_items:
            write_jsonl(out_f, item)

    print("[DONE] weighted segmentation dataset jsonl generated.")
    print(f"  out_jsonl           = {args.out_jsonl}")
    print(f"  real_count          = {len(real_items)}")
    print(f"  gen_count           = {len(gen_items)}")
    print(f"  pseudo_count        = {len(pseudo_items)}")
    print(f"  total_count         = {len(real_items) + len(gen_items) + len(pseudo_items)}")
    print(f"  synthetic_ratio     = {args.synthetic_ratio}")
    print(f"  pseudo_ratio        = {args.pseudo_ratio}")
    print(f"  real_missing_image  = {real_missing_image}")
    print(f"  real_missing_mask   = {real_missing_mask}")
    print(f"  bad_json            = {bad_json}")
    print(f"  bad_gen_weight      = {bad_gen_weight}")
    print(f"  missing_gen_file    = {missing_gen_file}")
    print(f"  missing_reliability = {missing_reliability}")
    print(f"  duplicate_gen       = {duplicate_gen}")
    print(f"  bad_pseudo          = {bad_pseudo}")


if __name__ == "__main__":
    main()
