import os
import json
import argparse
from typing import List, Optional


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

    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--adaptive_prior_dir", type=str, required=True)
    parser.add_argument("--difficulty_dir", type=str, default=None)
    parser.add_argument("--list_txt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="dermoscopic image")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ids = read_id_list(args.list_txt)

    written = 0
    missing_image = 0
    missing_mask = 0
    missing_prior = 0
    missing_difficulty = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for stem in ids:
            img_path = find_existing_file(args.img_dir, stem, IMG_EXTS)
            mask_path = find_mask(args.mask_dir, stem)
            prior_path = find_existing_file(args.adaptive_prior_dir, stem, MASK_EXTS)

            if img_path is None:
                missing_image += 1
                continue

            if mask_path is None:
                missing_mask += 1
                continue

            if prior_path is None:
                missing_prior += 1
                continue

            item = {
                "source": mask_path,
                "target": img_path,
                "prompt": args.prompt,
                "boundary_prior": prior_path,
            }

            if args.difficulty_dir is not None and args.difficulty_dir != "":
                difficulty_path = find_existing_file(args.difficulty_dir, stem, MASK_EXTS)
                if difficulty_path is not None:
                    item["difficulty"] = difficulty_path
                else:
                    missing_difficulty += 1

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print("[DONE] BEF prompt json generated.")
    print(f"  out                = {args.out}")
    print(f"  ids                = {len(ids)}")
    print(f"  written            = {written}")
    print(f"  missing_image      = {missing_image}")
    print(f"  missing_mask       = {missing_mask}")
    print(f"  missing_prior      = {missing_prior}")
    print(f"  missing_difficulty = {missing_difficulty}")


if __name__ == "__main__":
    main()
