import os
import csv
import json
import math
import random
import argparse
from typing import Dict, List, Optional, Tuple


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
LEVELS = ("easy", "medium", "hard")

# Feedback CSV difficulty-column aliases, in priority order.
DIFFICULTY_COLUMN_CANDIDATES = (
    "image_difficulty",
    "mean_difficulty",
)


def normalize_stem(value: str) -> str:
    stem = os.path.splitext(os.path.basename(str(value).strip()))[0]

    if stem.endswith("_segmentation"):
        stem = stem[: -len("_segmentation")]

    return stem


def read_id_list(list_txt: str) -> List[str]:
    ids: List[str] = []

    with open(list_txt, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()

            if not value or value.startswith("#"):
                continue

            ids.append(normalize_stem(value))

    return ids


def find_existing_file(
    dir_path: Optional[str],
    stem: str,
    exts: List[str],
) -> Optional[str]:
    if not dir_path:
        return None

    for ext in exts:
        path = os.path.join(dir_path, stem + ext)

        if os.path.isfile(path):
            return path

    return None


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    path = find_existing_file(
        mask_dir,
        stem + "_segmentation",
        MASK_EXTS,
    )

    if path is not None:
        return path

    return find_existing_file(mask_dir, stem, MASK_EXTS)


def read_image_difficulty_csv(csv_path: str) -> Dict[str, float]:
    """
    Read image-level difficulty values from a feedback summary CSV.

    Expected image identifier column:
        image_name

    Supported difficulty columns, in priority order:
        image_difficulty
        mean_difficulty

    The returned dictionary uses normalized image stems as keys.
    Difficulty values are clipped to [0, 1].
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Feedback CSV does not exist: {csv_path}"
        )

    values: Dict[str, float] = {}

    # utf-8-sig also handles ordinary UTF-8 files and removes a possible BOM.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        fieldname_set = set(fieldnames)

        if not fieldnames:
            raise RuntimeError(
                f"Feedback CSV has no header: {csv_path}"
            )

        if "image_name" not in fieldname_set:
            raise RuntimeError(
                f"Missing column 'image_name' in feedback CSV {csv_path}. "
                f"Existing columns: {fieldnames}"
            )

        difficulty_column: Optional[str] = None

        for candidate in DIFFICULTY_COLUMN_CANDIDATES:
            if candidate in fieldname_set:
                difficulty_column = candidate
                break

        if difficulty_column is None:
            raise RuntimeError(
                f"Missing difficulty column in feedback CSV {csv_path}. "
                f"Expected one of: {list(DIFFICULTY_COLUMN_CANDIDATES)}. "
                f"Existing columns: {fieldnames}"
            )

        if difficulty_column == "image_difficulty":
            print(
                "[INFO] Using feedback CSV column "
                "'image_difficulty' as image-level difficulty."
            )
        else:
            print(
                f"[WARN] Feedback CSV does not contain "
                f"'image_difficulty'; using '{difficulty_column}' instead."
            )

        invalid_image_name = 0
        invalid_difficulty = 0
        duplicate_image_name = 0

        for row_number, row in enumerate(reader, start=2):
            stem = normalize_stem(row.get("image_name", ""))

            if not stem:
                invalid_image_name += 1
                continue

            raw_value = row.get(difficulty_column, "")

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid_difficulty += 1
                continue

            if not math.isfinite(value):
                invalid_difficulty += 1
                continue

            # Clamp unexpected values to the normalized [0, 1] interval.
            value = float(min(max(value, 0.0), 1.0))

            if stem in values:
                duplicate_image_name += 1

            values[stem] = value

        if invalid_image_name > 0:
            print(
                f"[WARN] Ignored {invalid_image_name} rows with an "
                "empty or invalid image_name."
            )

        if invalid_difficulty > 0:
            print(
                f"[WARN] Ignored {invalid_difficulty} rows with an "
                f"invalid '{difficulty_column}' value."
            )

        if duplicate_image_name > 0:
            print(
                f"[WARN] Found {duplicate_image_name} duplicate image names; "
                "the last valid value was used."
            )

    if not values:
        raise RuntimeError(
            f"No valid difficulty values found in {csv_path}. "
            f"Difficulty column used: {difficulty_column}"
        )

    print(f"[INFO] Loaded image difficulty values: {len(values)}")
    print(f"[INFO] Difficulty source column: {difficulty_column}")

    return values


def validate_quantiles(
    easy_quantile: float,
    hard_quantile: float,
) -> None:
    if not (0.0 < easy_quantile < hard_quantile < 1.0):
        raise ValueError(
            "Expected 0 < easy_quantile < hard_quantile < 1, got "
            f"{easy_quantile}, {hard_quantile}"
        )


def normalized_ratios(
    easy_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
) -> Dict[str, float]:
    raw = {
        "easy": max(0.0, float(easy_ratio)),
        "medium": max(0.0, float(medium_ratio)),
        "hard": max(0.0, float(hard_ratio)),
    }

    total = sum(raw.values())

    if total <= 1e-12:
        raise ValueError(
            "easy_ratio + medium_ratio + hard_ratio must be > 0"
        )

    return {
        level: value / total
        for level, value in raw.items()
    }


def allocate_counts(
    total: int,
    ratios: Dict[str, float],
) -> Dict[str, int]:
    total = max(0, int(total))

    raw = {
        level: total * ratios[level]
        for level in LEVELS
    }

    counts = {
        level: int(math.floor(raw[level]))
        for level in LEVELS
    }

    remainder = total - sum(counts.values())

    order = sorted(
        LEVELS,
        key=lambda level: (
            raw[level] - counts[level],
            -LEVELS.index(level),
        ),
        reverse=True,
    )

    for level in order[:remainder]:
        counts[level] += 1

    return counts


def natural_bucket_counts(
    total: int,
    easy_quantile: float,
    hard_quantile: float,
) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    if total == 1:
        return 0, 1, 0

    if total == 2:
        return 1, 0, 1

    n_easy = max(
        1,
        int(round(total * easy_quantile)),
    )

    n_hard = max(
        1,
        int(round(total * (1.0 - hard_quantile))),
    )

    while n_easy + n_hard >= total:
        if n_easy >= n_hard and n_easy > 1:
            n_easy -= 1
        elif n_hard > 1:
            n_hard -= 1
        else:
            break

    n_medium = total - n_easy - n_hard

    return n_easy, n_medium, n_hard


def assign_difficulty_levels(
    valid: List[dict],
    easy_quantile: float,
    hard_quantile: float,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    ranked = sorted(
        valid,
        key=lambda item: (
            float(item["image_difficulty"]),
            str(item["source_stem"]),
        ),
    )

    n_total = len(ranked)

    n_easy, _, n_hard = natural_bucket_counts(
        total=n_total,
        easy_quantile=easy_quantile,
        hard_quantile=hard_quantile,
    )

    buckets: Dict[str, List[dict]] = {
        level: []
        for level in LEVELS
    }

    ranked_with_level: List[dict] = []

    for rank, item in enumerate(ranked):
        if rank < n_easy:
            level = "easy"
        elif rank >= n_total - n_hard:
            level = "hard"
        else:
            level = "medium"

        record = dict(item)
        record["difficulty_level"] = level
        record["difficulty_rank"] = int(rank)

        record["difficulty_rank_fraction"] = (
            0.0
            if n_total <= 1
            else float(rank) / float(n_total - 1)
        )

        ranked_with_level.append(record)
        buckets[level].append(record)

    return ranked_with_level, buckets


def export_item(
    item: dict,
    generation_index: Optional[int] = None,
    per_level_index: Optional[int] = None,
) -> dict:
    output = {
        "source": item["source"],
        "target": item["target"],
        "prompt": item["prompt"],
        "boundary_prior": item["boundary_prior"],
        "source_stem": item["source_stem"],
        "image_difficulty": float(item["image_difficulty"]),
        "difficulty_level": item["difficulty_level"],
        "difficulty_rank": int(item["difficulty_rank"]),
        "difficulty_rank_fraction": float(
            item["difficulty_rank_fraction"]
        ),
    }

    if item.get("difficulty"):
        output["difficulty"] = item["difficulty"]

    if generation_index is not None:
        output["generation_index"] = int(generation_index)

    if per_level_index is not None:
        output["per_level_generation_index"] = int(
            per_level_index
        )

    return output


def write_jsonl(
    path: str,
    records: List[dict],
) -> None:
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            )

            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--adaptive_prior_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--difficulty_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--feedback_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--list_txt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train_out",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--sample_out",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="dermoscopic image",
    )

    parser.add_argument(
        "--sample_multiplier",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--easy_quantile",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--hard_quantile",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--easy_ratio",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--medium_ratio",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--hard_ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--require_difficulty_map",
        action="store_true",
        help=(
            "Reject records whose per-pixel difficulty map "
            "is missing."
        ),
    )

    args = parser.parse_args()

    validate_quantiles(
        args.easy_quantile,
        args.hard_quantile,
    )

    ratios = normalized_ratios(
        args.easy_ratio,
        args.medium_ratio,
        args.hard_ratio,
    )

    if args.sample_multiplier <= 0.0:
        raise ValueError(
            "--sample_multiplier must be > 0"
        )

    ids = read_id_list(args.list_txt)

    if not ids:
        raise RuntimeError(
            f"No valid image IDs found in list file: "
            f"{args.list_txt}"
        )

    difficulty_map = read_image_difficulty_csv(
        args.feedback_csv
    )

    valid: List[dict] = []

    missing_image = 0
    missing_mask = 0
    missing_prior = 0
    missing_difficulty_map = 0
    missing_image_difficulty = 0

    for stem in ids:
        image_path = find_existing_file(
            args.img_dir,
            stem,
            IMG_EXTS,
        )

        mask_path = find_mask(
            args.mask_dir,
            stem,
        )

        prior_path = find_existing_file(
            args.adaptive_prior_dir,
            stem,
            MASK_EXTS,
        )

        if image_path is None:
            missing_image += 1
            continue

        if mask_path is None:
            missing_mask += 1
            continue

        if prior_path is None:
            missing_prior += 1
            continue

        if stem not in difficulty_map:
            missing_image_difficulty += 1
            continue

        difficulty_path = find_existing_file(
            args.difficulty_dir,
            stem,
            MASK_EXTS,
        )

        if difficulty_path is None:
            missing_difficulty_map += 1

            if args.require_difficulty_map:
                continue

        record = {
            "source_stem": stem,
            "source": mask_path,
            "target": image_path,
            "prompt": args.prompt,
            "boundary_prior": prior_path,
            "image_difficulty": float(
                difficulty_map[stem]
            ),
        }

        if difficulty_path is not None:
            record["difficulty"] = difficulty_path

        valid.append(record)

    if not valid:
        raise RuntimeError(
            "No valid samples were collected. Check list_txt, "
            "feedback_csv, image/mask/prior directories, and "
            "difficulty maps."
        )

    ranked, buckets = assign_difficulty_levels(
        valid=valid,
        easy_quantile=args.easy_quantile,
        hard_quantile=args.hard_quantile,
    )

    train_records = [
        export_item(item)
        for item in ranked
    ]

    sample_total = max(
        1,
        int(
            round(
                len(ranked) * args.sample_multiplier
            )
        ),
    )

    target_counts = allocate_counts(
        sample_total,
        ratios,
    )

    rng = random.Random(int(args.seed))

    sample_records: List[dict] = []
    global_generation_index = 0

    for level in LEVELS:
        candidates = list(buckets[level])
        target_count = int(target_counts[level])

        if target_count <= 0:
            continue

        if not candidates:
            raise RuntimeError(
                f"Difficulty bucket '{level}' is empty but "
                f"target count is {target_count}. Adjust "
                "easy_quantile/hard_quantile or sampling ratios."
            )

        rng.shuffle(candidates)

        for per_level_index in range(target_count):
            source_item = candidates[
                per_level_index % len(candidates)
            ]

            output = export_item(
                source_item,
                generation_index=global_generation_index,
                per_level_index=per_level_index,
            )

            output["sample_uid"] = (
                f"{level}_{per_level_index:06d}_"
                f"{source_item['source_stem']}"
            )

            sample_records.append(output)
            global_generation_index += 1

    rng.shuffle(sample_records)

    # Re-index after shuffle so each JSONL line has a deterministic
    # global generation index.
    for index, record in enumerate(sample_records):
        record["generation_index"] = int(index)

    write_jsonl(
        args.train_out,
        train_records,
    )

    write_jsonl(
        args.sample_out,
        sample_records,
    )

    print("[DONE] BEF train/sample prompt JSON generated.")
    print(f"  list_txt                 = {args.list_txt}")
    print(f"  feedback_csv             = {args.feedback_csv}")
    print(f"  valid                    = {len(valid)}")
    print(f"  train_out                = {args.train_out}")
    print(f"  train_records            = {len(train_records)}")
    print(f"  sample_out               = {args.sample_out}")
    print(f"  sample_records           = {len(sample_records)}")
    print(f"  natural_easy             = {len(buckets['easy'])}")
    print(f"  natural_medium           = {len(buckets['medium'])}")
    print(f"  natural_hard             = {len(buckets['hard'])}")
    print(f"  target_ratios            = {ratios}")
    print(f"  target_counts            = {target_counts}")
    print(f"  missing_image            = {missing_image}")
    print(f"  missing_mask             = {missing_mask}")
    print(f"  missing_prior            = {missing_prior}")
    print(
        f"  missing_difficulty_map   = "
        f"{missing_difficulty_map}"
    )
    print(
        f"  missing_image_difficulty = "
        f"{missing_image_difficulty}"
    )


if __name__ == "__main__":
    main()
