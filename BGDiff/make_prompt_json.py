import os
import json
import argparse
import glob

def build_index(mask_dir):
    """
    index by stem (filename without extension)
    """
    masks = glob.glob(os.path.join(mask_dir, "*"))
    idx = {}
    for m in masks:
        stem = os.path.splitext(os.path.basename(m))[0]
        idx[stem] = m
    return idx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--prompt", type=str, default="medical image")
    p.add_argument("--img_exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff")
    p.add_argument("--mask_exts", type=str, default=".png,.jpg,.jpeg,.bmp,.tif,.tiff")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    img_exts = tuple([e.strip().lower() for e in args.img_exts.split(",")])
    mask_exts = tuple([e.strip().lower() for e in args.mask_exts.split(",")])

    # list images
    imgs = []
    for pth in glob.glob(os.path.join(args.img_dir, "*")):
        if os.path.splitext(pth)[1].lower() in img_exts:
            imgs.append(pth)
    imgs = sorted(imgs)

    # build mask index
    mask_index = build_index(args.mask_dir)

    missing = 0
    written = 0
    with open(args.out, "w") as f:
        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]

            # direct match by stem
            mask_path = None
            if stem in mask_index:
                mask_path = mask_index[stem]
            else:
                # try relaxed matching: sometimes mask has suffix/prefix
                # scan keys containing stem
                for k, v in mask_index.items():
                    if k == stem or k.startswith(stem) or stem.startswith(k):
                        mask_path = v
                        break

            if mask_path is None or os.path.splitext(mask_path)[1].lower() not in mask_exts:
                missing += 1
                continue

            item = {
                "source": mask_path,
                "target": img_path,
                "prompt": args.prompt
            }
            f.write(json.dumps(item) + "\n")
            written += 1

    print(f"[DONE] out={args.out} written={written} missing_mask={missing}")

if __name__ == "__main__":
    main()
