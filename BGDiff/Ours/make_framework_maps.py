#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python: pip install opencv-python")


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def get_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def load_gray(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def otsu_threshold(gray):
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr) / 255.0


def largest_component(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num <= 1:
        return mask_u8.astype(np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = (labels == largest_idx).astype(np.float32)
    return out


def extract_mask(gray, foreground="dark"):
    """
    foreground:
        dark   : lesion/object is darker than background
        bright : lesion/object is brighter than background
    """
    gray = normalize01(gray)
    thr = otsu_threshold(gray)

    if foreground == "bright":
        mask = (gray > thr).astype(np.float32)
    else:
        mask = (gray < thr).astype(np.float32)

    mask = largest_component(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    return mask.astype(np.float32)


def make_soft_boundary_prior(mask, radius=24, tau=6.0, boundary_kernel=3):
    """
    Similar in spirit to ControlLDM._soft_boundary_prior_from_mask().
    Input:
        mask: [H,W], binary {0,1}
    Output:
        soft boundary prior: [H,W], [0,1]
    """
    mask_u8 = (mask > 0.5).astype(np.uint8)

    k = max(3, int(boundary_kernel))
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)

    dilation = cv2.dilate(mask_u8, kernel, iterations=1)
    erosion = cv2.erode(mask_u8, kernel, iterations=1)
    hard_boundary = np.clip(dilation - erosion, 0, 1).astype(np.float32)

    soft = hard_boundary.copy()
    prev = hard_boundary.copy()

    radius = max(1, int(radius))
    tau = max(1e-6, float(tau))

    for r in range(1, radius + 1):
        kr = 2 * r + 1
        kernel_r = np.ones((kr, kr), np.uint8)

        dilated = cv2.dilate(hard_boundary.astype(np.uint8), kernel_r, iterations=1)
        dilated = dilated.astype(np.float32)

        shell = np.clip(dilated - prev, 0.0, 1.0)
        weight = math.exp(-float(r) / tau)

        soft = np.maximum(soft, shell * weight)
        prev = dilated

    soft = cv2.GaussianBlur(soft, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0)
    soft = normalize01(soft)

    return soft.astype(np.float32)


def make_scale_shift_maps(soft_prior, mask):
    """
    Figure-style visualization of gamma_i and beta_i.

    gamma_i:
        boundary-strength-like scale map, high around soft boundary.

    beta_i:
        signed local shift map, positive outside boundary band,
        negative inside boundary band.

    Note:
        If you want true learned gamma/beta, load a Stage2 checkpoint and run
        BoundarySpatialModulator. This function is for clean paper-figure assets.
    """
    p = soft_prior.astype(np.float32)
    p = cv2.GaussianBlur(p, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)
    p = normalize01(p)

    gamma = np.tanh(2.4 * p)
    gamma = normalize01(gamma)

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)
    mask_blur = np.clip(mask_blur, 0.0, 1.0)

    beta = p * (1.0 - 2.0 * mask_blur)
    beta = np.tanh(2.4 * beta)
    beta = np.clip(beta, -1.0, 1.0)

    return gamma.astype(np.float32), beta.astype(np.float32)


def resize_map(x, size):
    img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    img = img.resize((size, size), get_resample())
    return np.asarray(img).astype(np.float32) / 255.0


def save_gray(path, x):
    x = normalize01(x)
    img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    img.save(path)


def colorize_teal(x):
    x = normalize01(x)[..., None]
    c0 = np.array([248, 252, 253], dtype=np.float32)
    c1 = np.array([116, 196, 196], dtype=np.float32)
    rgb = c0 * (1.0 - x) + c1 * x
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize_orange_boundary(x):
    x = normalize01(x)[..., None]
    c0 = np.array([255, 255, 255], dtype=np.float32)
    c1 = np.array([244, 146, 74], dtype=np.float32)
    rgb = c0 * (1.0 - x) + c1 * x
    return np.clip(rgb, 0, 255).astype(np.uint8)


def colorize_beta(beta):
    beta = np.clip(beta, -1.0, 1.0)
    mag = np.abs(beta)[..., None]

    zero = np.array([248, 248, 250], dtype=np.float32)
    pos = np.array([178, 143, 210], dtype=np.float32)
    neg = np.array([126, 170, 214], dtype=np.float32)

    color = np.where(beta[..., None] >= 0, pos, neg)
    rgb = zero * (1.0 - mag) + color * mag
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_rgb(path, rgb):
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def load_font(size=28):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def draw_card(content_rgb, title, out_path, border=(120, 170, 190)):
    """
    Save a rounded academic-style card with label.
    """
    content = Image.fromarray(content_rgb).convert("RGB")

    W, H = 420, 500
    pad = 38
    title_h = 62
    img_size = 330

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle(
        [10, 10, W - 10, H - 10],
        radius=24,
        outline=border,
        width=3,
        fill=(252, 254, 255),
    )

    font = load_font(28)
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = draw.textlength(title, font=font)

    draw.text(((W - tw) / 2, 28), title, fill=(28, 48, 78), font=font)

    content = content.resize((img_size, img_size), get_resample())
    x0 = (W - img_size) // 2
    y0 = title_h + pad

    draw.rounded_rectangle(
        [x0 - 6, y0 - 6, x0 + img_size + 6, y0 + img_size + 6],
        radius=20,
        outline=(220, 226, 232),
        width=2,
        fill=(255, 255, 255),
    )

    canvas.paste(content, (x0, y0))
    canvas.save(out_path)


def make_decoder_feature_cube(feature_map, out_path):
    """
    Draw a clean vector-style decoder feature h_i cube.
    This is for the architecture diagram, not a real hidden activation tensor.
    """
    fmap = resize_map(feature_map, 8)
    fmap = normalize01(fmap)

    W, H = 500, 420
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    border = (122, 93, 154)
    face = (229, 218, 242)
    top = (239, 232, 248)
    side = (214, 200, 232)

    x0, y0 = 150, 110
    cell = 28
    n = 6
    depth_x, depth_y = 52, -40

    front = [
        (x0, y0),
        (x0 + n * cell, y0),
        (x0 + n * cell, y0 + n * cell),
        (x0, y0 + n * cell),
    ]
    top_face = [
        (x0, y0),
        (x0 + depth_x, y0 + depth_y),
        (x0 + n * cell + depth_x, y0 + depth_y),
        (x0 + n * cell, y0),
    ]
    side_face = [
        (x0 + n * cell, y0),
        (x0 + n * cell + depth_x, y0 + depth_y),
        (x0 + n * cell + depth_x, y0 + n * cell + depth_y),
        (x0 + n * cell, y0 + n * cell),
    ]

    draw.polygon(top_face, fill=top, outline=border)
    draw.polygon(side_face, fill=side, outline=border)
    draw.polygon(front, fill=face, outline=border)

    for i in range(n):
        for j in range(n):
            v = float(fmap[min(i, fmap.shape[0] - 1), min(j, fmap.shape[1] - 1)])
            base = np.array([229, 218, 242], dtype=np.float32)
            dark = np.array([158, 128, 192], dtype=np.float32)
            color = base * (1.0 - 0.65 * v) + dark * (0.65 * v)
            color = tuple(np.clip(color, 0, 255).astype(np.uint8).tolist())

            x = x0 + j * cell
            y = y0 + i * cell
            draw.rectangle(
                [x, y, x + cell, y + cell],
                fill=color,
                outline=border,
                width=1,
            )

    for j in range(n + 1):
        x = x0 + j * cell
        draw.line([(x, y0), (x + depth_x, y0 + depth_y)], fill=border, width=1)

    for i in range(n + 1):
        y = y0 + i * cell
        draw.line(
            [(x0 + n * cell, y), (x0 + n * cell + depth_x, y + depth_y)],
            fill=border,
            width=1,
        )

    title_font = load_font(30)
    title = "Decoder feature h_i"

    try:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = draw.textlength(title, font=title_font)

    draw.text(((W - tw) / 2, 32), title, fill=(62, 39, 92), font=title_font)

    draw.rounded_rectangle(
        [14, 14, W - 14, H - 14],
        radius=26,
        outline=(151, 121, 179),
        width=3,
    )

    canvas.save(out_path)


def make_decoder_feature_heatmap(gray, soft_prior):
    """
    Make a pseudo decoder feature map for clean visualization.
    It mimics a high-level spatial feature by mixing image structure and boundary prior.
    """
    g = normalize01(gray)
    g_blur1 = cv2.GaussianBlur(g, ksize=(0, 0), sigmaX=3.0, sigmaY=3.0)
    g_blur2 = cv2.GaussianBlur(g, ksize=(0, 0), sigmaX=8.0, sigmaY=8.0)

    feat = 0.45 * g_blur1 + 0.35 * g_blur2 + 0.20 * soft_prior
    feat = normalize01(feat)
    return feat.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="/data/zjy_work/BGDiff/asset/Train_real_1.png",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/zjy_work/BGDiff/asset",
    )
    parser.add_argument(
        "--foreground",
        type=str,
        default="dark",
        choices=["dark", "bright"],
        help="Use dark if lesion/object is black on white background; use bright if lesion/object is white.",
    )
    parser.add_argument("--map_size", type=int, default=256)
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--tau", type=float, default=6.0)
    parser.add_argument("--boundary_kernel", type=int, default=3)

    args = parser.parse_args()

    mkdir(args.out_dir)

    gray = load_gray(args.input)
    mask = extract_mask(gray, foreground=args.foreground)

    soft_prior = make_soft_boundary_prior(
        mask=mask,
        radius=args.radius,
        tau=args.tau,
        boundary_kernel=args.boundary_kernel,
    )

    gamma, beta = make_scale_shift_maps(soft_prior, mask)

    soft_prior_r = resize_map(soft_prior, args.map_size)
    gamma_r = resize_map(gamma, args.map_size)

    beta_img = Image.fromarray(np.clip((beta + 1.0) * 127.5, 0, 255).astype(np.uint8))
    beta_img = beta_img.resize((args.map_size, args.map_size), get_resample())
    beta_r = np.asarray(beta_img).astype(np.float32) / 127.5 - 1.0
    beta_r = np.clip(beta_r, -1.0, 1.0)

    decoder_feature = make_decoder_feature_heatmap(gray, soft_prior)
    decoder_feature_r = resize_map(decoder_feature, args.map_size)

    # Raw grayscale outputs
    save_gray(os.path.join(args.out_dir, "soft_boundary_prior_B_gray.png"), soft_prior_r)
    save_gray(os.path.join(args.out_dir, "scale_map_gamma_i_gray.png"), gamma_r)
    save_gray(os.path.join(args.out_dir, "shift_map_beta_i_gray.png"), (beta_r + 1.0) / 2.0)
    save_gray(os.path.join(args.out_dir, "decoder_feature_hi_gray.png"), decoder_feature_r)

    # Colored outputs
    prior_rgb = colorize_orange_boundary(soft_prior_r)
    gamma_rgb = colorize_teal(gamma_r)
    beta_rgb = colorize_beta(beta_r)
    decoder_rgb = colorize_beta(decoder_feature_r * 2.0 - 1.0)

    save_rgb(os.path.join(args.out_dir, "soft_boundary_prior_B.png"), prior_rgb)
    save_rgb(os.path.join(args.out_dir, "scale_map_gamma_i.png"), gamma_rgb)
    save_rgb(os.path.join(args.out_dir, "shift_map_beta_i.png"), beta_rgb)
    save_rgb(os.path.join(args.out_dir, "decoder_feature_hi_heatmap.png"), decoder_rgb)

    # Paper-figure cards with labels
    draw_card(
        prior_rgb,
        "Soft Boundary Prior B",
        os.path.join(args.out_dir, "soft_boundary_prior_B_card.png"),
        border=(120, 160, 210),
    )

    draw_card(
        gamma_rgb,
        "Scale map γ_i",
        os.path.join(args.out_dir, "scale_map_gamma_i_card.png"),
        border=(98, 165, 170),
    )

    draw_card(
        beta_rgb,
        "Shift map β_i",
        os.path.join(args.out_dir, "shift_map_beta_i_card.png"),
        border=(151, 121, 179),
    )

    make_decoder_feature_cube(
        decoder_feature,
        os.path.join(args.out_dir, "decoder_feature_hi_card.png"),
    )

    # Also save numeric arrays for later paper plotting
    np.save(os.path.join(args.out_dir, "soft_boundary_prior_B.npy"), soft_prior_r)
    np.save(os.path.join(args.out_dir, "scale_map_gamma_i.npy"), gamma_r)
    np.save(os.path.join(args.out_dir, "shift_map_beta_i.npy"), beta_r)
    np.save(os.path.join(args.out_dir, "decoder_feature_hi.npy"), decoder_feature_r)

    print("[OK] Generated framework assets:")
    print("  soft_boundary_prior_B.png")
    print("  scale_map_gamma_i.png")
    print("  shift_map_beta_i.png")
    print("  decoder_feature_hi_heatmap.png")
    print("  soft_boundary_prior_B_card.png")
    print("  scale_map_gamma_i_card.png")
    print("  shift_map_beta_i_card.png")
    print("  decoder_feature_hi_card.png")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

# python /data/zjy_work/BGDiff/Ours/make_framework_maps.py --input /data/zjy_work/BGDiff/asset/Train_real_1.png --out_dir /data/zjy_work/BGDiff/asset/maps --foreground bright
