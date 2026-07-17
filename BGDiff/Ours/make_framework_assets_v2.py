import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_mask(mask_path, size=512):
    mask = Image.open(mask_path).convert("L")
    mask = ImageOps.fit(mask, (size, size), method=Image.Resampling.NEAREST)
    mask_np = np.array(mask)
    mask_bin = np.where(mask_np > 127, 255, 0).astype(np.uint8)
    return mask_bin


def read_rgb(image_path, size=512):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
    return np.array(image)


def save_gray(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)


def save_rgb(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)


# ============================================================
# A / B. Mask, hard boundary, soft boundary prior
# ============================================================

def build_boundary_from_mask(mask_bin, kernel_size=3):
    """
    Hard boundary:
        B_h = Dilate(M) - Erode(M)

    This is consistent with the morphological-gradient style used in your BGDiff dataset code.
    """
    mask01 = (mask_bin > 127).astype(np.uint8)

    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    kernel = np.ones((k, k), np.uint8)
    dilation = cv2.dilate(mask01, kernel, iterations=1)
    erosion = cv2.erode(mask01, kernel, iterations=1)

    boundary = np.clip(dilation - erosion, 0, 1).astype(np.uint8) * 255
    return boundary


def build_soft_boundary_prior(mask_bin, radius=12, tau=4.0, kernel_size=3):
    """
    Soft Boundary Prior, visually consistent with your ControlLDM logic.

    hard boundary:
        B_h = Dilate(M) - Erode(M)

    soft prior:
        multi-scale dilated shells with exponential decay.
    """
    hard = build_boundary_from_mask(mask_bin, kernel_size=kernel_size)
    hard01 = (hard > 127).astype(np.float32)

    soft = hard01.copy()
    prev = hard01.copy()

    for r in range(1, radius + 1):
        kr = 2 * r + 1
        kernel_r = np.ones((kr, kr), np.uint8)

        dilated = cv2.dilate(hard01.astype(np.uint8), kernel_r, iterations=1).astype(np.float32)
        shell = np.clip(dilated - prev, 0.0, 1.0)

        weight = np.exp(-float(r) / float(tau))
        soft = np.maximum(soft, shell * weight)
        prev = dilated

    soft = np.clip(soft, 0.0, 1.0)
    return soft


def soft_prior_to_orange_heatmap(soft_prior):
    """
    Convert soft boundary prior to orange heatmap band on black background.
    """
    h, w = soft_prior.shape

    bg = np.zeros((h, w, 3), dtype=np.float32)

    orange = np.zeros_like(bg)
    orange[..., 0] = 255
    orange[..., 1] = 150
    orange[..., 2] = 20

    alpha = np.clip(soft_prior[..., None], 0.0, 1.0)
    out = bg * (1.0 - alpha) + orange * alpha

    return out.astype(np.uint8)


# ============================================================
# C. Latent noise
# ============================================================

def make_latent_noise(size=512, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (size, size))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return (noise * 255).astype(np.uint8)


# ============================================================
# D. Progressive boundary guidance curve
# ============================================================

def make_progressive_guidance_curve(
    out_path,
    alpha_max=0.15,
    start_ratio=0.35,
    temperature=0.05,
):
    """
    Draw a simple curve for the figure:
    low influence in early denoising, high influence in late denoising.
    """
    progress = np.linspace(0, 1, 300)  # 0: early, 1: late
    t_norm = 1.0 - progress            # early corresponds to large t

    alpha = alpha_max / (1.0 + np.exp(-(start_ratio - t_norm) / temperature))

    plt.figure(figsize=(4.2, 2.5), dpi=300)
    plt.plot(progress, alpha, linewidth=3)
    plt.xlabel("Denoising progress")
    plt.ylabel("Boundary influence")
    plt.xticks([0, 1], ["early", "late"])
    plt.yticks([0, alpha_max], ["low", "high"])
    plt.title("Progressive guidance during late denoising")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()


# ============================================================
# E. Feature-map schematic for decoder refinement
# ============================================================

def make_feature_stack(size=512, seed=1, orange_refinement=False):
    """
    Create a schematic feature-map stack.
    This is not model output. It is only a clean visual asset for the framework figure.
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)

    if orange_refinement:
        colors = [
            (95, 120, 190, 130),
            (160, 140, 205, 130),
            (245, 150, 60, 150),
        ]
    else:
        colors = [
            (70, 95, 170, 130),
            (95, 120, 190, 130),
            (130, 150, 215, 130),
        ]

    for i, color in enumerate(colors):
        offset = i * 38
        x0 = 80 + offset
        y0 = 130 - offset // 3
        x1 = 360 + offset
        y1 = 410 - offset // 3

        layer = np.zeros_like(canvas)
        cv2.rectangle(layer, (x0, y0), (x1, y1), color, -1)

        for _ in range(25):
            px = rng.integers(x0, max(x0 + 1, x1 - 30))
            py = rng.integers(y0, max(y0 + 1, y1 - 30))
            pw = rng.integers(15, 55)
            ph = rng.integers(15, 55)

            if orange_refinement:
                patch_color = (255, 180, 80, 85)
            else:
                patch_color = (190, 205, 245, 70)

            cv2.rectangle(layer, (px, py), (px + pw, py + ph), patch_color, -1)

        alpha = layer[..., 3:4].astype(np.float32) / 255.0
        canvas[..., :3] = (
            canvas[..., :3].astype(np.float32) * (1.0 - alpha)
            + layer[..., :3].astype(np.float32) * alpha
        ).astype(np.uint8)
        canvas[..., 3] = np.maximum(canvas[..., 3], layer[..., 3])

    return canvas


# ============================================================
# F. Synthetic training pair
# ============================================================

def make_synthetic_training_pair(generated_rgb, train_mask_bin):
    h, w, _ = generated_rgb.shape
    mask_rgb = np.stack([train_mask_bin, train_mask_bin, train_mask_bin], axis=-1)

    sep = np.ones((h, 24, 3), dtype=np.uint8) * 255
    pair = np.concatenate([generated_rgb, sep, mask_rgb], axis=1)
    return pair


# ============================================================
# F. Test overlay for real test-set evaluation
# ============================================================

def make_test_overlay(test_rgb, gt_mask_bin=None, pred_mask_bin=None):
    overlay = test_rgb.copy()

    if gt_mask_bin is not None:
        gt = (gt_mask_bin > 127).astype(np.uint8)
        contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 60, 60), 3)

    if pred_mask_bin is not None:
        pred = (pred_mask_bin > 127).astype(np.uint8)
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (40, 180, 90), 3)

    return overlay


def make_metrics_bar(out_path, metrics=None):
    """
    Metrics shown in the framework figure.
    Replace these values with your real test-set results when available.
    """
    if metrics is None:
        metrics = {
            "Dice": 0.89,
            "IoU": 0.81,
            "Sen.": 0.88,
            "Spe.": 0.96,
        }

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(4.0, 2.6), dpi=300)
    plt.bar(names, values)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Test-set segmentation metrics")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close()


# ============================================================
# Main asset generation function
# ============================================================

def generate_framework_assets_v2(
    train_mask_path,
    generated_train_image_path,
    test_image_path,
    out_dir,
    test_gt_mask_path=None,
    pred_test_mask_path=None,
    size=512,
    metrics=None,
):
    ensure_dir(out_dir)

    # ------------------------------------------------------------
    # Training generation branch
    # ------------------------------------------------------------
    train_mask = read_mask(train_mask_path, size=size)
    generated_train_image = read_rgb(generated_train_image_path, size=size)

    boundary = build_boundary_from_mask(train_mask, kernel_size=3)
    soft_prior = build_soft_boundary_prior(
        train_mask,
        radius=12,
        tau=4.0,
        kernel_size=3,
    )
    soft_heatmap = soft_prior_to_orange_heatmap(soft_prior)

    save_gray(train_mask, os.path.join(out_dir, "A_training_mask.png"))
    save_gray(train_mask, os.path.join(out_dir, "B_mask_small.png"))
    save_gray(boundary, os.path.join(out_dir, "B_boundary_outline.png"))
    save_gray((soft_prior * 255).astype(np.uint8), os.path.join(out_dir, "B_soft_boundary_prior_gray.png"))
    save_rgb(soft_heatmap, os.path.join(out_dir, "B_soft_boundary_prior_heatmap.png"))

    save_rgb(generated_train_image, os.path.join(out_dir, "C_generated_train_image.png"))

    synthetic_pair = make_synthetic_training_pair(
        generated_train_image,
        train_mask,
    )
    save_rgb(synthetic_pair, os.path.join(out_dir, "F_synthetic_training_pair.png"))

    # ------------------------------------------------------------
    # Diffusion schematic assets
    # ------------------------------------------------------------
    latent_noise = make_latent_noise(size=size, seed=0)
    save_gray(latent_noise, os.path.join(out_dir, "C_latent_noise.png"))

    make_progressive_guidance_curve(
        os.path.join(out_dir, "D_progressive_guidance_curve.png"),
        alpha_max=0.15,
        start_ratio=0.35,
        temperature=0.05,
    )

    decoder_features = make_feature_stack(size=size, seed=1, orange_refinement=False)
    refined_features = make_feature_stack(size=size, seed=2, orange_refinement=True)

    Image.fromarray(decoder_features, mode="RGBA").save(os.path.join(out_dir, "E_decoder_features.png"))
    save_rgb(soft_heatmap, os.path.join(out_dir, "E_soft_boundary_band.png"))
    Image.fromarray(refined_features, mode="RGBA").save(os.path.join(out_dir, "E_refined_features.png"))

    # ------------------------------------------------------------
    # Real test-set evaluation branch
    # ------------------------------------------------------------
    test_image = read_rgb(test_image_path, size=size)
    save_rgb(test_image, os.path.join(out_dir, "F_test_image.png"))

    test_gt_mask = None
    if test_gt_mask_path is not None:
        test_gt_mask = read_mask(test_gt_mask_path, size=size)
        save_gray(test_gt_mask, os.path.join(out_dir, "F_test_gt_mask.png"))

    pred_test_mask = None
    if pred_test_mask_path is not None:
        pred_test_mask = read_mask(pred_test_mask_path, size=size)
        save_gray(pred_test_mask, os.path.join(out_dir, "F_predicted_mask.png"))
    else:
        # Optional placeholder only for drawing.
        # For a real paper, replace this with actual BGDNet prediction.
        pred_test_mask = test_gt_mask.copy() if test_gt_mask is not None else None
        if pred_test_mask is not None:
            save_gray(pred_test_mask, os.path.join(out_dir, "F_predicted_mask_placeholder.png"))

    overlay = make_test_overlay(
        test_rgb=test_image,
        gt_mask_bin=test_gt_mask,
        pred_mask_bin=pred_test_mask,
    )
    save_rgb(overlay, os.path.join(out_dir, "F_test_overlay.png"))

    make_metrics_bar(
        os.path.join(out_dir, "F_metrics_bar.png"),
        metrics=metrics,
    )

    print(f"[OK] Framework assets saved to: {out_dir}")


if __name__ == "__main__":
    generate_framework_assets_v2(
        train_mask_path="./train_mask.png",
        generated_train_image_path="./generated_train_image.png",
        test_image_path="./test_image.png",
        test_gt_mask_path="./test_gt_mask.png",
        pred_test_mask_path="./pred_test_mask.png",
        out_dir="./framework_assets_v2",
        size=512,
        metrics={
            "Dice": 0.89,
            "IoU": 0.81,
            "Sen.": 0.88,
            "Spe.": 0.96,
        },
    )

