import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
INPUT_PATH = "/data/zjy_work/BGDiff/asset/Train_real_1.png"
OUTPUT_DIR = "/data/zjy_work/BGDiff/asset/soft_boundary_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Method parameters
# =========================
THRESHOLD = 0.5

# Hard morphological boundary
BOUNDARY_KERNEL = 11

# Larger radius makes the soft boundary band easier to see in paper figures
MAX_RADIUS = 28

# Larger tau makes the exponential decay slower and visually clearer
TAU = 10.0


# =========================
# Visualization parameters
# =========================
# Only used for visualization, not for the numerical prior
VIS_GAIN = 1.8
VIS_GAMMA = 0.45
VIS_BLUR_SIGMA = 1.6

# Make the orange core contour more visible
CORE_VIS_THICKEN = 2

# Academic pastel orange palette
ORANGE_CORE = np.array([242, 145, 45]) / 255.0
ORANGE_MID = np.array([250, 184, 94]) / 255.0
ORANGE_LIGHT = np.array([255, 225, 186]) / 255.0

BLUE_MASK = np.array([205, 226, 248]) / 255.0
PALE_INSIDE = np.array([248, 250, 253]) / 255.0
WHITE = np.array([1.0, 1.0, 1.0])


# =========================
# Utility functions
# =========================
def read_binary_mask(path, threshold=0.5):
    """
    Read the original mask and binarize it:
        M_b(u,v) = I[M(u,v) > 0.5]

    The lesion shape is strictly inherited from the input mask.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    mb = (arr > threshold).astype(np.uint8)
    return mb


def morph_boundary(mb, k):
    """
    G = Dilate_k(M_b) - Erode_k(M_b)
    """
    assert k % 2 == 1, "BOUNDARY_KERNEL must be odd."

    kernel = np.ones((k, k), np.uint8)

    dilated = cv2.dilate(mb, kernel, iterations=1)
    eroded = cv2.erode(mb, kernel, iterations=1)

    g = np.clip(dilated - eroded, 0, 1).astype(np.float32)

    return g, dilated.astype(np.float32), eroded.astype(np.float32)


def build_soft_boundary_prior(g, max_radius, tau):
    """
    D_r = Dilate_{2r+1}(G)
    S_r = clip(D_r - D_{r-1}, 0, 1)
    w_r = exp(-r / tau)

    B = max(G, max_r w_r S_r)
    """
    d_prev = g.copy()
    soft_shells = np.zeros_like(g, dtype=np.float32)

    for r in range(1, max_radius + 1):
        kernel_size = 2 * r + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        d_r = cv2.dilate(g.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        s_r = np.clip(d_r - d_prev, 0, 1)

        w_r = np.exp(-r / tau)
        soft_shells = np.maximum(soft_shells, w_r * s_r)

        d_prev = d_r

    b = np.maximum(g, soft_shells)
    b = np.clip(b, 0, 1).astype(np.float32)

    return soft_shells, b


def enhance_for_visualization(x, gain=1.8, gamma=0.45, blur_sigma=1.6):
    """
    Enhance soft boundary only for paper visualization.

    This does not change the numerical prior B.
    """
    x_vis = np.clip(x * gain, 0, 1)
    x_vis = np.power(x_vis, gamma)

    if blur_sigma > 0:
        ksize = int(blur_sigma * 6 + 1)
        if ksize % 2 == 0:
            ksize += 1
        x_vis = cv2.GaussianBlur(x_vis, (ksize, ksize), blur_sigma)

    return np.clip(x_vis, 0, 1)


def alpha_blend(base, color, alpha):
    """
    base: H x W x 3
    color: RGB vector in [0, 1]
    alpha: H x W in [0, 1]
    """
    return base * (1 - alpha[..., None]) + color[None, None, :] * alpha[..., None]


def save_rgb(path, rgb):
    rgb = np.clip(rgb, 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.save(path)


def save_gray(path, gray):
    gray = np.clip(gray, 0, 1)
    img = Image.fromarray((gray * 255).astype(np.uint8))
    img.save(path)


def thicken_binary_map(x, radius=2):
    """
    Thicken the hard contour only for visualization.
    """
    if radius <= 0:
        return x.astype(np.float32)

    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    x_thick = cv2.dilate((x > 0).astype(np.uint8), kernel, iterations=1)
    return x_thick.astype(np.float32)


# =========================
# Main
# =========================
mb = read_binary_mask(INPUT_PATH, THRESHOLD)

g, dilated, eroded = morph_boundary(mb, BOUNDARY_KERNEL)
soft_shells, b = build_soft_boundary_prior(g, MAX_RADIUS, TAU)

# Enhanced maps for visualization only
soft_shells_vis = enhance_for_visualization(
    soft_shells,
    gain=VIS_GAIN,
    gamma=VIS_GAMMA,
    blur_sigma=VIS_BLUR_SIGMA,
)

b_vis = enhance_for_visualization(
    b,
    gain=VIS_GAIN,
    gamma=VIS_GAMMA,
    blur_sigma=VIS_BLUR_SIGMA,
)

g_vis = thicken_binary_map(g, CORE_VIS_THICKEN)

h, w = mb.shape


# =========================
# 1. Binary lesion mask visualization
# =========================
binary_mask_rgb = np.ones((h, w, 3), dtype=np.float32)
binary_mask_rgb[mb > 0] = BLUE_MASK


# =========================
# 2. Hard boundary visualization
# =========================
hard_boundary_rgb = np.ones((h, w, 3), dtype=np.float32)
hard_boundary_rgb[mb > 0] = PALE_INSIDE

hard_boundary_rgb = alpha_blend(
    hard_boundary_rgb,
    ORANGE_CORE,
    np.clip(g_vis * 0.95, 0, 1),
)


# =========================
# 3. Decayed boundary shells visualization
# =========================
shells_rgb = np.ones((h, w, 3), dtype=np.float32)
shells_rgb[mb > 0] = PALE_INSIDE

# broad pale orange shell
shells_rgb = alpha_blend(
    shells_rgb,
    ORANGE_LIGHT,
    np.clip(soft_shells_vis * 0.75, 0, 1),
)

# mid orange transition
shells_rgb = alpha_blend(
    shells_rgb,
    ORANGE_MID,
    np.clip(soft_shells_vis * 0.35, 0, 1),
)

# visible core contour
shells_rgb = alpha_blend(
    shells_rgb,
    ORANGE_CORE,
    np.clip(g_vis * 0.95, 0, 1),
)


# =========================
# 4. Final soft boundary prior visualization
# =========================
prior_rgb = np.ones((h, w, 3), dtype=np.float32)
prior_rgb[mb > 0] = PALE_INSIDE

# enhanced soft tolerance band
prior_rgb = alpha_blend(
    prior_rgb,
    ORANGE_LIGHT,
    np.clip(b_vis * 0.82, 0, 1),
)

# stronger mid band
prior_rgb = alpha_blend(
    prior_rgb,
    ORANGE_MID,
    np.clip(b_vis * 0.32, 0, 1),
)

# hard boundary as strongest response
prior_rgb = alpha_blend(
    prior_rgb,
    ORANGE_CORE,
    np.clip(g_vis * 0.98, 0, 1),
)


# =========================
# Save numerical maps
# =========================
save_gray(os.path.join(OUTPUT_DIR, "01_binary_mask_raw.png"), mb.astype(np.float32))
save_gray(os.path.join(OUTPUT_DIR, "02_dilate_k.png"), dilated)
save_gray(os.path.join(OUTPUT_DIR, "03_erode_k.png"), eroded)
save_gray(os.path.join(OUTPUT_DIR, "04_hard_boundary_G.png"), g)
save_gray(os.path.join(OUTPUT_DIR, "05_decayed_shells_wrSr.png"), soft_shells)
save_gray(os.path.join(OUTPUT_DIR, "06_soft_boundary_prior_B.png"), b)


# =========================
# Save enhanced visualization maps
# =========================
save_gray(os.path.join(OUTPUT_DIR, "05_decayed_shells_wrSr_visual_enhanced.png"), soft_shells_vis)
save_gray(os.path.join(OUTPUT_DIR, "06_soft_boundary_prior_B_visual_enhanced.png"), b_vis)

save_rgb(os.path.join(OUTPUT_DIR, "vis_01_binary_lesion_mask_Mb.png"), binary_mask_rgb)
save_rgb(os.path.join(OUTPUT_DIR, "vis_02_hard_boundary_G_orange.png"), hard_boundary_rgb)
save_rgb(os.path.join(OUTPUT_DIR, "vis_03_decayed_boundary_shells_wrSr_enhanced.png"), shells_rgb)
save_rgb(os.path.join(OUTPUT_DIR, "vis_04_soft_boundary_prior_B_enhanced.png"), prior_rgb)


# =========================
# Save overview figure
# =========================
fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5), dpi=300)

items = [
    ("Binary Lesion Mask\n$M_b$", binary_mask_rgb),
    ("Hard Boundary\n$G$", hard_boundary_rgb),
    ("Decayed Boundary Shells\n$w_r S_r$", shells_rgb),
    ("Soft Boundary Prior\n$B$", prior_rgb),
]

for ax, (title, img) in zip(axes, items):
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "overview_soft_boundary_prior_assets_enhanced.png"),
    facecolor="white",
    bbox_inches="tight",
    pad_inches=0.08,
)
plt.close()

print(f"Saved enhanced assets to: {OUTPUT_DIR}")
