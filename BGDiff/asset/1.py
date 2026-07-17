import os
import csv
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 1. 路径配置
# =========================

IMAGE_PATH = "/data/zjy_work/ISIC2018/train/Images/ISIC_0000000.jpg"
MASK_PATH = "/data/zjy_work/ISIC2018/train/Masks/ISIC_0000000_segmentation.png"
OUT_DIR = "/data/zjy_work/BGDiff/asset/middle"

# 如果你后续已有真实模型输出，可以填入下面路径；没有就保持为空字符串
PRED_PROB_PATH = ""     # 例如："/data/zjy_work/BGDiff/pred/ISIC_0000000_prob.png"
SYN_IMAGE_PATH = ""     # 例如："/data/zjy_work/BGDiff/syn/ISIC_0000000_syn.png"
FINAL_MASK_PATH = ""    # 例如："/data/zjy_work/BGDiff/final/ISIC_0000000_final.png"

OUT_SIZE = 512
RANDOM_SEED = 2026


# =========================
# 2. 基础工具函数
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_rgb(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (m > 127).astype(np.float32)


def norm01(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def save_rgb(path, arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_gray01(path, arr):
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray(rgb).save(path)


def save_cmap01(path, arr, cmap_name="turbo"):
    arr = norm01(arr)
    if cmap_name not in plt.colormaps():
        cmap_name = "jet"
    cmap = plt.get_cmap(cmap_name)
    rgb = (cmap(arr)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def square_box_from_mask(mask, margin_ratio=0.35):
    h, w = mask.shape
    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        side = min(h, w)
        cx, cy = w / 2, h / 2
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        side = int(max(bw, bh) * (1 + 2 * margin_ratio))
        side = max(side, 128)
        side = min(side, max(h, w))
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x3 = x0 + side
    y3 = y0 + side
    return x0, y0, x3, y3


def crop_pad(arr, box, fill_value=0):
    x0, y0, x1, y1 = box
    h, w = arr.shape[:2]
    out_h = y1 - y0
    out_w = x1 - x0

    if arr.ndim == 2:
        out = np.full((out_h, out_w), fill_value, dtype=arr.dtype)
    else:
        out = np.full((out_h, out_w, arr.shape[2]), fill_value, dtype=arr.dtype)

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def resize_img(arr, size=512, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(arr, (size, size), interpolation=interp)


def read_optional_gray(path, crop_box, size):
    if not path or not os.path.exists(path):
        return None
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if x is None:
        return None
    x = x.astype(np.float32)
    if x.max() > 1:
        x = x / 255.0
    x = crop_pad(x, crop_box, fill_value=0)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.clip(x, 0, 1)


def read_optional_rgb(path, crop_box, size):
    if not path or not os.path.exists(path):
        return None
    x = read_rgb(path)
    x = crop_pad(x, crop_box, fill_value=255)
    x = resize_img(x, size=size, is_mask=False)
    return x


# =========================
# 3. 边界、软边界、不确定性
# =========================

def morph_gradient(mask, k=5):
    m = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    return g.astype(np.float32)


def soft_boundary_prior(mask, k=3, R=18, tau=5.0):
    m = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    dil = cv2.dilate(m, kernel)
    ero = cv2.erode(m, kernel)
    g = np.clip(dil - ero, 0, 1).astype(np.uint8)

    b = g.astype(np.float32)
    prev = g.copy()

    for r in range(1, R + 1):
        kr = 2 * r + 1
        kernel_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))
        d = cv2.dilate(g, kernel_r)
        shell = np.clip(d - prev, 0, 1).astype(np.float32)
        b = np.maximum(b, math.exp(-r / tau) * shell)
        prev = d

    b = cv2.GaussianBlur(b, (0, 0), sigmaX=1.0)
    return norm01(b)


def signed_distance(mask):
    m = (mask > 0.5).astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return inside - outside


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def smooth_noise(h, w, scale=32, seed=2026):
    rng = np.random.default_rng(seed)
    small_h = max(4, h // scale)
    small_w = max(4, w // scale)
    n = rng.normal(0, 1, size=(small_h, small_w)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n = n - n.mean()
    n = n / (n.std() + 1e-8)
    n = np.clip(n / 2.5, -1, 1)
    return n


def simulate_pred_prob(mask, seed=2026):
    h, w = mask.shape
    sdf = signed_distance(mask)
    noise = smooth_noise(h, w, scale=36, seed=seed)

    # 通过低频扰动模拟低标注模型的边界漂移
    pred = sigmoid((sdf + 6.0 * noise - 1.0) / 3.5)
    pred = cv2.GaussianBlur(pred.astype(np.float32), (0, 0), sigmaX=1.5)
    return np.clip(pred, 0, 1)


def boundary_probability_from_prob(prob):
    gx = cv2.Sobel(prob.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
    return norm01(g)


def entropy_uncertainty(prob):
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    u = -p * np.log(p) - (1 - p) * np.log(1 - p)
    u = u / np.log(2.0)
    return np.clip(u, 0, 1)


def boundary_error_overlay(gt_mask, pred_prob):
    gt_edge = morph_gradient(gt_mask, k=5)
    pred_edge = morph_gradient(pred_prob > 0.5, k=5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gt_edge = cv2.dilate(gt_edge.astype(np.uint8), kernel).astype(bool)
    pred_edge = cv2.dilate(pred_edge.astype(np.uint8), kernel).astype(bool)

    rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

    # GT 绿色，预测红色，重叠黄色
    rgb[gt_edge] = [40, 220, 80]
    rgb[pred_edge] = [255, 45, 45]
    rgb[gt_edge & pred_edge] = [255, 220, 40]

    return rgb


def boundary_error_heat(gt_mask, pred_prob):
    gt_edge = morph_gradient(gt_mask, k=5)
    pred_edge = morph_gradient(pred_prob > 0.5, k=5)
    err = np.abs(gt_edge - pred_edge)
    err = cv2.dilate(err.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    err = cv2.GaussianBlur(err, (0, 0), sigmaX=2.0)
    return norm01(err)


# =========================
# 4. 合成图示意与图标
# =========================

def simulate_synthetic_image(img, mask, seed=2026):
    rng = np.random.default_rng(seed)
    img_f = img.astype(np.float32)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-5, 5), 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.18), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.92, 1.10), 0, 255)
    aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    h, w = mask.shape
    n = smooth_noise(h, w, scale=24, seed=seed + 17)
    n3 = np.stack([n * 10, n * 6, -n * 4], axis=-1)

    lesion_weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=6.0)
    lesion_weight = lesion_weight[..., None]

    syn = img_f * (1 - 0.25 * lesion_weight) + aug * (0.25 * lesion_weight)
    syn = syn + n3 * lesion_weight

    syn = cv2.GaussianBlur(syn.astype(np.uint8), (3, 3), sigmaX=0.5)
    return np.clip(syn, 0, 255).astype(np.uint8)


def make_icon_diffusion(path, size=512):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    blue = (30, 80, 180, 255)
    light = (220, 235, 255, 255)

    layers = [4, 5, 4, 3]
    xs = np.linspace(90, size - 90, len(layers))

    positions = []
    for li, num in enumerate(layers):
        ys = np.linspace(110, size - 110, num)
        layer_pos = []
        for y in ys:
            layer_pos.append((int(xs[li]), int(y)))
        positions.append(layer_pos)

    for li in range(len(positions) - 1):
        for p1 in positions[li]:
            for p2 in positions[li + 1]:
                draw.line([p1, p2], fill=(120, 160, 220, 160), width=3)

    for layer in positions:
        for x, y in layer:
            r = 15
            draw.ellipse([x - r, y - r, x + r, y + r], fill=light, outline=blue, width=4)

    img.save(path)


def make_icon_region(path, size=512):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    green = (70, 160, 85, 255)
    draw.ellipse([80, 80, size - 80, size - 80], fill=green)
    draw.line([(180, 260), (235, 315), (340, 200)], fill=(255, 255, 255, 255), width=36, joint="curve")
    img.save(path)


def make_icon_boundary(path, size=512):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    green = (70, 160, 85, 255)

    cx, cy = size // 2, size // 2
    pts = []
    for i in range(80):
        t = 2 * math.pi * i / 80
        r = 130 + 18 * math.sin(3 * t) + 10 * math.cos(5 * t)
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))

    for i in range(0, len(pts), 2):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        draw.line([p1, p2], fill=green, width=12)

    img.save(path)


def make_icon_weight(path, size=512):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    green = (70, 160, 85, 255)

    base_y = 390
    xs = [150, 220, 290, 360]
    hs = [80, 140, 210, 270]
    for x, h in zip(xs, hs):
        draw.rounded_rectangle([x, base_y - h, x + 45, base_y], radius=10, fill=green)

    draw.line([(120, base_y), (420, base_y)], fill=green, width=10)
    draw.line([(120, 120), (120, base_y)], fill=green, width=10)
    draw.line([(120, 120), (95, 160)], fill=green, width=10)
    draw.line([(120, 120), (145, 160)], fill=green, width=10)

    img.save(path)


def make_test_to_mask_pair(path, test_img, final_mask, size=512):
    canvas = Image.new("RGB", (size * 2 + 120, size), (255, 255, 255))
    left = Image.fromarray(test_img).resize((size, size))
    mask_img = (np.clip(final_mask, 0, 1) * 255).astype(np.uint8)
    mask_rgb = np.stack([mask_img, mask_img, mask_img], axis=-1)
    right = Image.fromarray(mask_rgb).resize((size, size))

    canvas.paste(left, (0, 0))
    canvas.paste(right, (size + 120, 0))

    draw = ImageDraw.Draw(canvas)
    green = (70, 170, 85)
    y = size // 2
    x1 = size + 25
    x2 = size + 95
    draw.line([(x1, y), (x2, y)], fill=green, width=22)
    draw.polygon([(x2, y), (x2 - 28, y - 28), (x2 - 28, y + 28)], fill=green)

    canvas.save(path)


def make_contact_sheet(asset_records, out_path, thumb=150, cols=5):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    rows = math.ceil(len(asset_records) / cols)
    sheet_w = cols * thumb
    sheet_h = rows * (thumb + 38)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    for idx, rec in enumerate(asset_records):
        filename = rec["filename"]
        path = os.path.join(OUT_DIR, filename)
        if not os.path.exists(path):
            continue
        r = idx // cols
        c = idx % cols
        x = c * thumb
        y = r * (thumb + 38)

        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb, thumb))
        px = x + (thumb - im.width) // 2
        py = y + (thumb - im.height) // 2
        sheet.paste(im, (px, py))

        short_name = filename.replace(".png", "")
        draw.text((x + 4, y + thumb + 4), short_name[:20], fill=(20, 20, 20), font=font)

    sheet.save(out_path)


# =========================
# 5. 主流程
# =========================

def main():
    ensure_dir(OUT_DIR)
    records = []

    def record(filename, desc):
        records.append({"filename": filename, "description": desc})

    img = read_rgb(IMAGE_PATH)
    mask = read_mask(MASK_PATH)

    crop_box = square_box_from_mask(mask, margin_ratio=0.35)

    img_crop = crop_pad(img, crop_box, fill_value=255)
    mask_crop = crop_pad(mask, crop_box, fill_value=0)

    img_crop = resize_img(img_crop, OUT_SIZE, is_mask=False)
    mask_crop = resize_img(mask_crop, OUT_SIZE, is_mask=True)
    mask_crop = (mask_crop > 0.5).astype(np.float32)

    # 1. 原始皮肤镜图像
    save_rgb(os.path.join(OUT_DIR, "01_dermoscopy_image.png"), img_crop)
    record("01_dermoscopy_image.png", "模块1：皮肤镜图像")

    # 2. 二值病灶Mask
    save_gray01(os.path.join(OUT_DIR, "02_binary_lesion_mask.png"), mask_crop)
    record("02_binary_lesion_mask.png", "模块1：二值病灶Mask")

    # 3. GT边界叠加图，可选备用
    gt_edge = morph_gradient(mask_crop, k=5)
    overlay = img_crop.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    edge_alpha = cv2.dilate(gt_edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    edge_alpha = edge_alpha[..., None]
    overlay = overlay * (1 - 0.65 * edge_alpha) + red * (0.65 * edge_alpha)
    save_rgb(os.path.join(OUT_DIR, "03_dermoscopy_with_gt_boundary.png"), overlay)
    record("03_dermoscopy_with_gt_boundary.png", "备用：原图叠加真实边界")

    # 4. 初始BGDNet预测概率图
    pred_prob = read_optional_gray(PRED_PROB_PATH, crop_box, OUT_SIZE)
    if pred_prob is None:
        pred_prob = simulate_pred_prob(mask_crop, seed=RANDOM_SEED)

    save_gray01(os.path.join(OUT_DIR, "04_initial_pred_mask_prob.png"), pred_prob)
    record("04_initial_pred_mask_prob.png", "模块2：初始预测Mask概率图")

    # 5. 初始预测二值Mask
    pred_bin = (pred_prob > 0.5).astype(np.float32)
    save_gray01(os.path.join(OUT_DIR, "05_initial_pred_mask_binary.png"), pred_bin)
    record("05_initial_pred_mask_binary.png", "备用：初始预测二值Mask")

    # 6. 边界概率图
    boundary_prob = boundary_probability_from_prob(pred_prob)
    save_gray01(os.path.join(OUT_DIR, "06_boundary_probability_map.png"), boundary_prob)
    record("06_boundary_probability_map.png", "模块2：边界概率图")

    # 7. 边界误差图
    err_overlay = boundary_error_overlay(mask_crop, pred_prob)
    save_rgb(os.path.join(OUT_DIR, "07_boundary_error_overlay.png"), err_overlay)
    record("07_boundary_error_overlay.png", "模块3：边界误差图，绿色GT，红色预测，黄色重叠")

    # 8. 不确定性图
    uncertainty = entropy_uncertainty(pred_prob)
    save_cmap01(os.path.join(OUT_DIR, "08_uncertainty_map.png"), uncertainty, cmap_name="turbo")
    record("08_uncertainty_map.png", "模块3：预测不确定性图")

    # 9. 软边界先验
    soft_b = soft_boundary_prior(mask_crop, k=3, R=20, tau=5.0)
    save_gray01(os.path.join(OUT_DIR, "09_soft_boundary_prior_gray.png"), soft_b)
    record("09_soft_boundary_prior_gray.png", "模块4：软边界先验灰度图")

    save_cmap01(os.path.join(OUT_DIR, "10_soft_boundary_prior_heat.png"), soft_b, cmap_name="inferno")
    record("10_soft_boundary_prior_heat.png", "备用：软边界先验热力图")

    # 10. 边界困难图 Boundary Difficulty Map
    err_heat = boundary_error_heat(mask_crop, pred_prob)
    bdm = norm01(0.75 * soft_b * uncertainty + 0.85 * err_heat)
    save_cmap01(os.path.join(OUT_DIR, "11_boundary_difficulty_map.png"), bdm, cmap_name="turbo")
    record("11_boundary_difficulty_map.png", "模块3：Boundary Difficulty Map边界困难热力图")

    # 11. 困难感知软边界条件
    bridge_boundary = np.clip(soft_b * (1.0 + 1.2 * bdm), 0, 1)
    bridge_boundary = norm01(bridge_boundary)
    save_cmap01(os.path.join(OUT_DIR, "12_difficulty_aware_boundary.png"), bridge_boundary, cmap_name="inferno")
    record("12_difficulty_aware_boundary.png", "模块4：困难感知软边界条件")

    # 12. 扩散模型图标
    make_icon_diffusion(os.path.join(OUT_DIR, "13_diffusion_model_icon.png"), size=OUT_SIZE)
    record("13_diffusion_model_icon.png", "模块4：扩散模型图标")

    # 13. 合成皮肤镜图像
    syn_img = read_optional_rgb(SYN_IMAGE_PATH, crop_box, OUT_SIZE)
    if syn_img is None:
        syn_img = simulate_synthetic_image(img_crop, mask_crop, seed=RANDOM_SEED)

    save_rgb(os.path.join(OUT_DIR, "14_synthetic_dermoscopy_demo.png"), syn_img)
    record("14_synthetic_dermoscopy_demo.png", "模块4：合成皮肤镜图像示意；真实论文图建议替换为SBG-Diff输出")

    # 14. 三个质量门控图标
    make_icon_region(os.path.join(OUT_DIR, "15_region_consistency_icon.png"), size=OUT_SIZE)
    record("15_region_consistency_icon.png", "模块5：区域一致性图标")

    make_icon_boundary(os.path.join(OUT_DIR, "16_boundary_consistency_icon.png"), size=OUT_SIZE)
    record("16_boundary_consistency_icon.png", "模块5：边界一致性图标")

    make_icon_weight(os.path.join(OUT_DIR, "17_quality_weight_icon.png"), size=OUT_SIZE)
    record("17_quality_weight_icon.png", "模块5：样本质量权重图标")

    # 15. 测试图像
    save_rgb(os.path.join(OUT_DIR, "18_test_image.png"), img_crop)
    record("18_test_image.png", "模块5：测试图像")

    # 16. 最终分割Mask
    final_mask = read_optional_gray(FINAL_MASK_PATH, crop_box, OUT_SIZE)
    if final_mask is None:
        # 没有最终模型输出时，使用GT生成一个干净示意Mask
        final_mask = mask_crop.copy()
        final_mask = cv2.morphologyEx(
            final_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ).astype(np.float32)

    save_gray01(os.path.join(OUT_DIR, "19_final_segmentation_mask.png"), final_mask)
    record("19_final_segmentation_mask.png", "模块5：最终分割Mask示意；真实论文图建议替换为最终模型输出")

    # 17. 测试图像到最终Mask的组合小图
    make_test_to_mask_pair(
        os.path.join(OUT_DIR, "20_test_to_final_mask_pair.png"),
        img_crop,
        final_mask,
        size=OUT_SIZE
    )
    record("20_test_to_final_mask_pair.png", "模块5：测试图像到最终分割Mask组合图")

    # 18. 输出索引表
    csv_path = os.path.join(OUT_DIR, "asset_index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "description"])
        writer.writeheader()
        writer.writerows(records)

    # 19. 总览图
    make_contact_sheet(records, os.path.join(OUT_DIR, "00_asset_overview.png"))

    print(f"Done. Assets saved to: {OUT_DIR}")
    print(f"Index saved to: {csv_path}")
    print("Generated files:")
    for r in records:
        print(f"  {r['filename']}  -  {r['description']}")


if __name__ == "__main__":
    main()
