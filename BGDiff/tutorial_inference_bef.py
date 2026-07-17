# BEF-SBG diffusion inference
import os
import sys
import argparse

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/data/zjy_work/BGDiff")
if os.path.isdir(PROJECT_DIR) and PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("BOUNDARY_PRIOR_MODE", "external")

import torch

# ============================================================
# CUDA / cuDNN stability configuration
# Must run before model creation and any CUDA operation.
# ============================================================
DIFF_DISABLE_CUDNN = os.environ.get("DIFF_DISABLE_CUDNN", "0") == "1"

if DIFF_DISABLE_CUDNN:
    torch.backends.cudnn.enabled = False
    print("[WARN] cuDNN disabled by DIFF_DISABLE_CUDNN=1")
else:
    torch.backends.cudnn.enabled = True
    print("[INFO] cuDNN enabled")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(
    "[CUDA Backend] "
    f"cudnn.enabled={torch.backends.cudnn.enabled}, "
    f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
    f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
)

import gc
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *  # noqa: F401,F403
from tutorial_dataset_sample_bef import MyDataset
from cldm.model import create_model, load_state_dict


GLOBAL_SEED = int(os.environ.get("DIFF_SEED", os.environ.get("PL_GLOBAL_SEED", "0")))
pl.seed_everything(GLOBAL_SEED, workers=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.environ.get("CKPT_PATH", "./merged_pytorch_model.pth"),
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        default=os.environ.get("PROMPT_JSON", "./data/prompt_test.json"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.environ.get(
            "OUT_DIR",
            "/data/zjy_work/Work3_BEF_SBG/results/generated_bef_sbg",
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "cuda:0"),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "1")),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", "4")),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=int(os.environ.get("N_SAMPLES", "2")),
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=int(os.environ.get("IMG_SIZE", "384")),
    )

    parser.add_argument(
        "--sample_seed_base",
        type=int,
        default=int(os.environ.get("SAMPLE_SEED_BASE", str(GLOBAL_SEED))),
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=int(os.environ.get("DDIM_STEPS", "70")),
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=float(os.environ.get("CFG", "9.0")),
    )

    parser.add_argument(
        "--use_image_control",
        action="store_true",
        default=os.environ.get("USE_IMAGE_CONTROL", "0") == "1",
    )

    return parser.parse_args()


args = parse_args()

RESULT_DIR = args.out_dir
os.makedirs(RESULT_DIR, exist_ok=True)

learning_rate = 1e-5
sd_locked = False
only_mid_control = False


def _set_sample_seed(seed: int):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model():
    model = create_model("./models/cldm_v15.yaml").cpu()

    state_dict = load_state_dict(args.ckpt, location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[Checkpoint Load]")
    print(f"  ckpt: {args.ckpt}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("  first missing keys:")
        for k in missing[:20]:
            print(f"    {k}")

    if len(unexpected) > 0:
        print("  first unexpected keys:")
        for k in unexpected[:20]:
            print(f"    {k}")

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    model.to(args.device)
    model.eval()

    return model


def _tensor_to_uint8_img(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert CHW tensor in [-1, 1] to HWC uint8 image.
    """
    image = (image_tensor + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return image


def _hwc01_batch_to_bchw_neg1_1(x: torch.Tensor):
    """
    Convert dataloader tensor from [B,H,W,C], [0,1]
    to [B,C,H,W], [-1,1].
    """
    if x is None or not isinstance(x, torch.Tensor):
        return None

    if x.ndim == 4 and x.shape[-1] in [1, 3, 4]:
        x = x.permute(0, 3, 1, 2).contiguous()

    x = x.float()
    x = torch.clamp(x, 0.0, 1.0)
    x = x * 2.0 - 1.0
    return x


def _save_tensor_images(
    tensor: torch.Tensor,
    save_root: str,
    global_index: int,
    sample_index: int = None,
    binary: bool = False,
    grayscale: bool = True,
):
    """
    Save BCHW tensor in [-1,1] to image files.
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return

    os.makedirs(save_root, exist_ok=True)

    for i, image in enumerate(tensor):
        img = _tensor_to_uint8_img(image)

        if grayscale and img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., 0]

        if sample_index is None:
            filename = f"id-{global_index:06}_idx-{i}.png"
        else:
            filename = f"id-{global_index:06}_s-{sample_index:02}_idx-{i}.png"

        path = os.path.join(save_root, filename)

        pil_img = Image.fromarray(img)

        if binary:
            pil_img = pil_img.convert("1")
        elif grayscale:
            pil_img = pil_img.convert("L")

        pil_img.save(path)


def log_local(
    save_dir: str,
    images: dict,
    batch: dict,
    global_index: int,
    sample_index: int = None,
):
    """
    Save BEF-SBG inference outputs.

    Output folders:
      images/
      masks/
      boundary_prior/
      difficulty/
      boundary_hard/
    """
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")
    boundary_prior_root = os.path.join(save_dir, "boundary_prior")
    difficulty_root = os.path.join(save_dir, "difficulty")
    boundary_hard_root = os.path.join(save_dir, "boundary_hard")

    # Generated images.
    if "samples" in images and isinstance(images["samples"], torch.Tensor):
        os.makedirs(samples_root, exist_ok=True)

        for i, image in enumerate(images["samples"]):
            img = _tensor_to_uint8_img(image)

            if sample_index is None:
                filename = f"id-{global_index:06}_idx-{i}.png"
            else:
                filename = f"id-{global_index:06}_s-{sample_index:02}_idx-{i}.png"

            path = os.path.join(samples_root, filename)
            Image.fromarray(img).save(path)

    # Main mask.
    if "control_mask" in images:
        _save_tensor_images(
            tensor=images["control_mask"],
            save_root=mask_root,
            global_index=global_index,
            sample_index=sample_index,
            binary=True,
            grayscale=True,
        )

    # External adaptive boundary prior from BEF.
    if "control_boundary_prior" in images:
        _save_tensor_images(
            tensor=images["control_boundary_prior"],
            save_root=boundary_prior_root,
            global_index=global_index,
            sample_index=sample_index,
            binary=False,
            grayscale=True,
        )

    # Hard boundary extracted from mask for visualization only.
    if "control_boundary_hard" in images:
        _save_tensor_images(
            tensor=images["control_boundary_hard"],
            save_root=boundary_hard_root,
            global_index=global_index,
            sample_index=sample_index,
            binary=True,
            grayscale=True,
        )

    # Difficulty map from dataset.
    if isinstance(batch, dict) and "difficulty" in batch:
        difficulty = _hwc01_batch_to_bchw_neg1_1(batch["difficulty"])
        if difficulty is not None:
            difficulty = difficulty.detach().cpu()
            _save_tensor_images(
                tensor=difficulty,
                save_root=difficulty_root,
                global_index=global_index,
                sample_index=sample_index,
                binary=False,
                grayscale=True,
            )


def print_bef_config():
    print("[BEF-SBG Inference Config]")

    keys = [
        "PROJECT_DIR",
        "CKPT_PATH",
        "PROMPT_JSON",
        "OUT_DIR",
        "DIFF_SEED",
        "PL_GLOBAL_SEED",
        "SAMPLE_SEED_BASE",
        "BOUNDARY_PRIOR_MODE",
        "ENABLE_SOFT_BOUNDARY_PRIOR",
        "BOUNDARY_PRIOR_TAU",
        "BOUNDARY_PRIOR_RADIUS",
        "BOUNDARY_DILATE_KERNEL",
        "ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE",
        "BOUNDARY_GUIDANCE_MAX",
        "BOUNDARY_GUIDANCE_START_RATIO",
        "BOUNDARY_GUIDANCE_TEMPERATURE",
        "BOUNDARY_BRANCH_SCALE",
        "ENABLE_BOUNDARY_MODULATION",
        "BOUNDARY_MOD_SCALE",
        "BOUNDARY_MOD_START_RATIO",
        "USE_IMAGE_CONTROL",
        "N_SAMPLES",
        "DDIM_STEPS",
        "CFG",
    ]

    for k in keys:
        print(f"  {k} = {os.environ.get(k, '<default>')}")


if __name__ == "__main__":
    print_bef_config()

    model = get_model()

    dataset = MyDataset(
        prompt_json=args.prompt_json,
        size=args.img_size,
    )

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    finaldir = RESULT_DIR
    os.makedirs(finaldir, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            for batch_id, batch in enumerate(dataloader):
                if args.batch_size != 1:
                    print(
                        "[WARN] batch_size=1 is recommended to ensure "
                        "n_samples images correspond strictly to each input sample."
                    )

                print(
                    f"Processing batch {batch_id}, "
                    f"batch_size={args.batch_size}, "
                    f"n_samples={args.n_samples}"
                )

                for s in range(args.n_samples):
                    cur_seed = int(args.sample_seed_base) + batch_id * 1000 + s
                    _set_sample_seed(cur_seed)

                    print(f"  sample_index={s}, sample_seed={cur_seed}")

                    images = model.log_images(
                        batch,
                        N=1,
                        sample=True,
                        ddim_steps=args.ddim_steps,
                        ddim_eta=0.0,
                        use_image_control=args.use_image_control,
                        unconditional_guidance_scale=args.cfg,
                    )

                    for k in images:
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            images[k] = torch.clamp(images[k], -1.0, 1.0)

                    log_local(
                        save_dir=finaldir,
                        images=images,
                        batch=batch,
                        global_index=batch_id,
                        sample_index=s,
                    )

    print(f"[OK] BEF-SBG inference finished. Results saved to: {finaldir}")
