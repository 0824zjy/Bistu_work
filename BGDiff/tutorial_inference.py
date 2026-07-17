#5.5-ms
import os
import sys
import argparse

# ---------------------------------------------------------------------
# Robust import path setup.
# This file can be copied to /data/zjy_work/BGDiff/tutorial_inference.py
# or kept under /data/zjy_work/BGDiff/Ours/ablation_sbg_ms/.
# In both cases, make sure project root is importable before importing:
#   share, tutorial_dataset_sample, cldm, ldm
# ---------------------------------------------------------------------
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/data/zjy_work/BGDiff")
if os.path.isdir(PROJECT_DIR) and PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *  # noqa: F401,F403
from tutorial_dataset_sample import MyDataset
from cldm.model import create_model, load_state_dict


GLOBAL_SEED = int(os.environ.get("DIFF_SEED", os.environ.get("PL_GLOBAL_SEED", "0")))
pl.seed_everything(GLOBAL_SEED, workers=True)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ckpt",
        type=str,
        default=os.environ.get("CKPT_PATH", "./merged_pytorch_model.pth"),
    )
    p.add_argument(
        "--prompt_json",
        type=str,
        default=os.environ.get("PROMPT_JSON", "./data/prompt_test.json"),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.environ.get(
            "OUT_DIR",
            "/data/zjy_work/BGDiff/Ours/results/exp_mask2img_sbp_pg/",
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "cuda:0"),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "1")),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", "4")),
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=int(os.environ.get("N_SAMPLES", "5")),
    )

    p.add_argument(
        "--img_size",
        type=int,
        default=int(os.environ.get("IMG_SIZE", "384")),
    )

    p.add_argument(
        "--sample_seed_base",
        type=int,
        default=int(os.environ.get("SAMPLE_SEED_BASE", str(GLOBAL_SEED))),
    )

    p.add_argument(
        "--sampler",
        type=str,
        default=os.environ.get("SAMPLER", "ddim"),
        choices=["ddim", "dpm", "hybrid"],
    )
    p.add_argument(
        "--ddim_steps",
        type=int,
        default=int(os.environ.get("DDIM_STEPS", "50")),
    )
    p.add_argument(
        "--dpm_steps",
        type=int,
        default=int(os.environ.get("DPM_STEPS", "20")),
    )
    p.add_argument(
        "--hybrid_split",
        type=float,
        default=float(os.environ.get("HYBRID_SPLIT", "0.5")),
    )
    p.add_argument(
        "--cfg",
        type=float,
        default=float(os.environ.get("CFG", "9.0")),
    )

    # Default: mask-only image input.
    # Boundary prior is automatically generated from mask / boundary according to
    # BOUNDARY_PRIOR_MODE inside model.get_input.
    p.add_argument(
        "--use_image_control",
        action="store_true",
        default=os.environ.get("USE_IMAGE_CONTROL", "0") == "1",
    )

    return p.parse_args()


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

    # strict=False is important because SBP-PG introduces weak boundary modules.
    # Old checkpoints can still be partially loaded.
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


def _tensor_to_uint8_img(image_tensor):
    """
    Convert CHW tensor in [-1,1] to HWC uint8 image.
    """
    image = (image_tensor + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def _save_tensor_images(
    tensor,
    save_root,
    global_index,
    sample_index=None,
    binary=False,
    grayscale=True,
):
    """
    Save tensor images.

    Args:
        tensor:
            [B,C,H,W], expected range [-1,1]
        save_root:
            target directory
        binary:
            if True, save as binary mask
        grayscale:
            if True and image has 3 channels, use first channel
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


def log_local(save_dir, images, global_index, sample_index=None):
    """
    Save inference outputs.

    SBP-PG / ablation output folders:
        images/
        masks/
        boundary_prior/
        boundary_hard/

    Compatibility:
        If old key "control_boundary" appears, also save it under boundaries/.
    """
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")

    boundary_prior_root = os.path.join(save_dir, "boundary_prior")
    boundary_hard_root = os.path.join(save_dir, "boundary_hard")

    # Old compatibility folder.
    boundary_root = os.path.join(save_dir, "boundaries")

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

    # Boundary prior visualization.
    if "control_boundary_prior" in images:
        _save_tensor_images(
            tensor=images["control_boundary_prior"],
            save_root=boundary_prior_root,
            global_index=global_index,
            sample_index=sample_index,
            binary=False,
            grayscale=True,
        )

    # Hard boundary visualization.
    if "control_boundary_hard" in images:
        _save_tensor_images(
            tensor=images["control_boundary_hard"],
            save_root=boundary_hard_root,
            global_index=global_index,
            sample_index=sample_index,
            binary=True,
            grayscale=True,
        )

    # Compatibility with previous region-boundary version.
    if "control_boundary" in images:
        _save_tensor_images(
            tensor=images["control_boundary"],
            save_root=boundary_root,
            global_index=global_index,
            sample_index=sample_index,
            binary=True,
            grayscale=True,
        )


def print_sbp_pg_config():
    print("[SBP-PG / Ablation Inference Config]")

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
        "ENABLE_TOLERANCE_BAND_LOSS",
        "LAMBDA_BAND",
        "BOUNDARY_BAND_T_GATE",
        "USE_IMAGE_CONTROL",
    ]

    for k in keys:
        print(f"  {k} = {os.environ.get(k, '<default>')}")


if __name__ == "__main__":
    print_sbp_pg_config()

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
                        sampler=args.sampler,
                        dpm_steps=args.dpm_steps,
                        hybrid_split=args.hybrid_split,
                        unconditional_guidance_scale=args.cfg,
                    )

                    for k in images:
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            images[k] = torch.clamp(images[k], -1.0, 1.0)

                    log_local(
                        finaldir,
                        images,
                        global_index=batch_id,
                        sample_index=s,
                    )

    print(f"[OK] inference finished. Results saved to: {finaldir}")
