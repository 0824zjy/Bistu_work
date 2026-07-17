import os
import argparse
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from share import *
from tutorial_dataset_sample import MyDataset
from cldm.model import create_model, load_state_dict

pl.seed_everything(0, workers=True)

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, default=os.environ.get("CKPT_PATH", "./merged_pytorch_model.pth"))
    p.add_argument("--prompt_json", type=str, default=os.environ.get("PROMPT_JSON", "./data/prompt_test.json"))
    p.add_argument("--out_dir", type=str, default=os.environ.get("OUT_DIR", "/data/zjy_work/BGDiff/Ours/results/exp_mask2img/"))
    p.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda:0"))

    # 吞吐 batch（一次取多少条样本）
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))

    # 每条样本生成多少张（多样性次数）
    p.add_argument("--n_samples", type=int, default=int(os.environ.get("N_SAMPLES", "5")))

    # sampling
    p.add_argument("--sampler", type=str, default=os.environ.get("SAMPLER", "ddim"), choices=["ddim", "dpm", "hybrid"])
    p.add_argument("--ddim_steps", type=int, default=int(os.environ.get("DDIM_STEPS", "50")))
    p.add_argument("--dpm_steps", type=int, default=int(os.environ.get("DPM_STEPS", "20")))
    p.add_argument("--hybrid_split", type=float, default=float(os.environ.get("HYBRID_SPLIT", "0.5")))
    p.add_argument("--cfg", type=float, default=float(os.environ.get("CFG", "9.0")))

    # optional enhancement
    p.add_argument("--use_image_control", action="store_true",
                   default=os.environ.get("USE_IMAGE_CONTROL", "0") == "1")

    return p.parse_args()

args = parse_args()

RESULT_DIR = args.out_dir
os.makedirs(RESULT_DIR, exist_ok=True)

learning_rate = 1e-5
sd_locked = False
only_mid_control = False

def get_model():
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location='cpu'), strict=False)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    model.to(args.device)
    model.eval()
    return model

def log_local(save_dir, images, global_index, sample_index=None):
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")
    boundary_root = os.path.join(save_dir, "boundaries")

    if "samples" in images and isinstance(images["samples"], torch.Tensor):
        for i, image in enumerate(images["samples"]):
            image = (image + 1.0) / 2.0
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)

            if sample_index is None:
                filename = f"id-{global_index:06}_idx-{i}.png"
            else:
                filename = f"id-{global_index:06}_s-{sample_index:02}_idx-{i}.png"

            path = os.path.join(samples_root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(image).save(path)

    if "control_mask" in images and isinstance(images["control_mask"], torch.Tensor):
        for i, image in enumerate(images["control_mask"]):
            image = image.permute(1, 2, 0).numpy()
            if image.shape[-1] > 1:
                image = image.mean(axis=-1)
            mask = (image * 255).astype(np.uint8)

            if sample_index is None:
                filename = f"id-{global_index:06}_idx-{i}.png"
            else:
                filename = f"id-{global_index:06}_s-{sample_index:02}_idx-{i}.png"

            path = os.path.join(mask_root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(mask).convert('1').save(path)

    if "control_boundary" in images and isinstance(images["control_boundary"], torch.Tensor):
        for i, image in enumerate(images["control_boundary"]):
            image = image.permute(1, 2, 0).numpy()
            if image.shape[-1] > 1:
                image = image.mean(axis=-1)
            bd = (image * 255).astype(np.uint8)

            if sample_index is None:
                filename = f"id-{global_index:06}_idx-{i}.png"
            else:
                filename = f"id-{global_index:06}_s-{sample_index:02}_idx-{i}.png"

            path = os.path.join(boundary_root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(bd).convert('L').save(path)

if __name__ == "__main__":
    model = get_model()

    dataset = MyDataset(prompt_json=args.prompt_json, size=384)

    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    finaldir = RESULT_DIR
    os.makedirs(finaldir, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            for batch_id, batch in enumerate(dataloader):
                #  batch_size=1；>1，这里也能跑，但 global_index 需要更复杂映射
                if args.batch_size != 1:
                    print("[WARN] batch_size=1 才能确保“每条样本生成 n_samples 张”语义严格对应。")

                print(f"Processing batch {batch_id}, batch_size={args.batch_size}, n_samples={args.n_samples}")

                # 对同一条（或同一批）数据重复采样 n_samples 次
                for s in range(args.n_samples):
                    images = model.log_images(
                        batch,
                        N=1,  # 每次采样输出1张（方案A）
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

                    # global_index：当 batch_size=1 时，batch_id 就是样本 id
                    # 文件名包含 s，避免覆盖
                    log_local(finaldir, images, global_index=batch_id, sample_index=s)
