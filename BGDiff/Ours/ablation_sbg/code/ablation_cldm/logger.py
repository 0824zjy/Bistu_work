import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    """
    Save only two sets of images:
      - *_latest.png : always overwritten each validation epoch
      - *_best.png   : overwritten only when monitored metric improves

    Images are saved under:
      <save_dir>/image_log/val/
    """

    def __init__(
        self,
        monitor="val/loss_simple_ema",
        mode="min",                 # "min" or "max"
        max_images=4,
        clamp=True,
        rescale=True,
        disabled=False,
        log_images_kwargs=None,
        save_split="val",           # save to image_log/val
    ):
        super().__init__()
        assert mode in ["min", "max"]
        self.monitor = monitor
        self.mode = mode
        self.max_images = max_images
        self.clamp = clamp
        self.rescale = rescale
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.save_split = save_split

        self.best_score = None
        self.cached_batch = None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    @rank_zero_only
    def _save_images_fixedname(self, save_dir, split, images, tag: str):
        """
        Save images with fixed filenames (overwrite).
        Example:
          control_mask_best.png
          samples_cfg_scale_9.00_mask_latest.png
          blended_best.png
        """
        root = os.path.join(save_dir, "image_log_Stage1-2", split)
        os.makedirs(root, exist_ok=True)

        path_image = {}

        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # [-1,1] -> [0,1]
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            img = Image.fromarray(grid)
            filename = f"{k}_{tag}.png"  # fixed filename
            img.save(os.path.join(root, filename))
            path_image[k] = img

        # blended (optional)
        if "control_mask" in path_image and "samples_cfg_scale_9.00_mask" in path_image:
            blended = Image.blend(
                path_image["control_mask"],
                path_image["samples_cfg_scale_9.00_mask"],
                alpha=0.5,
            )
            blended.save(os.path.join(root, f"blended_{tag}.png"))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Cache a fixed batch (batch_idx==0) for consistent visualization across epochs.
        """
        if self.disabled:
            return
        if batch_idx == 0 and self.cached_batch is None:
            self.cached_batch = batch

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        At each validation epoch end:
          - always save latest images (overwrite)
          - if metric improves, save best images (overwrite)
        """
        if self.disabled:
            return
        if self.cached_batch is None:
            return
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        # fetch metric
        metrics = trainer.callback_metrics
        score = None
        if self.monitor in metrics:
            try:
                score = metrics[self.monitor].detach().float().cpu().item()
            except Exception:
                score = None

        # switch to eval for logging
        was_train = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(self.cached_batch, split=self.save_split, **self.log_images_kwargs)

        # postprocess tensors
        for k in list(images.keys()):
            x = images[k]
            if isinstance(x, torch.Tensor):
                N = min(x.shape[0], self.max_images)
                x = x[:N].detach().cpu()
                if self.clamp:
                    x = torch.clamp(x, -1.0, 1.0)
                images[k] = x

        # 1) always overwrite latest
        self._save_images_fixedname(pl_module.logger.save_dir, self.save_split, images, tag="latest")

        # 2) overwrite best only if improved
        if score is not None and self._is_better(score):
            self.best_score = score
            self._save_images_fixedname(pl_module.logger.save_dir, self.save_split, images, tag="best")

        if was_train:
            pl_module.train()
