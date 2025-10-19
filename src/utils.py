import os
import torch
import numpy as np
from functools import partial
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete

torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def get_losses_and_metrics():
    """Return loss, postprocessing and metric objects for multi-class segmentation."""
    # Use one-hot encoding and softmax for multi-class case
    loss = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True
    )

    # Post-processing: use softmax + argmax
    post_softmax = Activations(softmax=True)
    post_pred = AsDiscrete(argmax=True)

    # Dice metric (multi-class)
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )

    return loss, post_softmax, post_pred, dice_metric


def get_inferer(model, roi, sw_batch_size, overlap):
    return partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )


def save_checkpoint(model, epoch, best_val, outdir, fname, optimizer=None, scheduler=None):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, fname)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val": best_val,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path


def find_latest_checkpoint(outdir):
    """Find the most recent checkpoint in the directory."""
    if not os.path.isdir(outdir):
        return None
    ckpts = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".pt")]
    if not ckpts:
        return None
    latest = max(ckpts, key=os.path.getmtime)
    print(f"Latest checkpoint found: {latest}")
    return latest
