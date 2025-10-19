import os
import yaml
import time
import math
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from monai.data import decollate_batch
from functools import partial
from tqdm import tqdm

from dataset import get_loaders
from model import create_model
from utils import (
    AverageMeter,
    get_losses_and_metrics,
    get_inferer,
    save_checkpoint,
    find_latest_checkpoint,
)

# -------------------------
# Load config
# -------------------------
cfg = yaml.safe_load(open("src/config.yaml", "r"))
os.makedirs(cfg["root_dir"], exist_ok=True)

# -------------------------
# Setup device & seed
# -------------------------
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])
if device.type == "cuda":
    torch.cuda.manual_seed_all(cfg["seed"])

# -------------------------
# Initialize MLflow
# -------------------------
if cfg.get("mlflow_tracking_uri"):
    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
mlflow.set_experiment(cfg["experiment_name"])

# -------------------------
# Data
# -------------------------
train_loader, val_loader = get_loaders(
    data_dir=cfg["data_dir"],
    json_path=cfg["json_list"],
    roi=tuple(cfg["roi"]),
    batch_size=cfg["batch_size"],
    num_workers=cfg["num_workers"],
)

print(f"Steps per epoch: {len(train_loader)}")
print(f"Validation steps: {len(val_loader)}")

# -------------------------
# Model, losses, optimizer
# -------------------------
model = create_model(
    in_channels=cfg["in_channels"],
    out_channels=cfg["out_channels"],
    feature_size=cfg["feature_size"],
    use_checkpoint=bool(cfg.get("use_checkpoint", False)),
).to(device)

dice_loss, post_softmax, post_pred, dice_metric = get_losses_and_metrics()
model_inferer = get_inferer(model, tuple(cfg["roi"]), cfg["sw_batch_size"], cfg["infer_overlap"])

optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

# -------------------------
# Resume from checkpoint
# -------------------------
start_epoch = 0
best_val = -math.inf
resume_ckpt = cfg.get("resume_ckpt") or find_latest_checkpoint(cfg["root_dir"])

if resume_ckpt and os.path.isfile(resume_ckpt):
    print(f"=> Loading checkpoint from {resume_ckpt}")
    checkpoint = torch.load(resume_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val = checkpoint.get("best_val", -math.inf)
    print(f"=> Resumed from epoch {start_epoch}, best_val={best_val:.4f}")
else:
    print("=> No checkpoint found. Starting from scratch.")

# -------------------------
# Train / Validation loops
# -------------------------
def train_epoch(model, loader, optimizer, loss_func, epoch):
    model.train()
    run_loss = AverageMeter()
    epoch_start = time.time()

    progress = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}", ncols=100)
    for idx, batch_data in progress:
        optimizer.zero_grad(set_to_none=True)
        data = batch_data["image"].to(device, non_blocking=True)
        target = batch_data["label"].to(device, non_blocking=True)

        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = data.shape[0]
        run_loss.update(loss.item(), n=bsz)

        # --- Step logging ---
        if idx % 20 == 0:
            mlflow.log_metric("train_step_loss", loss.item(), step=epoch * len(loader) + idx)

        progress.set_postfix({"loss": f"{run_loss.avg:.4f}"})

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} min | Avg Loss: {run_loss.avg:.4f}")
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, model_inferer, post_softmax, post_pred):
    model.eval()
    acc_func.reset()
    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"[Val] Epoch {epoch+1}", ncols=100):
            data = batch_data["image"].to(device, non_blocking=True)
            target = batch_data["label"].to(device, non_blocking=True)
            logits = model_inferer(data)

            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_softmax(o)) for o in val_outputs_list]
            acc_func(y_pred=val_output_convert, y=val_labels_list)

    acc, not_nans = acc_func.aggregate()
    acc_np = acc.cpu().numpy()
    mean_acc = float(np.nanmean(acc_np))
    return acc_np, mean_acc


# -------------------------
# Trainer
# -------------------------
def trainer():
    global best_val
    mlflow_run = mlflow.start_run(run_name=f"fold_{cfg.get('fold', 0)}")
    mlflow.log_params({k: str(v) for k, v in cfg.items()})

    history = {"tr_loss": [], "val_mean_dice": [], "val_dice_tc": [], "val_dice_wt": [], "val_dice_et": []}

    try:
        for epoch in range(start_epoch, cfg["max_epochs"]):
            print(f"\n===== Epoch {epoch+1}/{cfg['max_epochs']} =====")
            tr_loss = train_epoch(model, train_loader, optimizer, dice_loss, epoch)
            scheduler.step()

            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            history["tr_loss"].append(tr_loss)

            # Validation
            if (epoch + 1) % cfg["val_every"] == 0 or epoch == 0:
                acc_np, val_mean = val_epoch(model, val_loader, epoch, dice_metric, model_inferer, post_softmax, post_pred)
                val_tc, val_wt, val_et = [float(v) if len(acc_np) > i else float("nan") for i, v in enumerate(acc_np[:3])]
                print(f"[Val] mean_dice {val_mean:.4f} tc {val_tc:.4f} wt {val_wt:.4f} et {val_et:.4f}")

                mlflow.log_metrics(
                    {"val_mean_dice": val_mean, "val_tc": val_tc, "val_wt": val_wt, "val_et": val_et},
                    step=epoch,
                )

                if val_mean > best_val:
                    best_val = val_mean
                    ckpt_path = save_checkpoint(
                        model,
                        epoch,
                        best_val,
                        cfg["root_dir"],
                        f"best_fold{cfg.get('fold', 0)}.pt",
                        optimizer,
                        scheduler,
                    )
                    mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                    try:
                        mlflow.pytorch.log_model(model, artifact_path="model")
                    except Exception as e:
                        print("Warning: mlflow.pytorch.log_model failed:", e)

            # Periodic save
            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, epoch, best_val, cfg["root_dir"], "last.pt", optimizer, scheduler)

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        ckpt = save_checkpoint(model, epoch, best_val, cfg["root_dir"], "interrupted.pt", optimizer, scheduler)
        mlflow.log_artifact(ckpt, artifact_path="checkpoints")

    finally:
        mlflow.log_metric("best_val_mean_dice", best_val)
        mlflow.end_run()
    return history


if __name__ == "__main__":
    trainer()
