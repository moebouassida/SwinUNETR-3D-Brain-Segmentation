import os
import json
import torch
from monai import transforms
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from monai.transforms import MapTransform
import multiprocessing


def load_brats_data(data_dir, json_path, val_split=0.1):
    """
    Load BRATS dataset JSON and split into train/val sets.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    train_data = data["training"]
    for d in train_data:
        d["image"] = os.path.join(data_dir, d["image"].replace("./", ""))
        d["label"] = os.path.join(data_dir, d["label"].replace("./", ""))

    train_files, val_files = train_test_split(train_data, test_size=val_split, random_state=42)
    return train_files, val_files

def get_loaders(
    data_dir,
    json_path,
    roi=(128, 128, 128),
    batch_size=2,
    num_workers=None,
):
    """
    Return optimized train/val DataLoaders for BRATS dataset.
    """

    # Auto-optimize num_workers based on CPU cores
    if num_workers is None or num_workers <= 0:
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(16, max(4, cpu_count // 2))  # adaptive cap
        print(f"[INFO] Auto-selected num_workers={num_workers} based on {cpu_count} CPU cores")

    train_files, val_files = load_brats_data(data_dir, json_path)

    # ---------------------------
    # Training transforms
    # ---------------------------
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=[roi[0], roi[1], roi[2]],
            allow_smaller=True,
        ),
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])

    # ---------------------------
    # Validation transforms
    # ---------------------------
    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    # ---------------------------
    # Datasets
    # ---------------------------
    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_transform)

    # ---------------------------
    # DataLoaders (optimized)
    # ---------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,           # enables fast transfer to GPU
        persistent_workers=True,   # keeps workers alive between epochs
        prefetch_factor=4,         # each worker preloads batches
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(2, num_workers // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"[INFO] Train loader: {len(train_loader)} steps/epoch | Batch size: {batch_size}")
    print(f"[INFO] Val loader:   {len(val_loader)} steps/epoch")
    print(f"[INFO] num_workers={num_workers} | pin_memory=True | persistent_workers=True")

    return train_loader, val_loader
