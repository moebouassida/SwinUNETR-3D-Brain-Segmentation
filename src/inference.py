import os
import argparse
import time
import yaml
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from model import create_model
from utils import get_inferer, Activations, AsDiscrete
from dataset import load_brats_data
from monai.transforms import LoadImage

def load_config(path):
    return yaml.safe_load(open(path, "r"))


def save_nifti(arr, ref_path, out_path):
    try:
        ref = nib.load(str(ref_path))
        affine = ref.affine
    except Exception:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), affine), str(out_path))


def run_case(model, inferer, case_path, roi, device, out_path, post_softmax, post_pred):
    model.eval()
    loader = LoadImage(image_only=True)
    img, meta = loader(str(case_path))
    # Ensure shape (C,H,W,D)
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    elif img.ndim == 4:
        if img.shape[0] in (1,4):
            pass
        else:
            img = np.transpose(img, (3,0,1,2))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = inferer(img_tensor)
        # decollate-like postprocessing for single batch
        soft = post_softmax(logits)
        pred = post_pred(soft)
        out = pred.squeeze().cpu().numpy()
    t = time.time() - t0
    print(f"Inference done in {t:.2f}s | output shape: {out.shape}")
    os.makedirs(out_path.parent, exist_ok=True)
    save_nifti(out, case_path, out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--outdir", default="outputs/inference")
    parser.add_argument("--ckpt", default=None, help="optional checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = create_model(in_channels=cfg.get("in_channels", 4), out_channels=cfg.get("out_channels", 3), feature_size=cfg.get("feature_size", 48), use_checkpoint=cfg.get("use_checkpoint", False)).to(device)

    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"Loading checkpoint {args.ckpt}")
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"]) if "model_state_dict" in ck else model.load_state_dict(ck)

    post_softmax = Activations(softmax=True)
    post_pred = AsDiscrete(argmax=True)

    inferer = get_inferer(model, tuple(cfg.get("roi", (128,128,128))), cfg.get("sw_batch_size", 1), cfg.get("infer_overlap", 0.25))

    case_path = Path(args.case)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (case_path.stem + "_pred.nii.gz")

    run_case(model, inferer, case_path, cfg.get("roi", (128,128,128)), device, out_path, post_softmax, post_pred)


if __name__ == "__main__":
    main()