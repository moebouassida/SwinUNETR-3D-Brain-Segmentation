from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import torch
from model import create_model
from utils import get_inferer, Activations, AsDiscrete
from pathlib import Path
import yaml
import numpy as np
from monai.transforms import LoadImage
import nibabel as nib
import uuid

app = FastAPI()

cfg = yaml.safe_load(open("src/config.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(
    in_channels=cfg.get("in_channels", 4),
    out_channels=cfg.get("out_channels", 3),
    feature_size=cfg.get("feature_size", 48),
).to(device)
model.eval()
post_softmax = Activations(softmax=True)
post_pred = AsDiscrete(argmax=True)
inferer = get_inferer(model, tuple(cfg.get("roi", (128,128,128))), cfg.get("sw_batch_size", 1), cfg.get("infer_overlap", 0.25))

@app.post("/infer")
async def infer_case(file: UploadFile = File(...)):
    try:
        data = await file.read()
        tmp_path = Path("/tmp") / f"{uuid.uuid4()}_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(data)
        loader = LoadImage(image_only=True)
        img, _ = loader(str(tmp_path))
        if img.ndim == 3:
            img = np.expand_dims(img, 0)
        elif img.ndim == 4 and img.shape[0] not in (1,4):
            img = np.transpose(img, (3,0,1,2))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = inferer(img_tensor)
            soft = post_softmax(logits)
            pred = post_pred(soft)
            out = pred.squeeze().cpu().numpy()
        out_path = Path("outputs/api_preds") / f"{tmp_path.stem}_pred.nii.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ref = nib.load(str(tmp_path))
        affine = ref.affine if hasattr(ref, "affine") else np.eye(4)
        nib.save(nib.Nifti1Image(out.astype(np.uint8), affine), str(out_path))
        return FileResponse(path=str(out_path), filename=out_path.name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)