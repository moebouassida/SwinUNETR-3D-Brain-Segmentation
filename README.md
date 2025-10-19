# 🧠 3D Brain Segmentation with SwinUNETR  
![Python](https://img.shields.io/badge/Python-3.10-blue) ![Docker](https://img.shields.io/badge/Docker-GPU-green) ![License](https://img.shields.io/badge/License-MIT-orange)  

**3D brain tumor segmentation using SwinUNETR with FastAPI, GPU Docker, MLflow tracking.**  

🚧 **Live demo is currently in progress!**

---

## ✅ Features

- 📤 Upload NIfTI MRI scans (`.nii` / `.nii.gz`)  
- 🧠 Predict tumor segmentation masks using **SwinUNETR (MONAI + PyTorch)**  
- 📁 Output is downloadable as a **NIfTI file**  
- ⚡ GPU acceleration supported (CUDA + Docker)  
- 📊 **MLflow** integration for tracking metrics, parameters, and model checkpoints  
- 📝 Interactive API documentation via **Swagger UI** (`/docs`)  
- 🧱 Modular, clean project structure for easy extension  

---

## 📁 Project Structure

| File / Folder | Description |
|---------------|-------------|
| `src/api/main.py` | FastAPI app + inference logic |
| `models/best_model.pth` | Trained SwinUNETR model checkpoint |
| `mlruns/` | MLflow experiment tracking folder |
| `Dockerfile` | GPU-enabled Docker image |
| `docker-compose.yml` | Optional Docker Compose setup |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🚀 Setup & Run

### Clone the repository  
Add your trained model to `models/best_model.pth`.

### Run Locally (without Docker)
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker (CPU)
``` bash docker build -t brain-tumor-api .
docker run -p 8000:8000 brain-tumor-api
```

### Run with GPU (Recommended)
Requires NVIDIA GPU and NVIDIA Container Toolkit:
``` bash
docker build -t brain-tumor-api-gpu .
docker run --gpus all -p 8000:8000 brain-tumor-api-gpu
```
Or using Docker Compose
``` bash
docker-compose up --build
```

### 📤 API Endpoints
Method	Endpoint	Description
POST	/predict	Upload MRI (.nii/.nii.gz) → returns tumor mask
GET	/health	Health check
GET	/docs	Interactive Swagger UI

Example request:
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/scan.nii.gz" --output tumor_mask.nii.gz

### 📊 MLflow Integration
Metrics and models are logged to ./mlruns.

Start MLflow UI:
```
mlflow ui --backend-store-uri ./mlruns
Open in browser: http://localhost:5000
```

### Tracked items:

✅ Loss & Dice score
✅ Hyperparameters (LR, batch size, epochs)
✅ Saved models

### ⚙ Technologies Used

| Component |	Technology |
|---------------|-------------|
| Backend API |	FastAPI |
| Medical AI |	MONAI + PyTorch |
| Model Type |	3D SwinUNETR |
| Experiment | Tracking	MLflow |
| Containerization |	Docker + NVIDIA Runtime |
| Input Format |	NIfTI (BRATS MRI) |
| Output Format |	NIfTI Segmentation Mask |

### 🔄 Roadmap
✅ GPU-enabled Docker image

✅ MLflow integration

📊 Web dashboard for predictions & visualization (in progress)

☁ Cloud deployment (AWS)
