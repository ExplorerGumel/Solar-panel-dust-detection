import os
import io
import time
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F

from src.models.swin_classifier import build_model
from src.explainability.gradcam import UniversalGradCAM, generate_gradcam_overlay
from src.data.dataset import get_default_transforms

app = FastAPI(
    title="Solar Panel Defect Inspection API",
    description="Enterprise Edge AI Inference API for Solar Panel Quality & Anomaly Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Global Model & GradCAM Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, target_layer = build_model(architecture='swin_t', num_classes=len(CLASSES), pretrained=True)
model = model.to(device)
model.eval()

grad_cam_engine = UniversalGradCAM(model, target_layer)
_, val_transforms = get_default_transforms(img_size=224)

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    latency_ms: float
    gradcam_base64: str

@app.get("/")
def read_root():
    return {
        "service": "Solar Panel Defect Inspection API",
        "status": "online",
        "model_architecture": "Swin Transformer (Swin-T)",
        "num_classes": len(CLASSES),
        "classes": CLASSES
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_defect(file: UploadFile = File(...)):
    """
    Inference endpoint for solar panel defect classification and Grad-CAM localization.
    Accepts uploaded thermal or optical panel image (JPEG/PNG/BMP/TIFF).
    """
    t0 = time.time()

    # Read uploaded bytes
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse image file: {str(e)}")

    try:
        # Preprocess image
        img_tensor = val_transforms(img_pil).unsqueeze(0).to(device)

        # Compute Grad-CAM heatmap
        model.zero_grad()
        cam, target_cls, conf = grad_cam_engine(img_tensor)

        # Compute class probabilities
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        probs_dict = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
        predicted_label = CLASSES[target_cls]

        # Generate Grad-CAM overlay
        _, _, superimposed = generate_gradcam_overlay(img_pil, cam)

        # Convert overlay image to Base64 PNG
        overlay_pil = Image.fromarray(superimposed)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        gradcam_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        latency_ms = round((time.time() - t0) * 1000.0, 2)

        return PredictionResponse(
            predicted_class=predicted_label,
            confidence=round(float(conf), 4),
            probabilities=probs_dict,
            latency_ms=latency_ms,
            gradcam_base64=gradcam_base64
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR in /predict]: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
