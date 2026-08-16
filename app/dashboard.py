import os
import io
import time
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn.functional as F
from src.models.swin_classifier import build_model
from src.explainability.gradcam import UniversalGradCAM, generate_gradcam_overlay
from src.data.dataset import get_default_transforms

st.set_page_config(
    page_title="Solar Guard AI - Visual Defect Inspection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF8C00, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #FF8C00;
    }
</style>
""", unsafe_allow_html=True)

CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

@st.cache_resource
def load_local_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_layer = build_model('swin_t', num_classes=len(CLASSES), pretrained=True)
    model = model.to(device)
    model.eval()
    grad_cam_engine = UniversalGradCAM(model, target_layer)
    _, val_tfms = get_default_transforms(img_size=224)
    return model, target_layer, grad_cam_engine, val_tfms, device

model, target_layer, grad_cam_engine, val_tfms, device = load_local_model()

# Header
st.markdown('<div class="main-header">☀️ Solar Guard AI: Edge Inspection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Autonomous Photovoltaic Defect Classification & Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

# Sidebar Options
st.sidebar.header("⚙️ Platform Settings")
inference_mode = st.sidebar.radio("Inference Engine:", ["Local PyTorch (Swin-T)", "FastAPI REST Server"])
api_url = st.sidebar.text_input("FastAPI Endpoint:", "http://127.0.0.1:8000/predict")
confidence_threshold = st.sidebar.slider("Anomaly Alarm Threshold", 0.50, 0.99, 0.85)

# Main UI Columns
col_upload, col_display = st.columns([1, 1.2])

with col_upload:
    st.subheader("📥 Upload Solar Panel Image")
    uploaded_file = st.file_uploader("Choose a thermal or optical cell image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert('RGB')
        st.image(img_pil, caption="Uploaded Panel Inspection Image", use_container_width=True)

with col_display:
    st.subheader("🔍 Inspection & Explainability Analysis")

    if uploaded_file is not None:
        if st.button("⚡ Run AI Quality Inspection", type="primary", use_container_width=True):
            t0 = time.time()

            if inference_mode == "FastAPI REST Server":
                try:
                    uploaded_file.seek(0)
                    response = requests.post(api_url, files={"file": ("img.jpg", uploaded_file.getvalue(), "image/jpeg")})
                    if response.status_code == 200:
                        res = response.json()
                        pred_class = res["predicted_class"]
                        conf = res["confidence"]
                        probs = res["probabilities"]
                        latency_ms = res["latency_ms"]

                        # Decode base64
                        import base64
                        overlay_bytes = base64.b64decode(res["gradcam_base64"])
                        overlay_img = Image.open(io.BytesIO(overlay_bytes))
                    else:
                        st.error(f"FastAPI Error: {response.text}")
                        st.stop()
                except Exception as e:
                    st.warning(f"Could not connect to FastAPI server ({e}). Falling back to Local PyTorch engine.")
                    inference_mode = "Local PyTorch (Swin-T)"

            if inference_mode == "Local PyTorch (Swin-T)":
                img_tensor = val_tfms(img_pil).unsqueeze(0).to(device)
                cam, target_cls, conf = grad_cam_engine(img_tensor)

                with torch.no_grad():
                    logits = model(img_tensor)
                    probs_tensor = F.softmax(logits, dim=1)[0]
                
                probs = {CLASSES[i]: float(probs_tensor[i]) for i in range(len(CLASSES))}
                pred_class = CLASSES[target_cls]
                latency_ms = round((time.time() - t0) * 1000.0, 2)

                raw_img, heatmap, superimposed = generate_gradcam_overlay(img_pil, cam)
                overlay_img = Image.fromarray(superimposed)

            # Display Result Metric Cards
            m1, m2, m3 = st.columns(3)
            with m1:
                status_color = "🔴" if pred_class not in ["Clean"] else "🟢"
                st.metric("Status", f"{status_color} {pred_class}")
            with m2:
                st.metric("Confidence", f"{conf * 100:.1f}%")
            with m3:
                st.metric("Latency", f"{latency_ms} ms")

            # Display Grad-CAM Heatmap
            st.markdown("#### 🎯 Grad-CAM Defect Localization Heatmap")
            st.image(overlay_img, caption="Red highlights indicate exact panel anomaly focus regions", use_container_width=True)

            # Class Probability Breakdown
            st.markdown("#### 📊 Probability Distribution")
            for cls_name, prob in probs.items():
                st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                st.progress(float(prob))

    else:
        st.info("Please upload a solar panel image on the left panel to trigger automated inspection.")
