import pytest
import torch
from src.models.swin_classifier import build_model, SolarSwinClassifier
from src.models.losses import FocalLoss
from src.explainability.gradcam import UniversalGradCAM

@pytest.mark.parametrize("arch", ["resnet18", "convnext_tiny", "swin_t"])
def test_model_building(arch):
    model, target_layer = build_model(architecture=arch, num_classes=6, pretrained=False)
    assert model is not None
    assert target_layer is not None

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 6)

def test_focal_loss_forward():
    inputs = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 5, 3])
    weights = torch.tensor([1.0, 1.2, 0.8, 1.5, 2.0, 1.1])

    criterion = FocalLoss(alpha=weights, gamma=2.0)
    loss = criterion(inputs, targets)

    assert loss is not None
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0

def test_gradcam_engine():
    model, target_layer = build_model('resnet18', num_classes=6, pretrained=False)
    grad_cam = UniversalGradCAM(model, target_layer)

    dummy_input = torch.randn(1, 3, 224, 224)
    cam, target_cls, conf = grad_cam(dummy_input)

    assert cam is not None
    assert 0 <= target_cls < 6
    assert 0.0 <= conf <= 1.0

def test_fastapi_predict_endpoint():
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from app.api import app

    client = TestClient(app)
    img_pil = Image.new('RGB', (224, 224), color='blue')
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)

    response = client.post('/predict', files={'file': ('test.jpg', buf, 'image/jpeg')})
    assert response.status_code == 200
    data = response.json()
    assert 'predicted_class' in data
    assert 'confidence' in data
    assert 'probabilities' in data
    assert 'gradcam_base64' in data
