import os
import torch
import torch.nn as nn
import torchvision.models as models

def build_model(architecture='swin_t', num_classes=6, pretrained=True):
    """
    Constructs vision model and modifies final classifier head.
    Returns: (model, target_layer_for_gradcam)
    """
    arch = architecture.lower()

    if arch == 'swin_t':
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]

    elif arch == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]

    elif arch == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer = model.layer4[-1]

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model, target_layer

class SolarSwinClassifier(nn.Module):
    """
    Enterprise PyTorch Wrapper for Swin Transformer Solar Panel Classifier.
    """
    def __init__(self, architecture='swin_t', num_classes=6, pretrained=True):
        super(SolarSwinClassifier, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.model, self.target_layer = build_model(architecture, num_classes, pretrained)

    def forward(self, x):
        return self.model(x)

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'state_dict': self.model.state_dict()
        }, path)

    def load_checkpoint(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
