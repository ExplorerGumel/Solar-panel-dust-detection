import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalGradCAM:
    """
    Architecture-Agile Grad-CAM Engine compatible with CNNs (ResNet, ConvNeXt)
    and Vision Transformers (Swin-T). Handles both [B, C, H, W] and [B, H, W, C] layouts.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        probs = F.softmax(output, dim=1)
        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        if grads.ndim == 3:
            if grads.shape[0] > grads.shape[-1]:  # Layout: [C, H, W]
                weights = torch.mean(grads, dim=(1, 2), keepdim=True)
                cam = torch.sum(weights * acts, dim=0)
            else:  # Layout: [H, W, C] (Swin Transformer)
                weights = torch.mean(grads, dim=(0, 1), keepdim=True)
                cam = torch.sum(weights * acts, dim=-1)
        else:
            raise ValueError(f"Unexpected gradient activation shape: {grads.shape}")

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), target_class, confidence

def generate_gradcam_overlay(img_path_or_pil, cam, alpha=0.5):
    """
    Overlays JET colormap heatmap onto raw solar panel RGB image using matplotlib.
    Returns: (raw_img_np, heatmap_np, superimposed_np)
    """
    if isinstance(img_path_or_pil, str):
        img_pil = Image.open(img_path_or_pil).convert('RGB').resize((224, 224))
    else:
        img_pil = img_path_or_pil.convert('RGB').resize((224, 224))

    img_np = np.array(img_pil) / 255.0

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
    cam_np = np.array(cam_img) / 255.0

    colormap = plt.get_cmap('jet')
    heatmap = colormap(cam_np)[:, :, :3]

    superimposed = np.clip((1 - alpha) * img_np + alpha * heatmap, 0.0, 1.0)

    return (img_np * 255).astype(np.uint8), (heatmap * 255).astype(np.uint8), (superimposed * 255).astype(np.uint8)
