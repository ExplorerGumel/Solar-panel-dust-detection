import os
import sys
import glob
import time
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

# Set seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# ---------------------------------------------------------------------------
# 1. Dataset Resolution & Path Configuration
# ---------------------------------------------------------------------------
def find_dataset_path():
    possible_paths = [
        r"C:\Users\USER\.cache\kagglehub\datasets\salonipandagale\solar-panel-defect-classification-dl-project\versions\1\Data",
        r"C:\Users\USER\.cache\kagglehub\datasets\salonipandagale\solar-panel-defect-classification-dl-project\1\Data",
        "/kaggle/input/solar-panel-defect-classification-dl-project/Data",
        "./Data"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    
    # Fallback to kagglehub if available
    try:
        import kagglehub
        path = kagglehub.dataset_download('salonipandagale/solar-panel-defect-classification-dl-project')
        data_path = os.path.join(path, "Data")
        if os.path.exists(data_path):
            return data_path
        return path
    except Exception as e:
        raise FileNotFoundError(f"Could not locate Kaggle dataset. Searched: {possible_paths}. Error: {e}")

DATA_DIR = find_dataset_path()
print(f"[INFO] Using dataset directory: {DATA_DIR}")

# ---------------------------------------------------------------------------
# 2. Data Indexing & Analysis
# ---------------------------------------------------------------------------
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

print(f"[INFO] Detected {len(classes)} classes: {classes}")

image_paths = []
labels = []
for cls_name in classes:
    cls_folder = os.path.join(DATA_DIR, cls_name)
    for fname in os.listdir(cls_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            image_paths.append(os.path.join(cls_folder, fname))
            labels.append(class_to_idx[cls_name])

df = pd.DataFrame({'path': image_paths, 'label': labels, 'class_name': [idx_to_class[l] for l in labels]})
print("\n[INFO] Dataset Class Distribution:")
print(df['class_name'].value_counts())

# Train/Val Split (80/20 Stratified)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
print(f"\n[INFO] Train samples: {len(train_df)} | Val samples: {len(val_df)}")

# Calculate class frequencies for Focal Loss / Class Weighting
class_counts = train_df['label'].value_counts().sort_index().values
total_train = len(train_df)
class_weights = total_train / (len(classes) * class_counts.astype(np.float32))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"[INFO] Computed Class Weights: {dict(zip(classes, np.round(class_weights, 3)))}")

# ---------------------------------------------------------------------------
# 3. PyTorch Dataset & Augmentations
# ---------------------------------------------------------------------------
class SolarPanelDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['label']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

# PyTorch Torchvision Transforms (Compatible with Kaggle out-of-the-box)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SolarPanelDataset(train_df, transform=train_transforms)
val_dataset = SolarPanelDataset(val_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# 4. Focal Loss Implementation
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Tensor of weights per class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]
            focal_loss = at * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ---------------------------------------------------------------------------
# 5. Model Definition (ResNet18 / ResNet50 Transfer Learning)
# ---------------------------------------------------------------------------
def build_model(num_classes=6, backbone='resnet18', pretrained=True):
    if backbone == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model

# ---------------------------------------------------------------------------
# 6. Grad-CAM Explainability Engine
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate gradients average across spatial dimensions
        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # [C, 1, 1]
        cam = torch.sum(weights * activations, dim=0) # [H, W]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
            
        return cam.cpu().numpy(), target_class, F.softmax(output, dim=1)[0, target_class].item()

def overlay_heatmap(img_path, cam, alpha=0.5):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_np = np.array(img) / 255.0
    
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
    cam_np = np.array(cam_img) / 255.0
    
    # Apply JET colormap using matplotlib
    colormap = plt.get_cmap('jet')
    heatmap = colormap(cam_np)[:, :, :3] # Drop alpha
    
    superimposed = (1 - alpha) * img_np + alpha * heatmap
    superimposed = np.clip(superimposed, 0, 1)
    
    return (img_np * 255).astype(np.uint8), (heatmap * 255).astype(np.uint8), (superimposed * 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# 7. Training Execution & Evaluation Loop
# ---------------------------------------------------------------------------
def train_and_evaluate(epochs=10, use_focal_loss=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Training on device: {device}")
    
    model = build_model(num_classes=len(classes), backbone='resnet18', pretrained=True).to(device)
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights_tensor.to(device), gamma=2.0)
        print("[INFO] Using Loss: Focal Loss (gamma=2.0, weighted)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
        print("[INFO] Using Loss: Weighted Cross-Entropy")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0.0
    output_dir = "./experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        scheduler.step()
        train_acc = correct / total
        train_loss = train_loss / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(val_dataset)
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_solar_model.pth"))
            
    print(f"\n[SUCCESS] Training Completed! Best Val Macro F1-Score: {best_f1:.4f}")
    
    # Final Evaluation & Classification Report
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_solar_model.pth")))
    model.eval()
    
    val_preds = []
    val_targets = []
    sample_paths = []
    with torch.no_grad():
        for imgs, targets, paths in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(targets.numpy())
            sample_paths.extend(paths)
            
    print("\n--- Detailed Classification Report ---")
    print(classification_report(val_targets, val_preds, target_names=classes))
    
    # ---------------------------------------------------------------------------
    # 8. Grad-CAM Visualization Generation
    # ---------------------------------------------------------------------------
    print("\n[INFO] Generating Grad-CAM Explainability Heatmaps for Sample Images...")
    grad_cam = GradCAM(model, model.layer4[-1])
    
    fig, axes = plt.subplots(len(classes), 3, figsize=(12, 4 * len(classes)))
    plt.suptitle("Solar Panel Defect Classification: Grad-CAM Explainability", fontsize=16, y=1.002)
    
    for idx, cls_name in enumerate(classes):
        # Pick first validation sample matching this class
        cls_idx = class_to_idx[cls_name]
        match_indices = [i for i, t in enumerate(val_targets) if t == cls_idx]
        if not match_indices:
            continue
        sample_i = match_indices[0]
        sample_path = sample_paths[sample_i]
        
        img_pil = Image.open(sample_path).convert('RGB')
        img_tensor = val_transforms(img_pil).unsqueeze(0).to(device)
        
        cam, pred_cls, conf = grad_cam(img_tensor, target_class=cls_idx)
        orig_img, heatmap, superimposed = overlay_heatmap(sample_path, cam)
        
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"True: {cls_name}", fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title(f"Grad-CAM Heatmap", fontsize=11)
        axes[idx, 1].axis('off')
        
        pred_label_str = idx_to_class[pred_cls]
        axes[idx, 2].imshow(superimposed)
        axes[idx, 2].set_title(f"Pred: {pred_label_str} ({conf*100:.1f}%)", fontsize=11, color='green' if pred_cls == cls_idx else 'red')
        axes[idx, 2].axis('off')
        
    plt.tight_layout()
    viz_save_path = os.path.join(output_dir, "grad_cam_summary.png")
    plt.savefig(viz_save_path, dpi=200, bbox_inches='tight')
    print(f"[SUCCESS] Grad-CAM summary saved to: {viz_save_path}")

if __name__ == "__main__":
    train_and_evaluate(epochs=8, use_focal_loss=True)
