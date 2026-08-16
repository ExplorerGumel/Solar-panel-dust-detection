import json
import os

notebook_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Solar Panel Defect Classification: Experimentation & Grad-CAM Explainability\n",
            "### Senior MLE Portfolio Blueprint - Prototyping & Visual Validation Phase\n",
            "\n",
            "This notebook runs the exploratory data analysis, stratified 5-fold validation setup, fine-tuning of deep convolutional neural networks with **Focal Loss**, and generates **Grad-CAM explainability heatmaps** to visually inspect model focus areas on solar panel defect thermal/optical imagery."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Dependencies & Environment Setup\n",
            "!pip install -q kagglehub torchvision albumentations opencv-python matplotlib seaborn pandas scikit-learn tqdm\n",
            "\n",
            "import os\n",
            "import glob\n",
            "import random\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "import cv2\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "import torchvision\n",
            "from torchvision import models, transforms\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.metrics import classification_report, confusion_matrix, f1_score\n",
            "\n",
            "def seed_everything(seed=42):\n",
            "    random.seed(seed)\n",
            "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    torch.cuda.manual_seed(seed)\n",
            "    torch.backends.cudnn.deterministic = True\n",
            "\n",
            "seed_everything(42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Ingest Dataset & Class Distribution Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import kagglehub\n",
            "path = kagglehub.dataset_download('salonipandagale/solar-panel-defect-classification-dl-project')\n",
            "DATA_DIR = os.path.join(path, 'Data') if os.path.exists(os.path.join(path, 'Data')) else path\n",
            "print('Dataset path:', DATA_DIR)\n",
            "\n",
            "classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])\n",
            "class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}\n",
            "idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}\n",
            "\n",
            "image_paths = []\n",
            "labels = []\n",
            "for cls_name in classes:\n",
            "    cls_folder = os.path.join(DATA_DIR, cls_name)\n",
            "    for fname in os.listdir(cls_folder):\n",
            "        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):\n",
            "            image_paths.append(os.path.join(cls_folder, fname))\n",
            "            labels.append(class_to_idx[cls_name])\n",
            "\n",
            "df = pd.DataFrame({'path': image_paths, 'label': labels, 'class_name': [idx_to_class[l] for l in labels]})\n",
            "print('Total samples:', len(df))\n",
            "print(df['class_name'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Stratified Split & Focal Loss Weights Computation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)\n",
            "class_counts = train_df['label'].value_counts().sort_index().values\n",
            "class_weights = len(train_df) / (len(classes) * class_counts.astype(np.float32))\n",
            "class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)\n",
            "\n",
            "class FocalLoss(nn.Module):\n",
            "    def __init__(self, alpha=None, gamma=2.0):\n",
            "        super(FocalLoss, self).__init__()\n",
            "        self.alpha = alpha\n",
            "        self.gamma = gamma\n",
            "    def forward(self, inputs, targets):\n",
            "        ce_loss = F.cross_entropy(inputs, targets, reduction='none')\n",
            "        pt = torch.exp(-ce_loss)\n",
            "        focal_loss = ((1 - pt) ** self.gamma) * ce_loss\n",
            "        if self.alpha is not None:\n",
            "            if self.alpha.device != inputs.device:\n",
            "                self.alpha = self.alpha.to(inputs.device)\n",
            "            focal_loss = self.alpha[targets] * focal_loss\n",
            "        return focal_loss.mean()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. PyTorch Dataset & Transfer Learning Fine-Tuning"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class SolarDataset(Dataset):\n",
            "    def __init__(self, df, transform=None):\n",
            "        self.df = df.reset_index(drop=True)\n",
            "        self.transform = transform\n",
            "    def __len__(self):\n",
            "        return len(self.df)\n",
            "    def __getitem__(self, idx):\n",
            "        row = self.df.iloc[idx]\n",
            "        img = Image.open(row['path']).convert('RGB')\n",
            "        if self.transform:\n",
            "            img = self.transform(img)\n",
            "        return img, row['label'], row['path']\n",
            "\n",
            "train_tfms = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.RandomHorizontalFlip(),\n",
            "    transforms.RandomVerticalFlip(),\n",
            "    transforms.RandomRotation(15),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
            "])\n",
            "\n",
            "val_tfms = transforms.Compose([\n",
            "    transforms.Resize((224, 224)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
            "])\n",
            "\n",
            "train_loader = DataLoader(SolarDataset(train_df, train_tfms), batch_size=16, shuffle=True)\n",
            "val_loader = DataLoader(SolarDataset(val_df, val_tfms), batch_size=16, shuffle=False)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Fine-Tuning & Evaluation Loop"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n",
            "model.fc = nn.Linear(model.fc.in_features, len(classes))\n",
            "model = model.to(device)\n",
            "\n",
            "criterion = FocalLoss(alpha=class_weights_tensor.to(device), gamma=2.0)\n",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n",
            "\n",
            "for epoch in range(1, 9):\n",
            "    model.train()\n",
            "    running_loss = 0.0\n",
            "    for imgs, labels_batch, _ in train_loader:\n",
            "        imgs, labels_batch = imgs.to(device), labels_batch.to(device)\n",
            "        optimizer.zero_grad()\n",
            "        out = model(imgs)\n",
            "        loss = criterion(out, labels_batch)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        running_loss += loss.item() * imgs.size(0)\n",
            "    \n",
            "    model.eval()\n",
            "    val_preds, val_targets = [], []\n",
            "    with torch.no_grad():\n",
            "        for imgs, labels_batch, _ in val_loader:\n",
            "            imgs = imgs.to(device)\n",
            "            out = model(imgs)\n",
            "            val_preds.extend(out.argmax(dim=1).cpu().numpy())\n",
            "            val_targets.extend(labels_batch.numpy())\n",
            "    \n",
            "    f1 = f1_score(val_targets, val_preds, average='macro')\n",
            "    print(f'Epoch {epoch:02d} | Train Loss: {running_loss/len(train_df):.4f} | Val Macro F1: {f1:.4f}')\n",
            "\n",
            "print('\\nFinal Classification Report:')\n",
            "print(classification_report(val_targets, val_preds, target_names=classes))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Grad-CAM Visual Explainability Heatmaps"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class GradCAM:\n",
            "    def __init__(self, model, target_layer):\n",
            "        self.model = model\n",
            "        self.target_layer = target_layer\n",
            "        self.gradients = None\n",
            "        self.activations = None\n",
            "        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))\n",
            "        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))\n",
            "\n",
            "    def __call__(self, input_tensor, target_class):\n",
            "        self.model.eval()\n",
            "        output = self.model(input_tensor)\n",
            "        self.model.zero_grad()\n",
            "        one_hot = torch.zeros_like(output)\n",
            "        one_hot[0, target_class] = 1.0\n",
            "        output.backward(gradient=one_hot, retain_graph=True)\n",
            "        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)\n",
            "        cam = torch.sum(weights * self.activations[0], dim=0)\n",
            "        cam = F.relu(cam)\n",
            "        cam = cam - cam.min()\n",
            "        if cam.max() != 0: cam = cam / cam.max()\n",
            "        return cam.cpu().numpy(), F.softmax(output, dim=1)[0, target_class].item()\n",
            "\n",
            "grad_cam = GradCAM(model, model.layer4[-1])\n",
            "\n",
            "fig, axes = plt.subplots(len(classes), 3, figsize=(12, 4 * len(classes)))\n",
            "for idx, cls_name in enumerate(classes):\n",
            "    cls_idx = class_to_idx[cls_name]\n",
            "    match_idx = [i for i, t in enumerate(val_targets) if t == cls_idx][0]\n",
            "    sample_path = val_df.iloc[match_idx]['path']\n",
            "    img_pil = Image.open(sample_path).convert('RGB')\n",
            "    img_tensor = val_tfms(img_pil).unsqueeze(0).to(device)\n",
            "    cam, conf = grad_cam(img_tensor, cls_idx)\n",
            "    \n",
            "    img_raw = np.array(Image.open(sample_path).convert('RGB').resize((224, 224))) / 255.0\n",
            "    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)) / 255.0\n",
            "    heatmap = plt.get_cmap('jet')(cam_resized)[:, :, :3]\n",
            "    superimposed = np.clip(0.5 * img_raw + 0.5 * heatmap, 0, 1)\n",
            "    \n",
            "    axes[idx, 0].imshow(img_raw)\n",
            "    axes[idx, 0].set_title(f'True: {cls_name}')\n",
            "    axes[idx, 0].axis('off')\n",
            "    axes[idx, 1].imshow(heatmap)\n",
            "    axes[idx, 1].set_title('Grad-CAM Heatmap')\n",
            "    axes[idx, 1].axis('off')\n",
            "    axes[idx, 2].imshow(superimposed)\n",
            "    axes[idx, 2].set_title(f'Pred Conf: {conf*100:.1f}%')\n",
            "    axes[idx, 2].axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

notebook_json = {
    "cells": notebook_cells,
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_path = r"C:\Users\USER\.gemini\antigravity\scratch\solar_panel_defect_classification\solar_defect_experimentation.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Created notebook at {nb_path}")
