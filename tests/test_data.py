import os
import pytest
import torch
from PIL import Image
from src.data.datamodule import SolarDataModule
from src.data.dataset import SolarPanelDataset, get_default_transforms

CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
LOCAL_DATA_DIR = r"C:\Users\USER\.cache\kagglehub\datasets\salonipandagale\solar-panel-defect-classification-dl-project\versions\1\Data"

@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    """
    Returns local dataset directory if available, otherwise creates a synthetic
    dataset directory for CI/CD runners (e.g., GitHub Actions).
    """
    if os.path.exists(LOCAL_DATA_DIR):
        return LOCAL_DATA_DIR

    # Create synthetic dataset for CI/CD environment
    base_dir = tmp_path_factory.mktemp("synthetic_solar_data")
    for cls_name in CLASSES:
        cls_dir = base_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i * 40, 100, 150))
            img.save(cls_dir / f"img_{i}.jpg")

    return str(base_dir)

def test_data_dir_exists(dataset_path):
    assert os.path.exists(dataset_path), f"Dataset directory not found: {dataset_path}"

def test_datamodule_preparation(dataset_path):
    dm = SolarDataModule(data_dir=dataset_path, img_size=224, batch_size=4)
    assert len(dm.classes) == 6
    assert len(dm.train_df) + len(dm.val_df) == len(dm.df)
    assert dm.class_weights_tensor.shape[0] == 6

def test_train_dataloader_tensor_shapes(dataset_path):
    dm = SolarDataModule(data_dir=dataset_path, img_size=224, batch_size=4)
    train_loader = dm.train_dataloader()

    imgs, labels, paths = next(iter(train_loader))

    assert imgs.shape[0] <= 4
    assert imgs.shape[1:] == (3, 224, 224)
    assert len(labels) == len(imgs)
    assert len(paths) == len(imgs)
    assert imgs.dtype == torch.float32
