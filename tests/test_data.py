import os
import pytest
import torch
from src.data.datamodule import SolarDataModule
from src.data.dataset import SolarPanelDataset, get_default_transforms

DATA_DIR = r"C:\Users\USER\.cache\kagglehub\datasets\salonipandagale\solar-panel-defect-classification-dl-project\versions\1\Data"

def test_data_dir_exists():
    assert os.path.exists(DATA_DIR), f"Dataset directory not found: {DATA_DIR}"

def test_datamodule_preparation():
    dm = SolarDataModule(data_dir=DATA_DIR, img_size=224, batch_size=4)
    assert len(dm.classes) == 6
    assert len(dm.train_df) + len(dm.val_df) == len(dm.df)
    assert dm.class_weights_tensor.shape[0] == 6

def test_train_dataloader_tensor_shapes():
    dm = SolarDataModule(data_dir=DATA_DIR, img_size=224, batch_size=4)
    train_loader = dm.train_dataloader()

    imgs, labels, paths = next(iter(train_loader))

    assert imgs.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert len(paths) == 4
    assert imgs.dtype == torch.float32
