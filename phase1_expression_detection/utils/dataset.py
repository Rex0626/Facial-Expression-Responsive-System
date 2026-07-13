from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import (
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)

from utils.transforms import (get_train_transform, get_val_transform)


def get_dataloaders():
    train_dataset = ImageFolder(root=DATASET_DIR / "train", transform=get_train_transform())
    val_dataset = ImageFolder(root=DATASET_DIR / "val", transform=get_val_transform())
    test_dataset = ImageFolder(root=DATASET_DIR / "test", transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (train_loader, val_loader, test_loader)