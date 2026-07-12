from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from .transforms import (
    get_train_transform,
    get_test_transform,
)



def get_dataloaders(
    dataset_root,
    batch_size=32,
    num_workers=2,
):

    dataset_root = Path(dataset_root)


    # ======================
    # Load training dataset
    # ======================

    full_train_dataset = ImageFolder(
        dataset_root / "train",
        transform=get_train_transform()
    )


    # Split train / validation

    train_size = int(
        len(full_train_dataset) * 0.8
    )

    val_size = (
        len(full_train_dataset)
        - train_size
    )


    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [
            train_size,
            val_size
        ]
    )


    # Validation 不使用 augmentation

    val_dataset.dataset.transform = get_test_transform()



    # ======================
    # Test Dataset
    # ======================

    test_dataset = ImageFolder(
        dataset_root / "test",
        transform=get_test_transform()
    )


    # ======================
    # DataLoader
    # ======================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    classes = full_train_dataset.classes


    return (
        train_loader,
        val_loader,
        test_loader,
        classes
    )