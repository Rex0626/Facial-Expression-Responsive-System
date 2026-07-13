from utils.dataset import get_dataloaders

from config import CLASS_NAMES


train_loader, val_loader, test_loader = get_dataloaders()

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print(f"Train batches : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")
print(f"Test batches : {len(test_loader)}")

images, labels = next(iter(train_loader))

print("\nImage Shape :", images.shape)
print("Label Shape :", labels.shape)

print("\nFirst Label :", CLASS_NAMES[labels[0]])