import torch
import torch.nn as nn
import os 
from torch.optim import Adam
from config import (DEVICE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE,)
from utils.dataset import get_dataloaders
from models.resnet18_emotion import EmotionResNet18
from utils.trainer import Trainer


def main():
    print("Device:", DEVICE)

    # Dataset
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = EmotionResNet18(num_classes=NUM_CLASSES,pretrained=True)
    model.to(DEVICE)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(model.parameters(),lr=LEARNING_RATE)
    trainer = Trainer(model,optimizer,criterion,DEVICE)
    best_val_acc = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = trainer.train_one_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]\n"
            f"Train Loss:{train_loss:.4f}\n"
            f"Train Accuracy:{train_acc*100:.2f}%\n"
            f"Validation Accuracy:{val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Saved Best Model!")

if __name__ == "__main__":
    main()