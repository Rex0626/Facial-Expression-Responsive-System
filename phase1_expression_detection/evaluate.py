import os
import json
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from config import *
from models.resnet18_emotion import EmotionResNet18
from utils.dataset import get_dataloaders


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    print(f"Device: {DEVICE}")
    os.makedirs("results", exist_ok=True)

    # ===============================
    # Load Model
    # ===============================

    model = EmotionResNet18()
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # ===============================
    # Load Dataset
    # ===============================
    _, _, test_loader = get_dataloaders()

    # ===============================
    # Emotion Names
    # ===============================
    emotion_names = test_loader.dataset.classes
    all_preds = []
    all_labels = []

    # ===============================
    # Prediction
    # ===============================
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ===============================
    # Accuracy
    # ===============================
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy : {acc*100:.2f}%")

    # ===============================
    # Classification Report
    # ===============================
    report = classification_report(all_labels, all_preds, target_names=emotion_names)
    print("\nClassification Report\n")
    print(report)
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # ===============================
    # Confusion Matrix
    # ===============================
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()

    # ===============================
    # Save JSON
    # ===============================
    result = {"accuracy": float(acc)}
    with open("results/test_result.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nEvaluation Finished!")
    print("Results saved to results/")


if __name__ == "__main__":
    main()