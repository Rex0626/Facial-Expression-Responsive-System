"""
Project Configuration
Facial Expression & Responsive System
Phase 1 - Emotion Recognition
"""

import torch
from pathlib import Path

# ==========================
# Project Path
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "fer2013"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# Dataset
# ==========================
IMAGE_SIZE = 112
NUM_CLASSES = 7
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================
# Training
# ==========================
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42


# ==========================
# Model
# ==========================
MODEL_NAME = "resnet18"
PRETRAINED = True


# ==========================
# Device
# ==========================
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)