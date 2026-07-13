# Weekly Development Log

This document records the weekly development progress of the **Facial Expression & Responsive System**.

---

# Project Roadmap

```
Phase 1
Facial Expression Detection

    Step 1 Dataset
    Step 2 Model
    Step 3 Training Pipeline
    Step 4 Evaluation
    Step 5 Real-time Camera

Phase 2
Emotion Memory

Phase 3
Emotion-aware Response System
```

---

# Current Progress

## Phase 1 - Facial Expression Detection

| Step                     |     Status    |
| ------------------------ | :-----------: |
| Step 1 Dataset           |  ✅ Completed  |
| Step 2 Model             |  ✅ Completed  |
| Step 3 Training Pipeline |  ✅ Completed  |
| Step 4 Evaluation        | ⬜ In Progress |
| Step 5 Real-time Camera  | ⬜ Not Started |

## Phase 2 - Emotion Memory

⬜ Not Started

## Phase 3 - Emotion-aware Response System

⬜ Not Started

---

# Week 1

## Goal

Initialize the project and establish the complete development environment for Phase 1.

## Completed Tasks

### Project Initialization

* Created GitHub repository.
* Designed the three-phase project architecture.
* Planned the complete development roadmap.

### Development Environment

Successfully configured:

* Python 3.10
* PyTorch
* Torchvision
* OpenCV
* MediaPipe
* CUDA-enabled PyTorch

### Dataset

* Selected FER2013 as the facial expression dataset.
* Organized the dataset into ImageFolder format.
* Built the dataset directory structure for training, validation, and testing.

### Model

Implemented a ResNet18-based facial expression classifier.

Completed:

* Modified the first convolution layer for grayscale images.
* Modified the final fully connected layer for seven emotion classes.

Emotion Categories:

```
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

### Model Verification

Successfully completed forward inference.

Result:

```
Device: CPU

Output:
torch.Size([4, 7])
```

The model successfully accepts image tensors and outputs seven emotion prediction scores.

---

## Next Week Goals

* Implement DataLoader
* Complete image preprocessing
* Build the training pipeline
* Train the emotion classification model
* Save model checkpoints

---

# Week 2

## Goal

Complete the entire training pipeline for facial expression recognition.

## Development Summary

### Environment

* Installed CUDA-enabled PyTorch.
* Successfully enabled GPU acceleration.
* Verified RTX 3060 CUDA execution.

### Data Pipeline

Completed:

* Image preprocessing
* Data augmentation
* ImageFolder dataset loading
* PyTorch DataLoader

### Training Pipeline

Implemented:

* Trainer class
* Accuracy metrics
* Training loop
* Validation loop
* CUDA training
* Automatic best model checkpoint saving

### Training Result

Successfully trained the ResNet18 emotion classifier.

Current validation result:

```
Epoch 1
Train Accuracy      : 46.75%
Validation Accuracy : 55.91%

Epoch 2
Train Accuracy      : 57.25%
Validation Accuracy : 58.61%
```

Best model successfully saved:

```
checkpoints/
└── best_model.pth
```

### Problems Solved

During development, the following issues were resolved:

* Fixed Windows DataLoader multiprocessing issue (`num_workers` configuration).
* Installed CUDA-enabled PyTorch.
* Enabled GPU training successfully.
* Implemented model checkpoint saving.
* Fixed checkpoint initialization bug (`best_val_acc`).

## Next Week Goals

### Phase 1 Evaluation

Planned tasks:

* Build `evaluate.py`
* Evaluate the best model on the test dataset
* Generate confusion matrix
* Generate classification report
* Implement single-image inference (`predict.py`)

---

# Week 3

## Goal

Develop the Emotion Memory module.

## Progress

(To be updated)
