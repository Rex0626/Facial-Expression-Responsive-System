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
    Step 5 Image Inference
    Step 6 Real-time Camera Demo

Phase 2
Emotion Memory
    Step 1 Database Layer (SQLite)
    Step 2 Smoothing Algorithm (EMA)
    Step 3 Trend Detection
    Step 4 Camera Capture Integration

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
| Step 4 Evaluation        | ✅ Completed |
| Step 5 Image Inference  | ✅ Completed |
| Step 6 Real-time Camera  | ⚠️ Implemented / Not Tested |

## Phase 2 - Emotion Memory
| Step                          |     Status    |
| ------------------------------ | :-----------: |
| Step 1 資料庫層 (SQLite)       |  ✅ Completed  |
| Step 2 平滑演算法 (EMA)        |  ✅ Completed  |
| Step 3 趨勢判斷                |  ✅ Completed  |
| Step 4 攝影機擷取整合          | ⚠️ Implemented / Not Tested |

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

### Model Evaluation
Implemented evaluation pipeline:
Completed:
* Test dataset evaluation.
* Accuracy calculation.
* Classification report generation.
* Confusion matrix visualization.
* Evaluation result saving.

Test Result:
Test Accuracy:
59.71%

Generated files:
```
results/
├── classification_report.txt
├── confusion_matrix.png
└── test_result.json
```

### Image Inference
Implemented:
* Loaded trained model checkpoint.
* Added single image prediction.
* Added confidence score calculation using Softmax.

Example:
* Emotion:Angry
* Confidence:30.01%

The trained model can now predict emotions from external images.

---

### Real-time Camera Demo
Implemented:
* OpenCV camera pipeline.
* MediaPipe face detection.
* Real-time facial expression recognition.

Pipeline:
```
Camera
↓
Face Detection
↓
Face Crop
↓
ResNet18 Emotion Model
↓
Emotion Prediction
```

Current Status:
Implemented but not tested due to unavailable camera hardware.

### Problems Solved
During development, the following issues were resolved:
* Fixed Windows DataLoader multiprocessing issue (`num_workers` configuration).
* Installed CUDA-enabled PyTorch.
* Enabled GPU training successfully.
* Implemented model checkpoint saving.
* Fixed checkpoint initialization bug (`best_val_acc`).


## Next Week Goals
### Phase 1 Completion
Tasks:
* Organize Phase 1 documentation.
* Create Phase 1 README.
* Test complete Phase 1 pipeline from dataset to inference.
* Prepare project release.

### Phase 2 Preparation
Tasks:
* Design Emotion Memory architecture.
* Define emotion history data structure.
* Plan temporal emotion modeling approach.

---

# Week 3
## Goal
Start Phase 2: Emotion Memory Module.

## Planned Tasks
* Design emotion memory architecture.
* Implement emotion history recording.
* Explore temporal emotion modeling methods.
* Prepare LSTM / GRU / Transformer based memory model.

## Progress

### 架構設計
完成 Phase 2 資料夾架構規劃，對應 `repository.md` 原始設計：
* `database/` — 逐幀情緒紀錄儲存
* `memory_model/` — 保留給之後的時序模型（GRU/LSTM），目前為空

### 資料儲存層
實作 SQLite 資料庫層（`database/db.py`）：
* 逐幀紀錄包含 `session_id`、`timestamp`、`emotion`、`confidence`、完整機率分佈
* 用 `session_id` 區分每次攝影機觀察區間
* 保留原始（未平滑）資料，供之後訓練模型使用

### 情緒記憶核心邏輯
實作 `emotion_memory.py`：
* EMA（指數移動平均）平滑，處理單幀雜訊
* 情緒持續時間計算（容許短暫中斷）
* 趨勢判斷（improving / worsening / stable），採用機率加權平均效價，而非單純 argmax 比較

### 攝影機整合
實作 `capture_session.py`，沿用 Phase 1 訓練好的模型與 MediaPipe 人臉偵測，
逐幀寫入資料庫，並即時顯示平滑後的情緒。

### 測試
以模擬資料驗證核心邏輯（`emotion_memory.py` 的平滑與趨勢判斷），
邏輯正確。**尚未用真實攝影機資料實際測試**，這是下週的首要工作。

## Next Steps
* 實際執行 `capture_session.py`，錄製多段真實情緒序列
* 檢視平滑效果與趨勢判斷是否符合直覺
* 視情況調整 `SMOOTHING_WINDOW`、`TREND_THRESHOLD` 等參數
* 累積足夠資料後評估是否需要訓練時序模型取代目前的統計 baseline
