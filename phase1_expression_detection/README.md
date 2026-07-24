# Phase 1 - Facial Expression Detection

## 目標

建立一個基於深度學習的臉部表情辨識模型，從人臉影像判斷七種基本情緒，
作為整個系統（Facial Expression & Responsive System）的感知層，
後續 Phase 2、3 會建立在這個模型的輸出之上。

---

## 架構

```
影像輸入
    |
    v
人臉偵測 (MediaPipe)
    |
    v
影像前處理 (灰階化 / resize / normalize)
    |
    v
ResNet18 (修改為單通道輸入、7 類輸出)
    |
    v
情緒預測 (7 類機率分佈)
```

---

## 資料集

**FER2013**，七種情緒類別：

```
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

資料集以 `ImageFolder` 格式組織，切分為 `train / val / test` 三個子集，實際資料不進版控（見 `.gitignore`）。

---

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `config.py` | 集中管理路徑、資料集參數、訓練超參數、裝置設定 |
| `models/resnet18_emotion.py` | 模型定義：以 ImageNet 預訓練的 ResNet18 為基礎，修改第一層 conv 適應灰階輸入，修改最後 fc 層輸出 7 類 |
| `utils/dataset.py` | 建立 train / val / test 的 `DataLoader` |
| `utils/transforms.py` | 訓練集（含資料增強：翻轉、旋轉）與驗證/測試集的影像前處理 pipeline |
| `utils/trainer.py` | `Trainer` 類別，封裝一個 epoch 的訓練與驗證邏輯 |
| `utils/metrics.py` | 準確率計算 |
| `train.py` | 訓練主程式，訓練過程中儲存驗證準確率最高的模型 |
| `evaluate.py` | 在測試集上評估最佳模型，輸出準確率、分類報告、混淆矩陣 |
| `predict.py` | 讀取單張圖片，輸出預測情緒與信心分數 |
| `camera_demo.py` | 即時攝影機 Demo，結合 MediaPipe 人臉偵測與模型推論，畫面疊加預測結果 |
| `test_model.py` | 驗證模型能否正常做前向推論（不涉及真實資料） |
| `test_dataset.py` | 驗證 DataLoader 能否正常讀取資料集 |
| `test_cuda.py` | 驗證 CUDA 環境是否可用 |

---

## 使用方式

### 訓練

```bash
python train.py
```

模型會依驗證準確率自動儲存最佳權重到 `checkpoints/best_model.pth`。

### 評估

```bash
python evaluate.py
```

需要先有 `checkpoints/best_model.pth`，結果會輸出到 `results/`：
- `classification_report.txt`
- `confusion_matrix.png`
- `test_result.json`

### 單張圖片推論

```bash
python predict.py
```

預設讀取 `test_images/sample.jpg`，會印出預測情緒與信心分數。

### 即時攝影機 Demo

```bash
python camera_demo.py
```

需要攝影機硬體，按 `ESC` 結束。

---

## 目前結果

以 2 個 epoch 訓練所得的結果：

| Epoch | Train Accuracy | Validation Accuracy |
|---|---|---|
| 1 | 46.75% | 55.91% |
| 2 | 57.25% | 58.61% |

**測試集準確率：59.71%**

---

## 目前狀態

| Step | 狀態 |
|---|---|
| Dataset | ✅ Completed |
| Model | ✅ Completed |
| Training Pipeline | ✅ Completed |
| Evaluation | ✅ Completed |
| Image Inference | ✅ Completed |
| Real-time Camera Demo | ⚠️ Implemented / Not Tested（缺攝影機硬體，尚未實測） |

---

## 已知限制

* **訓練回合數過少**：目前僅訓練 2 個 epoch，準確率仍有明顯成長空間，尚未收斂
* **未固定隨機種子**：`config.py` 定義了 `RANDOM_SEED`，但目前訓練流程尚未實際套用，實驗結果尚不可完全重現
* **類別不平衡未處理**：FER2013 中 Disgust 類別樣本數明顯偏少，目前使用未加權的 `CrossEntropyLoss`，可能導致該類別辨識效果較差（尚待用混淆矩陣驗證）
* **FER2013 資料集本身的限制**：標註為人工判斷，主觀性較高，且影像解析度低，這是模型準確率的天花板之一，並非單純調參能突破
* **無 learning rate 動態調整機制**：目前為固定學習率訓練

---

## 之後可能的改進方向

* 增加訓練 epoch 數，觀察準確率是否能提升至 65-70% 區間
* 補上隨機種子設定，確保實驗可重現
* 依混淆矩陣結果，評估是否需要對 Disgust 類別加權或做過採樣
* 加入 learning rate scheduler
* 待有攝影機硬體後完成 `camera_demo.py` 實測
