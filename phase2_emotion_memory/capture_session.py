"""
Phase 2 - Camera Capture Session

沿用 Phase 1 訓練好的模型做即時推論，
但這裡多做兩件 Phase 1 camera_demo 沒做的事：
    1. 把每一幀的預測結果 (含完整機率分佈) 寫入資料庫
    2. 用 EmotionMemory 的 LiveSmoother 做即時平滑，畫面顯示更穩定的結果

執行方式：
    在 phase2_emotion_memory/ 目錄下執行
        python capture_session.py

按 ESC 結束，結束時會印出這次 session_id，
之後可以用 emotion_memory.summarize(session_id) 拿到摘要，
或直接用 phase3 的 response_engine 抓最新 session。
"""

import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import mediapipe as mp

# ==========================
# 讓這個腳本可以 import Phase 1 的模組
# ==========================
PHASE1_DIR = Path(__file__).resolve().parent.parent / "phase1_expression_detection"
sys.path.insert(0, str(PHASE1_DIR))

from models.resnet18_emotion import EmotionResNet18  # noqa: E402
from utils.transforms import get_val_transform  # noqa: E402

from config import DB_PATH, CLASS_NAMES  # noqa: E402
from database.db import EmotionDatabase  # noqa: E402
from emotion_memory import EmotionMemory  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = PHASE1_DIR / "checkpoints" / "best_model.pth"


def load_model():
    model = EmotionResNet18()
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    return model


def predict_probs(model, face, transform):
    """回傳完整的 7 類機率分佈 (list[float])，而不只是 argmax"""
    image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)

    return probs.squeeze(0).cpu().tolist()


def main():
    model = load_model()
    transform = get_val_transform()

    db = EmotionDatabase(DB_PATH)
    memory = EmotionMemory(DB_PATH)
    smoother = memory.make_live_smoother()
    session_id = db.new_session_id()
    print(f"Session started: {session_id}")

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.detections:
            h, w, _ = frame.shape
            for detection in result.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                face = frame[y:y + bh, x:x + bw]
                if face.size == 0:
                    continue

                probs = predict_probs(model, face, transform)
                raw_idx = max(range(len(probs)), key=lambda i: probs[i])
                raw_emotion = CLASS_NAMES[raw_idx]
                raw_confidence = probs[raw_idx]

                # 存下原始（未平滑）的逐幀結果，保留完整資訊
                db.insert_record(
                    session_id=session_id,
                    emotion=raw_emotion,
                    confidence=raw_confidence,
                    probs=probs,
                    timestamp=time.time(),
                )

                # 畫面上顯示平滑後的結果，比較穩定不會一直跳動
                smooth_emotion, smooth_confidence = smoother.update(probs)
                label = f"{smooth_emotion} {smooth_confidence*100:.1f}%"

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        cv2.imshow("Emotion Recognition (Phase 2 - Recording)", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Session ended: {session_id}")
    print("這個 session_id 可以拿去用 EmotionMemory.summarize() 產生摘要，")
    print("或直接執行 phase3_response_system 的 chat.py，它會自動抓最新 session。")


if __name__ == "__main__":
    main()
