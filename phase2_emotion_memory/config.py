"""
Project Configuration
Facial Expression & Responsive System
Phase 2 - Emotion Memory
"""

from pathlib import Path

# ==========================
# Project Path
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATABASE_DIR / "emotion_history.db"


# ==========================
# Emotion Labels
# 順序必須跟 Phase 1 的 CLASS_NAMES 一致
# ==========================
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# 簡單的情緒效價 (valence) 對照表
# 用來計算「情緒趨勢」是變好還是變差
# 這是一個起始的簡化假設，之後可以依實際需求調整權重
EMOTION_VALENCE = {
    "Angry": -0.8,
    "Disgust": -0.6,
    "Fear": -0.7,
    "Happy": 1.0,
    "Neutral": 0.0,
    "Sad": -1.0,
    "Surprise": 0.3,
}


# ==========================
# Smoothing / Memory Parameters
# ==========================

# 滑動視窗大小（幾筆記錄一組做平滑）
SMOOTHING_WINDOW = 15

# EMA 平滑係數，越接近 1 越重視最新的資料
EMA_ALPHA = 0.3

# 判斷「情緒持續多久」時，容許中斷的秒數
# (避免單幀誤判造成 duration 被打斷)
DURATION_GAP_TOLERANCE_SEC = 2.0

# 計算趨勢時，比較「最近 N 秒」跟「再前面 N 秒」的平均效價
TREND_WINDOW_SEC = 10.0

# 趨勢判斷的閾值，差異小於這個值視為 stable
TREND_THRESHOLD = 0.15
