"""
Emotion Memory - Core Module

這是 Phase 2 的核心：把 Phase 1 逐幀、雜訊較大的情緒預測，
轉換成一個「去雜訊後的情緒狀態摘要」，供 Phase 3 使用。

目前先用滑動視窗 + EMA 的統計方法做 baseline，
之後有足夠的 session 資料，可以再替換成 GRU/LSTM 模型
(介面設計成方便替換：只要維持 summarize() 回傳的格式一致即可)。
"""

from collections import deque

from config import (
    CLASS_NAMES,
    EMOTION_VALENCE,
    SMOOTHING_WINDOW,
    EMA_ALPHA,
    DURATION_GAP_TOLERANCE_SEC,
    TREND_WINDOW_SEC,
    TREND_THRESHOLD,
)
from database.db import EmotionDatabase


class EmotionMemory:
    def __init__(self, db_path):
        self.db = EmotionDatabase(db_path)

    # ==========================
    # 即時平滑 (給 camera_demo 逐幀呼叫用)
    # ==========================
    def make_live_smoother(self):
        """
        回傳一個 LiveSmoother 物件，camera_demo 每一幀呼叫一次 update()，
        就能得到平滑後的當下情緒，不用等 session 結束。
        """
        return LiveSmoother()

    # ==========================
    # 離線分析 (session 結束後 / Phase 3 呼叫用)
    # ==========================
    def summarize(self, session_id):
        """
        產生給 Phase 3 使用的結構化情緒摘要

        Returns:
            dict，格式為：
            {
                "session_id": str,
                "current_emotion": str,
                "confidence": float,
                "trend": "improving" | "worsening" | "stable" | "unknown",
                "duration_sec": float,          # 目前情緒已經持續多久
                "recent_history": [ {emotion, timestamp}, ... ],
            }
            若該 session 沒有任何紀錄，回傳 None
        """
        records = self.db.get_session_records(session_id)
        if not records:
            return None

        smoothed = self._smooth_records(records)
        current_emotion, current_confidence = self._current_state(smoothed)
        trend = self._compute_trend(smoothed)
        duration_sec = self._current_emotion_duration(smoothed, current_emotion)

        return {
            "session_id": session_id,
            "current_emotion": current_emotion,
            "confidence": round(current_confidence, 4),
            "trend": trend,
            "duration_sec": round(duration_sec, 1),
            "recent_history": [
                {"emotion": r["emotion"], "timestamp": r["timestamp"]}
                for r in records[-SMOOTHING_WINDOW:]
            ],
        }

    # ==========================
    # 內部方法
    # ==========================
    def _smooth_records(self, records):
        """
        對逐幀機率向量做 EMA 平滑，回傳
        [{"timestamp": t, "probs": [...]}, ...]
        """
        smoothed = []
        ema = None

        for r in records:
            probs = r["probs"]
            if ema is None:
                ema = list(probs)
            else:
                ema = [
                    EMA_ALPHA * p + (1 - EMA_ALPHA) * e
                    for p, e in zip(probs, ema)
                ]
            smoothed.append({"timestamp": r["timestamp"], "probs": list(ema)})

        return smoothed

    def _current_state(self, smoothed):
        """取平滑後最後一筆的 argmax 當作目前情緒"""
        last_probs = smoothed[-1]["probs"]
        idx = max(range(len(last_probs)), key=lambda i: last_probs[i])
        return CLASS_NAMES[idx], last_probs[idx]

    def _current_emotion_duration(self, smoothed, current_emotion):
        """
        從最後一筆往回找，計算目前這個情緒（平滑後 argmax）
        連續維持了多久，容許短暫中斷 (DURATION_GAP_TOLERANCE_SEC)
        """
        if not smoothed:
            return 0.0

        end_time = smoothed[-1]["timestamp"]
        start_time = end_time
        prev_time = end_time

        for record in reversed(smoothed):
            probs = record["probs"]
            idx = max(range(len(probs)), key=lambda i: probs[i])
            emotion = CLASS_NAMES[idx]
            gap = prev_time - record["timestamp"]

            if emotion != current_emotion or gap > DURATION_GAP_TOLERANCE_SEC:
                break

            start_time = record["timestamp"]
            prev_time = record["timestamp"]

        return end_time - start_time

    def _compute_trend(self, smoothed):
        """
        比較「最近 TREND_WINDOW_SEC 秒」跟「再往前 TREND_WINDOW_SEC 秒」
        的平均情緒效價，判斷情緒是變好、變差還是穩定。
        """
        if len(smoothed) < 2:
            return "unknown"

        end_time = smoothed[-1]["timestamp"]
        recent, previous = [], []

        for record in reversed(smoothed):
            age = end_time - record["timestamp"]
            valence = self._expected_valence(record["probs"])

            if age <= TREND_WINDOW_SEC:
                recent.append(valence)
            elif age <= 2 * TREND_WINDOW_SEC:
                previous.append(valence)
            else:
                break

        if not recent or not previous:
            return "unknown"

        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        diff = recent_avg - previous_avg

        if diff > TREND_THRESHOLD:
            return "improving"
        elif diff < -TREND_THRESHOLD:
            return "worsening"
        else:
            return "stable"

    def _expected_valence(self, probs):
        """用機率加權平均效價，而不是只看 argmax，減少單幀誤判的影響"""
        return sum(
            p * EMOTION_VALENCE[CLASS_NAMES[i]] for i, p in enumerate(probs)
        )


class LiveSmoother:
    """
    給即時攝影機畫面用的輕量平滑器，只維護最近 SMOOTHING_WINDOW 筆機率向量，
    不需要連到資料庫，avoid 每一幀都查 DB 造成延遲。
    """

    def __init__(self):
        self._window = deque(maxlen=SMOOTHING_WINDOW)
        self._ema = None

    def update(self, probs):
        """
        餵入新一幀的機率分佈，回傳平滑後的 (emotion, confidence)
        """
        if self._ema is None:
            self._ema = list(probs)
        else:
            self._ema = [
                EMA_ALPHA * p + (1 - EMA_ALPHA) * e
                for p, e in zip(probs, self._ema)
            ]

        self._window.append(list(self._ema))

        idx = max(range(len(self._ema)), key=lambda i: self._ema[i])
        return CLASS_NAMES[idx], self._ema[idx]
