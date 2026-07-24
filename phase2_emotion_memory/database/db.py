"""
Emotion Memory - Database Layer

負責把 Phase 1 camera_demo 逐幀預測的結果存下來，
並提供依 session 查詢歷史紀錄的介面。
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS emotion_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    probs TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_time
ON emotion_records (session_id, timestamp);
"""


class EmotionDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def new_session_id():
        """每次啟動 camera_demo 呼叫一次，代表一個新的觀察區間"""
        return uuid.uuid4().hex[:12]

    def insert_record(self, session_id, emotion, confidence, probs, timestamp=None):
        """
        新增一筆情緒紀錄

        Args:
            session_id: 這次攝影機觀察的 session 識別碼
            emotion: 預測的情緒名稱 (str)
            confidence: softmax 最大值 (float)
            probs: 完整的 7 類機率分佈 (list[float])
            timestamp: unix time，預設為當下時間
        """
        if timestamp is None:
            timestamp = time.time()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO emotion_records
                    (session_id, timestamp, emotion, confidence, probs)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, timestamp, emotion, confidence, json.dumps(probs)),
            )

    def get_session_records(self, session_id):
        """取得某個 session 的所有紀錄，依時間排序"""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, emotion, confidence, probs
                FROM emotion_records
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        return [
            {
                "timestamp": row[0],
                "emotion": row[1],
                "confidence": row[2],
                "probs": json.loads(row[3]),
            }
            for row in rows
        ]

    def get_latest_session_id(self):
        """取得最近一次的 session_id，方便 Phase 3 直接抓最新狀態"""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT session_id
                FROM emotion_records
                ORDER BY timestamp DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()

        return row[0] if row else None

    def list_sessions(self):
        """列出所有 session_id 跟各自的起訖時間，方便之後做資料檢視/清理"""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT session_id, MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM emotion_records
                GROUP BY session_id
                ORDER BY MIN(timestamp) DESC
                """
            )
            rows = cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "record_count": row[3],
            }
            for row in rows
        ]
