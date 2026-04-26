from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .paths import DB_PATH, ensure_dirs


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  level TEXT NOT NULL,
  source TEXT NOT NULL,
  message TEXT NOT NULL,
  data TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  updated_ts REAL NOT NULL,
  status TEXT NOT NULL,
  prompt TEXT NOT NULL,
  result TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  step_index INTEGER NOT NULL,
  phase TEXT NOT NULL,
  content TEXT NOT NULL,
  data TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS task_memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  checkpoint_index INTEGER NOT NULL,
  step_index INTEGER NOT NULL,
  summary TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        ensure_dirs()
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def event(self, level: str, source: str, message: str, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(ts, level, source, message, data) VALUES (?, ?, ?, ?, ?)",
                (time.time(), level, source, message, json.dumps(data or {}, sort_keys=True)),
            )
            self.conn.commit()

    def create_task(self, prompt: str) -> int:
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO tasks(ts, updated_ts, status, prompt) VALUES (?, ?, 'running', ?)",
                (now, now, prompt),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def add_step(self, task_id: int, step_index: int, phase: str, content: str, data: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO steps(task_id, ts, step_index, phase, content, data) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, time.time(), step_index, phase, content, json.dumps(data or {}, sort_keys=True)),
            )
            self.conn.execute("UPDATE tasks SET updated_ts = ? WHERE id = ?", (time.time(), task_id))
            self.conn.commit()

    def add_task_memory(self, task_id: int, checkpoint_index: int, step_index: int, summary: str) -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO task_memories(task_id, ts, checkpoint_index, step_index, summary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, now, checkpoint_index, step_index, summary),
            )
            self.conn.execute("UPDATE tasks SET updated_ts = ? WHERE id = ?", (now, task_id))
            self.conn.commit()

    def list_task_memories(self, task_id: int, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT checkpoint_index, step_index, summary
                FROM task_memories
                WHERE task_id = ?
                ORDER BY checkpoint_index DESC
                LIMIT ?
                """,
                (task_id, limit),
            ).fetchall()
        return [
            {
                "checkpoint_index": int(row["checkpoint_index"]),
                "step_index": int(row["step_index"]),
                "summary": str(row["summary"]),
            }
            for row in rows
        ]

    def finish_task(self, task_id: int, result: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE tasks SET updated_ts = ?, status = 'done', result = ? WHERE id = ?",
                (time.time(), result, task_id),
            )
            self.conn.commit()

    def fail_task(self, task_id: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE tasks SET updated_ts = ?, status = 'failed', error = ? WHERE id = ?",
                (time.time(), error, task_id),
            )
            self.conn.commit()

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, sort_keys=True)),
            )
            self.conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        return json.loads(row["value"])
