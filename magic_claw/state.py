from __future__ import annotations

import json
import sqlite3
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
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        ensure_dirs()
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def event(self, level: str, source: str, message: str, data: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, level, source, message, data) VALUES (?, ?, ?, ?, ?)",
            (time.time(), level, source, message, json.dumps(data or {}, sort_keys=True)),
        )
        self.conn.commit()

    def create_task(self, prompt: str) -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO tasks(ts, updated_ts, status, prompt) VALUES (?, ?, 'running', ?)",
            (now, now, prompt),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_step(self, task_id: int, step_index: int, phase: str, content: str, data: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO steps(task_id, ts, step_index, phase, content, data) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, time.time(), step_index, phase, content, json.dumps(data or {}, sort_keys=True)),
        )
        self.conn.execute("UPDATE tasks SET updated_ts = ? WHERE id = ?", (time.time(), task_id))
        self.conn.commit()

    def finish_task(self, task_id: int, result: str) -> None:
        self.conn.execute(
            "UPDATE tasks SET updated_ts = ?, status = 'done', result = ? WHERE id = ?",
            (time.time(), result, task_id),
        )
        self.conn.commit()

    def fail_task(self, task_id: int, error: str) -> None:
        self.conn.execute(
            "UPDATE tasks SET updated_ts = ?, status = 'failed', error = ? WHERE id = ?",
            (time.time(), error, task_id),
        )
        self.conn.commit()

    def set_setting(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, sort_keys=True)),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        return json.loads(row["value"])

