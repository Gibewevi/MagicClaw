from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_HOME = Path(os.environ.get("MAGIC_CLAW_HOME", PROJECT_ROOT / ".magicclaw"))
STATE_DIR = APP_HOME / "state"
MODEL_DIR = APP_HOME / "models"
RUNTIME_DIR = APP_HOME / "runtime"
LOG_DIR = APP_HOME / "logs"
WORKSPACE_DIR = APP_HOME / "workspace"
CONFIG_PATH = APP_HOME / "config.generated.json"
DB_PATH = STATE_DIR / "magic_claw.sqlite"
ENV_PATH = APP_HOME / ".env"


def ensure_dirs() -> None:
    for path in (APP_HOME, STATE_DIR, MODEL_DIR, RUNTIME_DIR, LOG_DIR, WORKSPACE_DIR):
        path.mkdir(parents=True, exist_ok=True)
