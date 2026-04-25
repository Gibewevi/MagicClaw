from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .paths import CONFIG_PATH, WORKSPACE_DIR, ensure_dirs


class RuntimeSettings(BaseModel):
    model_option_id: str = ""
    model_repo: str = ""
    model_file: str = ""
    model_path: str = ""
    quantization: str = "Q4_K_M"
    context_tokens: int = 8192
    batch_size: int = 256
    ubatch_size: int = 128
    gpu_layers: int = -1
    threads: int = 6
    parallel: int = 1
    port: int = 8080
    host: str = "127.0.0.1"
    llama_server_path: str = ""
    keepalive_seconds: int = 3600
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    step_max_tokens: int = 1024
    request_timeout_seconds: int = 420
    request_retries: int = 1
    request_retry_backoff_seconds: float = 2.0
    server_timeout_seconds: int = 900
    reasoning: str = "off"
    reasoning_budget_tokens: int = 0
    flash_attention: str = "on"
    kv_offload: bool = True

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class TelegramSettings(BaseModel):
    enabled: bool = False
    bot_token_env: str = "MAGIC_CLAW_TELEGRAM_TOKEN"
    allow_user_ids: list[int] = Field(default_factory=list)
    poll_timeout_seconds: int = 30


class MagicConfig(BaseModel):
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    workspace_dir: str = str(WORKSPACE_DIR)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)


def load_config(path: Path = CONFIG_PATH) -> MagicConfig:
    ensure_dirs()
    if not path.exists():
        return MagicConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return MagicConfig.model_validate(data)


def save_config(config: MagicConfig, path: Path = CONFIG_PATH) -> None:
    ensure_dirs()
    content = json.dumps(config.model_dump(), indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_runtime(config: MagicConfig, values: dict[str, Any]) -> MagicConfig:
    current = config.runtime.model_dump()
    current.update(values)
    return config.model_copy(update={"runtime": RuntimeSettings.model_validate(current)})
