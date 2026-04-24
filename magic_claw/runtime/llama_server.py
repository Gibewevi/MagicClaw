from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx

from magic_claw.config import RuntimeSettings
from magic_claw.paths import LOG_DIR, ensure_dirs


class LlamaServer:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.process: subprocess.Popen[str] | None = None

    @property
    def models_url(self) -> str:
        return f"{self.settings.api_base}/models"

    def command(self) -> list[str]:
        if not self.settings.llama_server_path:
            raise RuntimeError("llama_server_path is not configured.")
        if not self.settings.model_path:
            raise RuntimeError("model_path is not configured.")
        return [
            self.settings.llama_server_path,
            "--model",
            self.settings.model_path,
            "--host",
            self.settings.host,
            "--port",
            str(self.settings.port),
            "--ctx-size",
            str(self.settings.context_tokens),
            "--batch-size",
            str(self.settings.batch_size),
            "--ubatch-size",
            str(self.settings.ubatch_size),
            "--threads",
            str(self.settings.threads),
            "--parallel",
            str(self.settings.parallel),
            "--n-gpu-layers",
            str(self.settings.gpu_layers),
            "--cont-batching",
        ]

    def start(self) -> None:
        ensure_dirs()
        if self.process and self.process.poll() is None:
            return
        stdout = (LOG_DIR / "llama-server.stdout.log").open("a", encoding="utf-8")
        stderr = (LOG_DIR / "llama-server.stderr.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command(),
            stdout=stdout,
            stderr=stderr,
            text=True,
            cwd=str(Path(self.settings.model_path).parent),
        )

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def healthy(self, timeout_seconds: float = 3.0) -> bool:
        try:
            response = httpx.get(self.models_url, timeout=timeout_seconds)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    def wait_until_ready(self, timeout_seconds: int = 180) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                return False
            if self.healthy(timeout_seconds=5):
                return True
            time.sleep(2)
        return False

