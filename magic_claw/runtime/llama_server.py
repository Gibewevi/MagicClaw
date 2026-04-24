from __future__ import annotations

from collections.abc import Callable
import re
import subprocess
import time
from pathlib import Path

import httpx

from magic_claw.config import RuntimeSettings
from magic_claw.paths import LOG_DIR, ensure_dirs


StatusCallback = Callable[[str], None]
STDOUT_LOG = LOG_DIR / "llama-server.stdout.log"
STDERR_LOG = LOG_DIR / "llama-server.stderr.log"


class LlamaServer:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.process: subprocess.Popen[str] | None = None
        self._log_offsets: dict[Path, int] = {}

    @property
    def models_url(self) -> str:
        return f"{self.settings.api_base}/models"

    @property
    def process_id(self) -> int | None:
        return self.process.pid if self.process else None

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
        self._log_offsets = {
            path: path.stat().st_size if path.exists() else 0
            for path in (STDERR_LOG, STDOUT_LOG)
        }
        stdout = STDOUT_LOG.open("a", encoding="utf-8")
        stderr = STDERR_LOG.open("a", encoding="utf-8")
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

    def latest_log_line(self) -> str:
        for path in (STDERR_LOG, STDOUT_LOG):
            line = _tail_interesting_line(path, start_offset=self._log_offsets.get(path, 0))
            if line:
                return line
        return ""

    def latest_startup_stage(self) -> str:
        return _friendly_startup_stage(self.latest_log_line())

    def wait_until_ready(
        self,
        timeout_seconds: int = 180,
        on_status: StatusCallback | None = None,
        status_prefix: str = "Loading model",
    ) -> bool:
        started = time.monotonic()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            elapsed = int(time.monotonic() - started)
            if self.process and self.process.poll() is not None:
                _emit_status(
                    on_status,
                    f"{status_prefix} | stopped after {elapsed}s | check logs",
                )
                return False
            if self.healthy(timeout_seconds=1.0):
                _emit_status(on_status, f"{status_prefix} | ready in {elapsed}s")
                return True
            stage = self.latest_startup_stage()
            _emit_status(
                on_status,
                _startup_status(
                    status_prefix=status_prefix,
                    elapsed=elapsed,
                    stage=stage,
                ),
            )
            time.sleep(2)
        _emit_status(
            on_status,
            f"{status_prefix} | timeout after {timeout_seconds}s | check logs",
        )
        return False


def _startup_status(status_prefix: str, elapsed: int, stage: str) -> str:
    return f"{status_prefix} | {stage} | {elapsed}s"


def _emit_status(callback: StatusCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _tail_interesting_line(path: Path, start_offset: int = 0, max_bytes: int = 8192, max_length: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if size <= start_offset:
                return ""
            handle.seek(max(start_offset, size - max_bytes))
            text = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""

    for line in reversed(text.splitlines()):
        cleaned = line.strip()
        if not cleaned or set(cleaned) <= {"."}:
            continue
        if len(cleaned) > max_length:
            return cleaned[: max_length - 3] + "..."
        return cleaned
    return ""


def _friendly_startup_stage(line: str) -> str:
    lower = line.lower()
    if not lower:
        return "starting"
    if "getting device memory" in lower:
        return "checking VRAM"
    if "loading model tensors" in lower:
        return "loading model"
    if "offloading output layer" in lower:
        return "moving output layer to GPU"
    offloaded = re.search(r"offloaded\s+(\d+/\d+)\s+layers", lower)
    if offloaded:
        return f"GPU layers {offloaded.group(1)}"
    if "offloading" in lower and "layers" in lower:
        return "moving layers to GPU"
    if "model buffer size" in lower:
        return "mapping model memory"
    if "constructing llama_context" in lower:
        return "building context"
    if "kv cache" in lower:
        return "allocating KV cache"
    if "sched_reserve" in lower:
        return "preparing compute"
    if "warming up" in lower:
        return "warming up"
    if "initializing slots" in lower:
        return "initializing slots"
    if "listening" in lower or "server is listening" in lower:
        return "starting API"
    if "error" in lower or "failed" in lower:
        return "startup issue"
    return "loading"
