from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from magic_claw.agent import AgentLoop
from magic_claw.config import MagicConfig
from magic_claw.state import StateStore
from magic_claw.telegram import TelegramBot

from .llama_server import LlamaServer


@dataclass
class SupervisorStatus:
    model_healthy: bool
    telegram_enabled: bool
    restarts: int


class Supervisor:
    def __init__(self, config: MagicConfig, on_status=None) -> None:
        self.config = config
        self.state = StateStore()
        self.server = LlamaServer(config.runtime)
        self.stop_event = threading.Event()
        self.restarts = 0
        self.on_status = on_status or (lambda _message: None)

    def status(self) -> SupervisorStatus:
        return SupervisorStatus(
            model_healthy=self.server.healthy(),
            telegram_enabled=self.config.telegram.enabled,
            restarts=self.restarts,
        )

    def status_text(self) -> str:
        status = self.status()
        return (
            "Magic Claw status\n"
            f"model_healthy: {status.model_healthy}\n"
            f"telegram_enabled: {status.telegram_enabled}\n"
            f"restarts: {status.restarts}\n"
            f"api_base: {self.config.runtime.api_base}"
        )

    def run_agent_task(self, prompt: str) -> str:
        loop = AgentLoop(
            self.config.runtime,
            self.config.workspace_dir,
            self.state,
            on_status=self.on_status,
        )
        result = loop.run(prompt)
        return result.answer

    def _start_telegram_thread(self) -> threading.Thread | None:
        bot = TelegramBot(self.config.telegram, self.state)
        if not bot.enabled:
            return None
        thread = threading.Thread(
            target=bot.poll_forever,
            args=(self.stop_event, self.run_agent_task, self.status_text),
            daemon=True,
            name="magic-claw-telegram",
        )
        thread.start()
        return thread

    def _model_label(self) -> str:
        name = Path(self.config.runtime.model_path).stem
        for suffix in ("-GGUF", ".gguf"):
            name = name.replace(suffix, "")
        if len(name) > 28:
            return name[:25] + "..."
        return name

    def run_forever(self) -> None:
        backoff = 2.0
        telegram_thread: threading.Thread | None = None
        while not self.stop_event.is_set():
            try:
                model_label = self._model_label()
                startup_prefix = f"Loading {model_label}"
                self.on_status(f"Starting {model_label}")
                self.server.start()
                self.on_status(f"{startup_prefix} | waiting for API | 0s")
                if not self.server.wait_until_ready(
                    timeout_seconds=240,
                    on_status=self.on_status,
                    status_prefix=startup_prefix,
                ):
                    raise RuntimeError("Model server did not become ready.")
                self.state.event("info", "supervisor", "model server ready")
                self.on_status(f"Ready | {model_label}")
                if telegram_thread is None:
                    telegram_thread = self._start_telegram_thread()

                backoff = 2.0
                while not self.stop_event.is_set():
                    if not self.server.healthy(timeout_seconds=5):
                        raise RuntimeError("Model healthcheck failed.")
                    telegram_state = "on" if self.config.telegram.enabled else "off"
                    self.on_status(f"Ready | Telegram {telegram_state} | restarts {self.restarts}")
                    time.sleep(10)
            except KeyboardInterrupt:
                self.stop_event.set()
            except Exception as exc:
                self.restarts += 1
                self.state.event("error", "supervisor", "runtime failure", {"error": str(exc), "restarts": self.restarts})
                self.on_status(f"Runtime failure: {exc}; restarting in {backoff:.0f}s")
                self.server.stop()
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 120)
