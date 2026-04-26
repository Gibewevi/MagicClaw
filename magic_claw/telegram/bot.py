from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import httpx

from magic_claw.config import TelegramSettings
from magic_claw.state import StateStore
from magic_claw.status import AgentStatusView, clean_status_message


StatusCallback = Callable[[str], None]
TaskRunner = Callable[[str, StatusCallback | None], str]


class TelegramBot:
    status_update_interval_seconds = 2.0

    def __init__(self, settings: TelegramSettings, state: StateStore) -> None:
        self.settings = settings
        self.state = state
        self.token = os.environ.get(settings.bot_token_env, "")
        self.offset = 0

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self.token)

    @property
    def api(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    def send_message(self, chat_id: int, text: str) -> int | None:
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{self.api}/sendMessage", json={"chat_id": chat_id, "text": text[:3900]})
            response.raise_for_status()
            data = response.json()
        result = data.get("result") if isinstance(data, dict) else None
        if isinstance(result, dict) and result.get("message_id") is not None:
            return int(result["message_id"])
        return None

    def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{self.api}/editMessageText",
                json={"chat_id": chat_id, "message_id": message_id, "text": text[:3900]},
            )
            response.raise_for_status()

    def _allowed(self, user_id: int) -> bool:
        return not self.settings.allow_user_ids or user_id in self.settings.allow_user_ids

    def poll_forever(
        self,
        stop_event: threading.Event,
        on_task: TaskRunner,
        on_status: Callable[[], str],
    ) -> None:
        if not self.enabled:
            return
        self.state.event("info", "telegram", "telegram polling started")
        backoff = 2.0
        while not stop_event.is_set():
            try:
                updates = self._get_updates()
                backoff = 2.0
                for update in updates:
                    self._handle_update(update, on_task, on_status)
            except Exception as exc:
                self.state.event("error", "telegram", "polling failure", {"error": str(exc)})
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 60)

    def _get_updates(self) -> list[dict]:
        with httpx.Client(timeout=self.settings.poll_timeout_seconds + 10) as client:
            response = client.get(
                f"{self.api}/getUpdates",
                params={
                    "timeout": self.settings.poll_timeout_seconds,
                    "offset": self.offset,
                    "allowed_updates": '["message"]',
                },
            )
            response.raise_for_status()
            data = response.json()
        if not data.get("ok"):
            raise RuntimeError(str(data))
        updates = data.get("result", [])
        if updates:
            self.offset = max(int(item["update_id"]) for item in updates) + 1
        return updates

    def _handle_update(self, update: dict, on_task: TaskRunner, on_status: Callable[[], str]) -> None:
        message = update.get("message") or {}
        text = str(message.get("text") or "").strip()
        chat = message.get("chat") or {}
        user = message.get("from") or {}
        chat_id = int(chat.get("id"))
        user_id = int(user.get("id"))
        if not text:
            return
        if not self._allowed(user_id):
            self.send_message(chat_id, "Access denied.")
            return
        if text.startswith("/status"):
            self.send_message(chat_id, on_status())
            return
        if text.startswith("/help") or text.startswith("/start"):
            self.send_message(chat_id, "/status\nSend any other message to start a Magic Claw task.")
            return
        status_view = AgentStatusView(text)
        status_message_id = self.send_message(chat_id, status_view.update("Task accepted.").text)
        status_update = self._status_callback(chat_id, status_message_id, status_view)
        try:
            result = on_task(text, status_update)
        except Exception as exc:
            self.state.event("error", "telegram", "task failure", {"error": str(exc)})
            status_update("Task failed.")
            self.send_message(chat_id, f"Task failed: {exc}")
            return
        status_update("Task complete.")
        self.send_message(chat_id, result)

    def _status_callback(
        self,
        chat_id: int,
        message_id: int | None,
        status_view: AgentStatusView | None = None,
    ) -> StatusCallback:
        view = status_view or AgentStatusView("")
        last_text = ""
        last_sent_at = 0.0

        def update(message: str) -> None:
            nonlocal last_text, last_sent_at
            status = view.update(message).text
            if status == last_text:
                return
            now = time.monotonic()
            force = not last_text or _is_important_status(message)
            if not force and now - last_sent_at < self.status_update_interval_seconds:
                return
            try:
                if message_id is None:
                    self.send_message(chat_id, status)
                else:
                    self.edit_message_text(chat_id, message_id, status)
                last_text = status
                last_sent_at = now
            except Exception as exc:
                self.state.event("error", "telegram", "status update failure", {"error": str(exc)})

        return update


def _status_text(message: str) -> str:
    clean = clean_status_message(message)
    return f"Magic Claw status\n{clean[:3600]}"


def _is_important_status(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "thinking",
            "running command",
            "writing",
            "appending",
            "creating",
            "compacting",
            "finalizing",
            "task complete",
            "task failed",
        )
    )
