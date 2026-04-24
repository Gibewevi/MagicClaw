from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import httpx

from magic_claw.config import TelegramSettings
from magic_claw.state import StateStore


class TelegramBot:
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

    def send_message(self, chat_id: int, text: str) -> None:
        with httpx.Client(timeout=30) as client:
            client.post(f"{self.api}/sendMessage", json={"chat_id": chat_id, "text": text[:3900]})

    def _allowed(self, user_id: int) -> bool:
        return not self.settings.allow_user_ids or user_id in self.settings.allow_user_ids

    def poll_forever(
        self,
        stop_event: threading.Event,
        on_task: Callable[[str], str],
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

    def _handle_update(self, update: dict, on_task: Callable[[str], str], on_status: Callable[[], str]) -> None:
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
        if text.startswith("/help"):
            self.send_message(chat_id, "/status\nSend any other message to start a Magic Claw task.")
            return
        self.send_message(chat_id, "Magic Claw accepted the task. Working...")
        result = on_task(text)
        self.send_message(chat_id, result)

