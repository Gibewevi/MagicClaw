from __future__ import annotations

import time
from dataclasses import dataclass


def task_excerpt(prompt: str, max_chars: int = 96) -> str:
    clean = " ".join(str(prompt).strip().split())
    if not clean:
        return "(no task text)"
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def clean_status_message(message: str) -> str:
    clean = " ".join(str(message).strip().split())
    return clean or "Working..."


@dataclass(frozen=True)
class AgentStatusSnapshot:
    task: str
    current: str
    history: tuple[tuple[str, str], ...]

    @property
    def line(self) -> str:
        return self.current

    @property
    def latest(self) -> tuple[str, str]:
        if self.history:
            return self.history[-1]
        return ("", self.current)

    @property
    def text(self) -> str:
        lines = [
            "Magic Claw status",
            f"Task: {self.task}",
            f"Current action: {self.current}",
        ]
        if self.history:
            lines.append("Recent actions:")
            lines.extend(f"- {timestamp} {message}" for timestamp, message in self.history)
        return "\n".join(lines)


class AgentStatusView:
    def __init__(self, prompt: str, history_limit: int = 6) -> None:
        self.task = task_excerpt(prompt)
        self.history_limit = max(1, history_limit)
        self.current = "Queued"
        self.history: list[tuple[str, str]] = []

    def snapshot(self) -> AgentStatusSnapshot:
        return AgentStatusSnapshot(self.task, self.current, tuple(self.history))

    def update(self, message: str) -> AgentStatusSnapshot:
        clean = clean_status_message(message)
        if clean != self.current:
            self.current = clean
            self.history.append((time.strftime("%H:%M:%S"), clean))
            self.history = self.history[-self.history_limit :]
        return self.snapshot()
