from magic_claw.config import TelegramSettings
from magic_claw.state import StateStore
from magic_claw.status import AgentStatusView
from magic_claw.telegram.bot import TelegramBot


class RecordingTelegramBot(TelegramBot):
    status_update_interval_seconds = 0.0

    def __init__(self, tmp_path):
        super().__init__(TelegramSettings(enabled=True), StateStore(tmp_path / "state.sqlite"))
        self.sent: list[tuple[int, str]] = []
        self.edits: list[tuple[int, int, str]] = []

    def send_message(self, chat_id: int, text: str) -> int:
        self.sent.append((chat_id, text))
        return len(self.sent)

    def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        self.edits.append((chat_id, message_id, text))


def _message_update(text: str = "Creer un projet") -> dict:
    return {
        "message": {
            "text": text,
            "chat": {"id": 42},
            "from": {"id": 123},
        }
    }


def test_telegram_task_relays_agent_status_updates(tmp_path):
    bot = RecordingTelegramBot(tmp_path)

    def task(prompt, on_status):
        assert prompt == "Creer un projet"
        on_status("Thinking | step 1/60")
        on_status("Running command: npm run build | step 2/60")
        return "Projet termine."

    bot._handle_update(_message_update(), task, lambda: "status")

    assert bot.sent[0][0] == 42
    assert "Task: Creer un projet" in bot.sent[0][1]
    assert "Current action: Task accepted." in bot.sent[0][1]
    assert bot.sent[-1] == (42, "Projet termine.")
    assert any("Thinking | step 1/60" in edit[2] for edit in bot.edits)
    assert any("Running command: npm run build" in edit[2] for edit in bot.edits)
    assert any("Task complete." in edit[2] for edit in bot.edits)
    assert any("Recent actions:" in edit[2] for edit in bot.edits)


def test_telegram_task_failure_is_reported_to_chat(tmp_path):
    bot = RecordingTelegramBot(tmp_path)

    def task(_prompt, _on_status):
        raise RuntimeError("boom")

    bot._handle_update(_message_update(), task, lambda: "status")

    assert bot.sent[-1] == (42, "Task failed: boom")
    assert any("Task failed." in edit[2] for edit in bot.edits)
    assert not any("Task complete." in edit[2] for edit in bot.edits)


def test_agent_status_view_keeps_task_excerpt_and_recent_history():
    view = AgentStatusView("Créer un projet météo très moderne pour Sherbrooke")

    first = view.update("Thinking | step 1/60")
    second = view.update("Auto-validation failed: \u2717 Build failed in 391ms")

    assert "Task:" in second.text
    assert "Sherbrooke" in second.text
    assert "Current action: Auto-validation failed: x Build failed in 391ms" in second.text
    assert "Thinking | step 1/60" in second.text
    assert first.line == "Thinking | step 1/60"
    assert "task:" not in first.line
