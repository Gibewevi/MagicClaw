import os

import pytest

from magic_claw.config import MagicConfig
from magic_claw.telegram.setup import (
    TelegramSetupError,
    apply_telegram_bot_info,
    normalise_telegram_token,
    reset_telegram,
    save_telegram_token,
    verify_telegram_token,
)


VALID_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._data


class FakeClient:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get(self, url):
        assert VALID_TOKEN in url
        return FakeResponse(
            data={
                "ok": True,
                "result": {
                    "id": 123456789,
                    "is_bot": True,
                    "username": "magicclaw_bot",
                    "first_name": "MagicClaw",
                },
            }
        )


def test_telegram_token_format_validation():
    assert normalise_telegram_token(f"  {VALID_TOKEN}  ") == VALID_TOKEN
    with pytest.raises(TelegramSetupError):
        normalise_telegram_token("not-a-token")


def test_verify_telegram_token_returns_bot_info():
    info = verify_telegram_token(VALID_TOKEN, client_factory=FakeClient)

    assert info.bot_id == 123456789
    assert info.username == "magicclaw_bot"
    assert info.first_name == "MagicClaw"


def test_save_and_reset_telegram_token_preserves_env_file(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=value\nMAGIC_CLAW_TELEGRAM_TOKEN=old\n", encoding="utf-8")

    save_telegram_token(VALID_TOKEN, env_path=env_path)

    content = env_path.read_text(encoding="utf-8")
    assert "OTHER=value" in content
    assert f"MAGIC_CLAW_TELEGRAM_TOKEN={VALID_TOKEN}" in content
    assert os.environ["MAGIC_CLAW_TELEGRAM_TOKEN"] == VALID_TOKEN

    config = apply_telegram_bot_info(MagicConfig(), verify_telegram_token(VALID_TOKEN, client_factory=FakeClient))
    reset = reset_telegram(config, env_path=env_path)

    assert reset.telegram.enabled is False
    assert "MAGIC_CLAW_TELEGRAM_TOKEN" not in env_path.read_text(encoding="utf-8")
    assert "MAGIC_CLAW_TELEGRAM_TOKEN" not in os.environ
