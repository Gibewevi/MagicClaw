from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from magic_claw.config import MagicConfig, TelegramSettings
from magic_claw.paths import ENV_PATH, ensure_dirs


TELEGRAM_TOKEN_ENV = "MAGIC_CLAW_TELEGRAM_TOKEN"
TOKEN_PATTERN = re.compile(r"^\d{6,20}:[A-Za-z0-9_-]{30,}$")


class TelegramSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramBotInfo:
    bot_id: int
    username: str
    first_name: str


def normalise_telegram_token(token: str) -> str:
    value = token.strip()
    if not value:
        raise TelegramSetupError("Token Telegram manquant.")
    if not TOKEN_PATTERN.fullmatch(value):
        raise TelegramSetupError(
            "Format du token invalide. Il doit ressembler a 123456789:ABCdef_..."
        )
    return value


def verify_telegram_token(
    token: str,
    timeout_seconds: float = 10.0,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> TelegramBotInfo:
    value = normalise_telegram_token(token)
    try:
        with client_factory(timeout=timeout_seconds) as client:
            response = client.get(f"https://api.telegram.org/bot{value}/getMe")
            if response.status_code == 401:
                raise TelegramSetupError("Token refuse par Telegram. Verifie le token BotFather.")
            response.raise_for_status()
            data = response.json()
    except TelegramSetupError:
        raise
    except httpx.TimeoutException as exc:
        raise TelegramSetupError("Telegram ne repond pas assez vite. Reessaie dans quelques secondes.") from exc
    except httpx.RequestError as exc:
        raise TelegramSetupError(f"Impossible de contacter Telegram: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise TelegramSetupError(f"Erreur Telegram HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
    except ValueError as exc:
        raise TelegramSetupError("Reponse Telegram illisible.") from exc

    if not data.get("ok"):
        description = str(data.get("description") or data)
        raise TelegramSetupError(f"Telegram refuse ce token: {description}")

    result = data.get("result") or {}
    if not result.get("is_bot"):
        raise TelegramSetupError("Ce token ne correspond pas a un bot Telegram.")
    username = str(result.get("username") or "").strip()
    first_name = str(result.get("first_name") or username or "Telegram bot").strip()
    bot_id = int(result.get("id") or 0)
    if not bot_id or not username:
        raise TelegramSetupError("Telegram a valide le token, mais les infos du bot sont incompletes.")
    return TelegramBotInfo(bot_id=bot_id, username=username, first_name=first_name)


def set_telegram_bot_commands(token: str, timeout_seconds: float = 10.0) -> bool:
    value = normalise_telegram_token(token)
    commands = [
        {"command": "help", "description": "Afficher les commandes Magic Claw"},
        {"command": "status", "description": "Voir le statut Magic Claw"},
    ]
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(
                f"https://api.telegram.org/bot{value}/setMyCommands",
                json={"commands": commands},
            )
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return bool(data.get("ok"))


def apply_telegram_bot_info(
    config: MagicConfig,
    bot_info: TelegramBotInfo,
    allow_user_ids: list[int] | None = None,
) -> MagicConfig:
    current = config.telegram.model_dump()
    current.update(
        {
            "enabled": True,
            "bot_token_env": TELEGRAM_TOKEN_ENV,
            "bot_id": bot_info.bot_id,
            "bot_username": bot_info.username,
            "bot_first_name": bot_info.first_name,
            "last_validated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if allow_user_ids is not None:
        current["allow_user_ids"] = allow_user_ids
    return config.model_copy(update={"telegram": TelegramSettings.model_validate(current)})


def disable_telegram(config: MagicConfig) -> MagicConfig:
    current = config.telegram.model_dump()
    current["enabled"] = False
    return config.model_copy(update={"telegram": TelegramSettings.model_validate(current)})


def reset_telegram(config: MagicConfig, env_path: Path = ENV_PATH) -> MagicConfig:
    remove_env_value(env_path, config.telegram.bot_token_env or TELEGRAM_TOKEN_ENV)
    current = config.telegram.model_dump()
    current.update(
        {
            "enabled": False,
            "bot_id": None,
            "bot_username": "",
            "bot_first_name": "",
            "last_validated_at": "",
        }
    )
    return config.model_copy(update={"telegram": TelegramSettings.model_validate(current)})


def save_telegram_token(token: str, env_path: Path = ENV_PATH, key: str = TELEGRAM_TOKEN_ENV) -> None:
    value = normalise_telegram_token(token)
    write_env_value(env_path, key, value)
    os.environ[key] = value


def write_env_value(env_path: Path, key: str, value: str) -> None:
    ensure_dirs()
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    replacement = f"{key}={value}"
    updated: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
            if not found:
                updated.append(replacement)
                found = True
            continue
        updated.append(line)
    if not found:
        updated.append(replacement)
    _write_env_lines(env_path, updated)


def remove_env_value(env_path: Path, key: str) -> None:
    ensure_dirs()
    if not env_path.exists():
        os.environ.pop(key, None)
        return
    lines = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"export {key}="):
            continue
        lines.append(line)
    _write_env_lines(env_path, lines)
    os.environ.pop(key, None)


def _write_env_lines(env_path: Path, lines: list[str]) -> None:
    content = "\n".join(lines).rstrip()
    tmp = env_path.with_suffix(env_path.suffix + ".tmp")
    tmp.write_text((content + "\n") if content else "", encoding="utf-8")
    tmp.replace(env_path)
