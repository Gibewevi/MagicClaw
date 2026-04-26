from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt, Prompt

from .config import load_config, merge_runtime, save_config
from .hardware import diagnose_hardware
from .models import compatible_model_plans, get_option, recommended_models
from .models.downloader import ModelResolutionError, download_model, resolve_model_file
from .models.profiles import generation_token_limits
from .paths import CONFIG_PATH, ENV_PATH, ensure_dirs
from .runtime.llama_binary import LlamaBinaryError, ensure_llama_runtime_dependencies, ensure_llama_server_binary
from .runtime.supervisor import Supervisor
from .status import AgentStatusView
from .telegram import (
    TelegramSetupError,
    apply_telegram_bot_info,
    disable_telegram,
    reset_telegram,
    save_telegram_token,
    set_telegram_bot_commands,
    verify_telegram_token,
)
from .ui import MagicConsole


EXIT_COMMANDS = {"exit", "quit", "q", "stop"}
BUFFER_RUN_COMMANDS = {"go", "lancer", "run", "start"}
PASTE_DRAIN_IDLE_SECONDS = 0.08
PASTE_DRAIN_MAX_SECONDS = 0.35
PASTE_DRAIN_MAX_CHARS = 50000
SECTION_HEADINGS = {
    "but",
    "contraintes",
    "constraints",
    "contexte",
    "context",
    "objectif",
    "objective",
    "requirements",
    "tache",
    "task",
}


def _runtime_ready() -> bool:
    config = load_config()
    model_path = Path(config.runtime.model_path) if config.runtime.model_path else None
    llama_path = Path(config.runtime.llama_server_path) if config.runtime.llama_server_path else None
    return bool(model_path and model_path.exists() and llama_path and llama_path.exists())


def _model_ready(config) -> bool:
    model_path = Path(config.runtime.model_path) if config.runtime.model_path else None
    return bool(model_path and model_path.exists())


def _prepare_missing_runtime(console: MagicConsole, config, hardware=None):
    console.console.print("[yellow]Model already downloaded. Preparing missing runtime...[/]")
    if hardware is None:
        hardware = _diagnose(console)
    try:
        with console.task("Preparing llama.cpp server runtime") as status:
            status.update("Looking for llama-server")
            llama_path = str(ensure_llama_server_binary(hardware, auto_download=True))
            status.update(f"Runtime ready: {llama_path}")
    except LlamaBinaryError as exc:
        console.console.print(f"[red]Runtime preparation failed:[/] {exc}")
        return None
    config = merge_runtime(config, {"llama_server_path": llama_path})
    config = _apply_dynamic_runtime_tuning(config, hardware)
    save_config(config)
    console.console.print("[green]Runtime configured.[/]")
    return config


def _diagnose(console: MagicConsole):
    with console.task("Scanning CPU, RAM, GPU and VRAM") as status:
        status.update("Reading local hardware sensors")
        hardware = diagnose_hardware()
    console.print_hardware(hardware)
    return hardware


def cmd_diagnose(_args: argparse.Namespace) -> int:
    console = MagicConsole()
    console.print_header()
    _diagnose(console)
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    console = MagicConsole()
    console.print_header()
    hardware = _diagnose(console)
    plans = _find_compatible_models(console, hardware)
    console.print_model_plans(plans)
    return 0


def _find_compatible_models(console: MagicConsole, hardware):
    with console.task("Recherche des modèles récents...") as status:
        status.update("Interrogation de Hugging Face et filtrage GGUF compatible")
        plans = compatible_model_plans(hardware, include_recent=True)
        if not plans:
            status.update("Aucun modèle récent compatible; utilisation du catalogue local")
            plans = [
                plan
                for plan in recommended_models(hardware)
                if plan.compatibility in {"recommended", "compatible"}
            ]
    return plans


def _choose_model_plan(console: MagicConsole, plans):
    if not plans:
        raise RuntimeError("No compatible model found for this machine.")
    console.console.print("[bold]Mode de sélection du modèle[/]")
    console.console.print("[cyan]1[/] Automatique - sélection du meilleur modèle selon le matériel")
    console.console.print("[cyan]2[/] Manuel - afficher la liste compatible et choisir")
    mode = IntPrompt.ask("Choisir le mode", default=1)
    if mode not in (1, 2):
        raise ValueError("Invalid selection mode.")
    if mode == 1:
        plan = plans[0]
        console.console.print(
            f"[green]Mode automatique:[/] {plan.option.display_name} "
            f"({plan.quantization}, contexte {plan.context_tokens}, step {plan.step_max_tokens})"
        )
        return plan
    console.print_model_plans(plans)
    selected_index = IntPrompt.ask("Choose a model number", default=1)
    if selected_index < 1 or selected_index > len(plans):
        raise ValueError("Invalid model choice.")
    return plans[selected_index - 1]


def _human_bytes(value: float | None) -> str:
    if value is None:
        return "?"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.1f} TB"


def _shorten_filename(filename: str, max_length: int = 46) -> str:
    if len(filename) <= max_length:
        return filename
    head = max_length // 2 - 2
    tail = max_length - head - 3
    return f"{filename[:head]}...{filename[-tail:]}"


def _download_status(filename: str, downloaded: float, total: float | None) -> str:
    name = escape(_shorten_filename(filename))
    if total and total > 0:
        percent = min(100.0, max(0.0, downloaded * 100 / total))
        return f"Telechargement {name} - {percent:5.1f}% ({_human_bytes(downloaded)} / {_human_bytes(total)})"
    return f"Telechargement {name} - {_human_bytes(downloaded)}"


def _apply_dynamic_runtime_tuning(config, hardware):
    option = get_option(config.runtime.model_option_id) if config.runtime.model_option_id else None
    params_b = option.params_b if option else None
    generation = generation_token_limits(config.runtime.context_tokens, hardware, params_b)
    if (
        config.runtime.max_tokens == generation.max_tokens
        and config.runtime.step_max_tokens == generation.step_max_tokens
    ):
        return config
    return merge_runtime(
        config,
        {
            "max_tokens": generation.max_tokens,
            "step_max_tokens": generation.step_max_tokens,
        },
    )


def _normalise_cli_text(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _is_section_heading(value: str) -> bool:
    normalised = _normalise_cli_text(value).strip().strip(":")
    return normalised in SECTION_HEADINGS or (value.strip().endswith(":") and len(value.strip()) <= 64)


class InteractivePromptBuffer:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, prompt: str) -> tuple[str | None, bool]:
        stripped = prompt.strip()
        if "\n" in stripped:
            if self.parts:
                combined = "\n".join([*self.parts, stripped])
                self.parts.clear()
                return combined, True
            return stripped, True
        normalised = _normalise_cli_text(stripped)
        if self.parts and normalised in BUFFER_RUN_COMMANDS:
            combined = "\n".join(self.parts)
            self.parts.clear()
            return combined, True
        if stripped.endswith("\\") or stripped.endswith("..."):
            self.parts.append(stripped.rstrip("\\. "))
            return None, True
        if _is_section_heading(stripped):
            self.parts.append(stripped)
            return None, True
        if self.parts:
            self.parts.append(stripped)
            combined = "\n".join(self.parts)
            self.parts.clear()
            return combined, True
        return stripped, False


def _combine_pasted_prompt(first_line: str, extra_text: str) -> str:
    text = first_line
    if extra_text:
        text = f"{first_line}\n{extra_text}"
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


def _drain_pasted_stdin() -> str:
    if os.name != "nt":
        return ""
    try:
        import msvcrt
    except ImportError:
        return ""

    chars: list[str] = []
    start = time.monotonic()
    idle_deadline = start + PASTE_DRAIN_IDLE_SECONDS
    end_deadline = start + PASTE_DRAIN_MAX_SECONDS
    while time.monotonic() < end_deadline and len(chars) < PASTE_DRAIN_MAX_CHARS:
        if msvcrt.kbhit():
            char = msvcrt.getwch()
            if char == "\x03":
                raise KeyboardInterrupt
            if char in {"\x00", "\xe0"}:
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            chars.append(char)
            idle_deadline = time.monotonic() + PASTE_DRAIN_IDLE_SECONDS
            continue
        if time.monotonic() >= idle_deadline:
            break
        time.sleep(0.01)
    return "".join(chars)


def _read_interactive_prompt(console: MagicConsole) -> str:
    first_line = console.console.input("\n[bold cyan]Magic Claw > [/]")
    extra_text = _drain_pasted_stdin()
    prompt = _combine_pasted_prompt(first_line, extra_text).strip()
    if extra_text.strip():
        line_count = len(prompt.splitlines())
        console.console.print(f"[dim]Captured pasted instruction ({line_count} lines).[/]")
    return prompt


def _with_interactive_context(prompt: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return prompt
    recent = history[-3:]
    context = "\n\n".join(
        "Previous user request:\n"
        f"{_trim_cli_context(old_prompt, 1200)}\n"
        "Previous result:\n"
        f"{_trim_cli_context(old_answer, 1600)}"
        for old_prompt, old_answer in recent
    )
    return (
        "Interactive context from earlier terminal turns. Use it only when the "
        "current request is a continuation or asks about earlier work.\n\n"
        f"{context}\n\nCurrent user request:\n{prompt}"
    )


def _trim_cli_context(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n...[historique tronqué]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return value[:head] + marker + value[-tail:]


def _terminal_task_status_callback(console: MagicConsole, rich_status, prompt: str):
    status_view = AgentStatusView(prompt)
    last_line = ""
    header_printed = False

    def update(message: str) -> None:
        nonlocal last_line, header_printed
        snapshot = status_view.update(message)
        rich_status.update(f"[bold cyan]{escape(snapshot.current)}[/]")
        if not header_printed:
            console.console.print(f"[bold cyan]Task:[/] [dim]{escape(snapshot.task)}[/]")
            header_printed = True
        if snapshot.line != last_line:
            timestamp, action = snapshot.latest
            prefix = f"{timestamp} | " if timestamp else ""
            console.console.print(f"[dim]{prefix}{escape(action)}[/]")
            last_line = snapshot.line

    return update


def _telegram_token_configured(config) -> bool:
    key = config.telegram.bot_token_env or "MAGIC_CLAW_TELEGRAM_TOKEN"
    return bool(os.environ.get(key, "").strip())


def _activate_telegram_from_token(console: MagicConsole, config, token: str, user_ids: list[int] | None = None):
    with console.task("Validation du bot Telegram") as status:
        status.update("Verification du token avec Telegram")
        bot_info = verify_telegram_token(token)
        status.update(f"Bot valide: @{bot_info.username}")
        save_telegram_token(token, key=config.telegram.bot_token_env or "MAGIC_CLAW_TELEGRAM_TOKEN")
        commands_ok = set_telegram_bot_commands(token)
    config = apply_telegram_bot_info(config, bot_info, allow_user_ids=user_ids)
    console.console.print(f"[green]Telegram active:[/] @{bot_info.username}")
    if not commands_ok:
        console.console.print("[yellow]Le bot est connecte, mais les commandes /help et /status n'ont pas pu etre publiees.[/]")
    return config


def _prompt_telegram_setup(console: MagicConsole, config, user_ids: list[int] | None = None):
    console.console.print("[bold]Configuration Telegram[/]")
    console.console.print("Cree un bot dans Telegram avec [cyan]@BotFather[/], puis copie le token ici.")
    console.console.print("Le token ressemble a [dim]123456789:ABCdef...[/].")
    while True:
        token = Prompt.ask("Token du bot Telegram", password=True).strip()
        if not token:
            console.console.print("[yellow]Configuration Telegram ignoree.[/]")
            return disable_telegram(config)
        try:
            return _activate_telegram_from_token(console, config, token, user_ids=user_ids)
        except TelegramSetupError as exc:
            console.console.print(f"[red]Telegram non configure:[/] {exc}")
            if not Confirm.ask("Reessayer ?", default=True):
                return disable_telegram(config)


def _configure_telegram_during_init(console: MagicConsole, config, args: argparse.Namespace):
    user_ids = getattr(args, "telegram_user_id", None)
    token = getattr(args, "telegram_token", None)
    if token:
        try:
            return _activate_telegram_from_token(console, config, token, user_ids=user_ids)
        except TelegramSetupError as exc:
            console.console.print(f"[red]Token Telegram invalide:[/] {exc}")
            return None

    if getattr(args, "enable_telegram", False):
        return _prompt_telegram_setup(console, config, user_ids=user_ids)

    if Confirm.ask("Configurer Telegram pour piloter Magic Claw depuis ton telephone ?", default=False):
        return _prompt_telegram_setup(console, config, user_ids=user_ids)
    return config


def cmd_init(args: argparse.Namespace) -> int:
    console = MagicConsole()
    console.print_header()
    ensure_dirs()
    hardware = _diagnose(console)
    plans = _find_compatible_models(console, hardware)

    try:
        plan = _choose_model_plan(console, plans)
    except (RuntimeError, ValueError) as exc:
        console.console.print(f"[red]{exc}[/]")
        return 2

    console.console.print(
        f"Selected [bold]{plan.option.display_name}[/] with {plan.quantization}, "
        f"context {plan.context_tokens}, step {plan.step_max_tokens}, "
        f"estimated VRAM {plan.estimated_vram_gb:.1f} GB."
    )
    if plan.compatibility == "not_recommended":
        if not Confirm.ask("This model is not recommended for stable H24 use. Continue anyway?", default=False):
            return 1

    config = load_config()
    configured = _configure_telegram_during_init(console, config, args)
    if configured is None:
        return 6
    config = configured

    try:
        with console.task(f"Resolving GGUF file for {plan.option.display_name}") as status:
            status.update("Searching Hugging Face repositories")
            candidate = resolve_model_file(plan)
            status.update(f"Selected {candidate.repo_id}/{candidate.filename}")
    except ModelResolutionError as exc:
        console.console.print(f"[red]{exc}[/]")
        return 3

    if args.no_download:
        model_path: Path | None = None
    else:
        try:
            with console.task(_download_status(candidate.filename, 0, None)) as status:
                model_path = download_model(
                    candidate,
                    on_progress=lambda downloaded, total: status.update(
                        _download_status(candidate.filename, downloaded, total)
                    ),
                )
                status.update(_download_status(candidate.filename, 1, 1))
        except Exception as exc:
            console.console.print(f"[red]Model download failed:[/] {exc}")
            return 4

    llama_path = ""
    if not args.no_runtime:
        try:
            with console.task("Preparing llama.cpp server runtime") as status:
                status.update("Looking for llama-server")
                llama_path = str(ensure_llama_server_binary(hardware, auto_download=True))
                status.update(f"Runtime ready: {llama_path}")
        except LlamaBinaryError as exc:
            console.console.print(f"[yellow]Runtime warning: {exc}[/]")

    config = merge_runtime(
        config,
        {
            "model_option_id": plan.option.id,
            "model_repo": candidate.repo_id,
            "model_file": candidate.filename,
            "model_path": str(model_path) if model_path is not None else "",
            "quantization": candidate.quantization,
            "context_tokens": plan.context_tokens,
            "batch_size": plan.batch_size,
            "ubatch_size": plan.ubatch_size,
            "gpu_layers": plan.gpu_layers,
            "threads": plan.threads,
            "parallel": plan.parallel,
            "max_tokens": plan.max_tokens,
            "step_max_tokens": plan.step_max_tokens,
            "llama_server_path": llama_path,
        },
    )
    save_config(config)
    console.console.print(f"[green]Configuration written:[/] {CONFIG_PATH}")

    if args.start:
        if config.runtime.model_path and config.runtime.llama_server_path:
            console.console.print("[green]Configuration complete. Starting Magic Claw supervisor...[/]")
            return cmd_run(args)
        console.console.print(
            "[yellow]Configuration saved, but the runtime is incomplete. "
            "Fix the warning above, then run `python -m magic_claw run`.[/]"
        )
        return 5
    if config.runtime.model_path and config.runtime.llama_server_path:
        console.console.print("[green]Configuration complete. Run `python -m magic_claw run` to start H24 mode.[/]")
    else:
        console.console.print("[yellow]Configuration saved, but runtime startup is not ready yet.[/]")
    return 0


def _run_supervisor(console: MagicConsole, config) -> int:
    if not config.runtime.model_path:
        console.console.print("[red]No model configured. Run `python -m magic_claw init` first.[/]")
        return 2
    if not config.runtime.llama_server_path:
        if _model_ready(config):
            repaired = _prepare_missing_runtime(console, config)
            if repaired is None:
                return 5
            config = repaired
        else:
            console.console.print("[red]No llama-server runtime configured. Run init again without --no-runtime.[/]")
            return 2

    llama_path = Path(config.runtime.llama_server_path)
    if not llama_path.exists():
        if _model_ready(config):
            repaired = _prepare_missing_runtime(console, config)
            if repaired is None:
                return 5
            config = repaired
        else:
            console.console.print(f"[red]Configured llama-server runtime was not found:[/] {llama_path}")
            return 2

    if not Path(config.runtime.model_path).exists():
        console.console.print(f"[red]Configured model was not found:[/] {config.runtime.model_path}")
        return 2

    if not config.runtime.llama_server_path:
        console.console.print("[red]No llama-server runtime configured. Run init again without --no-runtime.[/]")
        return 2

    try:
        hardware = diagnose_hardware()
        config = _apply_dynamic_runtime_tuning(config, hardware)
        save_config(config)
        ensure_llama_runtime_dependencies(Path(config.runtime.llama_server_path), hardware, auto_download=True)
    except LlamaBinaryError as exc:
        console.console.print(f"[yellow]Runtime dependency warning:[/] {exc}")

    supervisor = Supervisor(config)
    try:
        with console.task("Starting local model") as status:
            supervisor.on_status = lambda message: status.update(message)
            supervisor.start_model_server(timeout_seconds=240)

        model_name = Path(config.runtime.model_path).stem
        console.console.print(f"[green]Model ready:[/] {model_name}")
        telegram_thread = supervisor.start_telegram_control()
        if telegram_thread:
            username = f"@{config.telegram.bot_username}" if config.telegram.bot_username else "Telegram"
            console.console.print(f"[green]Telegram ready:[/] send /help to {username}.")
        elif config.telegram.enabled:
            console.console.print(
                "[yellow]Telegram is enabled but no token is loaded. "
                "Run `python -m magic_claw telegram setup` to reconnect it.[/]"
            )
        console.console.print("Type a task and press Enter. Type [cyan]exit[/] to stop.")
        prompt_buffer = InteractivePromptBuffer()
        history: list[tuple[str, str]] = []

        while True:
            try:
                prompt = _read_interactive_prompt(console)
            except EOFError:
                break

            if not prompt:
                continue
            if prompt.lower() in EXIT_COMMANDS:
                break
            buffered_prompt, from_buffer = prompt_buffer.add(prompt)
            if buffered_prompt is None:
                console.console.print("[dim]Partial instruction saved; continue with the next line.[/]")
                continue

            with console.task("Processing task") as status:
                task_status = _terminal_task_status_callback(console, status, buffered_prompt)
                supervisor.on_status = task_status
                try:
                    supervisor.ensure_model_server_ready(timeout_seconds=240)
                    effective_prompt = _with_interactive_context(buffered_prompt, history)
                    result = supervisor.run_agent_task_result(
                        effective_prompt,
                        max_steps=60,
                        on_status=task_status,
                    )
                except Exception as exc:
                    status.update("Task failed")
                    console.console.print(f"[red]Task failed:[/] {exc}")
                    continue

            if result.status == "done":
                console.console.print(f"[green]{result.answer}[/]")
                history.append((buffered_prompt, result.answer))
                history = history[-5:]
            else:
                console.console.print(f"[red]{result.answer}[/]")
                if from_buffer:
                    history.append((buffered_prompt, result.answer))
                    history = history[-5:]
    finally:
        supervisor.stop()

    console.console.print("[yellow]Magic Claw stopped.[/]")
    return 0


def cmd_boot(_args: argparse.Namespace) -> int:
    console = MagicConsole()
    config = load_config()

    if _runtime_ready():
        console.print_header()
        return _run_supervisor(console, config)

    if _model_ready(config):
        console.print_header()
        repaired = _prepare_missing_runtime(console, config)
        if repaired is None:
            return 5
        console.console.print("[green]Starting Magic Claw supervisor...[/]")
        return _run_supervisor(console, repaired)

    return cmd_init(
        argparse.Namespace(
            no_download=False,
            no_runtime=False,
            start=True,
            telegram_token=None,
            enable_telegram=False,
            telegram_user_id=None,
        )
    )


def cmd_run(_args: argparse.Namespace) -> int:
    console = MagicConsole()
    console.print_header()
    config = load_config()
    return _run_supervisor(console, config)


def cmd_task(args: argparse.Namespace) -> int:
    console = MagicConsole()
    config = load_config()
    if not config.runtime.model_path:
        console.console.print("[red]No model configured. Run init first.[/]")
        return 2
    if not Path(config.runtime.model_path).exists():
        console.console.print(f"[red]Configured model was not found:[/] {config.runtime.model_path}")
        return 2
    if not config.runtime.llama_server_path or not Path(config.runtime.llama_server_path).exists():
        repaired = _prepare_missing_runtime(console, config)
        if repaired is None:
            return 5
        config = repaired

    try:
        hardware = diagnose_hardware()
        config = _apply_dynamic_runtime_tuning(config, hardware)
        save_config(config)
        ensure_llama_runtime_dependencies(Path(config.runtime.llama_server_path), hardware, auto_download=True)
    except LlamaBinaryError as exc:
        console.console.print(f"[yellow]Runtime dependency warning:[/] {exc}")

    supervisor = Supervisor(config, on_status=lambda message: console.console.print(f"[cyan]*[/] {message}"))
    try:
        with console.task("Starting local model") as status:
            supervisor.on_status = lambda message: status.update(message)
            supervisor.start_model_server(timeout_seconds=420)
        task_status = AgentStatusView(args.prompt)
        console.console.print(f"[bold cyan]Task:[/] [dim]{escape(task_status.task)}[/]")

        def print_task_status(message: str) -> None:
            snapshot = task_status.update(message)
            timestamp, action = snapshot.latest
            prefix = f"{timestamp} | " if timestamp else ""
            console.console.print(f"[cyan]*[/] {prefix}{escape(action)}")

        result = supervisor.run_agent_task_result(
            args.prompt,
            max_steps=args.max_steps,
            on_status=print_task_status,
        )
    except Exception as exc:
        console.console.print(f"[red]Task failed:[/] {exc}")
        return 1
    finally:
        supervisor.stop()
    if result.status == "done":
        console.console.print(f"[green]{result.answer}[/]")
        return 0
    console.console.print(f"[red]{result.answer}[/]")
    return 1


def cmd_telegram(args: argparse.Namespace) -> int:
    console = MagicConsole()
    config = load_config()
    command = getattr(args, "telegram_command", None) or "status"

    if command == "status":
        token_state = "configured" if _telegram_token_configured(config) else "missing"
        enabled = "enabled" if config.telegram.enabled else "disabled"
        bot = f"@{config.telegram.bot_username}" if config.telegram.bot_username else "(unknown bot)"
        console.console.print("Telegram status")
        console.console.print(f"state: {enabled}")
        console.console.print(f"token: {token_state}")
        console.console.print(f"bot: {bot}")
        if config.telegram.allow_user_ids:
            console.console.print(f"allowed users: {', '.join(str(item) for item in config.telegram.allow_user_ids)}")
        else:
            console.console.print("allowed users: all")
        return 0

    if command == "setup":
        try:
            if args.token:
                config = _activate_telegram_from_token(console, config, args.token, user_ids=args.user_id)
            else:
                config = _prompt_telegram_setup(console, config, user_ids=args.user_id)
        except TelegramSetupError as exc:
            console.console.print(f"[red]Telegram non configure:[/] {exc}")
            return 2
        save_config(config)
        console.console.print(f"[green]Configuration written:[/] {CONFIG_PATH}")
        return 0

    if command == "disable":
        config = disable_telegram(config)
        save_config(config)
        console.console.print("[yellow]Telegram disabled. Token kept for later reuse.[/]")
        return 0

    if command == "reset":
        config = reset_telegram(config)
        save_config(config)
        console.console.print("[yellow]Telegram reset. Token removed.[/]")
        return 0

    console.console.print(f"[red]Unknown telegram command:[/] {command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magic-claw", description="Magic Claw local terminal agent")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("diagnose", help="Run hardware diagnosis").set_defaults(func=cmd_diagnose)
    sub.add_parser("models", help="Show compatible model recommendations").set_defaults(func=cmd_models)

    init_parser = sub.add_parser("init", help="Interactive plug and play setup")
    init_parser.add_argument("--no-download", action="store_true", help="Resolve model but skip download")
    init_parser.add_argument("--no-runtime", action="store_true", help="Skip llama.cpp runtime preparation")
    init_parser.add_argument("--start", action="store_true", help="Start supervisor after setup")
    init_parser.add_argument("--enable-telegram", action="store_true", help="Enable Telegram control if token is available")
    init_parser.add_argument("--telegram-token", help="Store Telegram bot token in Magic Claw local .env")
    init_parser.add_argument("--telegram-user-id", action="append", type=int, help="Allowed Telegram user id; repeatable")
    init_parser.set_defaults(func=cmd_init)

    telegram_parser = sub.add_parser("telegram", help="Configure Telegram control")
    telegram_sub = telegram_parser.add_subparsers(dest="telegram_command")
    telegram_sub.add_parser("status", help="Show Telegram configuration status").set_defaults(func=cmd_telegram)
    telegram_setup = telegram_sub.add_parser("setup", help="Connect or change the Telegram bot")
    telegram_setup.add_argument("--token", help="Telegram bot token from BotFather")
    telegram_setup.add_argument("--user-id", action="append", type=int, help="Allowed Telegram user id; repeatable")
    telegram_setup.set_defaults(func=cmd_telegram)
    telegram_sub.add_parser("disable", help="Disable Telegram but keep the saved token").set_defaults(func=cmd_telegram)
    telegram_sub.add_parser("reset", help="Disable Telegram and remove the saved token").set_defaults(func=cmd_telegram)
    telegram_parser.set_defaults(func=cmd_telegram)

    sub.add_parser("run", help="Start the local terminal agent").set_defaults(func=cmd_run)

    task_parser = sub.add_parser("task", help="Run one agent task against the configured local model")
    task_parser.add_argument("prompt")
    task_parser.add_argument("--max-steps", type=int, default=60)
    task_parser.set_defaults(func=cmd_task)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ENV_PATH)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.command:
            return cmd_boot(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
