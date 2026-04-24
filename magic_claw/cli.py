from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.markup import escape
from rich.prompt import Confirm, IntPrompt

from .agent import AgentLoop
from .config import load_config, merge_runtime, save_config
from .hardware import diagnose_hardware
from .models import compatible_model_plans, recommended_models
from .models.downloader import ModelResolutionError, download_model, resolve_model_file
from .paths import CONFIG_PATH, ENV_PATH, ensure_dirs
from .runtime.llama_binary import LlamaBinaryError, ensure_llama_server_binary
from .runtime.supervisor import Supervisor
from .state import StateStore
from .ui import MagicConsole


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
            f"({plan.quantization}, contexte {plan.context_tokens})"
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
        f"context {plan.context_tokens}, estimated VRAM {plan.estimated_vram_gb:.1f} GB."
    )
    if plan.compatibility == "not_recommended":
        if not Confirm.ask("This model is not recommended for stable H24 use. Continue anyway?", default=False):
            return 1

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

    config = load_config()
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
            "llama_server_path": llama_path,
        },
    )
    if args.telegram_token:
        ENV_PATH.write_text(f"MAGIC_CLAW_TELEGRAM_TOKEN={args.telegram_token}\n", encoding="utf-8")
    if args.enable_telegram or args.telegram_token:
        config.telegram.enabled = True
    if args.telegram_user_id:
        config.telegram.allow_user_ids = args.telegram_user_id
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

    supervisor = Supervisor(config, on_status=lambda message: console.console.print(f"[cyan]*[/] {message}"))
    with console.task("Magic Claw supervisor running") as status:
        supervisor.on_status = lambda message: status.update(message)
        supervisor.run_forever()
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
    state = StateStore()
    if not config.runtime.model_path:
        console.console.print("[red]No model configured. Run init first.[/]")
        return 2
    loop = AgentLoop(
        config.runtime,
        config.workspace_dir,
        state,
        on_status=lambda message: console.console.print(f"[cyan]*[/] {message}"),
    )
    result = loop.run(args.prompt, max_steps=args.max_steps)
    if result.status == "done":
        console.console.print(f"[green]{result.answer}[/]")
        return 0
    console.console.print(f"[red]{result.answer}[/]")
    return 1


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

    sub.add_parser("run", help="Run H24 supervisor").set_defaults(func=cmd_run)

    task_parser = sub.add_parser("task", help="Run one agent task against the configured local model")
    task_parser.add_argument("prompt")
    task_parser.add_argument("--max-steps", type=int, default=40)
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
