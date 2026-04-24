from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TransferSpeedColumn
from rich.status import Status
from rich.table import Table

from magic_claw.hardware import HardwareInfo
from magic_claw.models.profiles import RuntimePlan


class MagicConsole:
    def __init__(self) -> None:
        self.console = Console()

    @contextmanager
    def task(self, message: str) -> Iterator[Status]:
        with self.console.status(f"[bold cyan]{message}[/]", spinner="dots") as status:
            yield status

    def print_header(self) -> None:
        self.console.print(
            Panel.fit(
                "[bold cyan]Magic Claw[/]\nLocal always-on terminal agent",
                border_style="cyan",
            )
        )

    def print_hardware(self, hardware: HardwareInfo) -> None:
        table = Table(title="Hardware diagnosis", show_lines=False)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Value")

        table.add_row("OS", hardware.os_name)
        table.add_row("CPU", hardware.cpu.name)
        table.add_row("Cores", f"{hardware.cpu.physical_cores} physical / {hardware.cpu.logical_cores} logical")
        if hardware.cpu.max_freq_mhz:
            table.add_row("CPU max", f"{hardware.cpu.max_freq_mhz:.0f} MHz")
        table.add_row("RAM", f"{hardware.memory.total_gb:.1f} GB total / {hardware.memory.available_gb:.1f} GB available")

        if hardware.gpus:
            for index, gpu in enumerate(hardware.gpus):
                table.add_row(f"GPU {index}", gpu.name)
                table.add_row("VRAM", f"{gpu.vram_total_gb:.1f} GB total / {gpu.vram_used_gb:.1f} GB used / {gpu.vram_free_gb:.1f} GB free")
                table.add_row("Driver", gpu.driver_version)
                if gpu.temperature_c is not None:
                    table.add_row("Temperature", f"{gpu.temperature_c} C")
                if gpu.power_draw_w is not None and gpu.power_limit_w is not None:
                    table.add_row("Power", f"{gpu.power_draw_w:.1f} W / {gpu.power_limit_w:.1f} W")
            table.add_row("Stable usable VRAM", f"{hardware.stable_usable_vram_gb:.1f} GB")
        else:
            table.add_row("GPU", "No NVIDIA GPU detected")

        self.console.print(table)

    def print_model_plans(self, plans: list[RuntimePlan]) -> None:
        table = Table(title="Compatible model choices")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Model")
        table.add_column("Status", no_wrap=True)
        table.add_column("Quant", no_wrap=True)
        table.add_column("Ctx", justify="right", no_wrap=True)
        table.add_column("Est. VRAM", justify="right", no_wrap=True)
        table.add_column("Source", no_wrap=True)
        table.add_column("Reason")

        colors = {
            "recommended": "green",
            "compatible": "cyan",
            "tight": "yellow",
            "not_recommended": "red",
        }
        for index, plan in enumerate(plans, start=1):
            color = colors.get(plan.compatibility, "white")
            table.add_row(
                str(index),
                f"{plan.option.display_name}\n[dim]{plan.option.id} | {plan.option.role}[/]",
                f"[{color}]{plan.compatibility}[/]",
                plan.quantization,
                str(plan.context_tokens),
                f"{plan.estimated_vram_gb:.1f} GB",
                plan.option.source,
                plan.reason,
            )
        self.console.print(table)

    def download_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
