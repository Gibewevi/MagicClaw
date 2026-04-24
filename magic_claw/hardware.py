from __future__ import annotations

import csv
import io
import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    driver_version: str
    temperature_c: int | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def vram_used_gb(self) -> float:
        return self.vram_used_mb / 1024

    @property
    def vram_free_gb(self) -> float:
        return max(0.0, self.vram_total_gb - self.vram_used_gb)


@dataclass(frozen=True)
class CpuInfo:
    name: str
    physical_cores: int
    logical_cores: int
    max_freq_mhz: float | None


@dataclass(frozen=True)
class MemoryInfo:
    total_gb: float
    available_gb: float


@dataclass(frozen=True)
class HardwareInfo:
    os_name: str
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo]

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def stable_usable_vram_gb(self) -> float:
        gpu = self.primary_gpu
        if not gpu:
            return 0.0
        reserve = 4.0 if "Windows" in self.os_name else 2.5
        return max(0.0, gpu.vram_total_gb - gpu.vram_used_gb - reserve)


def _parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except ValueError:
        return None


def _detect_gpus() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    gpus: list[GpuInfo] = []
    reader = csv.reader(io.StringIO(proc.stdout))
    for row in reader:
        if len(row) < 7:
            continue
        temp = _parse_int(row[4])
        draw = _parse_float(row[5])
        limit = _parse_float(row[6])
        try:
            total_mb = int(float(row[1].strip()))
            used_mb = int(float(row[2].strip()))
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                name=row[0].strip(),
                vram_total_mb=total_mb,
                vram_used_mb=used_mb,
                driver_version=row[3].strip(),
                temperature_c=temp,
                power_draw_w=draw,
                power_limit_w=limit,
            )
        )
    return gpus


def diagnose_hardware() -> HardwareInfo:
    freq = psutil.cpu_freq()
    vm = psutil.virtual_memory()
    cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu=CpuInfo(
            name=cpu_name,
            physical_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
            logical_cores=psutil.cpu_count(logical=True) or 1,
            max_freq_mhz=freq.max if freq else None,
        ),
        memory=MemoryInfo(
            total_gb=vm.total / (1024**3),
            available_gb=vm.available / (1024**3),
        ),
        gpus=_detect_gpus(),
    )

