from __future__ import annotations

import os
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path

import httpx

from magic_claw.hardware import HardwareInfo
from magic_claw.paths import RUNTIME_DIR, ensure_dirs


class LlamaBinaryError(RuntimeError):
    pass


def find_llama_server_binary() -> Path | None:
    env_path = os.environ.get("MAGIC_CLAW_LLAMA_SERVER")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    for name in ("llama-server.exe", "llama-server"):
        found = shutil.which(name)
        if found:
            return Path(found)

    if RUNTIME_DIR.exists():
        for candidate in RUNTIME_DIR.rglob("llama-server*"):
            if candidate.is_file() and candidate.suffix.lower() in ("", ".exe"):
                return candidate
    return None


def _asset_score(name: str, hardware: HardwareInfo) -> int:
    lower = name.lower()
    system = platform.system().lower()
    score = 0
    if system == "windows":
        if "win" not in lower or not lower.endswith(".zip"):
            return 9999
        score -= 50
    elif system == "linux":
        if "linux" not in lower:
            return 9999
        score -= 50
    elif system == "darwin":
        if "macos" not in lower and "darwin" not in lower:
            return 9999
        score -= 50

    if "x64" in lower or "x86_64" in lower:
        score -= 20
    if hardware.primary_gpu and ("cuda" in lower or "cu12" in lower):
        score -= 40
    if "avx2" in lower:
        score -= 8
    if "server" in lower:
        score -= 5
    if "vulkan" in lower:
        score += 10
    return score


def _latest_llama_asset(hardware: HardwareInfo) -> tuple[str, str]:
    url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
    assets = data.get("assets", [])
    candidates: list[tuple[int, str, str]] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        download_url = str(asset.get("browser_download_url", ""))
        if not download_url:
            continue
        score = _asset_score(name, hardware)
        if score < 9999:
            candidates.append((score, name, download_url))
    if not candidates:
        raise LlamaBinaryError("No compatible llama.cpp binary asset found in latest release.")
    candidates.sort(key=lambda item: item[0])
    _, name, download_url = candidates[0]
    return name, download_url


def _download_file(url: str, target: Path) -> None:
    with httpx.Client(timeout=None, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)


def _extract_archive(archive: Path, target_dir: Path) -> None:
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
        return
    if archive.suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"]) or archive.suffix.lower() in (".tgz", ".xz"):
        with tarfile.open(archive) as tf:
            tf.extractall(target_dir)
        return
    raise LlamaBinaryError(f"Unsupported llama.cpp archive: {archive.name}")


def ensure_llama_server_binary(hardware: HardwareInfo, auto_download: bool = True) -> Path:
    ensure_dirs()
    existing = find_llama_server_binary()
    if existing:
        return existing
    if not auto_download:
        raise LlamaBinaryError("llama-server binary not found.")

    name, url = _latest_llama_asset(hardware)
    archive = RUNTIME_DIR / name
    if not archive.exists():
        _download_file(url, archive)
    extract_dir = RUNTIME_DIR / archive.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract_archive(archive, extract_dir)

    found = find_llama_server_binary()
    if not found:
        raise LlamaBinaryError("Downloaded llama.cpp, but llama-server binary was not found after extraction.")
    return found

