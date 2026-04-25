from __future__ import annotations

import os
import platform
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import httpx

from magic_claw.hardware import HardwareInfo
from magic_claw.paths import RUNTIME_DIR, ensure_dirs


class LlamaBinaryError(RuntimeError):
    pass


Asset = tuple[str, str]


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
    if "cudart" in lower:
        return 9999
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
    if "cuda-12.4" in lower:
        score -= 3
    if "cuda-13" in lower:
        score += 3
    if "avx2" in lower:
        score -= 8
    if "server" in lower:
        score -= 5
    if "vulkan" in lower:
        score += 10
    return score


def _release_assets() -> list[Asset]:
    url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
    assets: list[Asset] = []
    for asset in data.get("assets", []):
        name = str(asset.get("name", ""))
        download_url = str(asset.get("browser_download_url", ""))
        if name and download_url:
            assets.append((name, download_url))
    return assets


def _select_llama_asset(assets: list[Asset], hardware: HardwareInfo) -> Asset:
    candidates: list[tuple[int, str, str]] = []
    for name, download_url in assets:
        score = _asset_score(name, hardware)
        if score < 9999:
            candidates.append((score, name, download_url))
    if not candidates:
        raise LlamaBinaryError("No compatible llama.cpp binary asset found in latest release.")
    candidates.sort(key=lambda item: item[0])
    _, name, download_url = candidates[0]
    return name, download_url


def _latest_llama_asset(hardware: HardwareInfo) -> tuple[str, str]:
    return _select_llama_asset(_release_assets(), hardware)


def _cuda_version(name: str) -> str | None:
    match = re.search(r"cuda[-_]?(\d+(?:\.\d+)?)", name.lower())
    return match.group(1) if match else None


def _cuda_dependency_score(name: str, primary_asset_name: str, hardware: HardwareInfo) -> int:
    lower = name.lower()
    system = platform.system().lower()
    if "cudart" not in lower:
        return 9999
    if not hardware.primary_gpu:
        return 9999
    if system == "windows":
        if "win" not in lower or not lower.endswith(".zip"):
            return 9999
    elif system == "linux":
        if "linux" not in lower:
            return 9999
    else:
        return 9999

    primary_cuda = _cuda_version(primary_asset_name)
    dependency_cuda = _cuda_version(name)
    if primary_cuda and dependency_cuda and primary_cuda != dependency_cuda:
        return 9999

    score = 0
    if "x64" in lower or "x86_64" in lower:
        score -= 10
    if "cuda-12.4" in lower:
        score -= 3
    if "cuda-13" in lower:
        score += 3
    return score


def _select_cuda_dependency_asset(
    assets: list[Asset],
    primary_asset_name: str,
    hardware: HardwareInfo,
) -> Asset | None:
    candidates: list[tuple[int, str, str]] = []
    for name, download_url in assets:
        score = _cuda_dependency_score(name, primary_asset_name, hardware)
        if score < 9999:
            candidates.append((score, name, download_url))
    if not candidates:
        return None
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


def _cuda_dependencies_present(runtime_dir: Path) -> bool:
    if not (runtime_dir / "ggml-cuda.dll").exists():
        return True
    local_dlls = {path.name.lower() for path in runtime_dir.glob("*.dll")}
    has_cudart = any(name.startswith("cudart64_") for name in local_dlls)
    has_cublas = any(name.startswith("cublas64_") for name in local_dlls)
    path_has_cudart = shutil.which("cudart64_12.dll") or shutil.which("cudart64_13.dll")
    path_has_cublas = shutil.which("cublas64_12.dll") or shutil.which("cublas64_13.dll")
    return bool((has_cudart or path_has_cudart) and (has_cublas or path_has_cublas))


def _copy_runtime_dlls(source_dir: Path, runtime_dir: Path) -> None:
    for dll in source_dir.rglob("*.dll"):
        target = runtime_dir / dll.name
        if target.exists():
            continue
        shutil.copy2(dll, target)


def _download_and_extract(name: str, url: str, extract_dir: Path) -> Path:
    archive = RUNTIME_DIR / name
    if not archive.exists():
        _download_file(url, archive)
    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract_archive(archive, extract_dir)
    return archive


def _ensure_cuda_dependencies(
    runtime_dir: Path,
    primary_asset_name: str,
    hardware: HardwareInfo,
    auto_download: bool,
) -> None:
    if not hardware.primary_gpu or _cuda_dependencies_present(runtime_dir):
        return
    if not auto_download:
        raise LlamaBinaryError("CUDA llama.cpp runtime dependencies are missing.")

    dependency = _select_cuda_dependency_asset(_release_assets(), primary_asset_name, hardware)
    if not dependency:
        raise LlamaBinaryError("No compatible CUDA dependency archive found for llama.cpp.")

    name, url = dependency
    dependency_dir = RUNTIME_DIR / Path(name).stem
    _download_and_extract(name, url, dependency_dir)
    _copy_runtime_dlls(dependency_dir, runtime_dir)

    if not _cuda_dependencies_present(runtime_dir):
        raise LlamaBinaryError("CUDA dependencies were downloaded, but required DLLs were not found.")


def ensure_llama_runtime_dependencies(
    binary_path: Path,
    hardware: HardwareInfo,
    auto_download: bool = True,
) -> None:
    runtime_dir = binary_path.resolve().parent
    primary_asset_name = runtime_dir.name + ".zip"
    _ensure_cuda_dependencies(runtime_dir, primary_asset_name, hardware, auto_download)


def ensure_llama_server_binary(hardware: HardwareInfo, auto_download: bool = True) -> Path:
    ensure_dirs()
    existing = find_llama_server_binary()
    if existing:
        ensure_llama_runtime_dependencies(existing, hardware, auto_download=auto_download)
        return existing
    if not auto_download:
        raise LlamaBinaryError("llama-server binary not found.")

    release_assets = _release_assets()
    name, url = _select_llama_asset(release_assets, hardware)
    archive = _download_and_extract(name, url, RUNTIME_DIR / Path(name).stem)
    extract_dir = RUNTIME_DIR / archive.stem
    _ensure_cuda_dependencies(extract_dir, name, hardware, auto_download=auto_download)

    found = find_llama_server_binary()
    if not found:
        raise LlamaBinaryError("Downloaded llama.cpp, but llama-server binary was not found after extraction.")
    return found
