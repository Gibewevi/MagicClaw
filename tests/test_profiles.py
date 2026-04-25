from magic_claw.hardware import CpuInfo, GpuInfo, HardwareInfo, MemoryInfo
from magic_claw.models.catalog import recommended_models
from magic_claw.models.recent import compatible_model_plans
from magic_claw.config import RuntimeSettings
from magic_claw.runtime.llama_binary import _asset_score, _cuda_dependencies_present, _select_cuda_dependency_asset
from magic_claw.runtime.llama_server import LlamaServer
from magic_claw.runtime.llama_server import _friendly_startup_stage, _startup_status


def test_rtx_3090_prefers_mid_sized_models_for_stability():
    hardware = HardwareInfo(
        os_name="Windows 11",
        cpu=CpuInfo(name="i7-9700K", physical_cores=8, logical_cores=8, max_freq_mhz=3600),
        memory=MemoryInfo(total_gb=16.0, available_gb=10.0),
        gpus=[
            GpuInfo(
                name="NVIDIA GeForce RTX 3090",
                vram_total_mb=24576,
                vram_used_mb=1800,
                driver_version="test",
            )
        ],
    )
    plans = recommended_models(hardware)
    assert plans[0].compatibility == "recommended"
    assert plans[0].option.params_b <= 16
    assert any(plan.option.id == "qwen3.6-35b-a3b" for plan in plans)


def test_compatible_model_plans_filters_unstable_large_models_without_network():
    hardware = HardwareInfo(
        os_name="Windows 11",
        cpu=CpuInfo(name="i7-9700K", physical_cores=8, logical_cores=8, max_freq_mhz=3600),
        memory=MemoryInfo(total_gb=16.0, available_gb=10.0),
        gpus=[
            GpuInfo(
                name="NVIDIA GeForce RTX 3090",
                vram_total_mb=24576,
                vram_used_mb=1800,
                driver_version="test",
            )
        ],
    )
    plans = compatible_model_plans(hardware, include_recent=False)
    assert plans
    assert all(plan.compatibility in {"recommended", "compatible"} for plan in plans)
    assert all(plan.option.id != "qwen3.6-35b-a3b" for plan in plans)


def test_llama_runtime_selection_ignores_cudart_dependency_archives():
    hardware = HardwareInfo(
        os_name="Windows 11",
        cpu=CpuInfo(name="i7-9700K", physical_cores=8, logical_cores=8, max_freq_mhz=3600),
        memory=MemoryInfo(total_gb=16.0, available_gb=10.0),
        gpus=[
            GpuInfo(
                name="NVIDIA GeForce RTX 3090",
                vram_total_mb=24576,
                vram_used_mb=1800,
                driver_version="test",
            )
        ],
    )
    assert _asset_score("cudart-llama-bin-win-cuda-12.4-x64.zip", hardware) == 9999
    assert _asset_score("llama-b8920-bin-win-cuda-12.4-x64.zip", hardware) < 9999


def test_llama_runtime_selects_matching_cuda_dependency_archive():
    hardware = HardwareInfo(
        os_name="Windows 11",
        cpu=CpuInfo(name="i7-9700K", physical_cores=8, logical_cores=8, max_freq_mhz=3600),
        memory=MemoryInfo(total_gb=16.0, available_gb=10.0),
        gpus=[
            GpuInfo(
                name="NVIDIA GeForce RTX 3090",
                vram_total_mb=24576,
                vram_used_mb=1800,
                driver_version="test",
            )
        ],
    )
    assets = [
        ("cudart-llama-bin-win-cuda-13.1-x64.zip", "https://example.invalid/13.zip"),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "https://example.invalid/12.zip"),
    ]

    selected = _select_cuda_dependency_asset(assets, "llama-b8924-bin-win-cuda-12.4-x64.zip", hardware)

    assert selected == ("cudart-llama-bin-win-cuda-12.4-x64.zip", "https://example.invalid/12.zip")


def test_cuda_dependency_probe_checks_local_runtime_dlls(tmp_path, monkeypatch):
    monkeypatch.setattr("magic_claw.runtime.llama_binary.shutil.which", lambda _name: None)
    (tmp_path / "ggml-cuda.dll").write_text("", encoding="utf-8")
    assert _cuda_dependencies_present(tmp_path) is False

    (tmp_path / "cudart64_12.dll").write_text("", encoding="utf-8")
    (tmp_path / "cublas64_12.dll").write_text("", encoding="utf-8")
    assert _cuda_dependencies_present(tmp_path) is True


def test_llama_server_command_uses_gpu_and_stability_flags():
    settings = RuntimeSettings(
        llama_server_path=r"D:\runtime\llama-server.exe",
        model_path=r"D:\models\model.gguf",
        gpu_layers=-1,
    )

    command = LlamaServer(settings).command()

    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("--reasoning") + 1] == "off"
    assert command[command.index("--reasoning-budget") + 1] == "0"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert "--kv-offload" in command


def test_llama_startup_status_stays_short_and_readable():
    stage = _friendly_startup_stage(
        "common_params_fit_impl: getting device memory data for initial parameters:"
    )
    assert stage == "checking VRAM"
    assert _startup_status("Loading Qwen3.6-27B-IQ4_XS", 42, stage) == (
        "Loading Qwen3.6-27B-IQ4_XS | checking VRAM | 42s"
    )
    assert _friendly_startup_stage("load_tensors: offloaded 65/65 layers to GPU") == "GPU layers 65/65"
