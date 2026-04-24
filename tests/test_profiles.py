from magic_claw.hardware import CpuInfo, GpuInfo, HardwareInfo, MemoryInfo
from magic_claw.models.catalog import recommended_models
from magic_claw.models.recent import compatible_model_plans


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
