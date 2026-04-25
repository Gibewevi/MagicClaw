from __future__ import annotations

from dataclasses import dataclass

from magic_claw.hardware import HardwareInfo


@dataclass(frozen=True)
class QuantProfile:
    name: str
    weight_gb_per_billion: float
    quality_rank: int


QUANTS: dict[str, QuantProfile] = {
    "IQ4_XS": QuantProfile("IQ4_XS", 0.48, 3),
    "Q4_K_M": QuantProfile("Q4_K_M", 0.56, 4),
    "Q5_K_M": QuantProfile("Q5_K_M", 0.68, 5),
    "Q6_K": QuantProfile("Q6_K", 0.82, 6),
    "Q8_0": QuantProfile("Q8_0", 1.05, 8),
}


@dataclass(frozen=True)
class ModelOption:
    id: str
    display_name: str
    family: str
    params_b: float
    role: str
    search_terms: list[str]
    hf_repo_hint: str | None = None
    priority: int = 50
    notes: str = ""
    source: str = "catalog"
    last_modified: str = ""


@dataclass(frozen=True)
class RuntimePlan:
    option: ModelOption
    quantization: str
    compatibility: str
    reason: str
    estimated_vram_gb: float
    context_tokens: int
    batch_size: int
    ubatch_size: int
    gpu_layers: int
    threads: int
    parallel: int


def estimate_vram_gb(params_b: float, quantization: str, context_tokens: int) -> float:
    quant = QUANTS[quantization]
    weights = params_b * quant.weight_gb_per_billion
    kv_cache = params_b * (context_tokens / 32768) * 0.18
    runtime_overhead = 1.4
    return weights + kv_cache + runtime_overhead


def _best_context(params_b: float, usable_vram_gb: float, ram_total_gb: float) -> int:
    if params_b >= 30:
        return 8192
    if params_b >= 24:
        return 12288 if usable_vram_gb >= 18 else 8192
    if params_b >= 14:
        return 16384
    return 32768 if usable_vram_gb >= 18 and ram_total_gb >= 24 else 16384


def _candidate_quants(params_b: float, usable_vram_gb: float) -> list[str]:
    if params_b <= 8:
        return ["Q6_K", "Q5_K_M", "Q4_K_M"]
    if params_b <= 16:
        return ["Q5_K_M", "Q4_K_M", "Q6_K"]
    if params_b <= 28:
        return ["Q4_K_M", "IQ4_XS", "Q5_K_M"]
    return ["IQ4_XS", "Q4_K_M"]


def build_runtime_plan(option: ModelOption, hardware: HardwareInfo) -> RuntimePlan:
    usable_vram = hardware.stable_usable_vram_gb
    ram_total = hardware.memory.total_gb
    context = _best_context(option.params_b, usable_vram, ram_total)
    threads = max(1, min(hardware.cpu.physical_cores - 1, 8))

    selected_quant = "Q4_K_M"
    selected_estimate = estimate_vram_gb(option.params_b, selected_quant, context)
    for quant in _candidate_quants(option.params_b, usable_vram):
        estimate = estimate_vram_gb(option.params_b, quant, context)
        if estimate <= usable_vram:
            selected_quant = quant
            selected_estimate = estimate
            break

    if not hardware.primary_gpu:
        return RuntimePlan(
            option=option,
            quantization="Q4_K_M",
            compatibility="not_recommended",
            reason="No NVIDIA GPU detected; CPU-only mode is not stable for this target.",
            estimated_vram_gb=selected_estimate,
            context_tokens=min(context, 8192),
            batch_size=128,
            ubatch_size=64,
            gpu_layers=0,
            threads=threads,
            parallel=1,
        )

    if selected_estimate <= usable_vram * 0.82 and option.params_b <= 16:
        compatibility = "recommended"
        reason = "Good VRAM margin for long-running use."
    elif selected_estimate <= usable_vram:
        compatibility = "compatible"
        reason = "Fits the GPU, but keep one worker and moderate context."
    elif selected_estimate <= usable_vram + 2.0 and ram_total >= 32:
        compatibility = "tight"
        reason = "May run with some offload, but less stable for H24."
    else:
        compatibility = "not_recommended"
        reason = "Too close to VRAM/RAM limits for reliable continuous use."

    batch = 512 if (option.params_b <= 14 and usable_vram >= 16) or usable_vram >= 18 else 256
    ubatch = 256 if batch >= 512 and usable_vram >= 20 else 128 if batch >= 256 else 64

    return RuntimePlan(
        option=option,
        quantization=selected_quant,
        compatibility=compatibility,
        reason=reason,
        estimated_vram_gb=selected_estimate,
        context_tokens=context,
        batch_size=batch,
        ubatch_size=ubatch,
        gpu_layers=-1 if compatibility != "not_recommended" else 0,
        threads=threads,
        parallel=1,
    )
