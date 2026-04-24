from __future__ import annotations

from magic_claw.hardware import HardwareInfo

from .profiles import ModelOption, RuntimePlan, build_runtime_plan


MODEL_CATALOG: list[ModelOption] = [
    ModelOption(
        id="qwen3.6-27b",
        display_name="Qwen 3.6 27B",
        family="qwen",
        params_b=27,
        role="general/coding",
        search_terms=["qwen3.6 27b gguf", "qwen 3.6 27b instruct gguf", "qwen3 27b gguf"],
        priority=30,
        notes="Requested large model; use Q4 only on 24 GB cards.",
    ),
    ModelOption(
        id="gemma-4-26b-a4b",
        display_name="Gemma 4 26B A4B",
        family="gemma",
        params_b=26,
        role="general",
        search_terms=["gemma 4 26b a4b gguf", "gemma 4 26b gguf", "gemma 26b gguf"],
        priority=35,
        notes="Requested large model; good only if GGUF quant exists and fits.",
    ),
    ModelOption(
        id="gemma-4-e4b",
        display_name="Gemma 4 E4B",
        family="gemma",
        params_b=4,
        role="fast/general",
        search_terms=["gemma 4 e4b gguf", "gemma 4b instruct gguf", "gemma 4b gguf"],
        priority=15,
        notes="Requested small model; best for fast stable control tasks.",
    ),
    ModelOption(
        id="qwen3.6-35b-a3b",
        display_name="Qwen 3.6 35B A3B",
        family="qwen",
        params_b=35,
        role="general/coding",
        search_terms=["qwen3.6 35b a3b gguf", "qwen 3.6 35b gguf", "qwen3 35b gguf"],
        priority=45,
        notes="Requested very large model; risky on RTX 3090 for H24.",
    ),
    ModelOption(
        id="qwen2.5-coder-14b",
        display_name="Qwen2.5 Coder 14B Instruct",
        family="qwen",
        params_b=14,
        role="coding/agentic",
        search_terms=["Qwen2.5-Coder-14B-Instruct-GGUF", "qwen coder 14b instruct gguf"],
        hf_repo_hint="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        priority=5,
        notes="Stable default for RTX 3090 class machines.",
    ),
    ModelOption(
        id="qwen2.5-coder-7b",
        display_name="Qwen2.5 Coder 7B Instruct",
        family="qwen",
        params_b=7,
        role="coding/fast",
        search_terms=["Qwen2.5-Coder-7B-Instruct-GGUF", "qwen coder 7b instruct gguf"],
        hf_repo_hint="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        priority=10,
        notes="Fast fallback when reliability matters more than raw reasoning.",
    ),
    ModelOption(
        id="deepseek-coder-v2-lite",
        display_name="DeepSeek Coder V2 Lite Instruct",
        family="deepseek",
        params_b=16,
        role="coding/agentic",
        search_terms=["DeepSeek-Coder-V2-Lite-Instruct-GGUF", "deepseek coder v2 lite gguf"],
        priority=20,
        notes="Good coding fallback if Qwen GGUF is unavailable.",
    ),
    ModelOption(
        id="llama-3.1-8b",
        display_name="Llama 3.1 8B Instruct",
        family="llama",
        params_b=8,
        role="general/fast",
        search_terms=["Meta-Llama-3.1-8B-Instruct-GGUF", "llama 3.1 8b instruct gguf"],
        priority=25,
        notes="Fast and conservative long-running option.",
    ),
]


def recommended_models(hardware: HardwareInfo) -> list[RuntimePlan]:
    plans = [build_runtime_plan(option, hardware) for option in MODEL_CATALOG]
    rank = {
        "recommended": 0,
        "compatible": 1,
        "tight": 2,
        "not_recommended": 3,
    }
    return sorted(plans, key=lambda plan: (rank[plan.compatibility], plan.option.priority))


def get_option(model_id: str) -> ModelOption | None:
    for option in MODEL_CATALOG:
        if option.id == model_id:
            return option
    return None

