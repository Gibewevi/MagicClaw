from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from huggingface_hub import HfApi

from magic_claw.hardware import HardwareInfo

from .catalog import MODEL_CATALOG
from .downloader import _candidate_from_repo
from .profiles import ModelOption, RuntimePlan, build_runtime_plan


RECENT_SEARCH_TERMS = [
    "GGUF coder instruct",
    "GGUF Qwen coder",
    "GGUF agent instruct",
    "GGUF Gemma instruct",
    "GGUF Llama instruct",
]

TRUSTED_RECENT_OWNERS = {
    "bartowski",
    "unsloth",
    "ggml-org",
    "lmstudio-community",
    "qwen",
    "google",
    "deepseek-ai",
    "microsoft",
    "mradermacher",
    "quantfactory",
}


def _owner(repo_id: str) -> str:
    return repo_id.split("/", 1)[0].lower()


def _trust_score(repo_id: str) -> int:
    owner = _owner(repo_id)
    if owner in TRUSTED_RECENT_OWNERS:
        return 0
    if owner.endswith("-community") or "quant" in owner:
        return 1
    return 2


def _normalize_family(repo_id: str) -> str:
    lower = repo_id.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    if "deepseek" in lower:
        return "deepseek"
    if "llama" in lower:
        return "llama"
    if "mistral" in lower or "mixtral" in lower:
        return "mistral"
    if "phi" in lower:
        return "phi"
    return "other"


def _guess_role(repo_id: str) -> str:
    lower = repo_id.lower()
    roles = []
    if "coder" in lower or "code" in lower:
        roles.append("coding")
    if "instruct" in lower:
        roles.append("instruct")
    if "agent" in lower:
        roles.append("agentic")
    if not roles:
        roles.append("general")
    return "/".join(roles)


def _clean_display_name(repo_id: str) -> str:
    name = repo_id.split("/", 1)[-1]
    name = re.sub(r"[-_]?gguf$", "", name, flags=re.IGNORECASE)
    name = name.replace("_", " ").replace("-", " ")
    return " ".join(part for part in name.split() if part)


def _guess_params_b(text: str) -> float | None:
    lower = text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\s*[x-]?\s*(?:b|billion)\b",
        r"\b(\d+(?:\.\d+)?)b\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            value = float(match.group(1))
            if 1 <= value <= 120:
                return value
    # MoE names often include active-parameter hints such as A3B/A4B. Use total
    # size when present, because VRAM must hold the quantized full file.
    return None


def _as_iso(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


def _model_infos(api: HfApi, term: str, limit: int) -> Iterable[object]:
    try:
        return api.list_models(search=term, sort="last_modified", limit=limit)
    except TypeError:
        return api.list_models(search=term, sort="lastModified", limit=limit)


def discover_recent_gguf_models(hardware: HardwareInfo, limit: int = 12, search_limit: int = 40) -> list[RuntimePlan]:
    api = HfApi()
    seen: set[str] = {option.hf_repo_hint for option in MODEL_CATALOG if option.hf_repo_hint}
    seen.update(option.id for option in MODEL_CATALOG)
    plans: list[RuntimePlan] = []

    for term in RECENT_SEARCH_TERMS:
        try:
            infos = _model_infos(api, term, search_limit)
        except Exception:
            continue
        for info in infos:
            repo_id = getattr(info, "id", "")
            if not repo_id or repo_id in seen:
                continue
            lower = repo_id.lower()
            if "gguf" not in lower:
                continue
            if any(blocked in lower for blocked in ("vision", "embed", "embedding", "reranker", "tts", "whisper")):
                continue
            seen.add(repo_id)

            candidate = _candidate_from_repo(api, repo_id, preferred_quant="Q4_K_M")
            if not candidate:
                continue
            params = _guess_params_b(repo_id + " " + candidate.filename)
            if params is None:
                continue

            option = ModelOption(
                id=repo_id,
                display_name=_clean_display_name(repo_id),
                family=_normalize_family(repo_id),
                params_b=params,
                role=_guess_role(repo_id),
                search_terms=[repo_id],
                hf_repo_hint=repo_id,
                priority=18,
                notes="Discovered from recent Hugging Face GGUF models.",
                source="huggingface",
                last_modified=_as_iso(getattr(info, "last_modified", "")),
            )
            plan = build_runtime_plan(option, hardware)
            if plan.compatibility in {"recommended", "compatible"}:
                plans.append(plan)
    plans.sort(
        key=lambda plan: (
            _trust_score(plan.option.hf_repo_hint or plan.option.id),
            0 if "coding" in plan.option.role else 1,
            0 if plan.compatibility == "recommended" else 1,
            plan.estimated_vram_gb,
        )
    )
    deduped: list[RuntimePlan] = []
    seen_shapes: set[tuple[str, float]] = set()
    for plan in plans:
        shape = (plan.option.family, round(plan.option.params_b, 1))
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        deduped.append(plan)
        if len(deduped) >= limit:
            break
    return deduped


def compatible_model_plans(hardware: HardwareInfo, include_recent: bool = True) -> list[RuntimePlan]:
    catalog_plans = []
    for option in MODEL_CATALOG:
        plan = build_runtime_plan(option, hardware)
        if plan.compatibility in {"recommended", "compatible"}:
            catalog_plans.append(plan)
    try:
        recent_plans = discover_recent_gguf_models(hardware, limit=5) if include_recent else []
    except Exception:
        recent_plans = []
    combined = recent_plans + catalog_plans

    rank = {"recommended": 0, "compatible": 1}
    combined.sort(
        key=lambda plan: (
            rank.get(plan.compatibility, 9),
            0 if plan.option.source == "catalog" and plan.option.priority <= 10 else 1,
            plan.option.priority,
            plan.estimated_vram_gb,
        )
    )
    deduped: list[RuntimePlan] = []
    seen_ids: set[str] = set()
    for plan in combined:
        key = (plan.option.hf_repo_hint or plan.option.id).lower()
        if key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(plan)
    return deduped
