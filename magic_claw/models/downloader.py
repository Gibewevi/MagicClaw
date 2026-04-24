from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from magic_claw.paths import MODEL_DIR, ensure_dirs

from .profiles import ModelOption, RuntimePlan


@dataclass(frozen=True)
class ModelFileCandidate:
    repo_id: str
    filename: str
    quantization: str


class ModelResolutionError(RuntimeError):
    pass


ProgressCallback = Callable[[float, float | None], None]


def _make_silent_tqdm(on_progress: ProgressCallback | None):
    callback = on_progress or (lambda _current, _total: None)

    class SilentDownloadProgress:
        def __init__(self, *args, **kwargs) -> None:
            self.total = kwargs.get("total")
            self.n = float(kwargs.get("initial") or 0)
            self.desc = kwargs.get("desc") or ""
            self._notify()

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            self.close()

        def update(self, amount: float = 1) -> None:
            self.n += float(amount or 0)
            self._notify()

        def close(self) -> None:
            self._notify()

        def reset(self, total: float | None = None) -> None:
            self.total = total
            self.n = 0
            self._notify()

        def refresh(self) -> None:
            return None

        def clear(self) -> None:
            return None

        def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
            self.desc = desc or ""
            if refresh:
                self._notify()

        def _notify(self) -> None:
            total = float(self.total) if self.total else None
            callback(self.n, total)

    return SilentDownloadProgress


def _is_gguf(filename: str) -> bool:
    return filename.lower().endswith(".gguf")


def _file_quant(filename: str) -> str | None:
    upper = filename.upper()
    for quant in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS"):
        if quant in upper:
            return quant
    match = re.search(r"\bQ[4568](?:_[A-Z0-9]+)?\b", upper)
    return match.group(0) if match else None


def _quant_order(preferred: str) -> list[str]:
    order = [preferred, "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q6_K", "Q8_0"]
    seen: set[str] = set()
    return [item for item in order if not (item in seen or seen.add(item))]


def _score_repo(repo_id: str, option: ModelOption) -> int:
    lower = repo_id.lower()
    score = 0
    if "gguf" in lower:
        score -= 20
    if "qwen" in lower and option.family == "qwen":
        score -= 10
    if "gemma" in lower and option.family == "gemma":
        score -= 10
    if "bartowski" in lower or "unsloth" in lower:
        score -= 8
    if str(int(option.params_b)) in lower:
        score -= 4
    return score


def _candidate_from_repo(api: HfApi, repo_id: str, preferred_quant: str) -> ModelFileCandidate | None:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception:
        return None
    gguf_files = [file for file in files if _is_gguf(file)]
    if not gguf_files:
        return None

    quant_order = _quant_order(preferred_quant)
    scored: list[tuple[int, str, str]] = []
    for filename in gguf_files:
        quant = _file_quant(filename)
        if not quant:
            continue
        try:
            quant_score = quant_order.index(quant)
        except ValueError:
            quant_score = 99
        # Prefer single-file non-imatrix names for predictable startup.
        penalty = 0
        lower = filename.lower()
        if "imatrix" in lower:
            penalty += 3
        if "mmproj" in lower or "mmproj" in repo_id.lower():
            penalty += 20
        if "part-" in lower or lower.endswith(".gguf.split"):
            penalty += 20
        scored.append((quant_score + penalty, filename, quant))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], len(item[1])))
    _, filename, quant = scored[0]
    return ModelFileCandidate(repo_id=repo_id, filename=filename, quantization=quant)


def resolve_model_file(plan: RuntimePlan) -> ModelFileCandidate:
    api = HfApi()
    option = plan.option

    if option.hf_repo_hint:
        candidate = _candidate_from_repo(api, option.hf_repo_hint, plan.quantization)
        if candidate:
            return candidate

    seen: set[str] = set()
    repo_ids: list[str] = []
    for term in option.search_terms:
        try:
            models = api.list_models(search=term, limit=25)
            for model in models:
                repo_id = model.id
                if repo_id not in seen:
                    seen.add(repo_id)
                    repo_ids.append(repo_id)
        except Exception as exc:
            raise ModelResolutionError(
                f"Hugging Face search failed while resolving {option.display_name}: {exc}"
            ) from exc

    repo_ids.sort(key=lambda repo_id: _score_repo(repo_id, option))
    for repo_id in repo_ids[:40]:
        candidate = _candidate_from_repo(api, repo_id, plan.quantization)
        if candidate:
            return candidate

    raise ModelResolutionError(
        f"No GGUF candidate found for {option.display_name}. Try another model or set HF_TOKEN for gated repos."
    )


def download_model(candidate: ModelFileCandidate, on_progress: ProgressCallback | None = None) -> Path:
    ensure_dirs()
    local_dir = MODEL_DIR / candidate.repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=candidate.repo_id,
        filename=candidate.filename,
        repo_type="model",
        local_dir=local_dir,
        tqdm_class=_make_silent_tqdm(on_progress),
    )
    return Path(path)
