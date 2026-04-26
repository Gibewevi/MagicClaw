from __future__ import annotations

import json
import re
import subprocess
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from magic_claw.config import RuntimeSettings
from magic_claw.state import StateStore

from .tools import AgentToolbox, ToolError, validation_command_for_target
from .workspace import (
    active_project_from_observation,
    remember_active_project,
    resolve_workspace_focus,
    workspace_contract_prompt,
)


SYSTEM_PROMPT = """You are Magic Claw, a local autonomous coding and operations agent.
You work inside the local workspace only. Use tools when the task requires
workspace inspection, file changes, shell commands, or verification.

Return exactly one JSON object per turn:
{
  "thought": "short private-free rationale",
  "tool": "list_dir|read_file|write_file|append_file|commit_file|make_dir|search_files|run_shell|final",
  "args": {}
}

Rules:
- Never write prose outside JSON.
- Never use Gemma tool-call wrappers such as <|tool_call> or call:tool.
- Return one tool call only. Never return multiple JSON objects in one turn.
- JSON strings must escape newlines and quotes correctly. For source files,
  prefer concise code. File writes are transactional: write_file creates a
  .tmp draft, append_file adds chunks to that draft, and commit_file validates
  and replaces the final file. Always call commit_file before final.
- Keep args.content under 1200 characters. Split larger files across multiple
  write_file/append_file calls at syntactically safe boundaries. If content is
  rejected as too large or incomplete, resend smaller chunks using append_file.
- For frontend work, avoid one massive App.jsx. Split UI into focused files
  such as src/data/weather.js, src/components/Header.jsx,
  src/components/WeatherHero.jsx, src/components/ForecastGrid.jsx,
  src/App.jsx, and src/index.css.
- For build, install, change, fix, or file-generation tasks, inspect the
  workspace, make the required changes with tools, and verify with a bounded
  command before final whenever the project provides one.
- After every committed code change, Magic Claw automatically runs a bounded
  validation command (build, lint, test, or a light syntax check). If automatic
  validation fails, fix that failure immediately before unrelated work or final.
- Do not use final for actionable workspace tasks until the required work is
  actually done. If verification cannot run, explain the concrete reason in
  the final answer.
- For simple questions or response-only requests that do not require local
  workspace changes, use final directly without listing files.
- Pour une question simple ou une demande de réponse seulement, utilise
  final directement sans explorer les fichiers.
- Use final only when the user task is complete.
- For final, args must be {"answer": "..."}.
- Keep shell commands bounded and deterministic.
- If the step window is compacted, continue from the supplied continuation
  memory. Do not restart the task from scratch.
- For npm scaffolding, use non-interactive commands and lowercase ASCII
  kebab-case project folders, for example `meteo-vite`.
- Do not run long-lived dev servers directly (for example npm run dev). Use
  npm run build, npm run lint, npm test, npx playwright test, or a bounded
  command that starts a server, verifies it, and stops it.
- If a shell command times out or stalls, inspect the observation, read the
  project scripts/config, and choose a bounded alternative. Never repeat the
  same timed-out npm command unchanged.
- If a tool fails, inspect the error and retry with a corrected action.
"""


ACTIONABLE_TERMS = {
    "add",
    "build",
    "building",
    "change",
    "changer",
    "corriger",
    "create",
    "created",
    "creer",
    "cree",
    "develop",
    "developpe",
    "developper",
    "ecrire",
    "fix",
    "fixed",
    "generate",
    "generer",
    "genere",
    "implement",
    "implemented",
    "implementing",
    "install",
    "installer",
    "mettre en place",
    "modifier",
    "scaffold",
    "update",
    "write",
}
QUESTION_PREFIXES = (
    "as tu ",
    "as-tu ",
    "did ",
    "do ",
    "est ce ",
    "est-ce ",
    "how ",
    "pourquoi ",
    "quoi ",
    "status",
    "tu as ",
    "what ",
    "why ",
)
MUTATING_TOOLS = {"append_file", "commit_file", "make_dir", "run_shell", "write_file"}
FILE_MUTATION_TOOLS = {"append_file", "commit_file", "write_file"}
SOURCE_SUFFIXES = {".css", ".html", ".js", ".json", ".jsx", ".py", ".ts", ".tsx"}
VALIDATION_REPAIR_TOOLS = {"append_file", "commit_file", "list_dir", "read_file", "search_files", "write_file"}


class NonConformingToolOutputError(ValueError):
    pass


class IncompleteFileActionError(ValueError):
    pass


class LoopRecoverySignal(RuntimeError):
    def __init__(self, message: str, *, kind: str, path: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.path = path

    @property
    def recovery_key(self) -> str:
        return f"{self.kind}:{self.path or '*'}"


@dataclass
class LoopSafetyGuard:
    max_mutation_retries_per_file: int = 20
    max_same_error_retries: int = 5
    max_write_file_per_task: int = 80
    max_repeated_similar_writes: int = 5
    max_same_final_rejections: int = 5

    def __post_init__(self) -> None:
        self.write_file_count = 0
        self.file_mutation_counts: Counter[str] = Counter()
        self.error_counts: Counter[str] = Counter()
        self.repeated_write_counts: Counter[tuple[str, str, int, str], int] = Counter()
        self.last_write_key: tuple[str, str, int, str] | None = None

    def record(self, tool: str, args: dict[str, Any], observation: dict[str, Any]) -> None:
        path = _path_from_action_or_observation(args, observation)
        if tool in FILE_MUTATION_TOOLS and path:
            self.file_mutation_counts[path] += 1
            if self.file_mutation_counts[path] > self.max_mutation_retries_per_file:
                raise RuntimeError(
                    "Mutation loop detected: too many file mutations for "
                    f"{path} ({self.file_mutation_counts[path]} attempts)."
                )

        if tool == "write_file":
            self.write_file_count += 1
            if self.write_file_count > self.max_write_file_per_task:
                raise RuntimeError(
                    "Mutation loop detected: too many write_file calls in one task "
                    f"({self.write_file_count})."
                )

        signature = _observation_error_signature(observation)
        if signature:
            self.error_counts[signature] += 1
            if self.error_counts[signature] >= self.max_same_error_retries:
                raise LoopRecoverySignal(
                    "Repeated error loop detected after "
                    f"{self.error_counts[signature]} retries: {signature}",
                    kind="same_error",
                    path=path,
                )

        if tool == "write_file" and path:
            size_bucket = _size_bucket(observation.get("bytes"))
            write_key = (tool, path, size_bucket, signature or "no-error")
            if self.last_write_key == write_key:
                self.repeated_write_counts[write_key] += 1
            else:
                self.repeated_write_counts[write_key] = 1
                self.last_write_key = write_key
            if self.repeated_write_counts[write_key] >= self.max_repeated_similar_writes:
                raise LoopRecoverySignal(
                    "Repeated write loop detected: same file, similar size, and same "
                    f"result for {path} ({self.repeated_write_counts[write_key]} attempts).",
                    kind="similar_write",
                    path=path,
                )


@dataclass
class AgentResult:
    task_id: int
    status: str
    answer: str


class LocalModelClient:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        max_tokens = max(1, min(self.settings.max_tokens, self.settings.step_max_tokens))
        request_messages = _fit_messages_for_context(messages, self.settings, max_tokens)
        used_minimal_context = False
        payload = {
            "model": "local",
            "messages": request_messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        timeout = httpx.Timeout(
            connect=10,
            read=self.settings.request_timeout_seconds,
            write=30,
            pool=10,
        )
        attempts = max(1, self.settings.request_retries + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(f"{self.settings.api_base}/chat/completions", json=payload)
                    response.raise_for_status()
                    data = response.json()
                return str(data["choices"][0]["message"]["content"])
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if _is_response_format_error(exc) and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue
                if _is_context_size_error(exc) and not used_minimal_context:
                    used_minimal_context = True
                    payload["messages"] = _minimal_messages_for_context(messages, self.settings, max_tokens)
                    continue
                retryable = exc.response.status_code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt >= attempts:
                    raise RuntimeError(
                        "Model request failed with HTTP "
                        f"{exc.response.status_code}: {exc.response.text[:500]}"
                    ) from exc
                time.sleep(self.settings.request_retry_backoff_seconds * attempt)
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(self.settings.request_retry_backoff_seconds * attempt)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(self.settings.request_retry_backoff_seconds * attempt)
        raise RuntimeError(
            "Model request failed after "
            f"{attempts} attempt(s); last error: {last_error}"
        )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(re.findall(r"\S+", text))
    return max((len(text) + 2) // 3, int(words * 1.3))


def _message_token_estimate(message: dict[str, str]) -> int:
    return 4 + _estimate_tokens(message.get("role", "")) + _estimate_tokens(message.get("content", ""))


def _messages_token_estimate(messages: list[dict[str, str]]) -> int:
    return sum(_message_token_estimate(message) for message in messages)


def _prompt_token_budget(settings: RuntimeSettings, max_tokens: int) -> int:
    safety_margin = max(256, min(1024, settings.context_tokens // 12))
    return max(512, settings.context_tokens - max_tokens - safety_margin)


def _fit_messages_for_context(
    messages: list[dict[str, str]],
    settings: RuntimeSettings,
    max_tokens: int | None = None,
) -> list[dict[str, str]]:
    if not messages:
        return messages

    output_tokens = max_tokens or max(1, min(settings.max_tokens, settings.step_max_tokens))
    budget = _prompt_token_budget(settings, output_tokens)
    compacted = [_compact_message_for_context(message) for message in messages]
    if _messages_token_estimate(compacted) <= budget:
        return compacted

    head_count = 0
    for message in compacted:
        if message.get("role") != "system":
            break
        head_count += 1

    head = compacted[:head_count]
    body = compacted[head_count:]
    if not body:
        return _force_fit_tail(head, budget)

    tail = [_force_fit_message(body[-1], max(384, budget - _messages_token_estimate(head)))]
    selected: list[dict[str, str]] = []
    running = _messages_token_estimate(head + tail)

    for message in reversed(body[:-1]):
        candidate = _compact_message_for_context(message, aggressive=True)
        candidate_tokens = _message_token_estimate(candidate)
        if running + candidate_tokens > budget:
            continue
        selected.append(candidate)
        running += candidate_tokens

    fitted = head + list(reversed(selected)) + tail
    if _messages_token_estimate(fitted) <= budget:
        return fitted
    return _force_fit_tail(fitted, budget)


def _minimal_messages_for_context(
    messages: list[dict[str, str]],
    settings: RuntimeSettings,
    max_tokens: int,
) -> list[dict[str, str]]:
    if not messages:
        return messages
    budget = max(384, _prompt_token_budget(settings, max_tokens) // 2)
    system_messages = [message for message in messages if message.get("role") == "system"]
    last_message = messages[-1]
    minimal: list[dict[str, str]] = []
    remaining = budget

    for system_message in system_messages[:2]:
        forced = _force_fit_message(system_message, max(128, min(remaining // 2, 512)))
        minimal.append(forced)
        remaining -= _message_token_estimate(forced)
        if remaining <= 128:
            break

    last_budget = max(128, budget - _messages_token_estimate(minimal))
    minimal.append(_force_fit_message(last_message, last_budget))
    return _force_fit_tail(minimal, budget)


def _compact_message_for_context(message: dict[str, str], aggressive: bool = False) -> dict[str, str]:
    content = message.get("content", "")
    if content.startswith("Observation:"):
        limit = 3500 if aggressive else 8000
    elif message.get("role") == "assistant":
        limit = 1000 if aggressive else 2500
    elif message.get("role") == "user":
        limit = 2500 if aggressive else 6000
    else:
        limit = 5000 if aggressive else 12000
    return {**message, "content": _trim_text_middle(content, limit)}


def _force_fit_tail(messages: list[dict[str, str]], token_budget: int) -> list[dict[str, str]]:
    fitted: list[dict[str, str]] = []
    running = 0
    for message in messages:
        available = token_budget - running
        if available <= 0:
            break
        forced = _force_fit_message(message, available)
        forced_tokens = _message_token_estimate(forced)
        if forced_tokens > available and fitted:
            break
        fitted.append(forced)
        running += forced_tokens
    return fitted


def _force_fit_message(message: dict[str, str], token_budget: int) -> dict[str, str]:
    if _message_token_estimate(message) <= token_budget:
        return message
    max_chars = max(256, token_budget * 2)
    return {**message, "content": _trim_text_middle(message.get("content", ""), max_chars)}


def _trim_text_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[context truncated]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _is_context_size_error(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 400:
        return False
    text = exc.response.text.lower()
    return "exceed_context_size" in text or "exceeds the available context" in text


def _is_response_format_error(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 400:
        return False
    text = exc.response.text.lower()
    return "response_format" in text or "json_object" in text or "grammar" in text


def _parse_exact_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise NonConformingToolOutputError("Empty model output.")
    if stripped.startswith("<|tool_call>") or "call:" in stripped[:80]:
        raise NonConformingToolOutputError(
            "Non-conforming tool-call wrapper. Return exactly one JSON object."
        )
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        if _looks_like_incomplete_file_action(stripped):
            raise IncompleteFileActionError(
                "Content incomplete. Please resend in smaller chunks using append_file."
            ) from exc
        raise
    if stripped[end:].strip():
        raise NonConformingToolOutputError(
            "Trailing content after JSON object. Return exactly one JSON object."
        )
    if not isinstance(obj, dict):
        raise NonConformingToolOutputError("Top-level JSON value must be an object.")
    return obj


def _parse_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(obj, dict):
            return obj
    if last_error:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _looks_like_incomplete_file_action(text: str) -> bool:
    if not re.search(r'"tool"\s*:\s*"(?:write_file|append_file)"', text):
        return False
    return '"content"' in text or "content:" in text or "import " in text or "export " in text


def _message_excerpt(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated invalid model output]"


def _invalid_json_feedback(exc: Exception, raw: str) -> str:
    if isinstance(exc, IncompleteFileActionError):
        return (
            "Content incomplete. Please resend in smaller chunks using append_file. "
            "Do not resend a partial JSON string; return exactly one valid JSON object."
        )
    if isinstance(exc, NonConformingToolOutputError):
        return (
            f"Invalid tool output: {exc}. Return exactly one valid JSON object and nothing else. "
            "Do not use <|tool_call>, call:tool wrappers, Markdown fences, prose, or multiple JSON objects."
        )
    guidance = [
        f"Invalid JSON: {exc}.",
        "Return exactly one valid JSON object and nothing else.",
        "Do not use <|tool_call>, call:tool wrappers, or multiple JSON objects.",
    ]
    if "write_file" in raw or "Unterminated string" in str(exc) or len(raw) > 3000:
        guidance.append(
            "If writing code, keep args.content under 1200 characters and valid JSON: "
            "escape newlines as \\n or use write_file for the first short chunk and "
            "append_file for later chunks."
        )
    return " ".join(guidance)


def _summarise_action(action: dict[str, Any]) -> str:
    tool = str(action.get("tool", "unknown"))
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    details: list[str] = []
    for key in ("path", "command", "query"):
        value = args.get(key)
        if isinstance(value, str) and value:
            details.append(f"{key}={_trim_text_middle(value, 240)!r}")
    content = args.get("content")
    if isinstance(content, str):
        details.append(f"content_chars={len(content)}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {tool}{suffix}"


def _summarise_observation_payload(payload: str) -> str:
    try:
        observation = json.loads(payload)
    except json.JSONDecodeError:
        return f"- {_trim_text_middle(payload, 500)}"
    if not isinstance(observation, dict):
        return f"- {_trim_text_middle(payload, 500)}"

    if observation.get("timed_out"):
        command = str(observation.get("command", ""))
        timeout_kind = str(observation.get("timeout_kind", "timeout"))
        diagnosis = str(observation.get("diagnosis", "")).strip()
        stderr = str(observation.get("stderr", "")).strip()
        stdout = str(observation.get("stdout", "")).strip()
        tail = stderr or stdout
        line = f"- run_shell timed out ({timeout_kind}) for {command!r}"
        if diagnosis:
            line += f"; diagnosis: {_trim_text_middle(diagnosis, 260)}"
        if tail:
            line += f"; output tail: {_trim_text_middle(tail, 300)}"
        return line

    error = observation.get("error")
    if error:
        tool = observation.get("tool", "tool")
        return f"- {tool} error: {_trim_text_middle(str(error), 500)}"

    command = observation.get("command")
    if isinstance(command, str):
        returncode = observation.get("returncode")
        stdout = str(observation.get("stdout", "")).strip()
        stderr = str(observation.get("stderr", "")).strip()
        tail = stderr or stdout
        line = f"- command {command!r} returned {returncode}"
        if tail:
            line += f"; output tail: {_trim_text_middle(tail, 320)}"
        return line

    path = observation.get("path")
    if isinstance(path, str):
        entries = observation.get("entries")
        if isinstance(entries, list):
            names = [str(item.get("name", "")) for item in entries if isinstance(item, dict)]
            shown = ", ".join(name for name in names[:12] if name)
            suffix = f"; entries: {shown}" if shown else ""
            return f"- listed {path}{suffix}"
        if "bytes" in observation:
            return f"- wrote {path} ({observation.get('bytes')} bytes)"
        if "content" in observation:
            truncated = " truncated" if observation.get("truncated") else ""
            return f"- read {path}{truncated}"
        return f"- path result: {path}"

    return f"- {_trim_text_middle(payload, 500)}"


def _path_from_action_or_observation(args: dict[str, Any], observation: dict[str, Any]) -> str | None:
    value = observation.get("path") or args.get("path")
    if isinstance(value, str) and value:
        return value.replace("\\", "/")
    return None


def _size_bucket(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return -1
    if size < 0:
        return -1
    return size // 256


def _normalise_error_text(text: str) -> str:
    lowered = text.lower()
    if "expected" in lowered and "eof" in lowered:
        return "syntax:eof"
    lowered = re.sub(r"[a-z]:[\\/][^\s)]+", "<path>", lowered)
    lowered = re.sub(r"\b\d+\b", "<n>", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered[:500]


def _observation_error_signature(observation: dict[str, Any]) -> str | None:
    error = observation.get("error")
    if isinstance(error, str) and error:
        return _normalise_error_text(error)
    for key in ("post_write_validation", "project_validation"):
        validation = observation.get(key)
        if not isinstance(validation, dict):
            continue
        returncode = validation.get("returncode")
        if returncode not in (None, 0):
            text = f"{validation.get('stderr', '')}\n{validation.get('stdout', '')}"
            return _normalise_error_text(text)
    if observation.get("timed_out"):
        command = str(observation.get("command", ""))
        return _normalise_error_text(f"timeout:{observation.get('timeout_kind')}:{command}")
    return None


def _validation_failure_summary(observation: dict[str, Any]) -> str | None:
    if observation.get("loop_recovery"):
        error = observation.get("error")
        if isinstance(error, str) and error:
            return _trim_text_middle(error, 1200)
    error = observation.get("error")
    if isinstance(error, str) and error:
        lowered = error.lower()
        if (
            "automatic validation" in lowered
            or "project validation failed" in lowered
            or "content incomplete" in lowered
            or "content invalid" in lowered
        ):
            return _trim_text_middle(error, 1200)
    for key in ("post_write_validation", "project_validation"):
        validation = observation.get(key)
        if not isinstance(validation, dict):
            continue
        returncode = validation.get("returncode")
        if returncode in (None, 0) and not validation.get("timed_out"):
            continue
        command = str(validation.get("command") or "validation command")
        diagnosis = str(validation.get("diagnosis") or "").strip()
        text = f"{validation.get('stderr', '')}\n{validation.get('stdout', '')}"
        detail = diagnosis or _normalise_error_text(text)
        return _trim_text_middle(f"{command} failed: {detail}", 1200)
    return None


def _validation_repair_feedback(summary: str | None) -> str:
    if not summary:
        return ""
    if "loop detected" in summary.lower():
        return (
            "A loop guard was triggered. Do not repeat the same action unchanged. "
            "Change strategy now: read the relevant files and package metadata, inspect "
            "the failing validation, then make the smallest different repair. For frontend "
            "tooling loops, prefer a simpler working setup such as plain CSS or the installed "
            "framework defaults instead of rewriting the same config again.\n"
            f"Loop diagnostic: {summary}"
        )
    return (
        "Automatic validation failed. Fix this failure before any unrelated action "
        "or final response. Inspect the failing file if needed, update the draft in "
        "small chunks, commit it again, and wait for automatic validation to pass.\n"
        f"Validation failure: {summary}"
    )


def _loop_recovery_observation(
    tool: str,
    args: dict[str, Any],
    observation: dict[str, Any],
    signal: LoopRecoverySignal,
    recovery_count: int,
) -> dict[str, Any]:
    recovered = dict(observation)
    recovered["error"] = str(signal)
    recovered["tool"] = tool
    recovered["loop_recovery"] = {
        "kind": signal.kind,
        "path": signal.path,
        "attempt": recovery_count,
        "instruction": (
            "The task is still active. Stop repeating the same action and change strategy. "
            "Read the relevant files or validation output, then repair with a different, "
            "smaller action before final."
        ),
    }
    if "path" not in recovered and isinstance(args.get("path"), str):
        recovered["path"] = args["path"]
    return recovered


def _is_uncommitted_transaction(observation: dict[str, Any]) -> bool:
    transaction = observation.get("transaction")
    return isinstance(transaction, dict) and transaction.get("committed") is False


def _is_committed_file_observation(tool: str, observation: dict[str, Any]) -> bool:
    if tool not in FILE_MUTATION_TOOLS:
        return False
    transaction = observation.get("transaction")
    return isinstance(transaction, dict) and transaction.get("committed") is True


def _build_failure_diagnosis(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "expected" in text and "eof" in text:
        return (
            "Build failed with EOF; this usually means a truncated or incomplete source file. "
            "Do not rewrite the same large file in one call. Continue with smaller append_file chunks "
            "or split the UI into smaller component files."
        )
    return ""


def _post_write_validation_command(project_dir: str, target_path: str) -> str | None:
    return validation_command_for_target(project_dir, target_path)


def _recent_action_lines(messages: list[dict[str, str]], limit: int = 8) -> list[str]:
    lines: list[str] = []
    for message in reversed(messages):
        if len(lines) >= limit:
            break
        if message.get("role") != "assistant":
            continue
        try:
            action = _parse_first_json_object(message.get("content", ""))
        except json.JSONDecodeError:
            continue
        lines.append(_summarise_action(action))
    return list(reversed(lines))


def _recent_observation_lines(messages: list[dict[str, str]], limit: int = 6) -> list[str]:
    lines: list[str] = []
    for message in reversed(messages):
        if len(lines) >= limit:
            break
        content = message.get("content", "")
        if message.get("role") != "user" or not content.startswith("Observation:"):
            continue
        lines.append(_summarise_observation_payload(content.removeprefix("Observation:").strip()))
    return list(reversed(lines))


def _build_continuation_memory(
    *,
    prompt: str,
    messages: list[dict[str, str]],
    completed_tools: list[str],
    active_project: str | None,
    checkpoint_index: int,
    total_steps: int,
    previous_memory: str | None,
) -> str:
    tool_counts = Counter(completed_tools)
    completed = ", ".join(f"{tool}={count}" for tool, count in sorted(tool_counts.items())) or "(none yet)"
    actions = _recent_action_lines(messages)
    observations = _recent_observation_lines(messages)
    sections = [
        "Autonomous continuation memory.",
        f"Checkpoint: {checkpoint_index}",
        f"Total completed steps before checkpoint: {total_steps}",
        f"Active project: {active_project or '(none selected yet)'}",
        "Original user request:",
        _trim_text_middle(prompt, 2200),
        "Completed tool counts:",
        completed,
    ]
    if previous_memory:
        sections.extend(
            [
                "Previous checkpoint memory:",
                _trim_text_middle(previous_memory, 5000),
            ]
        )
    sections.extend(
        [
            "Recent actions:",
            "\n".join(actions) if actions else "- (none)",
            "Recent observations and failures:",
            "\n".join(observations) if observations else "- (none)",
            "Continuation rules:",
            "- Continue the same task without asking for human intervention.",
            "- Do not restart from scratch or recreate sibling projects after failures.",
            "- If a command timed out or stalled, inspect the project and choose a bounded alternative instead of repeating it unchanged.",
            "- Verify with a bounded command before final when the project provides one.",
        ]
    )
    return "\n".join(sections)


def _messages_after_continuation_checkpoint(
    *,
    prompt: str,
    messages: list[dict[str, str]],
    memory: str,
    settings: RuntimeSettings,
    max_tokens: int,
) -> list[dict[str, str]]:
    system_messages = [message for message in messages if message.get("role") == "system"][:2]
    if not system_messages:
        system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    tail_count = max(0, settings.step_compaction_tail_messages)
    tail = messages[-tail_count:] if tail_count else []
    compacted = [
        *system_messages,
        {"role": "system", "content": memory},
        {"role": "user", "content": f"Original user request:\n{_trim_text_middle(prompt, 3000)}"},
        *tail,
        {
            "role": "user",
            "content": (
                "Step window reached and context was compacted into durable memory. "
                "Continue autonomously from the next useful action. If enough work is done, finalise."
            ),
        },
    ]
    return _fit_messages_for_context(compacted, settings, max_tokens)


class AgentLoop:
    def __init__(
        self,
        settings: RuntimeSettings,
        workspace_dir: str,
        state: StateStore,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.workspace_dir = workspace_dir
        self.state = state
        self.on_status = on_status or (lambda _message: None)
        self.client = LocalModelClient(settings)
        self.tools = AgentToolbox(workspace_dir, on_status=self.on_status)
        self._last_validation_failure: str | None = None

    def _parse_action(self, raw: str) -> dict[str, Any]:
        return _parse_exact_json_object(raw)

    def _normalise_prompt(self, prompt: str) -> str:
        decomposed = unicodedata.normalize("NFKD", prompt.casefold())
        return "".join(char for char in decomposed if not unicodedata.combining(char))

    def _current_request_text(self, prompt: str) -> str:
        marker = "Current user request:"
        marker_index = prompt.rfind(marker)
        if marker_index < 0:
            return prompt
        current = prompt[marker_index + len(marker) :].strip()
        return current or prompt

    def _requires_workspace_action(self, prompt: str) -> bool:
        normalised = self._normalise_prompt(self._current_request_text(prompt)).strip()
        if normalised.startswith(QUESTION_PREFIXES):
            return False
        for term in ACTIONABLE_TERMS:
            pattern = re.escape(term).replace(r"\ ", r"\s+")
            if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", normalised):
                return True
        return False

    def _completion_rejection(self, prompt: str, completed_tools: list[str]) -> str | None:
        if not self._requires_workspace_action(prompt):
            return None
        drafts = self.tools.open_transactions()
        if drafts:
            shown = ", ".join(item["path"] for item in drafts[:5])
            return (
                "Open file draft(s) are not committed yet: "
                f"{shown}. Use append_file for missing chunks or commit_file before final."
            )
        if self._last_validation_failure:
            return (
                "Latest post-write validation failed. Fix the reported build or syntax "
                f"error before final: {self._last_validation_failure}"
            )
        if not any(tool in MUTATING_TOOLS for tool in completed_tools):
            return (
                "This is an actionable workspace task. Inspect the workspace, "
                "make the required file or shell changes, and verify before final."
            )
        return None

    def _describe_tool(self, tool: str, args: dict[str, Any]) -> str:
        if tool == "list_dir":
            return f"Listing folder: {args.get('path', '.')}"
        if tool == "read_file":
            return f"Reading file: {args.get('path', '')}"
        if tool == "write_file":
            return f"Writing file: {args.get('path', '')}"
        if tool == "append_file":
            return f"Appending file: {args.get('path', '')}"
        if tool == "commit_file":
            return f"Committing file: {args.get('path', '')}"
        if tool == "make_dir":
            return f"Creating folder: {args.get('path', '')}"
        if tool == "search_files":
            return f"Searching files: {args.get('query', '')}"
        if tool == "run_shell":
            command = str(args.get("command", ""))
            if len(command) > 80:
                command = command[:77] + "..."
            return f"Running command: {command}"
        return f"Running tool: {tool}"

    def _attach_post_write_validation(self, tool: str, observation: dict[str, Any]) -> None:
        if not _is_committed_file_observation(tool, observation):
            return
        project_validation = observation.get("project_validation")
        if isinstance(project_validation, dict):
            if project_validation.get("returncode") == 0 and not project_validation.get("timed_out"):
                self._last_validation_failure = None
                return
        path_value = observation.get("path")
        if not isinstance(path_value, str):
            return
        if Path(path_value).suffix.lower() not in SOURCE_SUFFIXES:
            return
        command = _post_write_validation_command(str(self.tools.working_dir), path_value)
        if not command:
            if self._last_validation_failure and "loop detected" in self._last_validation_failure.lower():
                self._last_validation_failure = None
            return
        try:
            validation = self.tools.run_shell(command, timeout_seconds=180, inactivity_timeout_seconds=90)
        except Exception as exc:
            validation = {"error": str(exc), "returncode": None}
        if isinstance(validation, dict):
            stdout = str(validation.get("stdout", ""))
            stderr = str(validation.get("stderr", ""))
            compact_validation = {
                "command": validation.get("command", command),
                "returncode": validation.get("returncode"),
                "timed_out": validation.get("timed_out", False),
                "stdout": _trim_text_middle(stdout, 4000),
                "stderr": _trim_text_middle(stderr, 4000),
            }
            diagnosis = _build_failure_diagnosis(stdout, stderr)
            if diagnosis:
                compact_validation["diagnosis"] = diagnosis
            observation["post_write_validation"] = compact_validation
            if compact_validation.get("error"):
                self._last_validation_failure = str(compact_validation.get("error"))
            elif compact_validation.get("returncode") == 0 and not compact_validation.get("timed_out"):
                self._last_validation_failure = None
            else:
                self._last_validation_failure = (
                    str(compact_validation.get("diagnosis"))
                    or _normalise_error_text(f"{stderr}\n{stdout}")
                )

    def run(self, prompt: str, max_steps: int = 60) -> AgentResult:
        focus = resolve_workspace_focus(prompt, self.workspace_dir, self.state)
        self.tools.set_active_project(focus.active_project)
        task_id = self.state.create_task(prompt)
        step_window = max(1, max_steps)
        max_continuations = max(0, self.settings.step_compaction_max_cycles)
        max_tokens = max(1, min(self.settings.max_tokens, self.settings.step_max_tokens))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": workspace_contract_prompt(focus)},
            {"role": "user", "content": prompt},
        ]
        invalid_json_retries = 0
        completed_tools: list[str] = []
        total_steps = 0
        window_step = 0
        continuation_count = 0
        continuation_memory: str | None = None
        safety_guard = LoopSafetyGuard()
        loop_recovery_counts: Counter[str] = Counter()
        final_rejection_counts: Counter[str] = Counter()
        self._last_validation_failure = None

        try:
            while True:
                if window_step >= step_window:
                    if continuation_count >= max_continuations:
                        raise RuntimeError(
                            "Step continuation limit reached after "
                            f"{total_steps} step(s) and {continuation_count} compaction(s)."
                        )
                    continuation_count += 1
                    active = focus.active_relative or (focus.active_project.name if focus.active_project else None)
                    continuation_memory = _build_continuation_memory(
                        prompt=prompt,
                        messages=messages,
                        completed_tools=completed_tools,
                        active_project=active,
                        checkpoint_index=continuation_count,
                        total_steps=total_steps,
                        previous_memory=continuation_memory,
                    )
                    self.on_status(
                        f"Compacting context | checkpoint {continuation_count} after {total_steps} steps"
                    )
                    self.state.add_task_memory(
                        task_id,
                        continuation_count,
                        total_steps,
                        continuation_memory,
                    )
                    self.state.add_step(
                        task_id,
                        total_steps,
                        "compaction",
                        _trim_text_middle(continuation_memory, 12000),
                        {"checkpoint_index": continuation_count},
                    )
                    messages = _messages_after_continuation_checkpoint(
                        prompt=prompt,
                        messages=messages,
                        memory=continuation_memory,
                        settings=self.settings,
                        max_tokens=max_tokens,
                    )
                    window_step = 0

                total_steps += 1
                window_step += 1
                status_suffix = (
                    f" | continuation {continuation_count}" if continuation_count else ""
                )
                self.on_status(f"Thinking | step {window_step}/{step_window}{status_suffix}")
                raw = self.client.complete(messages)
                self.state.add_step(task_id, total_steps, "assistant_raw", raw)

                try:
                    action = self._parse_action(raw)
                    invalid_json_retries = 0
                except Exception as exc:
                    invalid_json_retries += 1
                    if invalid_json_retries > 3:
                        raise RuntimeError(f"Model returned invalid JSON repeatedly: {exc}") from exc
                    messages.append({"role": "assistant", "content": _message_excerpt(raw)})
                    messages.append({"role": "user", "content": _invalid_json_feedback(exc, raw)})
                    continue

                tool = str(action.get("tool", ""))
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                thought = str(action.get("thought", ""))
                self.state.add_step(task_id, total_steps, "action", tool, {"thought": thought, "args": args})

                if tool == "final":
                    rejection = self._completion_rejection(prompt, completed_tools)
                    if rejection:
                        final_rejection_counts[rejection] += 1
                        self.state.add_step(task_id, total_steps, "final_rejected", rejection)
                        if final_rejection_counts[rejection] >= safety_guard.max_same_final_rejections:
                            raise RuntimeError(
                                "Repeated completion rejection loop detected after "
                                f"{final_rejection_counts[rejection]} retries: {rejection}"
                            )
                        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
                        messages.append({"role": "user", "content": f"Completion check failed: {rejection}"})
                        continue
                    answer = str(args.get("answer", "Task complete."))
                    self.on_status("Finalizing response")
                    self.state.finish_task(task_id, answer)
                    return AgentResult(task_id=task_id, status="done", answer=answer)

                self.on_status(f"{self._describe_tool(tool, args)} | step {window_step}/{step_window}{status_suffix}")
                if self._last_validation_failure and tool not in VALIDATION_REPAIR_TOOLS:
                    observation = {
                        "error": (
                            "Automatic validation is failing. Repair the current code "
                            "before running unrelated tools or continuing."
                        ),
                        "tool": tool,
                        "validation_block": True,
                        "last_validation_failure": self._last_validation_failure,
                    }
                else:
                    try:
                        observation = self.tools.execute(tool, args)
                    except (ToolError, TypeError, OSError, subprocess.SubprocessError) as exc:  # type: ignore[name-defined]
                        observation = {"error": str(exc), "tool": tool}
                    except Exception as exc:
                        observation = {"error": f"Unexpected tool failure: {exc}", "tool": tool}

                if "error" not in observation and not observation.get("timed_out"):
                    self._attach_post_write_validation(tool, observation)
                validation_failure = _validation_failure_summary(observation)
                if validation_failure:
                    self._last_validation_failure = validation_failure
                try:
                    safety_guard.record(tool, args, observation)
                except LoopRecoverySignal as exc:
                    loop_recovery_counts[exc.recovery_key] += 1
                    recovery_count = loop_recovery_counts[exc.recovery_key]
                    if recovery_count > 3:
                        raise RuntimeError(
                            "Loop recovery failed after "
                            f"{recovery_count - 1} forced strategy change(s): {exc}"
                        ) from exc
                    self.on_status(
                        f"Loop recovery requested: {exc.kind}"
                        + (f" for {exc.path}" if exc.path else "")
                    )
                    observation = _loop_recovery_observation(
                        tool,
                        args,
                        observation,
                        exc,
                        recovery_count,
                    )
                    validation_failure = _validation_failure_summary(observation)
                    if validation_failure:
                        self._last_validation_failure = validation_failure

                observation_text = json.dumps(observation, ensure_ascii=False)[:24000]
                if (
                    "error" not in observation
                    and not observation.get("timed_out")
                    and not _is_uncommitted_transaction(observation)
                ):
                    completed_tools.append(tool)
                    new_active = active_project_from_observation(
                        observation,
                        self.workspace_dir,
                        current_active=focus.active_project,
                    )
                    if new_active:
                        remember_active_project(self.state, focus.workspace, new_active)
                        focus = resolve_workspace_focus(prompt, self.workspace_dir, self.state)
                        self.tools.set_active_project(focus.active_project)
                        active = focus.active_relative or new_active.name
                        observation_text = observation_text[:23500] + (
                            f'\nWorkspace focus set to active project: "{active}".'
                        )
                self.state.add_step(task_id, total_steps, "observation", observation_text)
                messages.append({"role": "assistant", "content": json.dumps(action)})
                observation_feedback = _validation_repair_feedback(validation_failure)
                observation_message = f"Observation: {_trim_text_middle(observation_text, 8000)}"
                if observation_feedback:
                    observation_message += f"\n\n{observation_feedback}"
                messages.append({"role": "user", "content": observation_message})
                time.sleep(0.1)
        except Exception as exc:
            self.state.fail_task(task_id, str(exc))
            return AgentResult(task_id=task_id, status="failed", answer=str(exc))
