from __future__ import annotations

import json
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from magic_claw.config import RuntimeSettings
from magic_claw.state import StateStore

from .tools import AgentToolbox, ToolError
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
  "tool": "list_dir|read_file|write_file|append_file|make_dir|search_files|run_shell|final",
  "args": {}
}

Rules:
- Never write prose outside JSON.
- Never use Gemma tool-call wrappers such as <|tool_call> or call:tool.
- Return one tool call only. Never return multiple JSON objects in one turn.
- JSON strings must escape newlines and quotes correctly. For source files,
  prefer concise code. For large files, use write_file for the first chunk and
  append_file for later chunks instead of one huge JSON string.
- Keep args.content under 1200 characters. Split larger files across multiple
  write_file/append_file calls at syntactically safe boundaries.
- For build, install, change, fix, or file-generation tasks, inspect the
  workspace, make the required changes with tools, and verify with a bounded
  command before final whenever the project provides one.
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
- For npm scaffolding, use non-interactive commands and lowercase ASCII
  kebab-case project folders, for example `meteo-vite`.
- Do not run long-lived dev servers directly (for example npm run dev). Use
  npm run build, npm run lint, npm test, npx playwright test, or a bounded
  command that starts a server, verifies it, and stops it.
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
MUTATING_TOOLS = {"append_file", "make_dir", "run_shell", "write_file"}


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


def _action_candidates(raw: str) -> list[str]:
    candidates = [raw.strip()]
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1).strip())
    for match in re.finditer(r"<\|tool_call\>(.*?)<tool_call\|>", raw, flags=re.DOTALL):
        candidates.append(match.group(1).strip())
    return [candidate for candidate in candidates if candidate]


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


def _parse_gemma_tool_call(raw: str) -> dict[str, Any] | None:
    payloads = [match.group(1).strip() for match in re.finditer(r"<\|tool_call\>(.*?)<tool_call\|>", raw, re.DOTALL)]
    if not payloads and raw.startswith("<|tool_call>"):
        payloads.append(raw.removeprefix("<|tool_call>").strip())

    for payload in payloads:
        action = _parse_gemma_payload(payload)
        if action:
            return action
    return None


def _parse_truncated_file_action(raw: str) -> dict[str, Any] | None:
    tool_match = re.search(r'"tool"\s*:\s*"(write_file|append_file)"', raw)
    if not tool_match:
        return None
    content_start = _find_json_string_value_start(raw, "content")
    if content_start is None:
        return None

    tool = tool_match.group(1)
    path = _extract_loose_string_field(raw, "path") or _infer_path_from_text(raw)
    if not path:
        return None

    encoded_content, closed = _read_json_string_fragment(raw, content_start)
    if not encoded_content:
        return None
    content = _decode_loose_string(encoded_content)
    if not closed:
        content += "\n"
    return {
        "thought": "Recovered a truncated file-write action from malformed JSON.",
        "tool": tool,
        "args": {"path": path, "content": content},
    }


def _parse_gemma_payload(payload: str) -> dict[str, Any] | None:
    payload = payload.strip()
    if payload.startswith("call:"):
        payload = payload[len("call:") :].strip()
    elif payload.startswith("call"):
        payload = payload[len("call") :].lstrip(":").strip()

    try:
        return _parse_first_json_object(payload)
    except json.JSONDecodeError:
        pass

    if payload.startswith("MagicClaw:"):
        payload = payload[len("MagicClaw:") :].strip()

    match = re.match(r"(?P<tool>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<body>\{.*)", payload, re.DOTALL)
    if not match:
        return None

    tool = match.group("tool")
    body = match.group("body")
    args: dict[str, Any] = {}
    for key in ("command", "path", "query", "content", "answer"):
        value = _extract_loose_string_field(body, key)
        if value is not None:
            args[key] = value
    return {"thought": "Recovered from Gemma tool-call wrapper.", "tool": tool, "args": args}


def _extract_loose_string_field(text: str, field: str) -> str | None:
    key = rf"(?<![A-Za-z0-9_])['\"]?{re.escape(field)}['\"]?\s*:\s*"
    for quote in ('"', "'"):
        pattern = key + re.escape(quote) + rf"((?:\\.|[^{re.escape(quote)}\\])*)" + re.escape(quote)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return _decode_loose_string(match.group(1))
    return None


def _find_json_string_value_start(text: str, field: str) -> int | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', text)
    if not match:
        return None
    return match.end()


def _read_json_string_fragment(text: str, start: int) -> tuple[str, bool]:
    chars: list[str] = []
    escaped = False
    for char in text[start:]:
        if escaped:
            chars.append("\\" + char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return "".join(chars), True
        chars.append(char)
    if escaped:
        chars.append("\\")
    return "".join(chars), False


def _infer_path_from_text(text: str) -> str | None:
    path_patterns = (
        r"`([^`]+\.(?:tsx|jsx|ts|js|css|html|json|md))`",
        r'"([^"]+\.(?:tsx|jsx|ts|js|css|html|json|md))"',
        r"'([^']+\.(?:tsx|jsx|ts|js|css|html|json|md))'",
        r"\b(src/[A-Za-z0-9_./-]+\.(?:tsx|jsx|ts|js|css|html|json|md))\b",
    )
    for pattern in path_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace("\\", "/")
    if "App.jsx" in text:
        return "src/App.jsx"
    if "App.tsx" in text:
        return "src/App.tsx"
    if "import React" in text and ("lucide-react" in text or "framer-motion" in text or "return (" in text):
        return "src/App.jsx"
    return None


def _decode_loose_string(value: str) -> str:
    escaped = value.replace("\r", "\\r").replace("\n", "\\n")
    try:
        return json.loads(f'"{escaped}"')
    except json.JSONDecodeError:
        return (
            value.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\'", "'")
            .replace(r"\n", "\n")
            .replace(r"\r", "\r")
            .replace(r"\t", "\t")
        )


def _message_excerpt(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated invalid model output]"


def _invalid_json_feedback(exc: Exception, raw: str) -> str:
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


class AgentLoop:
    def __init__(
        self,
        settings: RuntimeSettings,
        workspace_dir: str,
        state: StateStore,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.client = LocalModelClient(settings)
        self.workspace_dir = workspace_dir
        self.tools = AgentToolbox(workspace_dir)
        self.state = state
        self.on_status = on_status or (lambda _message: None)

    def _parse_action(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        for candidate in _action_candidates(raw):
            try:
                return _parse_first_json_object(candidate)
            except json.JSONDecodeError:
                continue
        gemma_action = _parse_gemma_tool_call(raw)
        if gemma_action:
            return gemma_action
        truncated_file_action = _parse_truncated_file_action(raw)
        if truncated_file_action:
            return truncated_file_action
        return _parse_first_json_object(raw)

    def _normalise_prompt(self, prompt: str) -> str:
        decomposed = unicodedata.normalize("NFKD", prompt.casefold())
        return "".join(char for char in decomposed if not unicodedata.combining(char))

    def _requires_workspace_action(self, prompt: str) -> bool:
        normalised = self._normalise_prompt(prompt).strip()
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

    def run(self, prompt: str, max_steps: int = 60) -> AgentResult:
        focus = resolve_workspace_focus(prompt, self.workspace_dir, self.state)
        self.tools.set_active_project(focus.active_project)
        task_id = self.state.create_task(prompt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": workspace_contract_prompt(focus)},
            {"role": "user", "content": prompt},
        ]
        invalid_json_retries = 0
        completed_tools: list[str] = []

        try:
            for step in range(1, max_steps + 1):
                self.on_status(f"Thinking | step {step}/{max_steps}")
                raw = self.client.complete(messages)
                self.state.add_step(task_id, step, "assistant_raw", raw)

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
                self.state.add_step(task_id, step, "action", tool, {"thought": thought, "args": args})

                if tool == "final":
                    rejection = self._completion_rejection(prompt, completed_tools)
                    if rejection:
                        self.state.add_step(task_id, step, "final_rejected", rejection)
                        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
                        messages.append({"role": "user", "content": f"Completion check failed: {rejection}"})
                        continue
                    answer = str(args.get("answer", "Task complete."))
                    self.on_status("Finalizing response")
                    self.state.finish_task(task_id, answer)
                    return AgentResult(task_id=task_id, status="done", answer=answer)

                self.on_status(f"{self._describe_tool(tool, args)} | step {step}/{max_steps}")
                try:
                    observation = self.tools.execute(tool, args)
                except (ToolError, TypeError, OSError, subprocess.SubprocessError) as exc:  # type: ignore[name-defined]
                    observation = {"error": str(exc), "tool": tool}
                except Exception as exc:
                    observation = {"error": f"Unexpected tool failure: {exc}", "tool": tool}

                observation_text = json.dumps(observation, ensure_ascii=False)[:24000]
                if "error" not in observation and not observation.get("timed_out"):
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
                self.state.add_step(task_id, step, "observation", observation_text)
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({"role": "user", "content": f"Observation: {_trim_text_middle(observation_text, 8000)}"})
                time.sleep(0.1)

            raise RuntimeError(f"Max steps reached ({max_steps}) before completion.")
        except Exception as exc:
            self.state.fail_task(task_id, str(exc))
            return AgentResult(task_id=task_id, status="failed", answer=str(exc))
