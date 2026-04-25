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


SYSTEM_PROMPT = """You are Magic Claw, a local autonomous coding and operations agent.
You work inside the local workspace only. Use tools when the task requires
workspace inspection, file changes, shell commands, or verification.

Return exactly one JSON object per turn:
{
  "thought": "short private-free rationale",
  "tool": "list_dir|read_file|write_file|make_dir|search_files|run_shell|final",
  "args": {}
}

Rules:
- Never write prose outside JSON.
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
MUTATING_TOOLS = {"make_dir", "run_shell", "write_file"}


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
        payload = {
            "model": "local",
            "messages": messages,
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


class AgentLoop:
    def __init__(
        self,
        settings: RuntimeSettings,
        workspace_dir: str,
        state: StateStore,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.client = LocalModelClient(settings)
        self.tools = AgentToolbox(workspace_dir)
        self.state = state
        self.on_status = on_status or (lambda _message: None)

    def _parse_action(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start : end + 1])
            raise

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
        task_id = self.state.create_task(prompt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"Invalid JSON: {exc}. Return only valid JSON."})
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
                self.state.add_step(task_id, step, "observation", observation_text)
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({"role": "user", "content": f"Observation: {observation_text}"})
                time.sleep(0.1)

            raise RuntimeError(f"Max steps reached ({max_steps}) before completion.")
        except Exception as exc:
            self.state.fail_task(task_id, str(exc))
            return AgentResult(task_id=task_id, status="failed", answer=str(exc))
