from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from magic_claw.config import RuntimeSettings
from magic_claw.state import StateStore

from .tools import AgentToolbox, ToolError


SYSTEM_PROMPT = """You are Magic Claw, a local autonomous coding and operations agent.
You work inside the local workspace only. Complete the task using tools.

Return exactly one JSON object per turn:
{
  "thought": "short private-free rationale",
  "tool": "list_dir|read_file|write_file|make_dir|search_files|run_shell|final",
  "args": {}
}

Rules:
- Never write prose outside JSON.
- Use final only when the user task is complete.
- For final, args must be {"answer": "..."}.
- Keep shell commands bounded and deterministic.
- If a tool fails, inspect the error and retry with a corrected action.
"""


@dataclass
class AgentResult:
    task_id: int
    status: str
    answer: str


class LocalModelClient:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }
        with httpx.Client(timeout=180) as client:
            response = client.post(f"{self.settings.api_base}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])


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

    def run(self, prompt: str, max_steps: int = 40) -> AgentResult:
        task_id = self.state.create_task(prompt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        invalid_json_retries = 0

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
                self.state.add_step(task_id, step, "observation", observation_text)
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({"role": "user", "content": f"Observation: {observation_text}"})
                time.sleep(0.1)

            raise RuntimeError(f"Max steps reached ({max_steps}) before completion.")
        except Exception as exc:
            self.state.fail_task(task_id, str(exc))
            return AgentResult(task_id=task_id, status="failed", answer=str(exc))
