from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    pass


class AgentToolbox:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        path = raw if raw.is_absolute() else self.workspace / raw
        resolved = path.resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ToolError(f"Path escapes workspace: {relative_path}")
        return resolved

    def list_dir(self, path: str = ".") -> dict[str, Any]:
        target = self._safe_path(path)
        if not target.exists():
            raise ToolError(f"Directory not found: {path}")
        if not target.is_dir():
            raise ToolError(f"Not a directory: {path}")
        entries = []
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            entries.append(
                {
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
        return {"path": str(target), "entries": entries[:500]}

    def read_file(self, path: str, max_bytes: int = 20000) -> dict[str, Any]:
        target = self._safe_path(path)
        if not target.exists() or not target.is_file():
            raise ToolError(f"File not found: {path}")
        data = target.read_bytes()[:max_bytes]
        text = data.decode("utf-8", errors="replace")
        return {"path": str(target), "content": text, "truncated": target.stat().st_size > max_bytes}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes": target.stat().st_size}

    def make_dir(self, path: str) -> dict[str, Any]:
        target = self._safe_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": str(target)}

    def search_files(self, query: str, path: str = ".", max_results: int = 50) -> dict[str, Any]:
        root = self._safe_path(path)
        if not root.exists():
            raise ToolError(f"Search root not found: {path}")
        matches = []
        lowered = query.lower()
        for item in root.rglob("*"):
            if len(matches) >= max_results:
                break
            if any(part.startswith(".") for part in item.relative_to(root).parts):
                continue
            if lowered in item.name.lower():
                matches.append(str(item.relative_to(self.workspace)))
                continue
            if item.is_file() and item.stat().st_size <= 512_000:
                try:
                    text = item.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if lowered in text.lower():
                    matches.append(str(item.relative_to(self.workspace)))
        return {"query": query, "matches": matches}

    def _shell_env(self, command: str) -> dict[str, str]:
        env = os.environ.copy()
        lowered = command.lower()
        if any(token in lowered for token in ("npm ", "npx ", "pnpm ", "yarn ", "create-vite")):
            env.setdefault("CI", "true")
            env.setdefault("npm_config_yes", "true")
            env.setdefault("YES", "1")
        return env

    def _validate_shell_command(self, command: str) -> None:
        match = re.search(
            r"(?:npm\s+(?:create|init)\s+vite(?:@latest)?|create-vite)\s+([^\s]+)",
            command,
            flags=re.IGNORECASE,
        )
        if not match:
            return

        project_name = match.group(1).strip("\"'")
        if project_name.startswith("-"):
            raise ToolError("Vite project name is missing; provide a non-interactive ASCII folder name.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", project_name):
            raise ToolError(
                "Invalid Vite project name for non-interactive npm scaffolding. "
                "Use lowercase ASCII kebab-case, for example: meteo-vite."
            )
        target = self._safe_path(project_name)
        if target.exists() and any(target.iterdir()):
            raise ToolError(
                f"Vite target directory is not empty: {project_name}. "
                "Choose a new lowercase ASCII folder or explicitly clean the directory first."
            )

    def run_shell(self, command: str, timeout_seconds: int = 300) -> dict[str, Any]:
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            raise ToolError("timeout_seconds must be between 1 and 3600")
        lowered = command.lower()
        blocked = ["format ", "diskpart", "bcdedit", "shutdown", "restart-computer"]
        if any(token in lowered for token in blocked):
            raise ToolError("Command blocked by safety policy.")
        self._validate_shell_command(command)
        try:
            proc = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=self._shell_env(command),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "command": command,
                "returncode": None,
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
                "stdout": _tail_text(exc.stdout, 20000),
                "stderr": _tail_text(exc.stderr, 12000),
            }
        return {
            "command": command,
            "returncode": proc.returncode,
            "timed_out": False,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-12000:],
        }

    def execute(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "make_dir": self.make_dir,
            "search_files": self.search_files,
            "run_shell": self.run_shell,
        }
        func = mapping.get(tool)
        if not func:
            raise ToolError(f"Unknown tool: {tool}")
        return func(**args)


def _tail_text(value: str | bytes | None, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[-max_chars:]
