from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .workspace import vite_scaffold_target


class ToolError(RuntimeError):
    pass


class AgentToolbox:
    def __init__(self, workspace: str | Path, active_project: str | Path | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.active_project: Path | None = None
        if active_project:
            self.set_active_project(active_project)

    @property
    def working_dir(self) -> Path:
        return self.active_project or self.workspace

    @property
    def active_project_name(self) -> str | None:
        return self.active_project.name if self.active_project else None

    def set_active_project(self, active_project: str | Path | None) -> None:
        if active_project is None:
            self.active_project = None
            return
        raw = Path(active_project)
        resolved = raw.resolve() if raw.is_absolute() else (self.workspace / raw).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ToolError(f"Active project escapes workspace: {active_project}")
        resolved.mkdir(parents=True, exist_ok=True)
        self.active_project = resolved

    def _base_path_for(self, raw: Path) -> Path:
        if raw.is_absolute():
            return raw
        if self.active_project:
            parts = raw.parts
            if parts and parts[0].casefold() == self.active_project.name.casefold():
                raw = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            return self.active_project / raw
        return self.workspace / raw

    def _safe_path(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        path = self._base_path_for(raw)
        resolved = path.resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ToolError(f"Path escapes workspace: {relative_path}")
        return resolved

    def _ensure_active_write(self, target: Path, requested_path: str) -> None:
        if not self.active_project:
            return
        try:
            target.relative_to(self.active_project)
        except ValueError:
            active = self.active_project.relative_to(self.workspace).as_posix()
            raise ToolError(
                f"Write target escapes active project '{active}': {requested_path}. "
                "Continue inside the active project instead of creating a sibling project."
            )

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
        self._ensure_active_write(target, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes": target.stat().st_size}

    def make_dir(self, path: str) -> dict[str, Any]:
        target = self._safe_path(path)
        self._ensure_active_write(target, path)
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

    def _shell_stdin(self, command: str) -> str | None:
        lowered = command.lower()
        if "create-vite" in lowered or "npm create vite" in lowered or "npm init vite" in lowered:
            return "n\n"
        return None

    def _validate_shell_command(self, command: str) -> None:
        lowered = command.lower()
        project_name = vite_scaffold_target(command)
        if self.active_project and project_name:
            if Path(project_name) == Path("."):
                return
            active = self.active_project.relative_to(self.workspace).as_posix()
            raise ToolError(
                f"Active project is '{active}'. Do not scaffold a new Vite project '{project_name}'. "
                "Initialize or update the active project in place, then verify it."
            )
        if self.active_project and "npm init" in lowered:
            if re.search(r"\b(?:cd|mkdir(?:\s+-p)?)\s+[a-z0-9][a-z0-9_-]*", lowered):
                active = self.active_project.relative_to(self.workspace).as_posix()
                raise ToolError(
                    f"Active project is '{active}'. Run npm initialization in that project root, "
                    "not in a newly created subfolder."
                )
        if not project_name:
            return

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
                cwd=self.working_dir,
                shell=True,
                capture_output=True,
                text=True,
                input=self._shell_stdin(command),
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
