from __future__ import annotations

import os
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
        if not str(resolved).lower().startswith(str(self.workspace).lower()):
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

    def run_shell(self, command: str, timeout_seconds: int = 120) -> dict[str, Any]:
        if timeout_seconds <= 0 or timeout_seconds > 1800:
            raise ToolError("timeout_seconds must be between 1 and 1800")
        lowered = command.lower()
        blocked = ["format ", "diskpart", "bcdedit", "shutdown", "restart-computer"]
        if any(token in lowered for token in blocked):
            raise ToolError("Command blocked by safety policy.")
        proc = subprocess.run(
            command,
            cwd=self.workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        return {
            "command": command,
            "returncode": proc.returncode,
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

