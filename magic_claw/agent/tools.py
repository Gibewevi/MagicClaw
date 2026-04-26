from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .workspace import vite_scaffold_target


class ToolError(RuntimeError):
    pass


MAX_FILE_CONTENT_CHARS = 1200
PROJECT_VALIDATION_SUFFIXES = {".css", ".html", ".js", ".json", ".jsx", ".py", ".ts", ".tsx"}


@dataclass(frozen=True)
class ShellTimeoutPolicy:
    timeout_seconds: int
    inactivity_timeout_seconds: int | None


class AgentToolbox:
    def __init__(
        self,
        workspace: str | Path,
        active_project: str | Path | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.active_project: Path | None = None
        self.on_status = on_status or (lambda _message: None)
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

    def _validate_file_write_args(self, path: str, content: str) -> None:
        if not isinstance(path, str) or not path.strip():
            raise ToolError("Missing file path.")
        if not isinstance(content, str):
            raise ToolError("Content incomplete. Please resend in smaller chunks using append_file.")
        if len(content) > MAX_FILE_CONTENT_CHARS:
            raise ToolError("Content too large. Use append_file with smaller chunks.")

    def _draft_path(self, target: Path) -> Path:
        return target.with_name(target.name + ".tmp")

    def _uses_transaction(self, target: Path) -> bool:
        return True

    def _write_large_file_content(self, tool: str, path: str, content: str) -> dict[str, Any]:
        target = self._safe_path(path)
        self._ensure_active_write(target, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        draft = self._draft_path(target)
        if tool == "append_file" and not draft.exists() and target.exists():
            draft.write_text(target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            mode = "a"
        else:
            mode = "a" if tool == "append_file" else "w"

        chunks = _split_content_chunks(content, MAX_FILE_CONTENT_CHARS)
        with draft.open(mode, encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk)

        draft_content = draft.read_text(encoding="utf-8", errors="replace")
        valid, reason = _validate_transactional_content(target, draft_content)
        if not valid:
            return {
                "error": reason,
                "tool": tool,
                "path": str(target),
                "tmp_path": str(draft),
                "auto_chunked": True,
                "chunks": len(chunks),
            }

        result = self._commit_validated_draft(target, draft)
        result.update(
            {
                "auto_chunked": True,
                "chunks": len(chunks),
            }
        )
        result["transaction"][
            "note"
        ] = "Oversized complete content was split internally and committed."
        return result

    def _project_validation_command(self, target: Path) -> str | None:
        return validation_command_for_target(self.working_dir, target)

    def _commit_validated_draft(self, target: Path, draft: Path) -> dict[str, Any]:
        command = self._project_validation_command(target)
        if not command:
            draft.replace(target)
            return {
                "path": str(target),
                "bytes": target.stat().st_size,
                "transaction": {"committed": True, "validation": "passed"},
            }

        candidate = draft.read_bytes()
        had_previous = target.exists()
        previous = target.read_bytes() if had_previous else b""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(candidate)
        self.on_status(f"Auto-validating: {command}")
        validation = self.run_shell(command, timeout_seconds=180, inactivity_timeout_seconds=90)
        if validation.get("returncode") != 0 or validation.get("timed_out"):
            if had_previous:
                target.write_bytes(previous)
            else:
                target.unlink(missing_ok=True)
            summary = _project_validation_failure_summary(validation)
            self.on_status(f"Auto-validation failed: {summary}")
            raise ToolError(
                "Project validation failed; final file was not replaced. "
                f"{summary}"
            )
        draft.unlink(missing_ok=True)
        self.on_status(f"Auto-validation passed: {command}")
        return {
            "path": str(target),
            "bytes": target.stat().st_size,
            "transaction": {
                "committed": True,
                "validation": "passed",
            },
            "project_validation": _compact_project_validation(validation),
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        self._validate_file_write_args(path, content)
        target = self._safe_path(path)
        self._ensure_active_write(target, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self._uses_transaction(target):
            target.write_text(content, encoding="utf-8")
            return {"path": str(target), "bytes": target.stat().st_size, "transaction": {"committed": True}}

        draft = self._draft_path(target)
        draft.write_text(content, encoding="utf-8")
        return {
            "path": str(target),
            "tmp_path": str(draft),
            "bytes": draft.stat().st_size,
            "transaction": {
                "committed": False,
                "state": "draft",
                "next": "Use append_file for more chunks, then commit_file with the final path.",
            },
        }

    def append_file(self, path: str, content: str) -> dict[str, Any]:
        self._validate_file_write_args(path, content)
        target = self._safe_path(path)
        self._ensure_active_write(target, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self._uses_transaction(target):
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
            return {"path": str(target), "bytes": target.stat().st_size, "transaction": {"committed": True}}

        draft = self._draft_path(target)
        if not draft.exists() and target.exists():
            draft.write_text(target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        with draft.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return {
            "path": str(target),
            "tmp_path": str(draft),
            "bytes": draft.stat().st_size,
            "transaction": {
                "committed": False,
                "state": "draft",
                "next": "Use append_file for more chunks, then commit_file with the final path.",
            },
        }

    def commit_file(self, path: str) -> dict[str, Any]:
        target = self._safe_path(path)
        self._ensure_active_write(target, path)
        draft = self._draft_path(target)
        if not draft.exists():
            raise ToolError(f"No draft exists for {path}. Use write_file before commit_file.")
        content = draft.read_text(encoding="utf-8", errors="replace")
        valid, reason = _validate_transactional_content(target, content)
        if not valid:
            raise ToolError(reason)
        target.parent.mkdir(parents=True, exist_ok=True)
        return self._commit_validated_draft(target, draft)

    def open_transactions(self) -> list[dict[str, Any]]:
        drafts: list[dict[str, Any]] = []
        root = self.working_dir
        if not root.exists():
            return drafts
        for draft in root.rglob("*.tmp"):
            if not draft.is_file():
                continue
            relative_parts = draft.relative_to(root).parts
            if any(part in {"node_modules", ".git", "dist"} for part in relative_parts):
                continue
            target = draft.with_name(draft.name.removesuffix(".tmp"))
            try:
                rel_target = target.relative_to(root).as_posix()
                rel_draft = draft.relative_to(root).as_posix()
            except ValueError:
                continue
            drafts.append({"path": rel_target, "tmp_path": rel_draft, "bytes": draft.stat().st_size})
        return drafts[:50]

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
        if _is_package_manager_command(lowered):
            env.setdefault("CI", "true")
            env.setdefault("npm_config_yes", "true")
            env.setdefault("npm_config_audit", "false")
            env.setdefault("npm_config_fund", "false")
            env.setdefault("npm_config_progress", "false")
            env.setdefault("npm_config_update_notifier", "false")
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
        if _is_unbounded_dev_server_command(lowered):
            raise ToolError(
                "Long-running dev server command blocked. Use a bounded verification command "
                "such as `npm run build`, `npm run lint`, `npm test`, `npx playwright test`, "
                "or a script that starts the server, verifies it, and stops it."
            )
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

    def _shell_timeout_policy(
        self,
        command: str,
        timeout_seconds: int,
        inactivity_timeout_seconds: int | None = None,
    ) -> ShellTimeoutPolicy:
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            raise ToolError("timeout_seconds must be between 1 and 3600")
        if inactivity_timeout_seconds is not None:
            if inactivity_timeout_seconds <= 0 or inactivity_timeout_seconds > timeout_seconds:
                raise ToolError("inactivity_timeout_seconds must be between 1 and timeout_seconds")
            return ShellTimeoutPolicy(timeout_seconds, inactivity_timeout_seconds)
        if _is_package_manager_command(command.lower()):
            return ShellTimeoutPolicy(timeout_seconds, min(timeout_seconds, 180))
        return ShellTimeoutPolicy(timeout_seconds, None)

    def run_shell(
        self,
        command: str,
        timeout_seconds: int = 300,
        inactivity_timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        policy = self._shell_timeout_policy(command, timeout_seconds, inactivity_timeout_seconds)
        lowered = command.lower()
        blocked = ["format ", "diskpart", "bcdedit", "shutdown", "restart-computer"]
        if any(token in lowered for token in blocked):
            raise ToolError("Command blocked by safety policy.")
        self._validate_shell_command(command)
        started = time.monotonic()
        proc = _start_shell_process(
            command=command,
            cwd=self.working_dir,
            env=self._shell_env(command),
            stdin_text=self._shell_stdin(command),
        )
        stdout, stderr, returncode, timeout_kind = _collect_shell_process(
            proc,
            timeout_seconds=policy.timeout_seconds,
            inactivity_timeout_seconds=policy.inactivity_timeout_seconds,
        )
        duration_seconds = round(time.monotonic() - started, 3)
        if timeout_kind:
            return {
                "command": command,
                "returncode": None,
                "timed_out": True,
                "timeout_kind": timeout_kind,
                "timeout_seconds": policy.timeout_seconds,
                "inactivity_timeout_seconds": policy.inactivity_timeout_seconds,
                "duration_seconds": duration_seconds,
                "stdout": _tail_text(stdout, 20000),
                "stderr": _tail_text(stderr, 12000),
                "diagnosis": _shell_timeout_diagnosis(command, timeout_kind),
                "next_actions": _shell_timeout_next_actions(command),
            }
        return {
            "command": command,
            "returncode": returncode,
            "timed_out": False,
            "timeout_seconds": policy.timeout_seconds,
            "inactivity_timeout_seconds": policy.inactivity_timeout_seconds,
            "duration_seconds": duration_seconds,
            "stdout": stdout[-20000:],
            "stderr": stderr[-12000:],
        }

    def execute(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        mapping = {
            "append_file": self.append_file,
            "commit_file": self.commit_file,
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
        if tool in {"append_file", "write_file"}:
            path = args.get("path")
            content = args.get("content")
            if not isinstance(path, str) or not path.strip():
                raise ToolError("Missing file path.")
            if not isinstance(content, str):
                raise ToolError("Content incomplete. Please resend in smaller chunks using append_file.")
            if len(content) > MAX_FILE_CONTENT_CHARS:
                return self._write_large_file_content(tool, path, content)
        if tool == "commit_file":
            path = args.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ToolError("Missing file path.")
        return func(**args)


def _validate_transactional_content(target: Path, content: str) -> tuple[bool, str]:
    if not content.strip():
        return False, "Content incomplete. Please resend in smaller chunks using append_file."
    suffix = target.suffix.lower()
    if suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            return False, f"Content incomplete or invalid JSON: {exc}. Please resend in smaller chunks using append_file."
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        reason = _js_like_incomplete_reason(content)
        if reason:
            return False, f"Content incomplete. {reason}. Please resend in smaller chunks using append_file."
        reason = _js_like_module_reason(content)
        if reason:
            return False, f"Content invalid. {reason}. Please resend corrected content using append_file."
    if suffix == ".css":
        reason = _balanced_text_incomplete_reason(content, pairs={"{": "}"})
        if reason:
            return False, f"Content incomplete. {reason}. Please resend in smaller chunks using append_file."
    if suffix == ".html" and re.search(r"<[^>]*$", content):
        return False, "Content incomplete. Unterminated HTML tag. Please resend in smaller chunks using append_file."
    return True, "ok"


def _js_like_module_reason(content: str) -> str | None:
    default_exports = re.findall(r"(?m)^\s*export\s+default\b", content)
    if len(default_exports) > 1:
        return "Multiple default exports detected"
    return None


def validation_command_for_target(working_dir: str | Path, target: str | Path) -> str | None:
    root = Path(working_dir).resolve()
    candidate = Path(target).resolve()
    if candidate.suffix.lower() not in PROJECT_VALIDATION_SUFFIXES:
        return None
    package_command = _package_validation_command(root / "package.json")
    if package_command:
        return package_command
    if candidate.suffix.lower() == ".py":
        return f"{_quote_shell_arg(sys.executable)} -m py_compile {_quote_shell_arg(_relative_shell_path(root, candidate))}"
    return None


def _package_validation_command(package_json: Path) -> str | None:
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    scripts = data.get("scripts")
    if not isinstance(scripts, dict):
        return None
    for script_name in ("build", "lint", "test"):
        script_value = scripts.get(script_name)
        if isinstance(script_value, str) and script_value.strip():
            return "npm test" if script_name == "test" else f"npm run {script_name}"
    return None


def _relative_shell_path(root: Path, target: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def _quote_shell_arg(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _compact_project_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": validation.get("command"),
        "returncode": validation.get("returncode"),
        "timed_out": validation.get("timed_out", False),
        "stdout": _tail_text(str(validation.get("stdout", "")), 4000),
        "stderr": _tail_text(str(validation.get("stderr", "")), 4000),
    }


def _project_validation_failure_summary(validation: dict[str, Any]) -> str:
    text = _strip_ansi(
        f"{validation.get('stderr', '')}\n{validation.get('stdout', '')}"
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if "error" in lowered or "failed" in lowered or "expected" in lowered:
            return stripped[:500]
    return "Validation command failed."


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def _split_content_chunks(content: str, max_chars: int) -> list[str]:
    if len(content) <= max_chars:
        return [content]
    chunks: list[str] = []
    current = ""
    for line in content.splitlines(keepends=True):
        if len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(line[index : index + max_chars] for index in range(0, len(line), max_chars))
            continue
        if len(current) + len(line) > max_chars and current:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def _balanced_text_incomplete_reason(content: str, pairs: dict[str, str]) -> str | None:
    stack: list[str] = []
    closers = {closer: opener for opener, closer in pairs.items()}
    for char in content:
        if char in pairs:
            stack.append(char)
        elif char in closers:
            if not stack or stack[-1] != closers[char]:
                return f"Unexpected closing {char!r}"
            stack.pop()
    if stack:
        return f"Unclosed {stack[-1]!r}"
    return None


def _js_like_incomplete_reason(content: str) -> str | None:
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {closer: opener for opener, closer in pairs.items()}
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = 0
    while index < len(content):
        char = content[index]
        nxt = content[index + 1] if index + 1 < len(content) else ""

        if line_comment:
            if char in "\r\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char == "/" and nxt == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and nxt == "*":
            block_comment = True
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char in pairs:
            stack.append(char)
        elif char in closers:
            if not stack or stack[-1] != closers[char]:
                return f"Unexpected closing {char!r}"
            stack.pop()
        index += 1

    if quote:
        return f"Unterminated {quote!r} string"
    if block_comment:
        return "Unterminated block comment"
    if stack:
        return f"Unclosed {stack[-1]!r}"
    return None


def _start_shell_process(
    *,
    command: str,
    cwd: Path,
    env: dict[str, str],
    stdin_text: str | None,
) -> subprocess.Popen[str]:
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        "text": True,
        "bufsize": 1,
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(command, **kwargs)
    if stdin_text is not None and proc.stdin:
        try:
            proc.stdin.write(stdin_text)
            proc.stdin.flush()
        except OSError:
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
    return proc


def _collect_shell_process(
    proc: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    inactivity_timeout_seconds: int | None,
) -> tuple[str, str, int | None, str | None]:
    output: queue.Queue[tuple[str, str]] = queue.Queue()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def reader(name: str, stream: Any) -> None:
        try:
            for chunk in iter(stream.readline, ""):
                if chunk:
                    output.put((name, chunk))
        finally:
            try:
                stream.close()
            except OSError:
                pass

    threads = []
    if proc.stdout:
        threads.append(threading.Thread(target=reader, args=("stdout", proc.stdout), daemon=True))
    if proc.stderr:
        threads.append(threading.Thread(target=reader, args=("stderr", proc.stderr), daemon=True))
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + timeout_seconds
    last_output = time.monotonic()
    timeout_kind: str | None = None

    while True:
        try:
            name, chunk = output.get(timeout=0.1)
            if name == "stdout":
                stdout_parts.append(chunk)
            else:
                stderr_parts.append(chunk)
            last_output = time.monotonic()
        except queue.Empty:
            pass

        now = time.monotonic()
        if proc.poll() is not None:
            break
        if now >= deadline:
            timeout_kind = "absolute"
            _terminate_process_tree(proc)
            break
        if inactivity_timeout_seconds is not None and now - last_output >= inactivity_timeout_seconds:
            timeout_kind = "inactivity"
            _terminate_process_tree(proc)
            break

    for thread in threads:
        thread.join(timeout=1)
    while True:
        try:
            name, chunk = output.get_nowait()
        except queue.Empty:
            break
        if name == "stdout":
            stdout_parts.append(chunk)
        else:
            stderr_parts.append(chunk)
    return "".join(stdout_parts), "".join(stderr_parts), proc.poll(), timeout_kind


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.SubprocessError:
        pass


def _tail_text(value: str | bytes | None, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[-max_chars:]


def _is_unbounded_dev_server_command(lowered_command: str) -> bool:
    command = re.sub(r"\s+", " ", lowered_command.strip())
    command = re.sub(r"^(?:cmd\.exe\s+/[a-z]\s+/[a-z]\s+/c\s+)+", "", command)
    if re.search(r"\b(start-server-and-test|timeout|timeout\.exe)\b", command):
        return False
    long_running_segments = (
        r"(?:^|[;&]\s*)npm(?:\.cmd)?\s+(?:run\s+)?(?:dev|start|serve|preview)\b",
        r"(?:^|[;&]\s*)pnpm(?:\.cmd)?\s+(?:run\s+)?(?:dev|start|serve|preview)\b",
        r"(?:^|[;&]\s*)yarn(?:\.cmd)?\s+(?:dev|start|serve|preview)\b",
        r"(?:^|[;&]\s*)bun\s+(?:run\s+)?(?:dev|start|serve|preview)\b",
        r"(?:^|[;&]\s*)npx(?:\.cmd)?\s+vite\b",
        r"(?:^|[;&]\s*)vite\b",
        r"(?:^|[;&]\s*)next\s+(?:dev|start)\b",
        r"(?:^|[;&]\s*)react-scripts\s+start\b",
        r"(?:^|[;&]\s*)webpack\s+serve\b",
        r"(?:^|[;&]\s*)astro\s+dev\b",
        r"(?:^|[;&]\s*)nuxt\s+dev\b",
    )
    return any(re.search(pattern, command) for pattern in long_running_segments)


def _is_package_manager_command(lowered_command: str) -> bool:
    package_manager = r"(?:npm|npx|pnpm|yarn)(?:\.cmd)?|bun(?:\.exe)?|corepack|create-vite"
    return bool(re.search(rf"(?:^|[;&|]\s*)(?:{package_manager})\b", lowered_command))


def _shell_timeout_diagnosis(command: str, timeout_kind: str) -> str:
    lowered = command.lower()
    if _is_package_manager_command(lowered):
        if timeout_kind == "inactivity":
            return (
                "Package-manager command produced no output before the inactivity watchdog fired. "
                "It may be waiting for input, stuck on lifecycle scripts, network fetches, or a long-lived server."
            )
        return (
            "Package-manager command exceeded its absolute timeout. It may be a long-lived script, "
            "a slow install/build, or a process that spawned children and did not exit."
        )
    if timeout_kind == "inactivity":
        return "Command produced no output before the inactivity watchdog fired."
    return "Command exceeded its absolute timeout and the process tree was terminated."


def _shell_timeout_next_actions(command: str) -> list[str]:
    lowered = command.lower()
    if _is_package_manager_command(lowered):
        return [
            "Inspect package.json scripts before choosing the next command.",
            "Prefer bounded checks such as npm run build, npm run lint, npm test, or npx playwright test.",
            "For installs, retry with non-interactive flags such as npm install --no-audit --no-fund --loglevel=warn.",
            "Do not repeat the same timed-out command unchanged.",
        ]
    return [
        "Inspect the partial output and command intent.",
        "Retry only with a bounded alternative or a longer explicit timeout when the command is expected to finish.",
    ]
