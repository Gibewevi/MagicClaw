from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from magic_claw.state import StateStore


ACTIVE_PROJECT_SETTING = "agent.active_project_path"


@dataclass(frozen=True)
class WorkspaceFocus:
    workspace: Path
    active_project: Path | None = None
    source: str = "none"

    @property
    def active_relative(self) -> str | None:
        if not self.active_project:
            return None
        return self.active_project.relative_to(self.workspace).as_posix()


def resolve_workspace_focus(prompt: str, workspace_dir: str | Path, state: StateStore) -> WorkspaceFocus:
    workspace = Path(workspace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    mentioned = _mentioned_existing_project(prompt, workspace)
    if mentioned:
        remember_active_project(state, workspace, mentioned)
        return WorkspaceFocus(workspace=workspace, active_project=mentioned, source="prompt")

    stored = _stored_active_project(state, workspace)
    if stored:
        return WorkspaceFocus(workspace=workspace, active_project=stored, source="state")

    candidates = _top_level_dirs(workspace)
    if len(candidates) == 1:
        remember_active_project(state, workspace, candidates[0])
        return WorkspaceFocus(workspace=workspace, active_project=candidates[0], source="single-project")

    return WorkspaceFocus(workspace=workspace)


def remember_active_project(state: StateStore, workspace: str | Path, project: str | Path) -> None:
    workspace_path = Path(workspace).resolve()
    project_path = _resolve_project_path(workspace_path, project)
    state.set_setting(ACTIVE_PROJECT_SETTING, project_path.relative_to(workspace_path).as_posix())


def active_project_from_observation(
    observation: dict[str, Any],
    workspace: str | Path,
    current_active: Path | None = None,
) -> Path | None:
    if current_active:
        return None
    workspace_path = Path(workspace).resolve()
    path_value = observation.get("path")
    if isinstance(path_value, str):
        project = _top_level_project_for_path(workspace_path, Path(path_value))
        if project:
            return project

    command = observation.get("command")
    if isinstance(command, str):
        target_name = vite_scaffold_target(command)
        if target_name:
            target = workspace_path / target_name
            if target.exists() and target.is_dir():
                return target.resolve()
    return None


def workspace_contract_prompt(focus: WorkspaceFocus) -> str:
    if focus.active_project:
        active = focus.active_relative or focus.active_project.name
        return (
            "Workspace contract:\n"
            f"- Active project root: {active}\n"
            "- Treat this as the only project for this task and future continuations.\n"
            "- Use paths relative to the active project root; do not prefix them with the project folder name.\n"
            "- Do not create sibling or nested replacement projects such as meteo-app, meteo-landing, or vite-project.\n"
            "- If a framework scaffold would create a new folder, initialize or update the active project in place instead.\n"
            "- Keep the original objective and continue repairing the same project after failures."
        )
    return (
        "Workspace contract:\n"
        "- No active project root has been selected yet.\n"
        "- If this task creates a project or folder, create exactly one top-level project and keep using it.\n"
        "- Do not create alternate project folders after failures; repair the same target or explain the blocker."
    )


def vite_scaffold_target(command: str) -> str | None:
    match = re.search(
        r"(?:npm\s+(?:create|init)\s+vite(?:@latest)?|npx\s+create-vite(?:@latest)?|create-vite(?:@latest)?)\s+([^\s]+)",
        command,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip("\"'")


def _stored_active_project(state: StateStore, workspace: Path) -> Path | None:
    stored = state.get_setting(ACTIVE_PROJECT_SETTING)
    if not isinstance(stored, str) or not stored:
        return None
    try:
        project = _resolve_project_path(workspace, stored)
    except ValueError:
        return None
    if project.exists() and project.is_dir():
        return project
    return None


def _mentioned_existing_project(prompt: str, workspace: Path) -> Path | None:
    normalised_prompt = _normalise(prompt)
    candidates = sorted(_top_level_dirs(workspace), key=lambda item: len(item.name), reverse=True)
    for candidate in candidates:
        name = re.escape(_normalise(candidate.name))
        if re.search(rf"(?<![a-z0-9_-]){name}(?![a-z0-9_-])", normalised_prompt):
            return candidate
    return None


def _top_level_dirs(workspace: Path) -> list[Path]:
    if not workspace.exists():
        return []
    return [
        item.resolve()
        for item in workspace.iterdir()
        if item.is_dir() and not item.name.startswith(".") and item.name != "node_modules"
    ]


def _top_level_project_for_path(workspace: Path, path: Path) -> Path | None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError:
        return None
    if not relative.parts:
        return None
    project = workspace / relative.parts[0]
    if project.exists() and project.is_dir():
        return project.resolve()
    return None


def _resolve_project_path(workspace: Path, project: str | Path) -> Path:
    raw = Path(project)
    resolved = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()
    resolved.relative_to(workspace)
    return resolved


def _normalise(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
