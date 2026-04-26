from magic_claw.agent.loop import AgentLoop
from magic_claw.agent.tools import AgentToolbox, ToolError
from magic_claw.agent.workspace import ACTIVE_PROJECT_SETTING, resolve_workspace_focus
from magic_claw.config import RuntimeSettings
from magic_claw.state import StateStore


class StubModelClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, _messages):
        if not self.responses:
            raise AssertionError("No stub model responses left")
        return self.responses.pop(0)


def test_workspace_focus_uses_project_named_in_prompt(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "Meteo").mkdir(parents=True)
    (workspace / "meteo-app").mkdir()
    state = StateStore(tmp_path / "state.sqlite")

    focus = resolve_workspace_focus("Continuer dans Meteo", workspace, state)

    assert focus.active_project == (workspace / "Meteo").resolve()
    assert state.get_setting(ACTIVE_PROJECT_SETTING) == "Meteo"


def test_toolbox_blocks_new_vite_scaffold_inside_active_project(tmp_path):
    workspace = tmp_path / "workspace"
    active = workspace / "Meteo"
    active.mkdir(parents=True)
    toolbox = AgentToolbox(workspace, active_project="Meteo")

    try:
        toolbox.run_shell("npm create vite@latest meteo-landing -- --template react")
    except ToolError as exc:
        assert "Active project is 'Meteo'" in str(exc)
    else:
        raise AssertionError("Expected Vite scaffolding to be blocked inside active project")


def test_toolbox_allows_vite_scaffold_in_active_project_root(tmp_path):
    workspace = tmp_path / "workspace"
    active = workspace / "Meteo"
    active.mkdir(parents=True)
    toolbox = AgentToolbox(workspace, active_project="Meteo")

    toolbox._validate_shell_command("npm create vite@latest . -- --template react")


def test_toolbox_maps_prefixed_paths_to_active_project_root(tmp_path):
    workspace = tmp_path / "workspace"
    active = workspace / "Meteo"
    active.mkdir(parents=True)
    toolbox = AgentToolbox(workspace, active_project="Meteo")

    toolbox.write_file("Meteo/index.html", "ok")
    toolbox.commit_file("Meteo/index.html")

    assert (active / "index.html").read_text(encoding="utf-8") == "ok"
    assert not (active / "Meteo").exists()


def test_agent_records_created_top_level_project_as_active(tmp_path):
    workspace = tmp_path / "workspace"
    state = StateStore(tmp_path / "state.sqlite")
    loop = AgentLoop(RuntimeSettings(), str(workspace), state)
    loop.client = StubModelClient(
        [
            '{"thought":"create","tool":"make_dir","args":{"path":"Meteo"}}',
            '{"thought":"done","tool":"final","args":{"answer":"Created."}}',
        ]
    )

    result = loop.run("Creer un dossier nomme Meteo", max_steps=2)

    assert result.status == "done"
    assert state.get_setting(ACTIVE_PROJECT_SETTING) == "Meteo"
