import sys

from magic_claw.agent.loop import AgentLoop
from magic_claw.agent.tools import AgentToolbox, ToolError
from magic_claw.config import RuntimeSettings
from magic_claw.state import StateStore


class StubModelClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, _messages):
        if not self.responses:
            raise AssertionError("No stub model responses left")
        return self.responses.pop(0)


def test_agent_tool_statuses_are_terminal_friendly(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    assert loop._describe_tool("read_file", {"path": "notes.txt"}) == "Reading file: notes.txt"
    assert loop._describe_tool("write_file", {"path": "out.txt"}) == "Writing file: out.txt"
    assert loop._describe_tool("run_shell", {"command": "python --version"}) == "Running command: python --version"


def test_agent_rejects_premature_final_for_actionable_workspace_task(tmp_path):
    workspace = tmp_path / "workspace"
    loop = AgentLoop(RuntimeSettings(), str(workspace), StateStore(tmp_path / "state.sqlite"))
    loop.client = StubModelClient(
        [
            '{"thought":"done","tool":"final","args":{"answer":"Created."}}',
            '{"thought":"write","tool":"write_file","args":{"path":"index.html","content":"ok"}}',
            '{"thought":"done","tool":"final","args":{"answer":"Created index.html."}}',
        ]
    )

    result = loop.run("Développer un site vitrine", max_steps=3)

    assert result.status == "done"
    assert result.answer == "Created index.html."
    assert (workspace / "index.html").read_text(encoding="utf-8") == "ok"


def test_agent_allows_direct_final_for_question(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))
    loop.client = StubModelClient(
        ['{"thought":"answer","tool":"final","args":{"answer":"Pas encore vérifié."}}']
    )

    result = loop.run("Tu as développer le site ?", max_steps=1)

    assert result.status == "done"
    assert result.answer == "Pas encore vérifié."


def test_agent_allows_direct_final_for_response_only_request(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))
    loop.client = StubModelClient(['{"thought":"answer","tool":"final","args":{"answer":"OK"}}'])

    result = loop.run("Réponds simplement: OK", max_steps=1)

    assert result.status == "done"
    assert result.answer == "OK"


def test_run_shell_timeout_is_observable_not_fatal(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    result = toolbox.run_shell(
        f'"{sys.executable}" -c "import time; time.sleep(2)"',
        timeout_seconds=1,
    )

    assert result["timed_out"] is True
    assert result["returncode"] is None


def test_run_shell_rejects_interactive_vite_project_name(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    try:
        toolbox.run_shell("npm create vite@latest Météo_Vite -- --template react")
    except ToolError as exc:
        assert "meteo-vite" in str(exc)
    else:
        raise AssertionError("Expected invalid Vite project name to be rejected")


def test_run_shell_rejects_non_empty_vite_target(tmp_path):
    (tmp_path / "meteo-vite").mkdir()
    (tmp_path / "meteo-vite" / "index.html").write_text("", encoding="utf-8")
    toolbox = AgentToolbox(tmp_path)

    try:
        toolbox.run_shell("npm create vite@latest meteo-vite -- --template react")
    except ToolError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("Expected non-empty Vite target to be rejected")


def test_vite_scaffolding_gets_non_interactive_stdin(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    assert toolbox._shell_stdin("npm create vite@latest meteo-landing -- --template react") == "n\n"
    assert toolbox._shell_stdin("npx create-vite@latest meteo-landing --template react") == "n\n"
    assert toolbox._shell_stdin("npm install") is None
