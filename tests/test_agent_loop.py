import sys

from magic_claw.agent.loop import (
    AgentLoop,
    _fit_messages_for_context,
    _messages_token_estimate,
    _minimal_messages_for_context,
    _prompt_token_budget,
)
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
    assert loop._describe_tool("append_file", {"path": "out.txt"}) == "Appending file: out.txt"
    assert loop._describe_tool("run_shell", {"command": "python --version"}) == "Running command: python --version"


def test_agent_parses_first_json_object_from_multiple_objects(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    action = loop._parse_action(
        '{"thought":"one","tool":"list_dir","args":{}}\n'
        '{"thought":"two","tool":"run_shell","args":{"command":"dir"}}'
    )

    assert action == {"thought": "one", "tool": "list_dir", "args": {}}


def test_agent_recovers_json_from_gemma_tool_call_wrapper(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    action = loop._parse_action(
        '<|tool_call>call:{"args":{"path":"src/App.css"},"thought":"read","tool":"read_file"}<tool_call|>'
    )

    assert action["tool"] == "read_file"
    assert action["args"] == {"path": "src/App.css"}


def test_agent_recovers_loose_gemma_tool_call_wrapper(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    action = loop._parse_action('<|tool_call>call:MagicClaw:run_shell{command: "dir /s tests"}<tool_call|>')

    assert action["tool"] == "run_shell"
    assert action["args"] == {"command": "dir /s tests"}


def test_agent_recovers_truncated_write_file_with_explicit_path(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    action = loop._parse_action(
        '{"thought":"write app","tool":"write_file","args":{"path":"src/App.jsx",'
        "\"content\":\"import React from \\'react\\';\\n<div className=\\\"weather\\\">"
    )

    assert action["tool"] == "write_file"
    assert action["args"]["path"] == "src/App.jsx"
    assert "className=\"weather\"" in action["args"]["content"]


def test_agent_recovers_truncated_write_file_by_inferred_path(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    action = loop._parse_action(
        '{"thought":"Rewrite `src/App.jsx` safely","tool":"write_file","args":{'
        "\"content\":\"import React from \\'react\\';\\nexport default function App() {"
    )

    assert action["tool"] == "write_file"
    assert action["args"]["path"] == "src/App.jsx"
    assert "export default function App()" in action["args"]["content"]


def test_agent_recovers_truncated_react_write_without_path(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    action = loop._parse_action(
        '{"thought":"create component","tool":"write_file","args":{'
        "\"content\":\"import React from \\'react\\';\\nimport { Sun } from \\'lucide-react\\';\\n"
    )

    assert action["tool"] == "write_file"
    assert action["args"]["path"] == "src/App.jsx"


def test_model_messages_are_compacted_under_context_budget():
    settings = RuntimeSettings(context_tokens=2048, max_tokens=512, step_max_tokens=512)
    messages = [
        {"role": "system", "content": "system" * 200},
        {"role": "system", "content": "workspace" * 200},
        {"role": "user", "content": "Initial request"},
    ]
    for index in range(20):
        messages.append(
            {
                "role": "assistant",
                "content": '{"thought":"read","tool":"read_file","args":{"path":"src/App.tsx"}}',
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {index} " + ("x" * 5000),
            }
        )
    messages.append({"role": "user", "content": "Current request must stay visible"})

    fitted = _fit_messages_for_context(messages, settings)

    assert _messages_token_estimate(fitted) <= _prompt_token_budget(settings, 512)
    assert fitted[0]["role"] == "system"
    assert fitted[1]["role"] == "system"
    assert fitted[-1]["content"] == "Current request must stay visible"
    assert len(fitted) < len(messages)


def test_minimal_messages_fit_after_context_error():
    settings = RuntimeSettings(context_tokens=2048, max_tokens=512, step_max_tokens=512)
    messages = [
        {"role": "system", "content": "system" * 2000},
        {"role": "system", "content": "workspace" * 2000},
        {"role": "user", "content": "old" * 4000},
        {"role": "assistant", "content": "tool" * 4000},
        {"role": "user", "content": "Current request " + ("x" * 10000)},
    ]

    fitted = _minimal_messages_for_context(messages, settings, 512)

    assert _messages_token_estimate(fitted) <= _prompt_token_budget(settings, 512) // 2
    assert fitted[-1]["role"] == "user"
    assert "Current request" in fitted[-1]["content"]


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


def test_append_file_appends_inside_workspace(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    toolbox.write_file("notes.txt", "one\n")
    toolbox.append_file("notes.txt", "two\n")

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "one\ntwo\n"


def test_run_shell_rejects_unbounded_dev_server(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    try:
        toolbox.run_shell("npm run dev")
    except ToolError as exc:
        assert "Long-running dev server command blocked" in str(exc)
    else:
        raise AssertionError("Expected unbounded dev server command to be rejected")


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
