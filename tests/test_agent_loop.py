import sys

from magic_claw.agent.loop import (
    AgentLoop,
    IncompleteFileActionError,
    LoopSafetyGuard,
    NonConformingToolOutputError,
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
    assert loop._describe_tool("commit_file", {"path": "out.txt"}) == "Committing file: out.txt"
    assert loop._describe_tool("run_shell", {"command": "python --version"}) == "Running command: python --version"


def test_agent_rejects_multiple_json_objects(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    try:
        loop._parse_action(
            '{"thought":"one","tool":"list_dir","args":{}}\n'
            '{"thought":"two","tool":"run_shell","args":{"command":"dir"}}'
        )
    except NonConformingToolOutputError as exc:
        assert "Trailing content" in str(exc)
    else:
        raise AssertionError("Expected multiple JSON objects to be rejected")


def test_agent_rejects_gemma_tool_call_wrapper(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    try:
        loop._parse_action(
            '<|tool_call>call:{"args":{"path":"src/App.css"},"thought":"read","tool":"read_file"}<tool_call|>'
        )
    except NonConformingToolOutputError as exc:
        assert "wrapper" in str(exc)
    else:
        raise AssertionError("Expected Gemma tool-call wrapper to be rejected")


def test_agent_rejects_loose_gemma_tool_call_wrapper(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    try:
        loop._parse_action('<|tool_call>call:MagicClaw:run_shell{command: "dir /s tests"}<tool_call|>')
    except NonConformingToolOutputError:
        pass
    else:
        raise AssertionError("Expected loose Gemma tool-call wrapper to be rejected")


def test_agent_rejects_truncated_write_file_with_explicit_path(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    try:
        loop._parse_action(
            '{"thought":"write app","tool":"write_file","args":{"path":"src/App.jsx",'
            "\"content\":\"import React from \\'react\\';\\n<div className=\\\"weather\\\">"
        )
    except IncompleteFileActionError as exc:
        assert "Content incomplete" in str(exc)
    else:
        raise AssertionError("Expected truncated write_file to be rejected")


def test_agent_rejects_truncated_write_file_without_rescuing_path(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    try:
        loop._parse_action(
            '{"thought":"Rewrite `src/App.jsx` safely","tool":"write_file","args":{'
            "\"content\":\"import React from \\'react\\';\\nexport default function App() {"
        )
    except IncompleteFileActionError:
        pass
    else:
        raise AssertionError("Expected truncated write_file to be rejected")


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
            '{"thought":"commit","tool":"commit_file","args":{"path":"index.html"}}',
            '{"thought":"done","tool":"final","args":{"answer":"Created index.html."}}',
        ]
    )

    result = loop.run("Développer un site vitrine", max_steps=4)

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


def test_agent_allows_direct_final_for_interactive_current_question(tmp_path):
    prompt = """Interactive context from earlier terminal turns. Use it only when the current request is a continuation or asks about earlier work.

Previous user request:
Creer un projet meteo complet avec Vite React
Previous result:
Projet cree.

Current user request:
Ressenti"""
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))
    loop.client = StubModelClient(
        ['{"thought":"answer","tool":"final","args":{"answer":"Le ressenti est affiche."}}']
    )

    result = loop.run(prompt, max_steps=1)

    assert result.status == "done"
    assert result.answer == "Le ressenti est affiche."


def test_agent_uses_current_interactive_request_for_actionable_detection(tmp_path):
    prompt = """Interactive context from earlier terminal turns. Use it only when the current request is a continuation or asks about earlier work.

Previous user request:
Status du projet
Previous result:
Le projet est present.

Current user request:
Modifier le header"""
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    assert loop._requires_workspace_action(prompt) is True


def test_agent_fails_repeated_final_rejection_loop(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))
    loop.client = StubModelClient(
        ['{"thought":"done","tool":"final","args":{"answer":"Created."}}'] * 5
    )

    result = loop.run("Create a landing page", max_steps=10)

    assert result.status == "failed"
    assert "Repeated completion rejection loop detected" in result.answer


def test_agent_compacts_and_continues_after_step_window(tmp_path):
    workspace = tmp_path / "workspace"
    state = StateStore(tmp_path / "state.sqlite")
    loop = AgentLoop(RuntimeSettings(step_compaction_max_cycles=2), str(workspace), state)
    loop.client = StubModelClient(
        [
            '{"thought":"write","tool":"write_file","args":{"path":"index.html","content":"ok"}}',
            '{"thought":"commit","tool":"commit_file","args":{"path":"index.html"}}',
            '{"thought":"done","tool":"final","args":{"answer":"Created index.html."}}',
        ]
    )

    result = loop.run("DÃ©velopper un site vitrine", max_steps=2)

    assert result.status == "done"
    assert result.answer == "Created index.html."
    assert (workspace / "index.html").read_text(encoding="utf-8") == "ok"
    memories = state.list_task_memories(result.task_id)
    assert len(memories) == 1
    assert "Autonomous continuation memory" in memories[0]["summary"]


def test_run_shell_timeout_is_observable_not_fatal(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    result = toolbox.run_shell(
        f'"{sys.executable}" -c "import time; time.sleep(2)"',
        timeout_seconds=1,
    )

    assert result["timed_out"] is True
    assert result["returncode"] is None


def test_run_shell_inactivity_timeout_kills_stalled_command(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    result = toolbox.run_shell(
        f'"{sys.executable}" -c "import time; print(\'start\', flush=True); time.sleep(5)"',
        timeout_seconds=10,
        inactivity_timeout_seconds=1,
    )

    assert result["timed_out"] is True
    assert result["timeout_kind"] == "inactivity"
    assert result["returncode"] is None
    assert "start" in result["stdout"]
    assert result["duration_seconds"] < 4
    assert result["diagnosis"]


def test_npm_commands_get_inactivity_guard(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    policy = toolbox._shell_timeout_policy("npm install", timeout_seconds=300)

    assert policy.timeout_seconds == 300
    assert policy.inactivity_timeout_seconds == 180


def test_append_file_appends_inside_workspace(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    toolbox.write_file("notes.txt", "one\n")
    toolbox.append_file("notes.txt", "two\n")
    toolbox.commit_file("notes.txt")

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "one\ntwo\n"


def test_source_file_writes_are_transactional_until_commit(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    toolbox.write_file("src/App.jsx", "export default function App() {\n")

    assert not (tmp_path / "src" / "App.jsx").exists()
    assert (tmp_path / "src" / "App.jsx.tmp").exists()

    toolbox.append_file("src/App.jsx", "  return null;\n}\n")
    result = toolbox.commit_file("src/App.jsx")

    assert result["transaction"]["committed"] is True
    assert (tmp_path / "src" / "App.jsx").read_text(encoding="utf-8") == (
        "export default function App() {\n  return null;\n}\n"
    )
    assert not (tmp_path / "src" / "App.jsx.tmp").exists()


def test_commit_file_rejects_incomplete_source_without_overwriting(tmp_path):
    toolbox = AgentToolbox(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.jsx").write_text("export default function App() { return null }\n", encoding="utf-8")

    toolbox.write_file("src/App.jsx", "export default function App() {\n")
    try:
        toolbox.commit_file("src/App.jsx")
    except ToolError as exc:
        assert "Content incomplete" in str(exc)
    else:
        raise AssertionError("Expected incomplete source draft to be rejected")

    assert (tmp_path / "src" / "App.jsx").read_text(encoding="utf-8") == (
        "export default function App() { return null }\n"
    )


def test_commit_file_rejects_multiple_default_exports_without_overwriting(tmp_path):
    toolbox = AgentToolbox(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.jsx").write_text("export default function App() { return null }\n", encoding="utf-8")

    toolbox.write_file("src/App.jsx", "export default 1;\nexport default 2;\n")
    try:
        toolbox.commit_file("src/App.jsx")
    except ToolError as exc:
        assert "Multiple default exports" in str(exc)
    else:
        raise AssertionError("Expected duplicate default exports to be rejected")

    assert (tmp_path / "src" / "App.jsx").read_text(encoding="utf-8") == (
        "export default function App() { return null }\n"
    )


def test_commit_file_rolls_back_when_project_validation_fails(tmp_path):
    toolbox = AgentToolbox(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "package.json").write_text(
        '{"scripts":{"build":"node scripts/check.js"}}',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "check.js").write_text(
        "const fs = require('fs');\n"
        "const text = fs.readFileSync('src/App.jsx', 'utf8');\n"
        "if (text.includes('FAIL_BUILD')) {\n"
        "  console.error('forced build failure');\n"
        "  process.exit(1);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "App.jsx").write_text("export default function App() { return null }\n", encoding="utf-8")

    toolbox.write_file("src/App.jsx", "export default function App() { return 'FAIL_BUILD'; }\n")
    try:
        toolbox.commit_file("src/App.jsx")
    except ToolError as exc:
        assert "Project validation failed" in str(exc)
    else:
        raise AssertionError("Expected failed project validation to roll back")

    assert (tmp_path / "src" / "App.jsx").read_text(encoding="utf-8") == (
        "export default function App() { return null }\n"
    )
    assert (tmp_path / "src" / "App.jsx.tmp").exists()


def test_commit_file_uses_lint_when_build_script_is_missing(tmp_path):
    toolbox = AgentToolbox(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "package.json").write_text(
        '{"scripts":{"lint":"node scripts/check.js"}}',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "check.js").write_text(
        "const fs = require('fs');\n"
        "fs.writeFileSync('lint-ran.txt', 'yes');\n",
        encoding="utf-8",
    )

    toolbox.write_file("src/App.jsx", "export default function App() { return null; }\n")
    result = toolbox.commit_file("src/App.jsx")

    assert result["project_validation"]["command"] == "npm run lint"
    assert (tmp_path / "lint-ran.txt").read_text(encoding="utf-8") == "yes"


def test_commit_file_uses_python_compile_for_python_sources(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    toolbox.write_file("src/app.py", "def ok():\n    return 1\n")
    result = toolbox.commit_file("src/app.py")

    assert "py_compile" in result["project_validation"]["command"]
    assert result["project_validation"]["returncode"] == 0


def test_agent_auto_validation_failure_forces_repair_before_final(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "scripts").mkdir()
    (workspace / "package.json").write_text(
        '{"scripts":{"build":"node scripts/check.js"}}',
        encoding="utf-8",
    )
    (workspace / "scripts" / "check.js").write_text(
        "const fs = require('fs');\n"
        "const countPath = 'validation-count.txt';\n"
        "const count = fs.existsSync(countPath) ? Number(fs.readFileSync(countPath, 'utf8')) : 0;\n"
        "fs.writeFileSync(countPath, String(count + 1));\n"
        "const text = fs.readFileSync('src/App.jsx', 'utf8');\n"
        "if (text.includes('FAIL_BUILD')) {\n"
        "  console.error('forced build failure');\n"
        "  process.exit(1);\n"
        "}\n",
        encoding="utf-8",
    )
    statuses: list[str] = []
    loop = AgentLoop(
        RuntimeSettings(),
        str(workspace),
        StateStore(tmp_path / "state.sqlite"),
        on_status=statuses.append,
    )
    loop.client = StubModelClient(
        [
            '{"thought":"write broken","tool":"write_file","args":{"path":"src/App.jsx","content":"export default function App() { return \'FAIL_BUILD\'; }\\n"}}',
            '{"thought":"commit broken","tool":"commit_file","args":{"path":"src/App.jsx"}}',
            '{"thought":"try unrelated command","tool":"run_shell","args":{"command":"echo unrelated"}}',
            '{"thought":"repair","tool":"write_file","args":{"path":"src/App.jsx","content":"export default function App() { return null; }\\n"}}',
            '{"thought":"commit repair","tool":"commit_file","args":{"path":"src/App.jsx"}}',
            '{"thought":"done","tool":"final","args":{"answer":"Fixed and validated."}}',
        ]
    )

    result = loop.run("Create a React app", max_steps=6)

    assert result.status == "done"
    assert result.answer == "Fixed and validated."
    assert (workspace / "src" / "App.jsx").read_text(encoding="utf-8") == (
        "export default function App() { return null; }\n"
    )
    assert (workspace / "validation-count.txt").read_text(encoding="utf-8") == "2"
    assert any("Auto-validating: npm run build" in status for status in statuses)
    assert any("Auto-validation failed" in status for status in statuses)


def test_write_file_rejects_oversized_content(tmp_path):
    toolbox = AgentToolbox(tmp_path)

    try:
        toolbox.write_file("src/App.jsx", "x" * 1201)
    except ToolError as exc:
        assert "Content too large" in str(exc)
    else:
        raise AssertionError("Expected oversized file content to be rejected")


def test_execute_auto_chunks_complete_oversized_file_content(tmp_path):
    toolbox = AgentToolbox(tmp_path)
    content = "export default function App() {\n  return null;\n}\n" + ("// filler\n" * 200)

    result = toolbox.execute("write_file", {"path": "src/App.jsx", "content": content})

    assert result["auto_chunked"] is True
    assert result["transaction"]["committed"] is True
    assert result["chunks"] > 1
    assert (tmp_path / "src" / "App.jsx").read_text(encoding="utf-8") == content
    assert not (tmp_path / "src" / "App.jsx.tmp").exists()


def test_execute_auto_chunk_rejects_incomplete_oversized_source_without_overwriting(tmp_path):
    toolbox = AgentToolbox(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.jsx").write_text("export default function App() { return null }\n", encoding="utf-8")
    content = "export default function App() {\n" + ("// filler\n" * 200)

    result = toolbox.execute("write_file", {"path": "src/App.jsx", "content": content})

    assert "error" in result
    assert "Content incomplete" in result["error"]
    assert (tmp_path / "src" / "App.jsx").read_text(encoding="utf-8") == (
        "export default function App() { return null }\n"
    )


def test_loop_safety_guard_stops_repeated_similar_writes():
    guard = LoopSafetyGuard(max_repeated_similar_writes=5)
    observation = {"path": "src/App.jsx", "bytes": 2600}

    for _index in range(4):
        guard.record("write_file", {"path": "src/App.jsx"}, observation)

    try:
        guard.record("write_file", {"path": "src/App.jsx"}, observation)
    except RuntimeError as exc:
        assert "Repeated write loop detected" in str(exc)
    else:
        raise AssertionError("Expected repeated similar writes to be rejected")


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
