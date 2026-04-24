from magic_claw.agent.loop import AgentLoop
from magic_claw.config import RuntimeSettings
from magic_claw.state import StateStore


def test_agent_tool_statuses_are_terminal_friendly(tmp_path):
    loop = AgentLoop(RuntimeSettings(), str(tmp_path / "workspace"), StateStore(tmp_path / "state.sqlite"))

    assert loop._describe_tool("read_file", {"path": "notes.txt"}) == "Reading file: notes.txt"
    assert loop._describe_tool("write_file", {"path": "out.txt"}) == "Writing file: out.txt"
    assert loop._describe_tool("run_shell", {"command": "python --version"}) == "Running command: python --version"
