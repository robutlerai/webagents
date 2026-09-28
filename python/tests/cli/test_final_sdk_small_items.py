"""
The smaller items of the CLI e2e pass (2026-09-28), in the Python CLI:

* B7: an agent file's `description:` reaches the A2A card and the listing (it
  was `""` in Python and the file's sentence in TypeScript);
* B8: an MCP stdio server's stderr goes to the profile's `logs/` folder, never
  to the terminal the chat draws on;
* B10: `doctor --json` and `sandbox setup --json`, the global option typed
  after the subcommand, answer the JSON document (fixture
  `cli/sandbox_default_global_options.json` holds the argv cases).
"""

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from webagents.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for name in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "OPENAI_API_KEY", "WEBAGENTS_AGENT_TOKEN"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")


def test_the_a2a_card_and_the_listing_carry_the_files_description():
    from webagents.cli.agent_builder import build_agent

    Path("AGENT.md").write_text("---\nname: helper-py\ndescription: The A2A peer that answers delegated requests\nskills:\n  - a2a\n---\nHelp.\n")
    built = asyncio.run(build_agent(Path("AGENT.md").resolve(), working_dir=Path.cwd(), initialize=False))
    assert built.agent.description == "The A2A peer that answers delegated requests"
    card = asyncio.run(built.agent.skills["a2a"].build_card(built.agent))
    assert card["description"] == "The A2A peer that answers delegated requests"


def test_an_mcp_servers_stderr_goes_to_the_profiles_log_folder(tmp_path):
    from webagents.agents.skills.local.mcp.skill import mcp_stderr_log

    log = mcp_stderr_log("echo/server one")
    try:
        log.write("a banner the chat never shows\n")
    finally:
        log.close()
    written = tmp_path / "home" / ".webagents" / "logs" / "mcp-echo_server_one.log"
    assert written.read_text() == "a banner the chat never shows\n"


def test_an_mcp_stdio_server_writing_stderr_does_not_reach_the_terminal(tmp_path, capfd):
    """A real stdio server that writes to stderr at start: nothing on this
    process's stderr, the lines in the log."""
    pytest.importorskip("mcp")
    from webagents.agents.skills.local.mcp.skill import LocalMcpSkill

    server = tmp_path / "noisy_server.py"
    server.write_text(
        "import sys\n"
        "print('noisy server starting', file=sys.stderr, flush=True)\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "app = FastMCP('noisy')\n"
        "@app.tool()\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
        "app.run()\n"
    )
    import sys

    skill = LocalMcpSkill({"mcp": {"noisy": {"command": sys.executable, "args": [str(server)]}}, "agent_name": "noisy"})

    class Agent:
        name = "noisy"

        def register_tool(self, *args, **kwargs):
            pass

    async def run():
        try:
            await skill.initialize(Agent())
        finally:
            await skill.cleanup()

    asyncio.run(run())
    captured = capfd.readouterr()
    assert "noisy server starting" not in captured.err
    log = tmp_path / "home" / ".webagents" / "logs" / "mcp-noisy.log"
    assert "noisy server starting" in log.read_text()


@pytest.mark.parametrize("argv", [["doctor", "--json"], ["--json", "doctor"]])
def test_doctor_json_after_the_subcommand(argv):
    Path("AGENT.md").write_text("---\nname: helper\n---\nHelp.\n")
    result = runner.invoke(app, argv)
    document = json.loads(result.stdout)
    assert "checks" in document["data"]


def test_sandbox_setup_json_after_the_subcommand():
    result = runner.invoke(app, ["sandbox", "setup", "--json"])
    assert "unknown option" not in result.output
    document = json.loads(result.stdout)
    assert [check["name"] for check in document["data"]["checks"]][:1] == ["platform"]
