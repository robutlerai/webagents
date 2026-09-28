"""
A port already in use is one sentence and exit 1, said BEFORE the address
line (2026-09-26, the new-developer e2e run): `webagents serve --port <busy>`
printed `[webagents] my-agent on http://127.0.0.1:<port>` and then uvicorn's
`[Errno 48] error while attempting to bind`. Pinned by
`tests/fixtures/cli/listen.json`, which the TypeScript suite runs too
(`tests/unit/server/port-in-use-e2efix.test.ts`): the probe, and the three
commands against a port this test holds, with uvicorn forbidden to run.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

# Imported HERE, outside any CliRunner invocation: `mcp.client.stdio.stdio_client`
# binds `sys.stderr` as a default argument at import time, and a first import
# inside the runner (whose stderr has no fileno) poisons every later
# `stdio_client` in the session (`test_mcp_serve.py`'s stdio case).
import mcp.client.stdio  # noqa: F401

from webagents.cli.listen import PORT_IN_USE, bind_or_refuse, port_in_use_sentence, port_is_free
from webagents.cli.main import app

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "listen.json").read_text())
runner = CliRunner()


@pytest.fixture
def busy():
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        yield holder.getsockname()[1]
    finally:
        holder.close()


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("ROBUTLER_API_URL", "http://127.0.0.1:9")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN", "WEBAGENTS_DEBUG", "WEBAGENTS_PUBLIC_URL"):
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENT.md").write_text("---\nname: my-agent\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n")
    monkeypatch.chdir(project)
    import uvicorn

    def never(*args, **kwargs):  # pragma: no cover - the point is that it is not reached
        raise AssertionError("uvicorn.run was called for a busy port")

    monkeypatch.setattr(uvicorn, "run", never)
    return project


def test_the_sentence_is_the_fixtures():
    assert PORT_IN_USE == FIXTURE["port_in_use"]
    assert port_in_use_sentence("127.0.0.1", 4242) == FIXTURE["port_in_use"].format(port=4242, host="127.0.0.1")


def test_the_probe_sees_a_busy_port_and_a_free_one(busy, capsys):
    assert port_is_free("127.0.0.1", busy) is False
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        free_port = free.getsockname()[1]
    assert port_is_free("127.0.0.1", free_port) is True
    with pytest.raises(SystemExit) as raised:
        bind_or_refuse("127.0.0.1", busy)
    assert raised.value.code == FIXTURE["exit"]
    assert capsys.readouterr().err.strip() == port_in_use_sentence("127.0.0.1", busy)


@pytest.mark.parametrize(
    "argv,address_line",
    [
        (["serve", "--port", "{port}"], "my-agent on http://127.0.0.1:{port}"),
        (["daemon", "--port", "{port}", "--no-cron"], "WebAgents daemon starting on"),
        (["mcp", "serve", "--http", "{port}"], "MCP on http://127.0.0.1:{port}"),
    ],
    ids=["serve", "daemon", "mcp serve --http"],
)
def test_each_command_says_the_sentence_and_never_the_address(busy, argv, address_line):
    result = runner.invoke(app, [a.replace("{port}", str(busy)) for a in argv])
    assert result.exit_code == FIXTURE["exit"], result.output
    assert port_in_use_sentence("127.0.0.1", busy) in result.output
    assert address_line.replace("{port}", str(busy)) not in result.output
    for never in FIXTURE["never_printed"]:
        assert never not in result.output
