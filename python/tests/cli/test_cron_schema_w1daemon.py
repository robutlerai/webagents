"""
The `cron:` block of an agent file (plan item 1.7, 2026-09-26): the same block
parses to the same schedules, and a wrong block gets the same sentence, in
both SDKs. The cases are `tests/fixtures/daemon/cron.json`; the TypeScript
suite runs them in `tests/unit/daemon/cron-schema-w1daemon.test.ts`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webagents.cli.loader import AgentFile, AgentFormatError, parse_frontmatter
from webagents.cli.loader.schedules import cron_expression, every_seconds, is_timezone, parse_cron_block

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "daemon" / "cron.json").read_text())


def test_the_agent_file_parses_to_the_fixture_s_schedules():
    data, _ = parse_frontmatter(FIXTURE["agent_file"])
    assert [s.to_dict() for s in parse_cron_block(data["cron"])] == FIXTURE["parsed"]


def test_the_loader_accepts_the_block_and_keeps_it_as_written(tmp_path):
    path = tmp_path / "AGENT.md"
    path.write_text(FIXTURE["agent_file"])
    agent = AgentFile(path)
    assert agent.metadata.name == "reporter"
    data, _ = parse_frontmatter(FIXTURE["agent_file"])
    assert agent.metadata.cron == data["cron"]


def test_the_loader_refuses_the_string_form_naming_the_file(tmp_path):
    path = tmp_path / "AGENT.md"
    path.write_text('---\nname: reporter\ncron: "0 9 * * *"\n---\nReport.\n')
    with pytest.raises(AgentFormatError) as refused:
        AgentFile(path)
    assert str(refused.value) == f"{path}: {FIXTURE['string_form_refused']}"


def test_an_empty_list_is_no_schedules():
    assert parse_cron_block([]) == []


@pytest.mark.parametrize("case", FIXTURE["errors"], ids=[c["message"][:60] for c in FIXTURE["errors"]])
def test_a_wrong_block_gets_the_fixture_s_sentence(case):
    with pytest.raises(AgentFormatError) as refused:
        parse_cron_block(case["cron"])
    assert str(refused.value) == case["message"]


@pytest.mark.parametrize("written, normalized", FIXTURE["expressions"]["valid"])
def test_valid_expressions(written, normalized):
    assert cron_expression(written) == normalized


@pytest.mark.parametrize("written", FIXTURE["expressions"]["invalid"])
def test_invalid_expressions(written):
    assert cron_expression(written) is None


def test_every_durations():
    for written, seconds in FIXTURE["every"]["valid"].items():
        assert every_seconds(written) == seconds, written
    for written in FIXTURE["every"]["invalid"]:
        assert every_seconds(written) is None, written


def test_timezones():
    assert is_timezone(FIXTURE["defaults"]["timezone"])
    assert is_timezone("Europe/Berlin")
    assert is_timezone("America/New_York")
    assert not is_timezone("Mars/Olympus")
    assert not is_timezone("")
    assert not is_timezone(5)


def test_the_daemon_registers_the_schedules(tmp_path, monkeypatch):
    """The daemon's registry carries the block for the runner, and a file
    whose block is wrong is not served (the scan skips a file that does not
    load)."""
    from webagents.cli.commands.daemon import daemon_server

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENT.md").write_text(FIXTURE["agent_file"])
    (root / "AGENT-broken.md").write_text('---\nname: broken\ncron: "0 9 * * *"\n---\nReport.\n')
    monkeypatch.chdir(root)
    server = daemon_server(cron=False)
    with TestClient(server.app) as client:
        assert client.get("/agents/reporter").status_code == 200
        assert client.get("/agents/broken").status_code == 404
    data, _ = parse_frontmatter(FIXTURE["agent_file"])
    assert server.registry.get("reporter").cron == data["cron"]
    assert [s.to_dict() for s in parse_cron_block(server.registry.get("reporter").cron)] == FIXTURE["parsed"]
