"""
Which agents `webagents daemon` serves, the same in both SDKs (2026-09-25).

The tree is `tests/fixtures/daemon/discovery.json`, which the TypeScript suite
builds too (`typescript/tests/unit/daemon/discovery.test.ts`): `AGENT.md` and
`AGENT-<name>.md` by exact name, anywhere under the folder, never inside a
tool's own directory. The daemon started without `-w` serves the working
directory's agents, as the TypeScript daemon now does; this one always did, but
its scan globbed, and on a case-insensitive disk a glob answers `agent.md` as
`AGENT.md` (`registry.discover_agent_files`).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webagents.cli.commands.daemon import daemon_server
from webagents.cli.daemon.registry import discover_agent_files

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "daemon" / "discovery.json").read_text())


def build(root: Path) -> None:
    for agent in FIXTURE["agents"]:
        path = root / agent["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {agent['name']}\n---\nHelp.\n")
    for name in FIXTURE["files"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Notes.\n")


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    # A placeholder, never used: without a model this daemon refuses to build
    # an agent at all (500 "No model for this agent"), and discovery is what
    # is under test here.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
    root = tmp_path / "project"
    root.mkdir()
    build(root)
    monkeypatch.chdir(root)
    return root


def test_the_files_the_daemon_reads(folder):
    found = sorted(str(p.relative_to(folder)) for p in discover_agent_files(folder))
    assert found == FIXTURE["discovered"]


def test_without_w_the_daemon_serves_this_folders_agents(folder):
    server = daemon_server(cron=False)
    with TestClient(server.app) as client:
        for name in FIXTURE["served"]:
            response = client.get(f"/agents/{name}")
            assert response.status_code == 200, name
            assert response.json()["name"] == name
        for name in FIXTURE["not_served"]:
            assert client.get(f"/agents/{name}").status_code == 404, name


def test_w_names_another_folder(folder, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "AGENT.md").write_text("---\nname: over-there\n---\nHelp.\n")
    server = daemon_server(watch=str(elsewhere), cron=False)
    with TestClient(server.app) as client:
        assert client.get("/agents/over-there").status_code == 200
        assert client.get("/agents/main-agent").status_code == 404
