"""
`webagents daemon` lists the served agents' schedules at `/agents/cron` and
at `/cron` (2026-09-27, the final e2e re-run), the two paths the TypeScript
daemon serves, pinned by `tests/fixtures/daemon/cron.json` `routes` and read
the same way by `tests/unit/daemon/cron-routes-parity-finalfix.test.ts`. A
server with no prefix has the one `/cron` route and no duplicate.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from webagents.server.core.app import create_server

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "daemon" / "cron.json").read_text())
ROUTES = FIXTURE["routes"]


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENT.md").write_text(FIXTURE["agent_file"])
    return root


def test_the_daemon_lists_schedules_at_both_paths(tmp_path, monkeypatch):
    from webagents.cli.commands.daemon import daemon_server

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
    monkeypatch.chdir(_project(tmp_path))
    server = daemon_server(cron=True)
    with TestClient(server.app) as client:
        bodies = []
        for route in ROUTES["list"]:
            res = client.get(route)
            assert res.status_code == 200, route
            bodies.append(res.json())
        expected = [(s["name"], s["kind"]) for s in FIXTURE["parsed"]]
        for body in bodies:
            assert [(s["name"], s["kind"]) for s in body[ROUTES["body_key"]]] == expected
        # Still read-only (S-273) at the alias too.
        for route in ROUTES["list"]:
            assert client.post(route, json={"id": "evil", "cron": "* * * * *", "agentName": "reporter", "task": "x"}).status_code in (404, 405)


def test_a_server_with_no_prefix_has_one_cron_route():
    server = create_server(url_prefix="", enable_cron=False, quiet=True)
    paths = [route.path for route in server.app.routes if getattr(route, "path", "") == "/cron"]
    assert paths == ["/cron"]
