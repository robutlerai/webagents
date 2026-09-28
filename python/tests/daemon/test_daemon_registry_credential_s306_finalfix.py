"""
S-306 (2026-09-27): `webagents daemon` runs `server/core/app.py`, whose
registry routes `POST /agents/` (register an agent file) and `DELETE
/agents/{name}` took no credential. The credential floor covers the billable
paths only, so with `--host` off loopback anyone who could reach the port
could deregister the owner's agents or register agent files readable on the
host. The Python twin of S-284 (the TypeScript daemon).

Fixed with the split the legacy daemon already uses (S-279): on loopback the
caller is local, as the CLI's `DaemonClient` is, and nothing is asked; off
loopback both routes take the credential the floor demands, refused with the
floor's own 401 before the body is read. `WebAgentsServer(loopback=...)` is
False unless `webagents daemon` says otherwise, from its bind address.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from webagents.server.core.app import create_server

AGENT_MD = """---
name: reg-probe
description: A probe agent.
skills:
  - session
---

# Probe
"""

AUTHED = {"authorization": "Bearer test-token"}


def _agent_file(tmp_path: Path) -> Path:
    path = tmp_path / "AGENT.md"
    path.write_text(AGENT_MD)
    return path


def _server(tmp_path: Path, **kwargs):
    return create_server(url_prefix="/agents", enable_file_watching=True, watch_dirs=[tmp_path], quiet=True, **kwargs)


class TestAnExposedDaemonRequiresACredential:
    def test_register_is_401_without_a_credential_and_registers_nothing(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        res = TestClient(server.app).post("/agents/", json={"path": str(_agent_file(tmp_path))})
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "unauthorized"
        assert server.registry.get("reg-probe") is None

    def test_register_is_refused_before_the_body_is_read(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        res = TestClient(server.app).post("/agents/", content="{not json", headers={"content-type": "application/json"})
        assert res.status_code == 401

    def test_a_bare_bearer_is_not_a_credential(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        res = TestClient(server.app).post("/agents/", json={"path": str(_agent_file(tmp_path))}, headers={"authorization": "Bearer"})
        assert res.status_code == 401

    def test_register_works_with_a_credential(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        res = TestClient(server.app).post("/agents/", json={"path": str(_agent_file(tmp_path))}, headers=AUTHED)
        assert res.status_code == 200, res.text
        assert res.json()["name"] == "reg-probe"
        assert server.registry.get("reg-probe") is not None

    def test_a_bad_body_with_a_credential_is_422_not_500(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        res = TestClient(server.app).post("/agents/", json={"nope": 1}, headers=AUTHED)
        assert res.status_code == 422

    def test_unregister_is_401_without_a_credential_and_removes_nothing(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        client = TestClient(server.app)
        assert client.post("/agents/", json={"path": str(_agent_file(tmp_path))}, headers=AUTHED).status_code == 200
        res = client.delete("/agents/reg-probe")
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "unauthorized"
        assert server.registry.get("reg-probe") is not None

    def test_unregister_works_with_a_credential_and_a_missing_agent_is_404(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        client = TestClient(server.app)
        assert client.post("/agents/", json={"path": str(_agent_file(tmp_path))}, headers=AUTHED).status_code == 200
        res = client.delete("/agents/reg-probe", headers=AUTHED)
        assert res.status_code == 200 and res.json() == {"status": "unregistered", "name": "reg-probe"}
        assert server.registry.get("reg-probe") is None
        assert client.delete("/agents/reg-probe", headers=AUTHED).status_code == 404

    def test_the_listing_stays_open(self, tmp_path):
        server = _server(tmp_path, loopback=False)
        res = TestClient(server.app).get("/agents/")
        assert res.status_code == 200 and "agents" in res.json()


class TestALoopbackDaemonStaysOpenForTheLocalCli:
    def test_register_and_unregister_need_no_credential_on_loopback(self, tmp_path):
        server = _server(tmp_path, loopback=True)
        client = TestClient(server.app)
        assert client.post("/agents/", json={"path": str(_agent_file(tmp_path))}).status_code == 200
        assert server.registry.get("reg-probe") is not None
        assert client.delete("/agents/reg-probe").status_code == 200
        assert server.registry.get("reg-probe") is None


class TestTheDaemonCommandWiresItsBind:
    def test_a_server_that_does_not_say_is_guarded(self, tmp_path):
        # `create_server` and `daemon_server` default to guarded: an embedder
        # that binds the server elsewhere is not open by omission.
        from webagents.cli.commands.daemon import daemon_server

        assert _server(tmp_path).loopback is False
        assert daemon_server(watch=str(tmp_path)).loopback is False
        assert daemon_server(watch=str(tmp_path), loopback=True).loopback is True

    def test_run_daemon_passes_the_bind_address_verdict(self, tmp_path, monkeypatch):
        import uvicorn

        from webagents.cli.commands import daemon as daemon_module

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
        monkeypatch.chdir(tmp_path)
        seen = []

        class FakeApp:
            def add_event_handler(self, *_args, **_kwargs):
                pass

        class FakeServer:
            app = FakeApp()

        monkeypatch.setattr(daemon_module, "daemon_server", lambda **kwargs: (seen.append(kwargs), FakeServer())[1])
        monkeypatch.setattr("webagents.cli.listen.bind_or_refuse", lambda host, port: None)
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

        daemon_module.run_daemon(port=18765, host="0.0.0.0")
        daemon_module.run_daemon(port=18765)
        assert [(k["loopback"], k["error_detail"]) for k in seen] == [(False, False), (True, True)]
