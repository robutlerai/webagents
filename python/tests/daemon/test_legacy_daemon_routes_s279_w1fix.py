"""
S-279 (2026-09-26): the legacy Python daemon class (`cli/daemon/server.py`
`WebAgentsDaemon`) has `POST /scan` and `POST /agents/`, which register agent
files a caller names, and neither checked who was calling. On loopback (the
default) no credential floor is installed; off loopback the floor covers only
the billable paths, not these, so anyone who could reach an exposed daemon
could register any agent file on the host's disk.

Fixed by the same loopback split the daemon already uses for its billable
floor: on loopback the caller is local (the CLI works unchanged), off loopback
both routes take the credential the exposed daemon demands. The watcher still
scans the watched directories, so nothing legitimate needs an anonymous scan.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from webagents.cli.daemon.server import WebAgentsDaemon

AGENT_MD = """---
name: reg-probe
description: A probe agent.
skills:
  - session
---

# Probe
"""

AUTHED = {"authorization": "Bearer test-token"}


def _agent_file(tmp_path):
    path = tmp_path / "AGENT.md"
    path.write_text(AGENT_MD)
    return path


class TestExposedDaemonRequiresACredential:
    def _client(self):
        # Bound off loopback: the exposed case the hole lived in.
        return TestClient(WebAgentsDaemon(port=0, host="0.0.0.0").app)

    def test_scan_is_401_without_a_credential(self):
        res = self._client().post("/scan?path=.")
        assert res.status_code == 401

    def test_scan_works_with_a_credential(self, tmp_path):
        _agent_file(tmp_path)
        res = self._client().post(f"/scan?path={tmp_path}", headers=AUTHED)
        assert res.status_code == 200
        assert res.json()["scanned"] >= 1

    def test_register_is_401_without_a_credential(self, tmp_path):
        res = self._client().post("/agents/", json={"path": str(_agent_file(tmp_path))})
        assert res.status_code == 401

    def test_register_works_with_a_credential(self, tmp_path):
        res = self._client().post("/agents/", json={"path": str(_agent_file(tmp_path))}, headers=AUTHED)
        assert res.status_code == 200

    def test_a_bare_bearer_is_not_a_credential(self, tmp_path):
        res = self._client().post("/agents/", json={"path": str(_agent_file(tmp_path))}, headers={"authorization": "Bearer"})
        assert res.status_code == 401


class TestLoopbackDaemonStaysOpenForTheLocalCli:
    def _client(self):
        # The default bind: local trust, as for the CLI and the harmonization tests.
        return TestClient(WebAgentsDaemon(port=0).app)

    def test_scan_needs_no_credential_on_loopback(self, tmp_path):
        _agent_file(tmp_path)
        res = self._client().post(f"/scan?path={tmp_path}")
        assert res.status_code == 200

    def test_register_needs_no_credential_on_loopback(self, tmp_path):
        res = self._client().post("/agents/", json={"path": str(_agent_file(tmp_path))})
        assert res.status_code == 200
