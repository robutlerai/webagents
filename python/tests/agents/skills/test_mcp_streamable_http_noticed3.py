"""
The MCP skill's Streamable HTTP transport under mcp's current name
(2026-09-29). mcp 1.24 renamed `streamablehttp_client` to
`streamable_http_client` (an `httpx.AsyncClient` in place of headers and
timeouts) and deprecated the old name; with mcp 1.26 every connect said
`DeprecationWarning: Use streamable_http_client instead`. The skill binds
whichever name the installed mcp has (`webagents/agents/skills/local/mcp/
skill.py`, `open_streamable_http`), because the floor in pyproject.toml is
`mcp>=1.0.0` and a pin was not the handbook's preference. Pinned here:

  * the current name is bound when the installed mcp is 1.24 or later, the
    old one only when it is not;
  * a real connect over Streamable HTTP raises no deprecation warning, and
    the connect-error unwrapping (`connect_errors.py`) still turns a 401
    into the "needs a credential" row through the new transport, over `http`
    and over `auto`;
  * with only the old name present the skill falls back to it, with the
    entry's headers as the old function took them;
  * with neither, an `http` entry is refused with the sentence, and `auto`
    goes to SSE.
"""

from __future__ import annotations

import asyncio
import json
import threading
import warnings
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version

import pytest

from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.local.mcp import skill as skill_module
from webagents.agents.skills.local.mcp.connect_errors import needs_credential_sentence
from webagents.agents.skills.local.mcp.skill import LocalMcpSkill, open_streamable_http, owner_reference_sources


def _mcp_version() -> tuple:
    return tuple(int(part) for part in version("mcp").split(".")[:2])


class _Refuses(BaseHTTPRequestHandler):
    """Every request is refused with 401, as the portal's `/mcp` refuses a request with no credential."""

    seen_headers: list = []

    def _refuse(self) -> None:
        _Refuses.seen_headers.append({k.lower(): v for k, v in self.headers.items()})
        body = json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": "Unauthorized"}, "id": None}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="portal"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _refuse
    do_POST = _refuse
    do_DELETE = _refuse

    def log_message(self, *_args) -> None:  # quiet
        return


@pytest.fixture
def refusing_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for var in ("WEBAGENTS_PROFILE", "WEBAGENTS_SECRETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    _Refuses.seen_headers.clear()
    server = HTTPServer(("127.0.0.1", 0), _Refuses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    finally:
        server.shutdown()
        server.server_close()


def _report(servers: dict) -> list:
    skill = LocalMcpSkill({"mcp": servers, "references": owner_reference_sources()})
    agent = BaseAgent(name="client", instructions="x", skills={"mcp": skill})

    async def go():
        await agent._ensure_skills_initialized()
        try:
            return skill.server_report()
        finally:
            await skill.cleanup()

    return asyncio.run(go())


def test_the_name_bound_is_the_installed_mcps():
    assert skill_module.streamable_http_available()
    if _mcp_version() >= (1, 24):
        assert skill_module.streamable_http_client is not None
        assert skill_module.create_mcp_http_client is not None
    else:
        assert skill_module.streamable_http_client is None
        assert skill_module.streamablehttp_client is not None


@pytest.mark.parametrize("transport", ["http", "auto"])
def test_a_connect_raises_no_deprecation_warning_and_a_401_is_still_the_credential_row(refusing_url, transport):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = _report({"robutler": {"url": refusing_url, "transport": transport, "headers": {"X-Probe": "yes"}}})
    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not [m for m in deprecations if "streamable" in m.lower()], deprecations
    row = report[0]
    assert row["connected"] is False
    assert row["error"] == needs_credential_sentence("robutler", 401)
    assert row["needs_credential"] is True
    assert "TaskGroup" not in json.dumps(report)
    # The entry's headers reached the server through the new client.
    assert any(h.get("x-probe") == "yes" for h in _Refuses.seen_headers)


def test_the_old_name_is_used_when_the_current_one_is_absent(monkeypatch):
    calls = []

    @asynccontextmanager
    async def old_name(url, headers=None):
        calls.append((url, dict(headers or {})))
        raise RuntimeError("the old name was called")
        yield  # pragma: no cover - the generator shape

    monkeypatch.setattr(skill_module, "streamable_http_client", None)
    monkeypatch.setattr(skill_module, "create_mcp_http_client", None)
    monkeypatch.setattr(skill_module, "streamablehttp_client", old_name)
    assert skill_module.streamable_http_available()
    report = _report({"legacy": {"url": "http://127.0.0.1:9/mcp", "transport": "http", "headers": {"X-Probe": "yes"}}})
    assert calls == [("http://127.0.0.1:9/mcp", {"X-Probe": "yes"})]
    assert report[0]["connected"] is False
    assert report[0]["error"] == "the old name was called"


def test_with_neither_name_an_http_entry_is_refused_and_auto_goes_to_sse(monkeypatch):
    monkeypatch.setattr(skill_module, "streamable_http_client", None)
    monkeypatch.setattr(skill_module, "create_mcp_http_client", None)
    monkeypatch.setattr(skill_module, "streamablehttp_client", None)
    assert not skill_module.streamable_http_available()

    async def neither():
        async with open_streamable_http("http://127.0.0.1:9/mcp"):
            pass

    with pytest.raises(RuntimeError, match="no Streamable HTTP client"):
        asyncio.run(neither())

    sse_calls = []

    @asynccontextmanager
    async def sse(url, headers=None):
        sse_calls.append(url)
        raise RuntimeError("sse was tried")
        yield  # pragma: no cover - the generator shape

    monkeypatch.setattr(skill_module, "sse_client", sse)
    report = _report({
        "strict": {"url": "http://127.0.0.1:9/mcp", "transport": "http"},
        "loose": {"url": "http://127.0.0.1:9/mcp", "transport": "auto"},
    })
    by_name = {row["name"]: row for row in report}
    assert by_name["strict"]["error"] == "Server 'strict' needs Streamable HTTP, which this mcp package does not have."
    assert by_name["loose"]["error"] == "sse was tried"
    assert sse_calls == ["http://127.0.0.1:9/mcp"]
