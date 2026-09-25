"""
The REST tool (`rest` skill, `rest_request`), ADR-0045 section 6.

Every scenario in `tests/fixtures/rest_tool/scenarios.json` runs here against a
local server with the routes that file lists, and must produce exactly the
result it pins; the TypeScript suite runs the same file (tests/unit/skills/rest.test.ts),
so the two SDKs answer every one of these requests identically. The tool
definition is pinned by `definition.json` the same way.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from webagents.agents.skills.local.rest.skill import TOOL_DEFINITION, RestSkill
from webagents.crypto.http_signature import SigningKey

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "rest_tool"
SCENARIOS = json.loads((FIXTURES / "scenarios.json").read_text())["scenarios"]
DEFINITION = json.loads((FIXTURES / "definition.json").read_text())["definition"]

JSON_BODY = b'{"hello":"world","n":1}'


class _Routes(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, status, body=b"", content_type=None, headers=(), head=False):
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head and body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _route(self, head=False):
        path = self.path.split("?", 1)[0]
        if path == "/json":
            return self._send(200, JSON_BODY, "application/json", head=head)
        if path == "/text":
            return self._send(404, b"not here", "text/plain; charset=utf-8")
        if path == "/bin":
            return self._send(200, bytes(range(256)), "application/octet-stream")
        if path == "/r1":
            return self._send(302, b"", "text/plain; charset=utf-8", [("Location", "/r2")])
        if path == "/r2":
            return self._send(301, b"", "text/plain; charset=utf-8", [("Location", "/json")])
        if path == "/loop":
            return self._send(302, b"", "text/plain; charset=utf-8", [("Location", "/loop")])
        if path == "/to-private":
            return self._send(302, b"", "text/plain; charset=utf-8", [("Location", "http://10.0.0.1/")])
        if path == "/headers":
            return self._send(
                200,
                b"ok",
                "text/plain; charset=utf-8",
                [
                    ("Link", '<https://example.com/next>; rel="next"'),
                    ("ETag", '"v1"'),
                    ("X-RateLimit-Remaining", "41"),
                    ("Set-Cookie", "session=secret"),
                ],
            )
        if path == "/big":
            return self._send(200, b"a" * (2 * 1024 * 1024), "text/plain; charset=utf-8")
        if path == "/slow":
            time.sleep(3)
            return self._send(200, b"late", "text/plain; charset=utf-8")
        return self._send(404, b"no route", "text/plain; charset=utf-8")

    def do_GET(self):
        self._route()

    def do_HEAD(self):
        self._route(head=True)

    def _echo(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else ""
        if self.path.split("?", 1)[0] != "/echo":
            # A POST to a redirect route answers like the GET.
            return self._route()
        payload = {
            "method": self.command,
            "content_type": self.headers.get("Content-Type"),
            "content_length": self.headers.get("Content-Length"),
            "body": body,
            "signed": self.headers.get("Signature-Input") is not None,
        }
        self._send(200, json.dumps(payload, separators=(",", ":")).encode(), "application/json")

    do_POST = _echo
    do_PUT = _echo


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Routes)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


def _skill(config=None, identity=None) -> RestSkill:
    skill = RestSkill({"allow_private": ["127.0.0.1"]} if config is None else config)
    skill.agent = SimpleNamespace(signing_identity=identity)
    return skill


def _expand(value, port):
    if isinstance(value, dict) and set(value) == {"repeat", "count"}:
        return value["repeat"] * value["count"]
    if isinstance(value, str):
        return value.replace("PORT", str(port))
    if isinstance(value, dict):
        return {k: _expand(v, port) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, port) for v in value]
    return value


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["name"])
def test_the_shared_scenarios(scenario, server, monkeypatch):
    for name, value in (scenario.get("env") or {}).items():
        monkeypatch.setenv(name, value)
    skill = _skill(scenario.get("config"))
    args = _expand(scenario["args"], server)
    result = json.loads(asyncio.run(skill.call(**{k: args.get(k) for k in ("method", "url", "headers", "body", "timeout_seconds")})))
    result.pop("elapsed_ms", None)
    assert result == _expand(scenario["expect"], server)


def test_the_definition_is_the_shared_one():
    assert TOOL_DEFINITION == DEFINITION


def test_the_result_keeps_its_key_order(server):
    raw = asyncio.run(_skill().call(method="GET", url=f"http://127.0.0.1:{server}/json"))
    assert list(json.loads(raw)) == [
        "ok", "status", "url", "redirects", "signed", "unsigned_reason",
        "content_type", "headers", "text", "bytes", "truncated", "elapsed_ms",
    ]
    assert raw.startswith('{"ok":true,"status":200,')


class TestConfig:
    def test_a_bad_sign_mode_is_refused(self):
        with pytest.raises(ValueError, match="sign must be auto, always or never"):
            RestSkill({"sign": "sometimes"})

    def test_a_bad_allow_entry_is_refused(self):
        with pytest.raises(ValueError, match="not an IP address or CIDR range"):
            RestSkill({"allow_private": ["localhost"]})


def _identity(issuer: str):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = SigningKey.from_private_key(ed25519.Ed25519PrivateKey.generate())
    return SimpleNamespace(issuer=issuer, held_keys=lambda: [key])


class TestSigning:
    def test_signed_when_the_agent_has_a_public_address(self, server, monkeypatch):
        # Plain http to the test server is signed only where the platform's
        # local overlay would verify it, which this switch stands for.
        monkeypatch.setenv("ROBUTLER_AGENT_URL_ALLOW_PRIVATE", "1")
        skill = _skill(identity=_identity("https://agents.example.com/agents/mini"))
        raw = asyncio.run(skill.call(method="POST", url=f"http://127.0.0.1:{server}/echo", body="{}"))
        result = json.loads(raw)
        assert result["signed"] is True
        assert result["signed_as"] == "https://agents.example.com/agents/mini"
        assert json.loads(result["text"])["signed"] is True

    def test_plain_http_is_not_signed_by_default(self, server, monkeypatch):
        monkeypatch.delenv("ROBUTLER_AGENT_URL_ALLOW_PRIVATE", raising=False)
        skill = _skill(identity=_identity("https://agents.example.com/agents/mini"))
        result = json.loads(asyncio.run(skill.call(method="POST", url=f"http://127.0.0.1:{server}/echo", body="{}")))
        assert result["signed"] is False
        assert result["unsigned_reason"] == "requests to plain http addresses are not signed."
        assert json.loads(result["text"])["signed"] is False

    def test_a_loopback_identity_cannot_sign(self):
        skill = _skill(identity=_identity("http://localhost:8080/agents/mini"))
        assert skill.signing_status()[1].startswith("this agent has no public address")

    def test_never(self):
        skill = _skill({"sign": "never"}, identity=_identity("https://agents.example.com/agents/mini"))
        assert skill.signing_status() == (None, "signing is turned off for this tool (sign: never).")


def test_the_prompt_says_whether_requests_are_signed():
    unsigned = _skill().rest_prompt()
    assert "Requests go out unsigned: this agent has no public address." in unsigned
    signed = _skill(identity=_identity("https://agents.example.com/agents/mini")).rest_prompt()
    assert "Requests are signed as https://agents.example.com/agents/mini (Web Bot Auth)" in signed
    assert '"signed":true' in signed


def test_the_tool_is_the_owners():
    from webagents.agents.core.base_agent import BaseAgent

    agent = BaseAgent(name="rest-probe", instructions="Probe.", skills={"rest": RestSkill({})})
    asyncio.run(agent._ensure_skills_initialized())
    assert "rest_request" not in {t["name"] for t in agent.get_tools_for_scopes([])}
    assert "rest_request" in {t["name"] for t in agent.get_tools_for_scopes(["owner"])}
