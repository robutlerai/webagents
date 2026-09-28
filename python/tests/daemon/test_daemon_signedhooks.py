"""
A daemon-served agent signs its `cron:` webhooks (2026-09-27, the
a2a-delegate-webhooks lane): `webagents daemon` (`create_server` with file
watching) gives each agent it loads an Ed25519 identity kept in the store the
server uses (`~/.webagents/keys`, 0600; S-309 moved it there out of the agent
folder), signs as `{public URL}/agents/{name}`,
serves that key set at `/agents/{name}/.well-known/jwks.json`, and a webhook
run by the runner verifies with `verify_web_bot_auth` against the set the
daemon serves. A restart holds the same key; `webagents cron run` signs with
it too; a daemon with no public URL posts unsigned and says why; a key file
that cannot be used is said and never replaced. The paths and words are the
shared fixture's (`tests/fixtures/daemon/signedhooks.json`); the TypeScript
daemon runs the same in `tests/unit/daemon/daemon-signedhooks.test.ts`.
"""

from __future__ import annotations

import asyncio
import json
import re
import stat
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from webagents.cli.cron_command import attach_daemon_identity
from webagents.cli.daemon.deliver import DeliveryContext, deliver_webhook
from webagents.cli.daemon.identity import daemon_public_url
from webagents.cli.loader import AgentFile
from webagents.cli.loader.schedules import DeliverTarget
from webagents.cli.daemon.deliver import RunResult
from webagents.crypto.web_bot_auth_verify import DiscoveredKey, InboundRequest, KeySetOutcome, MemoryNonceStore, verify_web_bot_auth

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "daemon" / "signedhooks.json").read_text())
PUBLIC_URL = "https://agent.example"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
    for name in ("WEBAGENTS_PUBLIC_URL", "WEBAGENTS_KEYS_DIR", "WEBAGENTS_PROFILE", "WEBAGENTS_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@dataclass
class Received:
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes


@dataclass
class Hook:
    url: str
    authority: str
    received: List[Received] = field(default_factory=list)
    server: Any = None


@pytest.fixture
def webhook_server():
    """A local webhook that keeps what it got and answers 200."""
    servers: List[ThreadingHTTPServer] = []

    def start() -> Hook:
        hook = Hook(url="", authority="")

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server's name
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                hook.received.append(Received(self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):  # quiet
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        hook.authority = f"127.0.0.1:{server.server_address[1]}"
        hook.url = f"http://{hook.authority}/hook"
        hook.server = server
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return hook

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def project(tmp_path: Path, url: str) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENT.md").write_text(FIXTURE["agent_file"].replace("{url}", url))
    return root


def key_file_of(root: Path, agent: str = "reporter") -> Path:
    """The key, in the store under the test's HOME (the fixture's `store.default`), never in `root` (S-309)."""
    return Path.home() / FIXTURE["store"]["default"] / FIXTURE["key_file"].format(agent=agent)


def daemon_on(root: Path, public_url=None):
    from webagents.cli.commands.daemon import daemon_server

    return daemon_server(watch=str(root), cron=True, public_url=public_url)


async def canned(messages, **kwargs):
    return {"choices": [{"message": {"role": "assistant", "content": "Nothing happened."}}]}


def verify(got: Received, authority: str, keys: List[Dict[str, str]]):
    class KeySets:
        async def get(self, discovery):
            return KeySetOutcome(keys=tuple(DiscoveredKey(thumbprint=k["kid"], x=k["x"]) for k in keys), ttl_s=300)

    return asyncio.run(
        verify_web_bot_auth(
            InboundRequest(method=got.method, target="/hook", headers=got.headers, body=got.body),
            authorities=[authority],
            scheme="http",
            key_sets=KeySets(),
            nonces=MemoryNonceStore(),
        )
    )


def jwks_route(agent: str = "reporter") -> str:
    return FIXTURE["jwks_route"].format(agent=agent)


# -- the public URL the daemon signs under ------------------------------------------------------


@pytest.mark.parametrize("case", FIXTURE["public_url_cases"], ids=[c["name"] for c in FIXTURE["public_url_cases"]])
def test_public_url_cases(case, monkeypatch):
    monkeypatch.setenv(FIXTURE["public_url_env"], case["env"])
    assert daemon_public_url(case["host"], case["port"], case.get("public_url")) == case["expect"]


# -- a daemon-served agent signs its webhooks ----------------------------------------------------


def test_keeps_a_0600_key_in_the_store_serves_its_key_set_and_the_webhook_verifies(tmp_path, monkeypatch, webhook_server):
    hook = webhook_server()
    root = project(tmp_path, hook.url)
    monkeypatch.chdir(root)
    issuer = FIXTURE["issuer"].format(public_url=PUBLIC_URL, agent="reporter")
    server = daemon_on(root, PUBLIC_URL)
    agent = asyncio.run(server.resolve_agent("reporter"))
    assert agent is not None
    assert agent.signing_identity.issuer == issuer
    kid = agent.signing_identity.held_keys()[0].thumbprint

    # The key, where the fixture says, with the modes the store guarantees; and nothing in the folder (S-309).
    key_file = key_file_of(root)
    assert key_file.exists()
    assert oct(stat.S_IMODE(key_file.stat().st_mode)) == "0o" + FIXTURE["key_mode"]
    assert oct(stat.S_IMODE(key_file.parent.stat().st_mode)) == "0o" + FIXTURE["dir_mode"]
    assert not (root / ".webagents" / "keys").exists()

    # The key set, where a static agent's is served (no lifespan: the cron loop stays off).
    client = TestClient(server.app)
    res = client.get(jwks_route())
    assert res.status_code == 200
    assert res.headers["cache-control"] == FIXTURE["jwks_cache_control"]
    jwks = res.json()
    assert [k["kid"] for k in jwks["keys"]] == [kid]
    assert client.get(jwks_route("nobody")).status_code == FIXTURE["no_identity_status"]

    # The runner's own delivery, on a canned turn: signed as the daemon's identity.
    server.cron.sync_from_registry(server.registry)
    agent.run = canned
    record = asyncio.run(server.cron.run_now("reporter", "hook"))
    assert record["outcome"] == "delivered"
    assert record["detail"] == FIXTURE["signed_detail"].format(url=hook.url, issuer=issuer)
    assert len(hook.received) == 1
    outcome = verify(hook.received[0], hook.authority, jwks["keys"])
    assert outcome.ok, outcome.refusal
    assert outcome.agent.principal == issuer
    # Signature-Agent names the daemon's own key-set route under the issuer.
    assert outcome.agent.identifier == issuer + jwks_route()[len("/agents/reporter") :]
    assert outcome.agent.thumbprints == (kid,)

    # A restart holds the same key: the file is read, never regenerated.
    again = daemon_on(root, PUBLIC_URL)
    reloaded = asyncio.run(again.resolve_agent("reporter"))
    assert reloaded.signing_identity.held_keys()[0].thumbprint == kid
    assert TestClient(again.app).get(jwks_route()).json()["keys"][0]["kid"] == kid


def test_cron_run_signs_with_the_daemon_s_key(tmp_path, monkeypatch, webhook_server):
    hook = webhook_server()
    root = project(tmp_path, hook.url)
    monkeypatch.chdir(root)
    issuer = FIXTURE["issuer"].format(public_url=PUBLIC_URL, agent="reporter")
    kid = asyncio.run(daemon_on(root, PUBLIC_URL).resolve_agent("reporter")).signing_identity.held_keys()[0].thumbprint

    # `webagents cron run`, with the public URL the daemon would read from the environment.
    monkeypatch.setenv(FIXTURE["public_url_env"], PUBLIC_URL)

    class FakeAgent:
        name = "reporter"

    said: List[str] = []
    agent = attach_daemon_identity(FakeAgent(), AgentFile(root / "AGENT.md"), error=said.append)
    assert said == []
    assert agent.signing_identity.issuer == issuer
    assert agent.signing_identity.held_keys()[0].thumbprint == kid
    result = RunResult(agent="reporter", schedule="hook", kind="cron", prompt="Report.", content="Report.", ran_at="2026-09-27T10:00:00Z")
    outcome = asyncio.run(deliver_webhook(DeliverTarget(kind="webhook", url=hook.url, timeout=5, retries=0), result, DeliveryContext(agent_dir=root, agent=agent)))
    assert outcome == ("delivered", FIXTURE["signed_detail"].format(url=hook.url, issuer=issuer))
    assert re.search(r'keyid="([^"]+)"', hook.received[0].headers["signature-input"]).group(1) == kid


def test_with_no_public_url_the_webhook_goes_out_unsigned_and_the_record_says_why(tmp_path, monkeypatch, webhook_server):
    hook = webhook_server()
    root = project(tmp_path, hook.url)
    monkeypatch.chdir(root)
    server = daemon_on(root, daemon_public_url("127.0.0.1", 8765))
    agent = asyncio.run(server.resolve_agent("reporter"))
    assert agent.signing_identity.issuer == "http://127.0.0.1:8765/agents/reporter"
    server.cron.sync_from_registry(server.registry)
    agent.run = canned
    record = asyncio.run(server.cron.run_now("reporter", "hook"))
    assert record["outcome"] == "delivered"
    assert record["detail"].startswith(f"webhook {hook.url} (unsigned: this agent's key could not sign: ")
    assert "loopback" in record["detail"]
    assert "signature" not in hook.received[0].headers


def test_a_key_file_that_cannot_be_used_is_said_never_replaced_and_the_agent_serves_unsigned(tmp_path, monkeypatch, webhook_server):
    from unittest.mock import MagicMock

    from webagents.server.extensions import local_file_source

    hook = webhook_server()
    root = project(tmp_path, hook.url)
    monkeypatch.chdir(root)
    key_file = key_file_of(root)
    key_file.parent.mkdir(parents=True)
    key_file.write_text("not a key")
    # The source's own logger (the SDK logger writes to the terminal it captured at import, not to caplog).
    said: List[str] = []
    recording = MagicMock()
    recording.error.side_effect = lambda message, *args, **kwargs: said.append(str(message))
    monkeypatch.setattr(local_file_source, "logger", recording)
    server = daemon_on(root, PUBLIC_URL)
    agent = asyncio.run(server.resolve_agent("reporter"))
    assert agent is not None
    assert getattr(agent, "signing_identity", None) is None
    assert any(str(key_file) in line and "webhooks go out unsigned" in line for line in said)
    assert key_file.read_text() == "not a key"
    assert TestClient(server.app).get(jwks_route()).status_code == FIXTURE["no_identity_status"]
    server.cron.sync_from_registry(server.registry)
    agent.run = canned
    record = asyncio.run(server.cron.run_now("reporter", "hook"))
    assert record["detail"] == f"webhook {hook.url} (unsigned: this agent holds no signing key)"
