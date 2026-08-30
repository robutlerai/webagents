"""Runnable tests that mirror the highest-value doc examples.

Each test references the doc file and section it validates.
No real API keys or LLM calls needed -- tests validate SDK wiring only.
"""

import pytest

# HARD imports, deliberately. These used to be wrapped in try/except with a
# module-wide skipif, so a broken install (F-040) reported as SKIPPED — a
# green suite over a package nobody could import. An ImportError here must
# FAIL the suite.
from webagents.agents.core.base_agent import BaseAgent
from webagents.agents.skills.base import Skill
from webagents.agents.tools.decorators import tool, hook, http
from webagents.agents.skills.robutler.payments.skill import pricing

HAS_PRICING = True


# ---------------------------------------------------------------------------
# Validates: quickstart.md -- "Create Your First Agent"
# ---------------------------------------------------------------------------


class TestQuickstartAgent:
    def test_create_agent_with_model_string(self):
        agent = BaseAgent(
            name="quickstart-agent",
            instructions="You are a helpful assistant.",
            model="openai/gpt-4o",
        )
        assert agent.name == "quickstart-agent"
        assert "openai" in agent.skills or len(agent.skills) > 0

    def test_create_agent_with_skills_dict(self):
        """quickstart.md -- 'Connect to the Network' (Python)"""
        skill = Skill()
        agent = BaseAgent(
            name="with-skills",
            model="openai/gpt-4o",
            skills={"test_skill": skill},
        )
        assert "test_skill" in agent.skills


# ---------------------------------------------------------------------------
# Validates: agent/tools.md -- @pricing decorator
# ---------------------------------------------------------------------------


class TestPricingDecorator:
    @pytest.mark.skipif(not HAS_PRICING, reason="pricing not importable")
    def test_pricing_attaches_metadata(self):
        """pricing() should set _pricing_config on the function."""

        @pricing(credits_per_call=0.10)
        @tool(name="lookup", description="Look something up")
        async def lookup(query: str) -> str:
            return query

        assert hasattr(lookup, "_webagents_pricing")
        assert lookup._webagents_pricing["credits_per_call"] == 0.10


# ---------------------------------------------------------------------------
# Validates: agent/tools.md -- @tool(scope=...)
# ---------------------------------------------------------------------------


class TestToolScoping:
    def test_tool_scope_is_stored(self):
        @tool(name="admin_reset", description="Reset data", scope="admin")
        async def admin_reset() -> str:
            return "done"

        assert admin_reset._tool_scope == "admin"

    def test_tool_scope_default_is_all(self):
        @tool(name="public_info", description="Get info")
        async def public_info() -> str:
            return "info"

        assert public_info._tool_scope == "all"

    def test_agent_filters_tools_by_scope(self):
        """get_tools_for_scope should exclude tools above the caller's level."""

        @tool(name="public_tool", description="Public", scope="all")
        async def public_tool() -> str:
            return "public"

        @tool(name="owner_tool", description="Owner only", scope="owner")
        async def owner_tool() -> str:
            return "owner"

        agent = BaseAgent(
            name="scoped-agent",
            model="openai/gpt-4o",
            tools=[public_tool, owner_tool],
        )

        all_scope_tools = agent.get_tools_for_scope("all")
        owner_scope_tools = agent.get_tools_for_scope("owner")

        all_names = {t.get("name") or t.get("function", {}).get("name", "") for t in all_scope_tools}
        owner_names = {t.get("name") or t.get("function", {}).get("name", "") for t in owner_scope_tools}

        assert "public_tool" in all_names
        assert "owner_tool" not in all_names
        assert "public_tool" in owner_names
        assert "owner_tool" in owner_names


# ---------------------------------------------------------------------------
# Validates: agent/endpoints.md -- @http decorator
# ---------------------------------------------------------------------------


class TestHttpEndpoints:
    def test_http_decorator_marks_function(self):
        @http("/health", method="get")
        async def health_check(request):
            return {"status": "ok"}

        assert hasattr(health_check, "_http_subpath")
        assert health_check._http_subpath == "/health"
        assert health_check._http_method == "get"


# ---------------------------------------------------------------------------
# Validates: agent/lifecycle.md -- @hook decorator and hook names
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    def test_hook_decorator_stores_event(self):
        @hook("on_connection", priority=1)
        async def on_conn(context):
            return context

        assert on_conn._hook_event_type == "on_connection"
        assert on_conn._hook_priority == 1

    def test_valid_hook_names_accepted(self):
        valid_hooks = [
            "on_connection",
            "before_llm_call",
            "after_llm_call",
            "before_toolcall",
            "after_toolcall",
            "on_message",
            "on_chunk",
            "finalize_connection",
        ]
        for name in valid_hooks:

            @hook(name)
            async def handler(context):
                return context

            assert handler._hook_event_type == name

    def test_skill_hook_methods_marked(self):
        """A skill with @hook methods should have decorator metadata on them."""

        class LogSkill(Skill):
            @hook("on_connection", priority=10)
            async def on_connect(self, context):
                return context

            @hook("finalize_connection", priority=99)
            async def on_finalize(self, context):
                return context

        skill = LogSkill()
        assert skill.on_connect._hook_event_type == "on_connection"
        assert skill.on_finalize._hook_event_type == "finalize_connection"
        assert skill.on_connect._hook_priority == 10
        assert skill.on_finalize._hook_priority == 99




# ---------------------------------------------------------------------------
# Validates: skills/platform/portal-connect.md Quick Start + quickstart.md
# "Serve as an API" — by EXECUTING the example files the doc snippets are
# generated from, against a stub portal. No network beyond loopback, no
# model call, no API key.
#
# There are no wrapper entry points left to test: an example builds an agent,
# builds a server, and runs it, so these tests drive the SAME objects the
# example defines (`module.server.app`, `module.agent`) rather than a
# convenience function that only the docs ever called.
# ---------------------------------------------------------------------------

import asyncio
import base64 as _b64
import re
import importlib.util
import json as _json
import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_example(name: str):
    """Import an example file as a module (executes its top-level code)."""
    path = _EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_agent_token(sub: str = "owner-1", agent_id: str = "agent-1") -> str:
    """A structurally valid JWT (unsigned) for the startup shape check —
    the stub portal never verifies it."""
    def seg(obj):
        return _b64.urlsafe_b64encode(_json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": sub}
    if agent_id:
        payload["agent_id"] = agent_id
    return f"{seg({'alg': 'none'})}.{seg(payload)}.x"


class StubPortal:
    """A loopback /ws that speaks the platform's REAL frames: answers
    `session.create` with `session.created`, then pushes an `input.text` turn
    carrying full history and a payment token, and records everything sent."""

    def __init__(self):
        self.frames: "asyncio.Queue[dict]" = asyncio.Queue()
        self.server = None
        self.socket = None
        # The session the platform ACKs is not the one it later addresses:
        # turns arrive with a per-request `req_...` id the SDK has never
        # seen, and must be resolved through the frame's own `agent` field.
        self.created_session_id = "sess_1"
        self.session_id = "req_stub_1"

    async def start(self) -> int:
        import websockets

        self.server = await websockets.serve(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, socket):
        import websockets

        self.socket = socket
        try:
            async for raw in socket:
                frame = _json.loads(raw)
                await self.frames.put(frame)
                if frame.get("type") == "session.create":
                    await socket.send(_json.dumps({
                        "type": "session.created",
                        "session_id": self.created_session_id,
                        "session": {"agent": frame["session"]["agent"]},
                    }))
                    await self.push_turn(frame["session"]["agent"])
        except websockets.exceptions.ConnectionClosed:
            pass

    async def push_turn(self, agent_name: str) -> None:
        # PER-REQUEST session_id (`req_...`) plus the frame's own `agent`
        # field — the shape the platform actually sends. The SDK must fall
        # back to `agent` for a sid it has never seen.
        await self.socket.send(_json.dumps({
            "type": "input.text",
            "session_id": self.session_id,
            "agent": agent_name,
            "text": "Hi",
            "messages": [
                {"role": "user", "content": "earlier turn"},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "Hi"},
            ],
            "payment_token": "pt_test",
        }))

    async def next_frame(self, of_type=None, timeout=10.0) -> dict:
        async def _pull():
            while True:
                frame = await self.frames.get()
                if of_type is None or frame.get("type") == of_type:
                    return frame
        return await asyncio.wait_for(_pull(), timeout=timeout)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


def _stub_streaming(sink: list):
    async def fake_run_streaming(messages, tools=None, **kwargs):
        sink.append(list(messages))
        yield {"choices": [{"delta": {"content": "hello from mini"}}]}
    return fake_run_streaming


class TestOwnUrlMinimalExample:
    """python/examples/own_url_minimal.py, executed without binding a port.

    Drives `module.server.app` — the exact object the example hands to
    uvicorn. Nothing here is reachable only through a test helper.
    """

    async def test_example_serves_completions_and_a_registration_complete_card(
        self, monkeypatch, tmp_path
    ):
        import httpx

        monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path))
        module = _load_example("own_url_minimal.py")
        agent = module.agent

        seen: list = []
        monkeypatch.setattr(agent, "run_streaming", _stub_streaming(seen))

        transport = httpx.ASGITransport(app=module.server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # The card must be at the ORIGIN (the platform resolves it
            # origin-relative and DISCARDS the agent path) AND under the agent
            # prefix, and must carry metadata.publicKey as SPKI PEM —
            # registration hard-requires it. `create_server` alone must
            # satisfy this: there is no wrapper left to add it.
            for path in ("/.well-known/agent.json", f"/{agent.name}/.well-known/agent.json"):
                r = await client.get(path)
                assert r.status_code == 200, path
                card = r.json()
                assert card["name"] == agent.name
                key = card.get("metadata", {}).get("publicKey", "")
                assert key.startswith("-----BEGIN PUBLIC KEY-----"), path

            for path in ("/.well-known/jwks.json", f"/{agent.name}/.well-known/jwks.json"):
                r = await client.get(path)
                assert r.status_code == 200, path
                assert r.json().get("keys"), path

            # The endpoint the platform dials. It runs the model on the
            # OWNER's credit, so an unauthenticated call is refused BEFORE the
            # provider is reached. This assertion used to pin the opposite
            # (200 with no Authorization at all) while docs/quickstart.md told
            # the reader a 401 was returned — the doc asserted a security
            # property the Python SDK did not have, and an unauthenticated POST
            # really did spend the owner's provider quota.
            r = await client.post(
                f"/{agent.name}/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )
            assert r.status_code == 401, r.text
            assert not seen, "the model was reached on an unauthenticated request"

            # A blank / bearer-only header is not a credential either.
            for bogus in ("", "   ", "Bearer", "bearer "):
                r = await client.post(
                    f"/{agent.name}/chat/completions",
                    json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
                    headers={"Authorization": bogus},
                )
                assert r.status_code == 401, (bogus, r.text)

            # Each of the three credential headers clears the floor — the
            # SAME list as `CREDENTIAL_HEADERS` in
            # typescript/src/server/handler.ts, so the two SDKs agree about
            # what reaches the model.
            for header in ("Authorization", "X-Api-Key", "X-Owner-Assertion"):
                r = await client.post(
                    f"/{agent.name}/chat/completions",
                    json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
                    headers={header: "Bearer test-token"},
                )
                assert r.status_code == 200, (header, r.text)
                assert "hello from mini" in r.text

    async def test_the_key_survives_a_restart(self, monkeypatch, tmp_path):
        """Registration pins the card's public key, so a key regenerated per
        boot works exactly until the first restart. Two independent loads of
        the example must publish the SAME key."""
        import httpx

        monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path))

        async def card_key():
            module = _load_example("own_url_minimal.py")
            transport = httpx.ASGITransport(app=module.server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return (await c.get("/.well-known/agent.json")).json()["metadata"]["publicKey"]

        assert await card_key() == await card_key()


class TestPortalConnectMinimalExample:
    """python/examples/portal_connect_minimal.py, executed end to end."""

    def test_skill_refuses_owner_key_without_agent_binding(self):
        """The F-045 trap is a start-time error with the fix in the text, and
        it lives on the SKILL now — so it guards every entry point, not just
        the one function the docs used to name."""
        from webagents.agents.skills.robutler.portal_connect import (
            PortalCredentialError,
            check_agent_token,
        )

        with pytest.raises(PortalCredentialError) as exc:
            check_agent_token(_fake_agent_token(sub="owner-1", agent_id=""))
        assert "api-key" in str(exc.value)

        with pytest.raises(PortalCredentialError):
            check_agent_token("")

        check_agent_token(_fake_agent_token())  # a bound key is accepted

    async def test_refusal_reaches_the_skill_start_path(self, monkeypatch):
        """Not just the helper: `start()` itself must refuse before opening
        a socket."""
        from webagents.agents.skills.robutler.portal_connect import (
            PortalConnectSkill,
            PortalCredentialError,
        )

        monkeypatch.delenv("WEBAGENTS_ALLOW_UNBOUND_TOKEN", raising=False)
        skill = PortalConnectSkill({
            "portal_ws_url": "ws://127.0.0.1:1/ws",
            "agents": [{"name": "mini", "token": _fake_agent_token(agent_id="")}],
        })
        with pytest.raises(PortalCredentialError):
            await skill.start()
        assert skill._connection_task is None

    async def test_server_startup_fails_on_an_unbound_token(self, monkeypatch, tmp_path):
        """The guard must survive the LIFECYCLE it was moved onto.

        `_start_portal_connect_skills` used to wrap initialize()/start() in a
        broad `except Exception` that logged and continued, so an owner-subject
        token with no agent binding produced one ERROR line and then a server
        that was up, answering /health 200, with no socket and no possible
        turn — verbatim the state the guard exists to prevent.

        `test_refusal_reaches_the_skill_start_path` above calls `start()`
        directly, which is exactly why it never caught this. This one drives
        the REAL server startup event.
        """
        from webagents.agents.skills.robutler.portal_connect import (
            PortalConnectSkill,
            PortalCredentialError,
        )

        monkeypatch.delenv("WEBAGENTS_ALLOW_UNBOUND_TOKEN", raising=False)
        monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path))
        monkeypatch.setenv("WEBAGENTS_PORTAL_URL", "ws://127.0.0.1:1/ws")
        # An OWNER key: a `sub`, no `agent_id`. The platform accepts the
        # socket and routes nothing to the agent (F-045).
        monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", _fake_agent_token(agent_id=""))

        module = _load_example("portal_connect_minimal.py")
        with pytest.raises(PortalCredentialError):
            await module.server.app.router.startup()

        skill = module.agent.skills["portal"]
        assert isinstance(skill, PortalConnectSkill)
        assert skill._connection_task is None, "a socket was opened despite the refusal"

    async def test_transient_start_failure_still_only_logs(self, monkeypatch, tmp_path):
        """The broad catch is kept for TRANSIENT I/O: the bridge owns its own
        reconnect loop, so a portal that is merely down must not take the
        process with it. Only credential/config errors are fatal."""
        monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path))
        monkeypatch.setenv("WEBAGENTS_PORTAL_URL", "ws://127.0.0.1:1/ws")
        monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", _fake_agent_token())

        module = _load_example("portal_connect_minimal.py")
        skill = module.agent.skills["portal"]

        async def boom() -> None:
            raise OSError("connection refused")

        monkeypatch.setattr(skill, "start", boom)
        await module.server.app.router.startup()  # must NOT raise
        await module.server.app.router.shutdown()

    async def test_example_serves_a_turn_over_the_bridge(self, monkeypatch, tmp_path):
        """The whole example, start to finish: the server's own lifecycle
        starts the attached PortalConnectSkill, the skill reads the env, and a
        turn pushed by the stub portal comes back as deltas.

        Load-bearing: session.create goes out, a PER-REQUEST `req_...`
        session_id resolves via the frame's `agent` field, response.delta /
        response.done come back, and the FULL history reaches the run.
        """
        portal = StubPortal()
        port = await portal.start()

        monkeypatch.setenv("WEBAGENTS_KEYS_DIR", str(tmp_path))
        monkeypatch.setenv("WEBAGENTS_PORTAL_URL", f"ws://127.0.0.1:{port}/ws")
        monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", _fake_agent_token())

        module = _load_example("portal_connect_minimal.py")
        seen: list = []
        monkeypatch.setattr(module.agent, "run_streaming", _stub_streaming(seen))

        # The server's startup event — the documented lifecycle, the thing
        # `connect()` used to stand in for.
        await module.server.app.router.startup()
        try:
            created = await portal.next_frame("session.create")
            assert created["session"]["agent"] == "mini"
            assert created["session"]["token"] == _fake_agent_token()

            delta = await portal.next_frame("response.delta")
            assert delta["delta"]["text"] == "hello from mini"
            done = await portal.next_frame("response.done")
            assert done["session_id"] == portal.session_id

            assert seen and len(seen[0]) == 3
        finally:
            await module.server.app.router.shutdown()
            await module.agent.skills["portal"].disconnect()
            await portal.stop()


class TestPortalConnectSocketOnlyExample:
    """python/examples/portal_connect_socket_only.py — no HTTP surface."""

    async def test_example_main_connects_and_serves_a_turn(self, monkeypatch):
        portal = StubPortal()
        port = await portal.start()

        monkeypatch.setenv("WEBAGENTS_PORTAL_URL", f"ws://127.0.0.1:{port}/ws")
        monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", _fake_agent_token())

        module = _load_example("portal_connect_socket_only.py")
        seen: list = []
        monkeypatch.setattr(module.agent, "run_streaming", _stub_streaming(seen))

        task = asyncio.create_task(module.main())
        try:
            await portal.next_frame("session.create")
            delta = await portal.next_frame("response.delta")
            assert delta["delta"]["text"] == "hello from mini"
            await portal.next_frame("response.done")
            assert seen and len(seen[0]) == 3
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            await module.portal.disconnect()
            await portal.stop()


_GUARD_CONFIG = _REPO_ROOT / "scripts" / "removed-api-guard.json"


def scan_for_removed_api(repo_root: Path) -> list:
    """Every line under the guard's roots that names the deleted
    `connect()` / `host()` wrappers and is not an allowlisted narrative
    mention. Shared config with the TypeScript half of the guard."""
    config = _json.loads(_GUARD_CONFIG.read_text())
    patterns = [re.compile(p) for p in config["patterns"]]
    allowlist = config["allowlist"]

    scanned = []
    for root in config["roots"]:
        base = repo_root / root["dir"]
        exts = set(root["ext"])
        scanned.extend(
            path
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix in exts
        )
    # Individually named files: the package READMEs and the other top-level
    # markdown cannot be expressed as roots (walking `python/` would descend
    # into .venv, walking the repo root into node_modules).
    for name in config.get("files", []):
        path = repo_root / name
        if path.is_file():
            scanned.append(path)

    offenders = []
    for path in scanned:
        rel = path.relative_to(repo_root).as_posix()
        exempt = allowlist.get(rel, [])
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip() in exempt:
                continue
            if any(p.search(line) for p in patterns):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    return offenders


class TestDocSnippetsAreGenerated:
    """The doc snippets must equal the example files — retyped snippets are
    how a syntactically valid quickstart shipped that connected never."""

    def test_generated_doc_blocks_match_example_files(self):
        spec = importlib.util.spec_from_file_location(
            "sync_doc_examples", _REPO_ROOT / "scripts" / "sync_doc_examples.py"
        )
        sync_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync_mod)
        assert sync_mod.sync(check_only=True) == 0, (
            "doc snippets drifted from examples/; run scripts/sync_doc_examples.py"
        )

    def test_no_documented_wrapper_entry_points_remain(self):
        """`connect()` / `host()` are gone. Everything they added moved into
        `create_server` and `PortalConnectSkill`; anything that still names
        them is teaching an import that no longer resolves.

        The predecessor of this test grepped `docs/**/*.md` for four exact
        import strings. That is not a guard: it missed `.mdx`, every call
        shape, every `webagents.portal` reference, and — because it never
        looked at shipped source at all — a public docstring on
        `JWKSManager.mint_aoauth_token` and a runtime `console.warn` that both
        told developers to call the deleted functions.

        The scan now covers docs (`.md` AND `.mdx`), `python/webagents/**`,
        `typescript/src/**` and both examples trees, matching call shapes
        rather than import lines. Deliberate narrative ("the deleted `host()`
        wrapper") is exempted by EXACT LINE in
        `scripts/removed-api-guard.json`, so rewording one re-raises it.
        The TypeScript half of the guard reads the same file
        (`typescript/tests/unit/examples.test.ts`).
        """
        import webagents

        assert not hasattr(webagents, "connect")
        assert not hasattr(webagents, "host")

        offenders = scan_for_removed_api(_REPO_ROOT)
        assert not offenders, "stale references to the deleted connect()/host() API:\n" + "\n".join(
            offenders
        )

    def test_the_guard_looks_where_readers_look(self):
        """The guard's blind spots are the interesting part of it.

        Its roots originally covered docs and shipped source only, which left
        out the three package READMEs — the most-read documents in the repo,
        and the ones that go out inside the npm and PyPI packages —
        CONTRIBUTING, RELEASE, CHANGELOG, and BOTH test trees. A stale
        reference to the deleted wrapper, in a test docstring, survived
        precisely because of that gap — so this pins the coverage rather than
        trusting the scan to have been pointed at the right places.
        """
        config = _json.loads(_GUARD_CONFIG.read_text())
        covered = {r["dir"] for r in config["roots"]} | set(config.get("files", []))
        for required in (
            "README.md",
            "python/README.md",
            "typescript/README.md",
            "CONTRIBUTING.md",
            "RELEASE.md",
            "CHANGELOG.md",
            "python/tests",
            "typescript/tests",
            "docs",
            "python/webagents",
            "typescript/src",
        ):
            assert required in covered, f"removed-api guard no longer covers {required}"

        # Every named file and root must actually exist, or "covered" is a
        # spelling that scans nothing.
        for r in config["roots"]:
            assert (_REPO_ROOT / r["dir"]).is_dir(), r["dir"]
        for name in config.get("files", []):
            assert (_REPO_ROOT / name).is_file(), name
