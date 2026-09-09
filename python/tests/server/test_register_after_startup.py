"""
``register_after_startup``: registering without deadlocking the server.

THE TRAP THIS REMOVES. Registration is a call the platform ANSWERS BY CALLING
BACK: it fetches the agent's card from this very server while the registering
request is still in flight. uvicorn serves nothing at all until every
``on_event("startup")`` handler has returned, so awaiting the registration
inside one deadlocks the callback against the handler waiting for it. From
outside that is a 502 on the card fetch and a bare 401 on the registering
call, and neither message points at the ordering: it cost a live debugging run
to find (2026-09-08 registration pass).

So the properties worth pinning are about SHAPE rather than about OAuth:

  1. The startup handler returns without waiting for the registration. If this
     regresses, every agent using the helper hangs on boot and the symptom is
     two misleading HTTP errors.
  2. The task is strongly referenced while it runs. ``asyncio.create_task``
     returns the only strong reference there is; drop it and CPython may
     collect the task before it completes, which produces a registration that
     silently never happened. That is documented behaviour, not a quirk.
  3. Nothing here can take startup down. An agent that cannot register still
     has to serve the callers that can reach it, same as the heartbeat: a
     failing registration, a raising registration and a raising callback are
     all warnings.
"""

import asyncio

import pytest

from webagents.server.core import registration as reg_mod
from webagents.server.core.registration import register_after_startup


class FakeApp:
    """Just enough FastAPI to hold an ``on_event("startup")`` handler."""

    def __init__(self):
        self.startup_handlers = []

    def on_event(self, name):
        def decorator(fn):
            if name == "startup":
                self.startup_handlers.append(fn)
            return fn

        return decorator

    async def run_startup(self):
        for fn in self.startup_handlers:
            await fn()


class FakeServer:
    """What ``create_server`` returns: an object carrying ``.app``."""

    def __init__(self):
        self.app = FakeApp()


@pytest.mark.asyncio
async def test_startup_handler_returns_without_waiting_for_registration(monkeypatch):
    """The whole point. Startup completes while the registration is still in flight."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_register(agent_name, **kwargs):
        started.set()
        await release.wait()
        return {"ok": True, "username": "a", "user_id": "1"}

    monkeypatch.setattr(reg_mod, "register_with_platform", slow_register)

    server = FakeServer()
    register_after_startup(server, "selfreg")

    # If the helper awaited the registration, this would never return: nothing
    # sets `release` until after it does.
    await asyncio.wait_for(server.app.run_startup(), timeout=1.0)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    release.set()
    await asyncio.sleep(0)  # let the task finish


@pytest.mark.asyncio
async def test_the_task_is_referenced_while_it_runs(monkeypatch):
    """Item 2: an unreferenced task can be collected mid-flight, silently."""
    release = asyncio.Event()

    async def slow_register(agent_name, **kwargs):
        await release.wait()
        return {"ok": True, "username": "a", "user_id": "1"}

    monkeypatch.setattr(reg_mod, "register_with_platform", slow_register)

    server = FakeServer()
    register_after_startup(server, "selfreg")
    await server.app.run_startup()

    assert len(reg_mod._PENDING_REGISTRATIONS) == 1
    release.set()
    await asyncio.sleep(0.01)
    # And released once done, so the set is not a leak.
    assert len(reg_mod._PENDING_REGISTRATIONS) == 0


@pytest.mark.asyncio
async def test_the_result_reaches_the_callback(monkeypatch):
    seen = []

    async def fake_register(agent_name, **kwargs):
        return {"ok": True, "username": "selfreg-agent", "user_id": "u-1", "reused": False}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    server = FakeServer()
    register_after_startup(server, "selfreg", on_result=seen.append)
    await server.app.run_startup()
    await asyncio.sleep(0.01)

    assert seen == [{"ok": True, "username": "selfreg-agent", "user_id": "u-1", "reused": False}]


@pytest.mark.asyncio
async def test_arguments_are_passed_straight_through(monkeypatch):
    captured = {}

    async def fake_register(agent_name, **kwargs):
        captured["agent_name"] = agent_name
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    server = FakeServer()
    register_after_startup(
        server,
        "selfreg",
        public_url="https://agent.example.com",
        platform_url="https://platform.example.com",
        scopes="read",
    )
    await server.app.run_startup()
    await asyncio.sleep(0.01)

    assert captured["agent_name"] == "selfreg"
    assert captured["public_url"] == "https://agent.example.com"
    assert captured["platform_url"] == "https://platform.example.com"
    assert captured["scopes"] == "read"
    # `on_result` is the helper's own argument and must not leak downstream.
    assert "on_result" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "returns_not_ok",
        "raises",
        "callback_raises",
    ],
)
async def test_no_failure_mode_takes_startup_down(monkeypatch, failure):
    """Item 3. An agent that cannot register still has to serve."""

    async def fake_register(agent_name, **kwargs):
        if failure == "raises":
            raise RuntimeError("network is on fire")
        return {"ok": False, "error": "401 from the platform"}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    def boom(_result):
        raise ValueError("callback is broken")

    server = FakeServer()
    register_after_startup(
        server,
        "selfreg",
        on_result=boom if failure == "callback_raises" else None,
    )

    await server.app.run_startup()
    await asyncio.sleep(0.01)
    # Nothing raised, and nothing is left holding a reference.
    assert len(reg_mod._PENDING_REGISTRATIONS) == 0


@pytest.mark.asyncio
async def test_accepts_the_app_directly_as_well_as_the_server(monkeypatch):
    """`create_server(...)` or `create_server(...).app`, both work."""

    async def fake_register(agent_name, **kwargs):
        return {"ok": True}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    app = FakeApp()
    register_after_startup(app, "selfreg")
    assert len(app.startup_handlers) == 1
    await app.run_startup()
    await asyncio.sleep(0.01)


class ServerWithPrefix(FakeServer):
    """`create_server(url_prefix=...)` exposes the prefix agents mount behind."""

    def __init__(self, url_prefix):
        super().__init__()
        self.url_prefix = url_prefix


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url_prefix,expected",
    [
        ("", ""),           # mounted at the root: agents are /alpha, /beta
        ("/agents", "/agents"),
        ("/agents/", "/agents"),  # a trailing slash would double the separator
    ],
)
async def test_agent_path_defaults_to_the_mount_prefix(monkeypatch, url_prefix, expected):
    """Finding 3 of the 2026-09-07 DCR pass, the Python half.

    `agent_registrations.agent_url` is unique and the platform composes it as
    ``issuer + agent_path + "/" + sub``. Until 2026-09-08 the portal read
    ``agent_path`` with a truthiness check, so ``""`` was indistinguishable
    from absent: every agent on one origin registered as the bare issuer, a
    host serving N agents could register exactly ONE, and the rest verified
    against the first one's key and were refused. The portal now tests
    ``!== undefined``, so an explicit empty prefix is meaningful and this must
    send it.

    The PREFIX goes here, never the agent's own name: the platform appends
    ``sub``, which is already the agent name.
    """
    captured = {}

    async def fake_register(agent_name, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    server = ServerWithPrefix(url_prefix)
    register_after_startup(server, "alpha")
    await server.app.run_startup()
    await asyncio.sleep(0.01)

    assert captured["agent_path"] == expected
    # Never the agent's own name: that arrives as `sub` and would be doubled.
    assert "alpha" not in (captured["agent_path"] or "")


@pytest.mark.asyncio
async def test_an_explicit_agent_path_wins_over_the_prefix(monkeypatch):
    captured = {}

    async def fake_register(agent_name, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    server = ServerWithPrefix("/agents")
    register_after_startup(server, "alpha", agent_path="/somewhere/else")
    await server.app.run_startup()
    await asyncio.sleep(0.01)

    assert captured["agent_path"] == "/somewhere/else"


@pytest.mark.asyncio
async def test_a_bare_app_sends_no_agent_path_at_all(monkeypatch):
    """Nothing to derive it from, so it stays absent and the platform falls
    back to the bare issuer, exactly as before this default existed."""
    captured = {}

    async def fake_register(agent_name, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(reg_mod, "register_with_platform", fake_register)

    app = FakeApp()
    register_after_startup(app, "alpha")
    await app.run_startup()
    await asyncio.sleep(0.01)

    assert "agent_path" not in captured
