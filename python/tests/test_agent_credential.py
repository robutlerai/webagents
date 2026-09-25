"""
The agent's own credential is found, not exported by hand (2026-09-24).

`webagents deploy` stored the agent's key in the keystore, and the docs then
told the developer to read it back and export it as WEBAGENTS_AGENT_TOKEN; most
platform skills read WEBAGENTS_API_KEY instead, and payments read a
`robutler_api_key` config key with no environment fallback. Every skill already
fell back to `agent.api_key`, which did not exist. One resolver now.
"""

import json

import pytest

from webagents import BaseAgent
from webagents.server.core.registration import resolve_agent_token
from webagents.utils.agent_credential import (
    agent_key_name,
    link_matches,
    resolve_agent_credential,
)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A clean HOME with a file-backed store, and a project directory."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("WEBAGENTS_SECRETS_BACKEND", "file")
    for name in ("WEBAGENTS_PROFILE", "WEBAGENTS_AGENT_TOKEN", "WEBAGENTS_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(project)
    return home, project


def _link(project, platform_name):
    (project / ".webagents").mkdir(exist_ok=True)
    (project / ".webagents" / "config.json").write_text(
        json.dumps({"link.agentId": "a-1", "link.agentName": platform_name})
    )


def _store(home, name, value):
    from webagents.agents.skills.local.secrets.store import open_secret_store

    store = open_secret_store(
        namespace="providers", secrets_dir=str(home / ".webagents" / "secrets"), quiet=True
    )
    store.set(name, value)


def test_the_key_name_matches_what_deploy_writes():
    assert agent_key_name("alice.my-agent") == "AGENT_KEY_ALICE_MY_AGENT"


def test_finds_the_stored_key_for_the_linked_agent(machine):
    home, project = machine
    _link(project, "alice.helper")
    _store(home, "AGENT_KEY_ALICE_HELPER", "rok_STORED")
    # THE FRICTION: this took `secrets get ... --show` and an export.
    assert resolve_agent_credential("helper") == ("rok_STORED", "keystore:AGENT_KEY_ALICE_HELPER")


def test_a_second_agent_in_the_directory_does_not_get_the_linked_key(machine):
    home, project = machine
    _link(project, "alice.helper")
    _store(home, "AGENT_KEY_ALICE_HELPER", "rok_STORED")
    assert not link_matches("alice.helper", "planner")
    assert resolve_agent_credential("planner") is None


def test_the_environment_overrides_the_store_and_code_overrides_both(machine, monkeypatch):
    home, project = machine
    _link(project, "alice.helper")
    _store(home, "AGENT_KEY_ALICE_HELPER", "rok_STORED")
    monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", "from-env")
    assert resolve_agent_credential("helper")[1] == "env:WEBAGENTS_AGENT_TOKEN"
    assert resolve_agent_credential("helper", explicit="in-code")[0] == "in-code"


def test_every_skill_sees_it_through_agent_api_key(machine):
    home, project = machine
    _link(project, "alice.helper")
    _store(home, "AGENT_KEY_ALICE_HELPER", "rok_STORED")
    agent = BaseAgent(name="helper", instructions="x", model="openai/gpt-4o-mini")
    # THE BUG: 55 skill fallbacks read `agent.api_key`, and it did not exist.
    assert agent.api_key == "rok_STORED"
    agent.api_key = "explicit"
    assert agent.api_key == "explicit"


def test_the_heartbeat_ignores_the_older_name_which_often_holds_an_owner_key(machine, monkeypatch):
    monkeypatch.setenv("WEBAGENTS_API_KEY", "owner-key")
    agent = BaseAgent(name="helper", instructions="x", model="openai/gpt-4o-mini")
    # Platform skills still accept it...
    assert agent.api_key == "owner-key"
    # ...but the heartbeat needs an agent-bound key, and the route refuses an owner's.
    assert resolve_agent_token(agent) is None
    monkeypatch.setenv("WEBAGENTS_AGENT_TOKEN", "agent-key")
    assert resolve_agent_token(agent) == "agent-key"


def test_registration_hands_its_bearer_to_the_heartbeat(monkeypatch):
    """The bearer used to be the caller's to export as WEBAGENTS_AGENT_TOKEN
    and restart with; the bridge then refused that same variable (2026-09-24)."""
    import asyncio

    from webagents.server.core import registration as reg

    started = []

    async def fake_loop(url, token, name, interval):
        started.append((url, token, name))

    monkeypatch.setattr(reg, "run_heartbeat_loop", fake_loop)
    monkeypatch.setenv("ROBUTLER_API_URL", "https://platform.example.com")

    class Server:
        _heartbeat_tasks = []
        _heartbeat_agents = set()

    async def scenario():
        server = Server()
        reg._start_heartbeat_with_bearer(server, "mini", {"ok": True, "access_token": "bearer"}, None)
        # A second registration result must not start a second heartbeat.
        reg._start_heartbeat_with_bearer(server, "mini", {"ok": True, "access_token": "bearer"}, None)
        # Nothing to beat with: nothing started.
        reg._start_heartbeat_with_bearer(server, "other", {"ok": False}, None)
        await asyncio.gather(*server._heartbeat_tasks)
        return server

    server = asyncio.run(scenario())
    assert started == [("https://platform.example.com", "bearer", "mini")]
    assert server._heartbeat_agents == {"mini"}
