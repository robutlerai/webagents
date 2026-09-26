"""
The session skill, served (2026-09-25): an agent keeps each verified caller's
conversation when the request names it (`agents/skills/local/session/skill.py`).

Pinned here through the skill's own hooks, over a context shaped like a served
turn's: the owner's conversation is kept where the chat keeps theirs, another
caller's in a namespace of their own, and nothing is kept for an anonymous
caller or a request that names no session. Two callers naming the same id get
two conversations. The skill answers nothing itself: no command and no route
(the old one's `/sessions` routes answered anyone, S-249).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from webagents.agents.skills.local.session.skill import SessionSkill
from webagents.cli.sessions import caller_sessions_dir, sessions_dir

SESSION = "0f1e2d3c-4b5a-4968-8776-655443322110"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("WEBAGENTS_PROFILE", raising=False)
    return tmp_path


class Turn:
    """A served turn's context: the request's messages, its metadata, the caller."""

    def __init__(self, auth: Any, metadata: Dict[str, Any]):
        self.auth = auth
        self.metadata = metadata
        self.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Plan the launch."},
        ]
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


async def serve_turn(skill: SessionSkill, auth: Any, metadata: Dict[str, Any], reply: str = "Step one: the date.") -> None:
    turn = Turn(auth, metadata)
    await skill.remember_request(turn)
    turn.messages.append({"role": "assistant", "content": reply})
    await skill.keep_conversation(turn)


def skill_for(folder: Path) -> SessionSkill:
    return SessionSkill({"agent_path": str(folder), "agent_name": "helper"})


OWNER = SimpleNamespace(authenticated=True, provider="platform", scope="owner", user_id="owner-1")
ALICE = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="alice")
BOB = SimpleNamespace(authenticated=True, provider="platform", scope="user", user_id="bob")
NOBODY = SimpleNamespace(authenticated=False, provider="anonymous", scope="all", user_id=None)


def kept(directory: Path) -> Dict[str, Any]:
    return json.loads((directory / f"{SESSION}.json").read_text())


@pytest.mark.asyncio
async def test_the_owners_conversation_is_kept_where_the_chat_keeps_theirs(tmp_path):
    folder = tmp_path / "agent"
    folder.mkdir()
    await serve_turn(skill_for(folder), OWNER, {"session_id": SESSION})

    data = kept(sessions_dir(folder, "helper"))
    assert data["messages"] == [
        {"role": "user", "content": "Plan the launch."},
        {"role": "assistant", "content": "Step one: the date."},
    ]
    assert data["agent_name"] == "helper" and data["metadata"]["sdk"] == "python"
    assert oct((sessions_dir(folder, "helper") / f"{SESSION}.json").stat().st_mode & 0o777) == "0o600"


@pytest.mark.asyncio
async def test_each_caller_has_a_namespace_of_their_own(tmp_path):
    folder = tmp_path / "agent"
    folder.mkdir()
    skill = skill_for(folder)
    await serve_turn(skill, ALICE, {"session_id": SESSION}, reply="for alice")
    await serve_turn(skill, BOB, {"session_id": SESSION}, reply="for bob")

    alice = kept(caller_sessions_dir(folder, "helper", "user:alice"))
    bob = kept(caller_sessions_dir(folder, "helper", "user:bob"))
    assert alice["messages"][-1]["content"] == "for alice"
    assert bob["messages"][-1]["content"] == "for bob"
    # Neither is the owner's.
    assert not (sessions_dir(folder, "helper") / f"{SESSION}.json").exists()


@pytest.mark.asyncio
async def test_nothing_is_kept_for_an_anonymous_caller_or_without_a_session(tmp_path):
    folder = tmp_path / "agent"
    folder.mkdir()
    skill = skill_for(folder)
    await serve_turn(skill, NOBODY, {"session_id": SESSION})
    await serve_turn(skill, OWNER, {})
    await serve_turn(skill, OWNER, {"session_id": "../../escape"})

    root = sessions_dir(folder, "helper")
    assert not root.exists() or not any(root.rglob("*.json"))


@pytest.mark.asyncio
async def test_the_next_turn_replaces_the_conversation_and_keeps_when_it_began(tmp_path):
    folder = tmp_path / "agent"
    folder.mkdir()
    skill = skill_for(folder)
    await serve_turn(skill, OWNER, {"session_id": SESSION})
    first = kept(sessions_dir(folder, "helper"))
    await serve_turn(skill, OWNER, {"session_id": SESSION}, reply="Step two.")
    second = kept(sessions_dir(folder, "helper"))

    assert second["created_at"] == first["created_at"]
    assert second["messages"][-1]["content"] == "Step two."


def test_it_answers_nothing_itself():
    from webagents.agents.core.base_agent import BaseAgent

    agent = BaseAgent(name="helper", instructions="Help.", skills={"session": SessionSkill({"agent_name": "helper"})})
    assert [c["path"] for c in agent.list_commands()] == []
    assert not [h for h in agent.get_all_http_handlers() if "session" in str(h.get("path", ""))]


def test_a_backend_it_does_not_know_is_refused_with_the_fix():
    with pytest.raises(ValueError, match='backend must be "local" or "robutler"'):
        SessionSkill({"backend": "cloud"})
