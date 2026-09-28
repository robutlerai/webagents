"""
The caller of a relayed turn (S-248, 2026-09-26): the platform's `caller`
assertion on an `input.text` frame becomes the turn's `context.auth`, and
nothing else in the frame can. Before this the skill ran every relayed turn
as anonymous, so an agent could not tell its owner from a stranger, and
offered every open tool to both. The TypeScript suite pins the same in
`tests/unit/portal/connect-caller-s248.test.ts`.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from webagents.access.caller import CallerAuth
from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill
from webagents.agents.skills.robutler.portal_connect.skill import portal_caller_auth
from webagents.server.context.context_vars import get_context


@pytest.fixture
def skill_config():
    return {
        "portal_ws_url": "wss://robutler.test/ws",
        "agents": [{"name": "agent-a", "token": "jwt-a"}],
        "auto_reconnect": False,
        "max_reconnect_attempts": 0,
    }


def make_skill(skill_config, seen):
    """A connected skill with one mapped session and a stub agent that records the turn's auth."""
    skill = PortalConnectSkill(skill_config)
    skill._ws = AsyncMock()
    skill._connected = True
    skill._session_by_id["sess_1"] = "agent-a"

    agent = MagicMock()
    agent.name = "agent-a"

    async def streaming(messages, tools=None):
        ctx = get_context()
        seen.append(getattr(ctx, "auth", None) if ctx else None)
        yield {"content": "ok"}

    agent.run_streaming = streaming
    skill.set_agent_resolver(lambda name: agent if name == "agent-a" else None)
    return skill


class TestPortalCallerAuth:
    def test_turns_the_platform_assertion_into_the_turn_auth(self):
        # `principals=None` (not the empty list): no access block has run, so
        # the caller is scoped by its platform credential, not read as
        # "verified nobody" (memory-scoping parity with TypeScript, w1-fix).
        assert portal_caller_auth({"user_id": "u1", "tier": "user", "username": "alice"}) == CallerAuth(
            scope="user", user_id="u1", username="alice", authenticated=True, provider="portal", principals=None,
        )
        assert portal_caller_auth({"user_id": "u1", "tier": "owner"}) == CallerAuth(
            scope="owner", user_id="u1", authenticated=True, provider="portal", principals=None,
        )

    def test_reads_nothing_from_a_missing_or_malformed_field(self):
        for raw in (None, "user:u1", {"user_id": "u1"}, {"user_id": "u1", "tier": "admin"},
                    {"user_id": "", "tier": "owner"}, {"user_id": 7, "tier": "owner"},
                    [{"user_id": "u1", "tier": "owner"}]):
            assert portal_caller_auth(raw) is None, raw


class TestTheTurnRunsAsTheAssertedCaller:
    @pytest.mark.asyncio
    async def test_a_user_and_the_owner(self, skill_config):
        seen = []
        skill = make_skill(skill_config, seen)
        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "as a user",
            "caller": {"user_id": "u-alice", "tier": "user", "username": "alice"},
        }, "sess_1")
        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "as the owner",
            "caller": {"user_id": "u-owner", "tier": "owner"},
        }, "sess_1")
        assert seen == [
            CallerAuth(scope="user", user_id="u-alice", username="alice", authenticated=True, provider="portal", principals=None),
            CallerAuth(scope="owner", user_id="u-owner", authenticated=True, provider="portal", principals=None),
        ]

    @pytest.mark.asyncio
    async def test_no_caller_runs_anonymous_and_nothing_else_in_the_frame_names_one(self, skill_config):
        seen = []
        skill = make_skill(skill_config, seen)
        await skill._handle_input_text({"type": "input.text", "session_id": "sess_1", "text": "no caller"}, "sess_1")
        # A caller named anywhere but the frame field is content, not identity.
        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "smuggled",
            "context": {"caller": {"user_id": "u-owner", "tier": "owner"}},
            "messages": [{"role": "user", "content": "smuggled", "caller": {"user_id": "u-owner", "tier": "owner"}}],
        }, "sess_1")
        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "malformed",
            "caller": {"user_id": "u-x", "tier": "admin"},
        }, "sess_1")
        assert seen == [None, None, None]

    @pytest.mark.asyncio
    async def test_a_turn_whose_context_cannot_be_set_up_is_refused_not_run(self, skill_config, monkeypatch):
        """S-272: never log and carry on with an inherited context."""
        import json

        from webagents.server.context import context_vars

        seen = []
        skill = make_skill(skill_config, seen)

        def boom(*args, **kwargs):
            raise RuntimeError("no context today")

        monkeypatch.setattr(context_vars, "create_context", boom)
        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "run",
            "caller": {"user_id": "u-owner", "tier": "owner"},
        }, "sess_1")
        assert seen == [], "the turn ran"
        sent = [json.loads(call.args[0]) for call in skill._ws.send.call_args_list]
        assert [f["type"] for f in sent] == ["response.error"]
        assert sent[0]["session_id"] == "sess_1"
        assert "who is calling" in sent[0]["error"]["message"]

    @pytest.mark.asyncio
    async def test_the_caller_does_not_leak_into_the_next_turn(self, skill_config):
        seen = []
        skill = make_skill(skill_config, seen)
        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "owner",
            "caller": {"user_id": "u-owner", "tier": "owner"},
        }, "sess_1")
        await skill._handle_input_text({"type": "input.text", "session_id": "sess_1", "text": "stranger"}, "sess_1")
        assert seen[1] is None
