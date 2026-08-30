"""
Frame handling in PortalConnectSkill: what may start a run, what the run is
given, and what the daemon does with the control frames it used to ignore.

The old suite only ever fed the parser the shapes the parser already expected,
which is how it stayed green while the parser dropped every piece of real
production traffic: the conversation history, the payment token, and the
`session.error` that is the only signal a rejected token produces.
"""

import asyncio
import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill
from webagents.agents.skills.robutler.portal_connect import skill as skill_module
from webagents.agents.skills.robutler.portal_connect.skill import (
    PortalConnectConfigError,
    sanitize_portal_messages,
)


class _Recorder(logging.Handler):
    """The SDK installs its own non-propagating handlers, so `caplog` sees
    nothing. Attach directly to the module logger instead."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def text_at(self, level):
        return "\n".join(
            r.getMessage() for r in self.records if r.levelno >= level
        )


@pytest.fixture
def logs():
    handler = _Recorder()
    logger = skill_module.logger
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


@pytest.fixture
def skill_config():
    return {
        "portal_ws_url": "wss://robutler.test/ws",
        "agents": [{"name": "agent-a", "token": "jwt-a"}],
        "auto_reconnect": False,
        "max_reconnect_attempts": 0,
    }


def make_skill(skill_config, *, streaming=None, record=None):
    """A connected skill with one mapped session and a stub agent."""
    skill = PortalConnectSkill(skill_config)
    skill._ws = AsyncMock()
    skill._connected = True
    skill._session_by_id["sess_1"] = "agent-a"

    agent = MagicMock()
    agent.name = "agent-a"

    async def default_streaming(messages, tools=None):
        if record is not None:
            record.append(list(messages))
        yield {"content": "ok"}

    agent.run_streaming = streaming or default_streaming
    skill.set_agent_resolver(lambda name: agent if name == "agent-a" else None)
    return skill


# ---------------------------------------------------------------------------
# _normalize: COMMAND vs BROADCAST
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_top_level_type_is_a_command(self):
        kind, _ = PortalConnectSkill._normalize({"type": "input.text", "session_id": "s"})
        assert kind == "command"

    def test_event_without_top_level_type_is_a_broadcast(self):
        kind, _ = PortalConnectSkill._normalize({
            "event": {"type": "message.created"}, "chatId": "c1", "_origin": "inst_1",
        })
        assert kind == "broadcast"

    def test_anything_else_is_logged_as_unknown(self, logs):
        kind, _ = PortalConnectSkill._normalize({"foo": 1, "bar": 2})
        assert kind == "unknown"
        assert "keys=" in logs.text_at(logging.WARNING)

    @pytest.mark.asyncio
    async def test_broadcast_never_starts_a_run(self, skill_config):
        # Both shapes arrive for the SAME turn, so mapping message.created onto
        # the input.text path would run the model twice and bill twice.
        runs = []
        skill = make_skill(skill_config, record=runs)

        await skill._handle_message(json.dumps({
            "event": {
                "type": "message.created",
                "message": {"content": "hello", "role": "user"},
            },
            "chatId": "chat-1",
            "_origin": "inst_1",
        }))
        await asyncio.sleep(0)

        assert runs == []
        skill._ws.send.assert_not_called()


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestMessageSanitizer:
    def test_strips_the_non_standard_content_items_key(self):
        out = sanitize_portal_messages([
            {"role": "user", "content": "look", "content_items": [{"type": "image"}]},
        ])
        assert out == [{"role": "user", "content": "look"}]

    def test_drops_tool_rows_and_the_assistant_tool_calls_they_pair_with(self):
        # Keeping either half alone is a 400: a tool row with no preceding
        # tool_calls, or tool_calls with no tool replies.
        out = sanitize_portal_messages([
            {"role": "user", "content": "search for cats"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "3 results", "tool_call_id": "c1", "name": "search"},
            {"role": "assistant", "content": "Found three."},
        ])
        assert out == [
            {"role": "user", "content": "search for cats"},
            {"role": "assistant", "content": "Found three."},
        ]

    def test_flattens_multimodal_content_arrays_to_text(self):
        out = sanitize_portal_messages([
            {"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ]},
        ])
        assert out == [{"role": "user", "content": "what is this"}]

    def test_non_list_input_is_empty(self):
        assert sanitize_portal_messages(None) == []
        assert sanitize_portal_messages({"role": "user"}) == []


class TestHistoryForwarding:
    @pytest.mark.asyncio
    async def test_uses_the_history_the_portal_sent(self, skill_config):
        runs = []
        skill = make_skill(skill_config, record=runs)

        await skill._handle_input_text({
            "type": "input.text",
            "session_id": "sess_1",
            "text": "and then?",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "and then?"},
            ],
        }, "sess_1")

        assert runs[0] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and then?"},
        ]

    @pytest.mark.asyncio
    async def test_falls_back_to_the_single_turn_when_no_history_arrives(self, skill_config):
        runs = []
        skill = make_skill(skill_config, record=runs)

        await skill._handle_input_text(
            {"type": "input.text", "session_id": "sess_1", "text": "solo"}, "sess_1",
        )

        assert runs[0] == [{"role": "user", "content": "solo"}]

    @pytest.mark.asyncio
    async def test_falls_back_when_the_history_sanitizes_to_nothing(self, skill_config):
        runs = []
        skill = make_skill(skill_config, record=runs)

        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "solo",
            "messages": [{"role": "tool", "content": "x", "tool_call_id": "c1"}],
        }, "sess_1")

        assert runs[0] == [{"role": "user", "content": "solo"}]


# ---------------------------------------------------------------------------
# Payment token
# ---------------------------------------------------------------------------


class TestPaymentToken:
    @pytest.mark.asyncio
    async def test_payment_token_lands_on_the_run_context(self, skill_config):
        from webagents.server.context.context_vars import get_context

        seen = {}

        async def streaming(messages, tools=None):
            ctx = get_context()
            seen["token"] = getattr(ctx, "payment_token", None)
            yield {"content": "ok"}

        skill = make_skill(skill_config, streaming=streaming)

        await skill._handle_input_text({
            "type": "input.text", "session_id": "sess_1", "text": "pay me",
            "payment_token": "pay_abc123",
        }, "sess_1")

        assert seen["token"] == "pay_abc123"

    @pytest.mark.asyncio
    async def test_no_payment_token_leaves_the_context_alone(self, skill_config):
        from webagents.server.context.context_vars import CONTEXT, get_context

        # Another test in the same session may have leaked a context into the
        # ContextVar (it is process-global); this test is about THIS turn not
        # attaching a token, so start from a clean slate.
        reset_token = CONTEXT.set(None)
        seen = {}

        async def streaming(messages, tools=None):
            ctx = get_context()
            seen["token"] = getattr(ctx, "payment_token", None) if ctx else None
            yield {"content": "ok"}

        try:
            skill = make_skill(skill_config, streaming=streaming)
            await skill._handle_input_text(
                {"type": "input.text", "session_id": "sess_1", "text": "free"}, "sess_1",
            )
            assert seen["token"] is None
        finally:
            CONTEXT.reset(reset_token)


# ---------------------------------------------------------------------------
# Control frames
# ---------------------------------------------------------------------------


class TestControlFrames:
    @pytest.mark.asyncio
    async def test_session_error_logs_the_code_at_error_and_starts_no_run(self, skill_config, logs):
        runs = []
        skill = make_skill(skill_config, record=runs)

        await skill._handle_message(json.dumps({
            "type": "session.error",
            "event_id": "evt_1",
            "error": {"code": "forbidden", "message": "Not authorized for this agent"},
        }))
        await asyncio.sleep(0)

        assert "forbidden" in logs.text_at(logging.ERROR)
        assert runs == []

    @pytest.mark.asyncio
    async def test_response_cancel_stops_the_run_and_acknowledges(self, skill_config):
        started = asyncio.Event()

        async def slow(messages, tools=None):
            started.set()
            await asyncio.sleep(30)
            yield {"content": "never"}

        skill = make_skill(skill_config, streaming=slow)

        await skill._handle_message(json.dumps({
            "type": "input.text", "session_id": "sess_1", "text": "go",
        }))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert "sess_1" in skill._runs

        await skill._handle_message(json.dumps({
            "type": "response.cancel", "session_id": "sess_1",
        }))
        await asyncio.sleep(0)

        sent = [json.loads(c[0][0]) for c in skill._ws.send.call_args_list]
        assert any(f["type"] == "response.cancelled" and f["session_id"] == "sess_1" for f in sent)
        # A deliberate stop is not an error.
        assert not any(f["type"] == "response.error" for f in sent)

    @pytest.mark.asyncio
    async def test_payment_frames_are_handled_without_starting_a_run(self, skill_config, logs):
        runs = []
        skill = make_skill(skill_config, record=runs)

        await skill._handle_message(json.dumps({
            "type": "payment.submit", "session_id": "sess_1",
            "payment": {"scheme": "robutler", "amount": "0.10", "token": "pay_1"},
        }))
        await skill._handle_message(json.dumps({
            "type": "payment.error", "session_id": "sess_1",
            "code": "insufficient_balance", "message": "no funds", "can_retry": False,
        }))
        await asyncio.sleep(0)

        assert runs == []
        assert "payment.submit" in logs.text_at(logging.INFO)
        assert "insufficient_balance" in logs.text_at(logging.ERROR)


# ---------------------------------------------------------------------------
# Task isolation and configuration
# ---------------------------------------------------------------------------


class TestTurnIsolation:
    @pytest.mark.asyncio
    async def test_a_slow_turn_does_not_block_the_read_loop(self, skill_config):
        entered = asyncio.Event()

        async def slow(messages, tools=None):
            entered.set()
            await asyncio.sleep(30)
            yield {"content": "never"}

        skill = make_skill(skill_config, streaming=slow)

        # If the turn were awaited inline this would not return until the model
        # finished — stalling `session.end` and every other multiplexed agent.
        await asyncio.wait_for(
            skill._handle_message(json.dumps({
                "type": "input.text", "session_id": "sess_1", "text": "go",
            })),
            timeout=1,
        )
        await asyncio.wait_for(entered.wait(), timeout=1)

        # session.end is still serviceable while the turn runs.
        await skill._handle_message(json.dumps({
            "type": "session.end", "session_id": "sess_1",
        }))
        assert "sess_1" not in skill._session_by_id
        assert "sess_1" not in skill._runs


class TestEmptyTokenIsLoud:
    @pytest.mark.asyncio
    async def test_an_empty_token_raises_instead_of_silently_doing_nothing(self, skill_config):
        skill = PortalConnectSkill(skill_config)
        skill._ws = AsyncMock()
        skill._connected = True

        with pytest.raises(PortalConnectConfigError):
            await skill._send_session_create("agent-a", "")
        skill._ws.send.assert_not_called()
