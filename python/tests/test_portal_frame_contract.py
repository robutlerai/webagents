"""T4 — the portal→SDK frame contract, on RECORDED production frames.

Every fixture under tests/fixtures/portal_frames/ is the byte-shape a real
portal emitter puts on the agent socket (see the README there for
per-fixture provenance). This file replaces test_portal_connect_skill.py,
whose 333 lines of hand-written fixtures were written to match what the
parser already expected — which is how a green suite coexisted with a
parser that dropped all production traffic (the conversation history, the
payment token, and the session.error that is the only signal a rejected
token produces).

The load-bearing assertion: NO frame other than a COMMAND `input.text`
ever reaches the run path. Both the COMMAND shape and the wrapped
BROADCAST envelope arrive on the same socket for the same turn; treating
`message.created` as work runs the model twice and bills twice (F-043's
proposed mapping).
"""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from webagents.agents.skills.robutler.portal_connect import PortalConnectSkill
from webagents.agents.skills.robutler.portal_connect import skill as skill_module

FRAMES_DIR = Path(__file__).parent / "fixtures" / "portal_frames"
ALL_FRAMES = sorted(FRAMES_DIR.glob("*.json"))
FRAME_IDS = [p.name for p in ALL_FRAMES]

assert len(ALL_FRAMES) >= 10, "portal frame fixtures went missing"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


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
def harness(monkeypatch):
    """A connected skill with the fixture session mapped, a stub agent, and
    spies on the run path and the payment-token sink."""
    skill = PortalConnectSkill({
        "portal_ws_url": "wss://robutler.test/ws",
        "agents": [{"name": "mini", "token": "jwt-mini"}],
        "auto_reconnect": False,
    })
    skill._ws = AsyncMock()
    skill._connected = True
    skill._session_by_id["sess_fixture_1"] = "mini"

    agent = MagicMock()
    agent.name = "mini"

    runs = []

    async def run_streaming(messages, tools=None):
        runs.append(list(messages))
        yield {"choices": [{"delta": {"content": "answer"}}]}

    agent.run_streaming = run_streaming
    skill.agent = agent

    payment_tokens = []
    monkeypatch.setattr(
        skill, "_apply_payment_token", lambda token: payment_tokens.append(token)
    )

    async def _drain():
        # let the per-turn task run to completion
        for _ in range(20):
            await asyncio.sleep(0)
            if not skill._runs:
                break

    return skill, runs, payment_tokens, _drain


@pytest.mark.parametrize("fixture", ALL_FRAMES, ids=FRAME_IDS)
async def test_only_command_input_text_starts_a_run(fixture, harness, logs):
    """The contract's core: BROADCAST envelopes, control frames, and
    payment frames NEVER reach run_streaming."""
    skill, runs, _tokens, drain = harness
    await skill._handle_message(json.dumps(load(fixture)))
    await drain()
    starts_run = fixture.name in (
        "input_text.json",
        "input_text_unmapped_session.json",
        # outbound_input_text.json has NO session_id: not a bridge work order
    )
    assert len(runs) == (1 if starts_run else 0), fixture.name


async def test_input_text_delivers_full_history_not_a_synthesized_turn(harness):
    skill, runs, tokens, drain = harness
    frame = load(FRAMES_DIR / "input_text.json")
    await skill._handle_message(json.dumps(frame))
    await drain()
    assert len(runs) == 1
    delivered = runs[0]
    # Sanitized, not truncated to one turn: the recorded frame carries 7 rows;
    # the tool-pairing rows and the content_items row survive as plain text or
    # are dropped per sanitize_portal_messages, but the CONVERSATION arrives.
    assert len(delivered) > 1, "history collapsed to a synthesized single turn"
    assert delivered[-1]["content"] == frame["text"]
    roles = {m["role"] for m in delivered}
    assert "tool" not in roles, "role:'tool' rows must not reach a plain completions API"
    for m in delivered:
        assert "content_items" not in m, "content_items must be stripped"

    # The payment token reaches the payments slot.
    assert tokens == ["REDACTED.PAYMENT.TOKEN"]


async def test_broadcast_shapes_are_recognized_not_unknown(harness, logs):
    """Both wrapped envelopes classify as BROADCAST (never 'unknown', never
    a command) — the discriminator is `event` key with no top-level type."""
    skill, _runs, _tokens, _drain = harness
    for name in ("broadcast_message_created.json", "emit_to_user_redis.json"):
        kind, _ = skill._normalize(load(FRAMES_DIR / name))
        assert kind == "broadcast", name
    # The local emitToUser shape is a top-level typed event — a command-shaped
    # frame whose type the handler ignores without starting a run (covered by
    # the parametrized test above).
    kind, _ = skill._normalize(load(FRAMES_DIR / "emit_to_user_local.json"))
    assert kind == "command"


async def test_session_created_maps_the_callers_agent_string_and_echoes_no_token(harness):
    skill, _runs, _tokens, _drain = harness
    frame = load(FRAMES_DIR / "session_created.json")
    assert "token" not in frame["session"], (
        "recorded session.created must never carry the caller's token — "
        "re-record if the portal regressed to echoing it"
    )
    skill._session_by_id.clear()
    await skill._handle_message(json.dumps(frame))
    # Keyed on the agent string the platform echoed (the string the daemon
    # sent), never on agent_username.
    assert skill._session_by_id == {"sess_fixture_1": "mini"}


async def test_session_error_logs_at_error_with_the_code(harness, logs):
    skill, runs, _tokens, _drain = harness
    await skill._handle_message(json.dumps(load(FRAMES_DIR / "session_error.json")))
    assert not runs
    errors = [r for r in logs.records if r.levelno >= logging.ERROR]
    assert errors, "a refused credential must be loud, not a healthy-looking idle socket"
    assert "unauthorized" in errors[0].getMessage()


async def test_response_cancel_cancels_the_run_and_acknowledges(harness):
    skill, runs, _tokens, drain = harness

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_run(messages, tools=None):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield {}

    skill.agent.run_streaming = slow_run

    await skill._handle_message(json.dumps(load(FRAMES_DIR / "input_text.json")))
    await asyncio.wait_for(started.wait(), timeout=5)
    await skill._handle_message(json.dumps(load(FRAMES_DIR / "response_cancel.json")))
    await asyncio.wait_for(cancelled.wait(), timeout=5)

    sent = [json.loads(c.args[0]) for c in skill._ws.send.call_args_list]
    assert any(f.get("type") == "response.cancelled" for f in sent), (
        "the portal handles response.cancelled; the daemon must acknowledge"
    )


async def test_unmapped_session_falls_back_to_the_frame_agent(harness):
    """The M5 prerequisite (R9): a per-request session id that never appeared
    in a local session.created must not mute the daemon — the frame's own
    `agent` field routes it. This inverts the old suite's assertion that an
    unknown session id sends nothing."""
    skill, runs, _tokens, drain = harness
    await skill._handle_message(
        json.dumps(load(FRAMES_DIR / "input_text_unmapped_session.json"))
    )
    await drain()
    assert len(runs) == 1
    sent = [json.loads(c.args[0]) for c in skill._ws.send.call_args_list]
    dones = [f for f in sent if f.get("type") == "response.done"]
    assert dones and dones[0]["session_id"] == "sess_perrequest_9"


async def test_outbound_tier_frame_without_session_id_is_ignored(harness):
    """The outbound-WS-tier shape has no session_id; the bridge parser must
    not invent one."""
    skill, runs, _tokens, drain = harness
    await skill._handle_message(json.dumps(load(FRAMES_DIR / "outbound_input_text.json")))
    await drain()
    assert not runs
