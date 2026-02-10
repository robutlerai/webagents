"""
Tests for A2A transport payment handling.

When payment is required, task returns failed status with payment info in error;
retry with X-Payment-Token header succeeds.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from webagents.agents.skills.core.transport.a2a.skill import A2ATransportSkill, TaskState
from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
from webagents.uamp import (
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ContentDelta,
    ResponseOutput,
    ContentItem,
)


@pytest.fixture
def a2a_skill():
    skill = A2ATransportSkill()
    skill._adapter = MagicMock()
    skill._adapter.to_uamp = MagicMock(return_value=[])
    skill._adapter.from_uamp_streaming = MagicMock(return_value=None)
    return skill


def _parse_sse(raw: str):
    """Parse SSE event string into (event_type, data_dict)."""
    lines = raw.strip().split("\n")
    event_type = None
    data_str = None
    for line in lines:
        if line.startswith("event: "):
            event_type = line[len("event: "):]
        elif line.startswith("data: "):
            data_str = line[len("data: "):]
    return event_type, json.loads(data_str) if data_str else None


@pytest.mark.asyncio
async def test_a2a_task_failed_on_payment_required(a2a_skill):
    """When process_uamp raises PaymentTokenRequiredError, task.failed SSE contains payment info."""
    err = PaymentTokenRequiredError(agent_name="test-agent")
    err.context["accepts"] = [{"scheme": "x402", "amount": "0.01"}]

    async def raise_payment(events):
        raise err
        yield  # make it an async generator

    mock_agent = MagicMock()
    mock_agent.process_uamp = raise_payment
    ctx = MagicMock()
    ctx.agent = mock_agent
    ctx.request = None
    a2a_skill.agent = mock_agent
    a2a_skill.get_context = MagicMock(return_value=ctx)

    events = []
    async for chunk in a2a_skill.create_task(message={"role": "user", "parts": [{"type": "text", "text": "hi"}]}):
        events.append(chunk)

    # Should have task.started then task.failed
    assert len(events) >= 2

    # Find task.failed
    failed_events = [(t, d) for t, d in ((_parse_sse(e)) for e in events) if t == "task.failed"]
    assert len(failed_events) == 1
    _, data = failed_events[0]
    assert data["status"] == "failed"
    assert data["code"] == "payment_required"
    assert data["status_code"] == 402
    assert data["accepts"] == [{"scheme": "x402", "amount": "0.01"}]


@pytest.mark.asyncio
async def test_a2a_task_completed_with_payment_token(a2a_skill):
    """When context has payment_token, process_uamp streams and task completes."""
    async def stream_ok(events):
        yield ResponseDeltaEvent(
            response_id="r1",
            delta=ContentDelta(type="text", text="Hello"),
        )
        yield ResponseDoneEvent(
            response_id="r1",
            response=ResponseOutput(
                id="r1",
                status="completed",
                output=[ContentItem(type="text", text="Hello")],
            ),
        )

    mock_agent = MagicMock()
    mock_agent.process_uamp = stream_ok

    mock_request = MagicMock()
    mock_request.headers = {"X-Payment-Token": "tok_abc"}
    ctx = MagicMock()
    ctx.agent = mock_agent
    ctx.request = mock_request
    a2a_skill.agent = mock_agent
    a2a_skill.get_context = MagicMock(return_value=ctx)

    events = []
    async for chunk in a2a_skill.create_task(message={"role": "user", "parts": [{"type": "text", "text": "hi"}]}):
        events.append(chunk)

    # payment_token was set on context from header
    assert ctx.payment_token == "tok_abc"

    # Last event should be task.completed
    last_type, last_data = _parse_sse(events[-1])
    assert last_type == "task.completed"
    assert last_data["status"] == "completed"


@pytest.mark.asyncio
async def test_a2a_payment_error_shape():
    """PaymentTokenRequiredError has status_code 402 and proper context shape."""
    err = PaymentTokenRequiredError(agent_name="foo")
    err.context["accepts"] = [{"scheme": "token", "amount": "0.01"}]
    assert err.status_code == 402
    assert err.error_code == "PAYMENT_TOKEN_REQUIRED"
    assert "accepts" in err.context
