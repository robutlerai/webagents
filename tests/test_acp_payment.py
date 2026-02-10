"""
Tests for ACP transport payment handling.

JSON-RPC error -32402 with payment requirements when payment required;
retry with payment_token in params succeeds.
"""

import json
import pytest
from unittest.mock import MagicMock

from webagents.agents.skills.core.transport.acp.skill import (
    ACPTransportSkill,
    ACPErrorCode,
)
from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError
from webagents.uamp import (
    ResponseDeltaEvent,
    ResponseDoneEvent,
    ContentDelta,
    ResponseOutput,
    ContentItem,
)


@pytest.fixture
def acp_skill():
    skill = ACPTransportSkill()
    skill._adapter = MagicMock()
    skill._adapter.to_uamp = MagicMock(return_value=[])
    skill._adapter.from_uamp_streaming = MagicMock(return_value=None)
    skill._sessions = {}
    return skill


def test_acp_json_rpc_error_32402():
    """ACP uses JSON-RPC error code -32402 for payment required."""
    assert ACPErrorCode.PAYMENT_REQUIRED == -32402


@pytest.mark.asyncio
async def test_acp_payment_required_yields_jsonrpc_error(acp_skill):
    """When process_uamp raises PaymentTokenRequiredError, SSE yields error with code -32402."""
    err = PaymentTokenRequiredError(agent_name="test-agent")
    err.context["accepts"] = [{"scheme": "token", "amount": "0.01"}]

    async def raise_payment(events):
        raise err
        yield  # make it an async generator

    mock_agent = MagicMock()
    mock_agent.process_uamp = raise_payment
    ctx = MagicMock()
    ctx.agent = mock_agent
    acp_skill.agent = mock_agent
    acp_skill.get_context = MagicMock(return_value=ctx)

    chunks = []
    async for chunk in acp_skill._handle_session_prompt_streaming(
        "req-1", {"sessionId": "s1", "prompt": []}
    ):
        chunks.append(chunk)

    # Find the error chunk (raw SSE data: ...)
    error_chunks = [c for c in chunks if "error" in c]
    assert len(error_chunks) >= 1

    # Parse the JSON-RPC error
    raw = error_chunks[0]
    data_str = raw.replace("data: ", "").strip()
    obj = json.loads(data_str)
    assert obj["jsonrpc"] == "2.0"
    assert obj["id"] == "req-1"
    assert obj["error"]["code"] == ACPErrorCode.PAYMENT_REQUIRED
    assert "data" in obj["error"]
    assert obj["error"]["data"]["accepts"] == [{"scheme": "token", "amount": "0.01"}]


@pytest.mark.asyncio
async def test_acp_payment_token_set_from_params(acp_skill):
    """When params include payment_token, context.payment_token is set before process_uamp."""
    async def stream_ok(events):
        yield ResponseDeltaEvent(
            response_id="r1",
            delta=ContentDelta(type="text", text="OK"),
        )
        yield ResponseDoneEvent(
            response_id="r1",
            response=ResponseOutput(
                id="r1",
                status="completed",
                output=[ContentItem(type="text", text="OK")],
            ),
        )

    mock_agent = MagicMock()
    mock_agent.process_uamp = stream_ok
    ctx = MagicMock()
    ctx.agent = mock_agent
    acp_skill.agent = mock_agent
    acp_skill.get_context = MagicMock(return_value=ctx)

    chunks = []
    async for chunk in acp_skill._handle_session_prompt_streaming(
        "req-2",
        {"sessionId": "s2", "prompt": [], "payment_token": "tok_retry"},
    ):
        chunks.append(chunk)

    # payment_token should have been set on context
    assert ctx.payment_token == "tok_retry"

    # Should have streamed response chunks (notifications) + final response, not error
    error_chunks = [c for c in chunks if str(ACPErrorCode.PAYMENT_REQUIRED) in c]
    assert len(error_chunks) == 0

    # Last chunk should be the stop reason response
    last_data = json.loads(chunks[-1].replace("data: ", "").strip())
    assert last_data.get("result", {}).get("stopReason") == "end_turn"


@pytest.mark.asyncio
async def test_acp_internal_error_does_not_leak_payment_code(acp_skill):
    """Non-payment exceptions yield INTERNAL_ERROR, not PAYMENT_REQUIRED."""
    async def raise_generic(events):
        raise RuntimeError("Something broke")
        yield

    mock_agent = MagicMock()
    mock_agent.process_uamp = raise_generic
    ctx = MagicMock()
    ctx.agent = mock_agent
    acp_skill.agent = mock_agent
    acp_skill.get_context = MagicMock(return_value=ctx)

    chunks = []
    async for chunk in acp_skill._handle_session_prompt_streaming(
        "req-3", {"sessionId": "s3", "prompt": []}
    ):
        chunks.append(chunk)

    error_chunks = [c for c in chunks if "error" in c]
    assert len(error_chunks) >= 1
    obj = json.loads(error_chunks[0].replace("data: ", "").strip())
    assert obj["error"]["code"] == ACPErrorCode.INTERNAL_ERROR
