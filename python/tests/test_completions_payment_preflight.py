"""
Tests for HTTP Completions 402 pre-flight.

When payment is required, the first chunk from execute_handoff raises PaymentTokenRequiredError;
the server returns 402 JSON. PaymentTokenRequiredError carries status_code and context.accepts.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch

from webagents.agents.skills.core.transport.completions.skill import CompletionsTransportSkill
from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError


@pytest.fixture
def skill():
    return CompletionsTransportSkill()


@pytest.mark.asyncio
async def test_completions_returns_402_on_payment_required(skill):
    """When execute_handoff raises PaymentTokenRequiredError, stream raises before yielding 200 SSE."""
    err = PaymentTokenRequiredError(agent_name="test")
    err.context["accepts"] = [{"scheme": "token", "amount": "0.01"}]

    async def raise_payment(*args, **kwargs):
        raise err
        yield  # make it an async generator

    with patch.object(skill, "execute_handoff", new=raise_payment):
        gen = skill.chat_completions(messages=[{"role": "user", "content": "Hi"}], stream=True)
        with pytest.raises(PaymentTokenRequiredError) as exc_info:
            async for _ in gen:
                pass
        assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_completions_402_body_has_requirements():
    """PaymentTokenRequiredError has status_code 402 and can carry context.accepts."""
    err = PaymentTokenRequiredError(agent_name="test")
    err.context["accepts"] = [{"scheme": "token", "amount": "0.01", "currency": "USD"}]
    assert err.status_code == 402
    assert "accepts" in err.context
    assert err.context["accepts"][0]["amount"] == "0.01"


@pytest.mark.asyncio
async def test_completions_retry_with_header_succeeds(skill):
    """With valid token (no raise), stream yields SSE chunks."""
    async def stream_ok(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "Hi"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    with patch.object(skill, "execute_handoff", new=stream_ok):
        gen = skill.chat_completions(messages=[{"role": "user", "content": "Hi"}], stream=True)
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert len(chunks) >= 2
        assert "data:" in chunks[0]
        assert "[DONE]" in chunks[-1]
