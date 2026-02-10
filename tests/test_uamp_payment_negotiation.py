"""
Tests for UAMP transport payment negotiation.

When billing is enabled and no token is provided, the UAMP skill sends payment.required,
waits for payment.submit, sets context.payment_token and retries; sends payment.accepted on success.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from webagents.agents.skills.core.transport.uamp.skill import (
    UAMPTransportSkill,
    UAMPSession,
)


@pytest.fixture
def uamp_skill():
    return UAMPTransportSkill()


@pytest.fixture
def mock_ws():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


def test_uamp_session_has_payment_token_field():
    """UAMPSession supports payment_token for pre-load and retry."""
    session = UAMPSession()
    assert hasattr(session, "payment_token")
    session.payment_token = "tok_abc"
    assert session.payment_token == "tok_abc"


def test_uamp_skill_has_pending_payment_futures(uamp_skill):
    """Skill has _pending_payment_futures for waiting on payment.submit."""
    assert hasattr(uamp_skill, "_pending_payment_futures")
    assert isinstance(uamp_skill._pending_payment_futures, dict)


@pytest.mark.asyncio
async def test_uamp_handle_payment_submit_sets_session_token(uamp_skill, mock_ws):
    """Receiving payment.submit sets session.payment_token and resolves pending future."""
    session_id = "sess_123"
    session = UAMPSession()
    session.id = session_id
    uamp_skill._sessions[session_id] = session

    fut = asyncio.get_event_loop().create_future()
    uamp_skill._pending_payment_futures[session_id] = fut

    event = {
        "type": "payment.submit",
        "payment": {"scheme": "token", "amount": "0.01", "token": "jwt_xyz"},
    }
    await uamp_skill._handle_payment_submit(mock_ws, session, event)

    assert session.payment_token == "jwt_xyz"
    assert fut.done()
    assert fut.result() == "jwt_xyz"
    assert session_id not in uamp_skill._pending_payment_futures


@pytest.mark.asyncio
async def test_uamp_session_update_preloads_token(uamp_skill, mock_ws):
    """session.update with payment_token is stored for subsequent requests."""
    session_id = "sess_456"
    session = UAMPSession()
    session.id = session_id
    uamp_skill._sessions[session_id] = session

    event = {
        "type": "session.update",
        "session": {"payment_token": "preloaded_tok"},
    }
    await uamp_skill._handle_session_update(mock_ws, session, event)

    assert session.payment_token == "preloaded_tok"
