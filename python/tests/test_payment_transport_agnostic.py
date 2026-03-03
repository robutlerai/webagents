"""
Tests for transport-agnostic payment token extraction and PaymentTokenRequiredError.

PaymentSkill reads context.payment_token first, then falls back to HTTP headers/query.
"""

import pytest

try:
    import robutler
    HAS_ROBUTLER = True
except ImportError:
    HAS_ROBUTLER = False

if not HAS_ROBUTLER:
    pytest.skip("robutler not installed", allow_module_level=True)

import logging
from unittest.mock import Mock
from webagents.agents.skills.robutler.payments import PaymentSkill
from webagents.agents.skills.robutler.payments.exceptions import PaymentTokenRequiredError


class MockContext:
    """Context with optional payment_token and request."""
    def __init__(self, payment_token=None, request=None):
        self.payment_token = payment_token
        self.request = request


def _make_skill():
    """Create a PaymentSkill with a logger (normally set by initialize)."""
    skill = PaymentSkill()
    skill.logger = logging.getLogger("test.payment")
    return skill


def test_extract_from_context_payment_token():
    """Token set on context.payment_token is found."""
    skill = _make_skill()
    ctx = MockContext(payment_token="tok_abc123")
    assert skill._extract_payment_token(ctx) == "tok_abc123"


def test_extract_from_http_headers_fallback():
    """Falls back to X-Payment-Token header when context.payment_token is None."""
    skill = _make_skill()
    req = Mock()
    req.headers = {"X-Payment-Token": "header_tok"}
    req.query_params = {}
    ctx = MockContext(payment_token=None, request=req)
    assert skill._extract_payment_token(ctx) == "header_tok"


def test_extract_prefers_context_over_headers():
    """context.payment_token takes priority over headers."""
    skill = _make_skill()
    req = Mock()
    req.headers = {"X-Payment-Token": "header_tok"}
    req.query_params = {}
    ctx = MockContext(payment_token="context_tok", request=req)
    assert skill._extract_payment_token(ctx) == "context_tok"


def test_extract_with_no_request_no_context():
    """Returns None gracefully when both are absent."""
    skill = _make_skill()
    ctx = MockContext(payment_token=None, request=None)
    assert skill._extract_payment_token(ctx) is None


def test_extract_query_param_fallback():
    """Falls back to query param payment_token when no header."""
    skill = _make_skill()
    req = Mock()
    req.headers = {}
    req.query_params = {"payment_token": "query_tok"}
    ctx = MockContext(payment_token=None, request=req)
    assert skill._extract_payment_token(ctx) == "query_tok"


def test_payment_required_error_includes_accepts():
    """PaymentTokenRequiredError can carry accepts array (set by skill when raising)."""
    err = PaymentTokenRequiredError(agent_name="test")
    err.context["accepts"] = [{"scheme": "token", "amount": "0.01", "currency": "USD"}]
    assert "accepts" in err.context
    assert err.context["accepts"][0]["amount"] == "0.01"


def test_payment_required_error_has_status_code_402():
    """Error carries status_code=402."""
    err = PaymentTokenRequiredError(agent_name="test")
    assert err.status_code == 402
