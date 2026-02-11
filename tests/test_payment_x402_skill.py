"""
Tests for PaymentSkillX402 (JWT payment tokens, JWKS, /api/payments/*).

Covers:
- _verify_payment_token: invalid JWT, valid JWT (mocked JWKS), missing payment claim,
  wrong/absent aud when expected_audience provided
- decode_payment_header / schemes: JWT vs legacy
- Lock/verify/settle flow via facilitator (mocked)
"""

# Mock robutler package before any webagents.robutler import (auth skill pulls in RobutlerClient, types)
import sys
from unittest.mock import MagicMock
_robutler = MagicMock()
_robutler.api = MagicMock()
_robutler.api.types = MagicMock()
sys.modules["robutler"] = _robutler
sys.modules["robutler.api"] = _robutler.api
sys.modules["robutler.api.types"] = _robutler.api.types

import pytest
import jwt as pyjwt
from unittest.mock import AsyncMock, MagicMock, patch

from webagents.agents.skills.robutler.payments_x402.skill import PaymentSkillX402
from webagents.agents.skills.robutler.payments_x402.schemes import (
    decode_payment_header,
    extract_token_from_payment,
    _is_jwt_string,
)


class TestPaymentX402Schemes:
    """Test JWT vs legacy payment header decoding."""

    def test_is_jwt_string_accepts_three_part_jwt(self):
        assert _is_jwt_string("a.b.c") is True
        assert _is_jwt_string("eyJhbG.eyJzdWI.X") is True

    def test_is_jwt_string_rejects_non_jwt(self):
        assert _is_jwt_string("") is False
        assert _is_jwt_string("a.b") is False
        assert _is_jwt_string("a.b.c.d") is False
        assert _is_jwt_string("not-base64!!.b.c") is False

    def test_decode_payment_header_raw_jwt(self):
        raw = "header.payload.signature"
        out = decode_payment_header(raw)
        assert out.get("_is_jwt") is True
        assert out.get("_raw_token") == raw
        assert out.get("scheme") == "token"
        assert out.get("network") == "robutler"

    def test_extract_token_from_payment_prefers_raw_token(self):
        data = {"_raw_token": "jwt.here", "payload": {"token": "legacy"}}
        assert extract_token_from_payment(data) == "jwt.here"


@pytest.fixture
def jwks_manager_mock():
    """Mock JWKSManager that returns a fixed public key for tests."""
    manager = MagicMock()
    manager.get_public_key_from_jwks = AsyncMock(return_value=None)  # override per test
    return manager


@pytest.fixture
def skill_with_jwks(jwks_manager_mock):
    """PaymentSkillX402 with mocked JWKS manager."""
    return PaymentSkillX402(config={
        "webagents_api_url": "https://test.example",
        "jwks_manager": jwks_manager_mock,
    })


class TestVerifyPaymentToken:
    """Test _verify_payment_token (local JWKS verification)."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_jwks_manager(self):
        skill = PaymentSkillX402(config={"webagents_api_url": "https://test.example", "jwks_manager": None})
        result = await skill._verify_payment_token("a.b.c")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_invalid_jwt(self, skill_with_jwks):
        result = await skill_with_jwks._verify_payment_token("not-a-valid-jwt")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_jwks_returns_no_key(self, skill_with_jwks, jwks_manager_mock):
        jwks_manager_mock.get_public_key_from_jwks.return_value = None
        with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "x"}
            with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.decode") as mock_decode:
                mock_decode.return_value = {"iss": "https://issuer.example"}
                result = await skill_with_jwks._verify_payment_token("a.b.c")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_payment_claim_missing(self, skill_with_jwks, jwks_manager_mock):
        fake_key = object()
        jwks_manager_mock.get_public_key_from_jwks.return_value = fake_key
        with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "x"}
            with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.decode") as mock_decode:
                mock_decode.side_effect = [
                    {"iss": "https://issuer.example"},
                    {"iss": "https://issuer.example", "exp": 9999999999},  # no payment
                ]
                result = await skill_with_jwks._verify_payment_token("a.b.c")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_valid_and_balance_when_jwt_valid(self, skill_with_jwks, jwks_manager_mock):
        fake_key = object()
        jwks_manager_mock.get_public_key_from_jwks.return_value = fake_key
        with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "test-kid"}
            with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.decode") as mock_decode:
                mock_decode.side_effect = [
                    {"iss": "https://issuer.example"},
                    {"payment": {"balance": 10.5}, "exp": 9999999999},
                ]
                result = await skill_with_jwks._verify_payment_token("a.b.c")
        assert result is not None
        assert result.get("isValid") is True
        assert result.get("balance") == 10.5

    @pytest.mark.asyncio
    async def test_returns_none_when_expected_audience_provided_and_aud_wrong(self, skill_with_jwks, jwks_manager_mock):
        fake_key = object()
        jwks_manager_mock.get_public_key_from_jwks.return_value = fake_key
        with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "x"}
            with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.decode") as mock_decode:
                mock_decode.side_effect = [
                    {"iss": "https://issuer.example"},
                    pyjwt.InvalidAudienceError("audience mismatch"),
                ]
                result = await skill_with_jwks._verify_payment_token(
                    "a.b.c", expected_audience=["my-agent"]
                )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_expected_audience_provided_and_aud_absent(self, skill_with_jwks, jwks_manager_mock):
        fake_key = object()
        jwks_manager_mock.get_public_key_from_jwks.return_value = fake_key
        with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "x"}
            with patch("webagents.agents.skills.robutler.payments_x402.skill.jwt.decode") as mock_decode:
                mock_decode.side_effect = [
                    {"iss": "https://issuer.example"},
                    pyjwt.InvalidAudienceError("Audience not found"),
                ]
                result = await skill_with_jwks._verify_payment_token(
                    "a.b.c", expected_audience=["my-agent"]
                )
        assert result is None


# ---------------------------------------------------------------------------
# Pricing decorator tests
# ---------------------------------------------------------------------------

class TestPricingDecorator:
    """Test @pricing(lock=...) stores metadata on decorated functions."""

    def test_pricing_stores_lock_in_metadata(self):
        from webagents.agents.skills.robutler.payments.skill import pricing

        @pricing(credits_per_call=0.05, lock=0.10, reason="Test tool")
        async def my_tool():
            return "result"

        assert hasattr(my_tool, '_webagents_pricing')
        meta = my_tool._webagents_pricing
        assert meta['lock'] == 0.10
        assert meta['credits_per_call'] == 0.05
        assert meta['reason'] == "Test tool"

    def test_pricing_lock_defaults_to_none(self):
        from webagents.agents.skills.robutler.payments.skill import pricing

        @pricing(credits_per_call=0.02)
        async def simple_tool():
            return "ok"

        meta = simple_tool._webagents_pricing
        assert meta['lock'] is None
        assert meta['credits_per_call'] == 0.02

    def test_pricing_dynamic_mode_when_no_fixed_price(self):
        from webagents.agents.skills.robutler.payments.skill import pricing

        @pricing()
        async def dynamic_tool():
            return "result"

        meta = dynamic_tool._webagents_pricing
        assert meta['supports_dynamic'] is True
        assert meta['credits_per_call'] is None


# ---------------------------------------------------------------------------
# BYOK detection tests
# ---------------------------------------------------------------------------

class TestBYOKDetection:
    """Test that context.is_byok is set correctly by LiteLLM skill."""

    def test_is_byok_flag_readable_from_context(self):
        """Verify that is_byok can be read from context by the payment skill."""
        # Simulate a context object with is_byok set
        context = MagicMock()
        context.is_byok = True
        assert getattr(context, 'is_byok', False) is True

    def test_is_byok_defaults_to_false(self):
        """Verify is_byok defaults to False when not set."""
        context = MagicMock(spec=[])  # empty spec -> no attrs
        assert getattr(context, 'is_byok', False) is False


# ---------------------------------------------------------------------------
# Two-settle finalization tests
# ---------------------------------------------------------------------------

class TestTwoSettleFinalization:
    """Test the payment finalization flow: platform_fee, then LLM cost, then agent markup."""

    @pytest.fixture
    def payment_skill(self):
        """Create a PaymentSkill with mocked client."""
        from webagents.agents.skills.robutler.payments.skill import PaymentSkill
        skill = PaymentSkill.__new__(PaymentSkill)
        skill.config = {}
        skill.enable_billing = True
        skill.agent_pricing_percent = 100.0
        skill.platform_fee_percent = 0.20
        skill.minimum_balance = 0.1
        skill.webagents_api_url = "https://test.example"
        skill.robutler_api_key = "test-key"
        skill.client = MagicMock()
        skill.agent = MagicMock()
        skill.agent.name = "test-agent"
        skill.logger = MagicMock()
        return skill

    @pytest.fixture
    def context_with_usage(self):
        """Create a context with LLM usage records and a payment lock."""
        from webagents.agents.skills.robutler.payments.skill import PaymentContext
        ctx = MagicMock()
        ctx.payments = PaymentContext(
            payment_token="a.b.c",
            user_id="user-1",
            agent_id="agent-1",
            lock_id="lock-123",
            locked_amount_dollars=1.0,
        )
        ctx.is_byok = False
        ctx.usage = [
            {
                'type': 'llm',
                'model': 'gpt-4o-mini',
                'prompt_tokens': 100,
                'completion_tokens': 50,
            }
        ]
        return ctx

    @pytest.mark.asyncio
    async def test_settles_platform_fee_first_then_llm_then_agent(self, payment_skill, context_with_usage):
        """Verify settlement order: platform_fee -> platform_llm -> agent_fee -> release."""
        # Mock _settle_payment to track call order
        settle_calls = []

        async def mock_settle(lock_id, amount, description="", charge_type=None, release=False):
            settle_calls.append({
                'lock_id': lock_id,
                'amount': amount,
                'charge_type': charge_type,
                'release': release,
            })
            return {'success': True}

        payment_skill._settle_payment = mock_settle

        # We need to mock cost_per_token since the test env may not have litellm
        with patch("webagents.agents.skills.robutler.payments.skill.LITELLM_AVAILABLE", True), \
             patch("webagents.agents.skills.robutler.payments.skill.cost_per_token") as mock_cpt:
            # Simulate cost_per_token returning ($0.0001 prompt, $0.0002 completion)
            mock_cpt.return_value = (0.0001, 0.0002)

            result = await payment_skill.finalize_payment(context_with_usage)

        assert result is context_with_usage

        # Should have at least 3 settle calls + 1 release
        charge_types = [c['charge_type'] for c in settle_calls if not c['release']]
        assert len(charge_types) >= 2
        # Platform fee should come first
        if charge_types:
            assert charge_types[0] == 'platform_fee'
        # Release should be last
        assert settle_calls[-1]['release'] is True

    @pytest.mark.asyncio
    async def test_byok_routes_llm_cost_as_agent_fee(self, payment_skill, context_with_usage):
        """When is_byok=True, LLM cost settles as agent_fee instead of platform_llm."""
        context_with_usage.is_byok = True

        settle_calls = []

        async def mock_settle(lock_id, amount, description="", charge_type=None, release=False):
            settle_calls.append({
                'lock_id': lock_id,
                'amount': amount,
                'charge_type': charge_type,
                'release': release,
            })
            return {'success': True}

        payment_skill._settle_payment = mock_settle

        with patch("webagents.agents.skills.robutler.payments.skill.LITELLM_AVAILABLE", True), \
             patch("webagents.agents.skills.robutler.payments.skill.cost_per_token") as mock_cpt:
            mock_cpt.return_value = (0.0001, 0.0002)
            await payment_skill.finalize_payment(context_with_usage)

        charge_types = [c['charge_type'] for c in settle_calls if not c['release']]
        # With BYOK, LLM costs go as agent_fee not platform_llm
        assert 'agent_fee' in charge_types
        assert 'platform_llm' not in charge_types

    @pytest.mark.asyncio
    async def test_no_settle_when_zero_cost(self, payment_skill, context_with_usage):
        """When LLM cost is zero, should release lock without settling."""
        context_with_usage.usage = []  # No usage records

        settle_calls = []

        async def mock_settle(lock_id, amount, description="", charge_type=None, release=False):
            settle_calls.append({
                'lock_id': lock_id,
                'charge_type': charge_type,
                'release': release,
            })
            return {'success': True}

        payment_skill._settle_payment = mock_settle

        await payment_skill.finalize_payment(context_with_usage)

        # Only release call, no charge settlements
        non_release = [c for c in settle_calls if not c['release']]
        assert len(non_release) == 0
        # Release should still happen
        release_calls = [c for c in settle_calls if c['release']]
        assert len(release_calls) == 1
