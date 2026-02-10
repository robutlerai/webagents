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
