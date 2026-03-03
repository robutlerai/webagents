"""
Tests for daemon BYOK (Bring Your Own Key) integration.

Covers:
- Auto model resolution (models.py)
- BYOK claim parsing from JWT (PaymentSkill)
- LiteLLM key priority (LiteLLMSkill._get_api_key_for_model)
"""

import base64
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ===== Auto model resolution (models.py) =====

class TestAutoModelResolution:
    """Test resolve_auto_model and get_provider_from_model from models.py"""

    def test_resolve_auto_fastest_google(self):
        """auto/fastest with ["google"] -> gemini-2.5-flash"""
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        result = resolve_auto_model("auto/fastest", ["google"])
        assert result == "gemini-2.5-flash"

    def test_resolve_auto_smartest_openai(self):
        """auto/smartest with ["openai"] -> gpt-4.1"""
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        result = resolve_auto_model("auto/smartest", ["openai"])
        assert result == "gpt-4.1"

    def test_resolve_auto_balanced_anthropic(self):
        """auto/balanced with ["anthropic"] -> claude-3-5-sonnet"""
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        result = resolve_auto_model("auto/balanced", ["anthropic"])
        assert result == "claude-3-5-sonnet"

    def test_resolve_no_providers(self):
        """No providers returns None"""
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        assert resolve_auto_model("auto/fastest", []) is None
        assert resolve_auto_model("auto/smartest", []) is None

    def test_resolve_multiple_providers_priority(self):
        """Multiple providers picks per priority order"""
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        # auto/fastest priority: google, openai, anthropic
        result = resolve_auto_model("auto/fastest", ["anthropic", "openai", "google"])
        assert result == "gemini-2.5-flash"  # google first
        # auto/smartest priority: anthropic, openai, google
        result = resolve_auto_model("auto/smartest", ["google", "openai", "anthropic"])
        assert result == "claude-3-5-sonnet"  # anthropic first

    def test_get_provider_from_model(self):
        """Provider extraction from model IDs"""
        from webagents.agents.skills.core.llm.models import get_provider_from_model
        assert get_provider_from_model("openai/gpt-4o") == "openai"
        assert get_provider_from_model("anthropic/claude-3-5-sonnet") == "anthropic"
        assert get_provider_from_model("gpt-4o") == "openai"
        assert get_provider_from_model("claude-3-5-sonnet") == "anthropic"
        assert get_provider_from_model("gemini-2.5-flash") == "google"
        assert get_provider_from_model("grok-4") == "xai"


# ===== BYOK claim parsing (PaymentSkill) =====

def _make_jwt(claims: dict) -> str:
    """Create a minimal JWT with given claims (header.payload.sig)."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b'=').decode()
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b'=').decode()
    return f"{header}.{payload}.sig"


@pytest.fixture
def payment_skill():
    """PaymentSkill instance for BYOK tests. Skips if robutler not installed."""
    robutler = pytest.importorskip("robutler")
    from webagents.agents.skills.robutler.payments.skill import PaymentSkill
    skill = PaymentSkill({
        "enable_billing": True,
        "minimum_balance": 0.01,
        "per_message_lock": 0.005,
    })
    skill.agent = MagicMock(name="test-agent")
    skill.logger = MagicMock()
    return skill


class TestByokClaimParsing:
    """Test BYOK claim parsing from JWT in PaymentSkill.setup_payment_context"""

    @pytest.mark.asyncio
    async def test_byok_claim_parsed_from_jwt(self, payment_skill):
        """Mock JWT with byok claim, verify context.byok_providers is set"""
        jwt = _make_jwt({"byok": ["openai", "google"], "sub": "user-123"})

        mock_client = MagicMock()
        mock_client.tokens.validate_with_balance = AsyncMock(
            return_value={"valid": True, "balance": 1.0}
        )
        mock_client.tokens.lock = AsyncMock(
            return_value={"lockId": "lock-1", "lockedAmountDollars": 0.005}
        )
        payment_skill.client = mock_client

        context = MagicMock()
        context.payment_token = jwt
        context.auth = MagicMock(user_id="user-123", agent_id=None)
        context.request = None

        with patch.object(payment_skill, "_extract_payment_token", return_value=jwt):
            result = await payment_skill.setup_payment_context(context)

        assert result.byok_providers == ["openai", "google"]
        assert result.byok_user_id == "user-123"

    @pytest.mark.asyncio
    async def test_missing_byok_claim(self, payment_skill):
        """Missing byok claim results in empty providers list"""
        jwt = _make_jwt({"sub": "user-456"})  # no byok claim

        mock_client = MagicMock()
        mock_client.tokens.validate_with_balance = AsyncMock(
            return_value={"valid": True, "balance": 1.0}
        )
        mock_client.tokens.lock = AsyncMock(
            return_value={"lockId": "lock-2", "lockedAmountDollars": 0.005}
        )
        payment_skill.client = mock_client

        # Use plain object so byok_providers is not auto-created when unset
        class SimpleContext:
            pass

        context = SimpleContext()
        context.payment_token = jwt
        context.auth = MagicMock(user_id="user-456", agent_id=None)
        context.request = None

        with patch.object(payment_skill, "_extract_payment_token", return_value=jwt):
            result = await payment_skill.setup_payment_context(context)

        assert getattr(result, "byok_providers", []) == []


# ===== LiteLLM key priority (LiteLLMSkill._get_api_key_for_model) =====

@pytest.fixture
def litellm_skill():
    """LiteLLMSkill instance for key priority tests."""
    pytest.importorskip("litellm")
    from webagents.agents.skills.core.llm.litellm.skill import LiteLLMSkill
    return LiteLLMSkill({"model": "gpt-4o-mini"})


def _get_litellm_skill():
    """Import and return LiteLLMSkill for tests that need a fresh instance."""
    from webagents.agents.skills.core.llm.litellm.skill import LiteLLMSkill
    return LiteLLMSkill


class TestLiteLLMKeyPriority:
    """Test LiteLLM _get_api_key_for_model key selection and flags"""

    def test_user_byok_key_takes_priority(self, litellm_skill):
        """When context.byok_keys has a provider key, it's used over env"""
        context = MagicMock()
        context.byok_keys = {
            "openai": {"key": "byok-openai-key", "tokenId": "tk-123"},
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=False):
            key = litellm_skill._get_api_key_for_model("gpt-4o", context)
        assert key == "byok-openai-key"
        assert context.is_byok is True
        assert context.is_agent_key is False

    def test_agent_key_detected(self, litellm_skill):
        """When agent config key differs from env, is_agent_key is set"""
        LiteLLMSkill = _get_litellm_skill()
        skill = LiteLLMSkill({
            "model": "gpt-4o-mini",
            "api_keys": {"openai": "agent-configured-key"},
        })
        context = MagicMock()
        context.byok_keys = None
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            key = skill._get_api_key_for_model("gpt-4o", context)
        assert key == "agent-configured-key"
        assert context.is_byok is False
        assert context.is_agent_key is True

    def test_flags_are_mutually_exclusive(self, litellm_skill):
        """is_byok and is_agent_key can't both be true"""
        context = MagicMock()
        context.byok_keys = {"openai": {"key": "byok-key", "tokenId": "tk-1"}}
        litellm_skill._get_api_key_for_model("gpt-4o", context)
        assert context.is_byok is True
        assert context.is_agent_key is False
        assert not (context.is_byok and context.is_agent_key)

        context2 = MagicMock()
        context2.byok_keys = {}
        LiteLLMSkill = _get_litellm_skill()
        skill_with_agent_key = LiteLLMSkill({
            "model": "gpt-4o-mini",
            "api_keys": {"openai": "agent-key"},
        })
        with patch.dict(os.environ, {"OPENAI_API_KEY": "different-env-key"}, clear=False):
            skill_with_agent_key._get_api_key_for_model("gpt-4o", context2)
        assert context2.is_byok is False
        assert context2.is_agent_key is True
        assert not (context2.is_byok and context2.is_agent_key)
