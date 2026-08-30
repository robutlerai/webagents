# PARTIALLY REMOVED (M4 suite triage): the cases that pinned the retired
# litellm/byok settle architecture (LITELLM_AVAILABLE flag,
# PaymentContext.byok_providers, is_byok/byok_llm settles) are gone; the
# current payment flow is covered by test_payment_transport_agnostic.py,
# test_completions_payment_preflight.py and the tests in this file.
#
# The triage over-reached and has been walked back: cases that were NOT about
# the retired architecture are restored below, each marked RESTORED with what
# (if anything) had to be adapted to the current code.
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
    """`resolve_auto_model` / `get_provider_from_model` — RESTORED.

    These were removed with the BYOK settle triage, but they cover neither:
    both functions are live (core/llm/models.py) and unrelated to the retired
    litellm settle architecture. Restored with the assertions derived from
    AUTO_MODEL_MAP / AUTO_PROVIDER_PRIORITY instead of the model IDs the
    original spelled out — those IDs (gemini-2.5-flash, gpt-4.1,
    claude-3-5-sonnet) have all been superseded in the catalog since, so a
    verbatim restore would have pinned a stale catalog rather than the
    resolution logic it is actually about.
    """

    def test_resolve_auto_fastest_google(self):
        from webagents.agents.skills.core.llm.models import AUTO_MODEL_MAP, resolve_auto_model
        assert resolve_auto_model("auto/fastest", ["google"]) == AUTO_MODEL_MAP["auto/fastest"]["google"]

    def test_resolve_auto_smartest_openai(self):
        from webagents.agents.skills.core.llm.models import AUTO_MODEL_MAP, resolve_auto_model
        assert resolve_auto_model("auto/smartest", ["openai"]) == AUTO_MODEL_MAP["auto/smartest"]["openai"]

    def test_resolve_auto_balanced_anthropic(self):
        from webagents.agents.skills.core.llm.models import AUTO_MODEL_MAP, resolve_auto_model
        assert resolve_auto_model("auto/balanced", ["anthropic"]) == AUTO_MODEL_MAP["auto/balanced"]["anthropic"]

    def test_resolve_no_providers(self):
        """No providers returns None"""
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        assert resolve_auto_model("auto/fastest", []) is None
        assert resolve_auto_model("auto/smartest", []) is None

    def test_resolve_unknown_tier_returns_none(self):
        from webagents.agents.skills.core.llm.models import resolve_auto_model
        assert resolve_auto_model("auto/nonexistent", ["openai"]) is None

    def test_resolve_multiple_providers_priority(self):
        """Multiple providers picks per the tier's priority order, not the
        caller's argument order."""
        from webagents.agents.skills.core.llm.models import (
            AUTO_MODEL_MAP,
            AUTO_PROVIDER_PRIORITY,
            resolve_auto_model,
        )
        for tier, priority in AUTO_PROVIDER_PRIORITY.items():
            providers = list(reversed(priority))
            expected = AUTO_MODEL_MAP[tier][priority[0]]
            assert resolve_auto_model(tier, providers) == expected, tier

    def test_get_provider_from_model(self):
        """Provider extraction from model IDs"""
        from webagents.agents.skills.core.llm.models import get_provider_from_model
        assert get_provider_from_model("openai/gpt-4o") == "openai"
        assert get_provider_from_model("anthropic/claude-3-5-sonnet") == "anthropic"
        assert get_provider_from_model("gpt-4o") == "openai"
        assert get_provider_from_model("claude-3-5-sonnet") == "anthropic"
        assert get_provider_from_model("gemini-2.5-flash") == "google"
        assert get_provider_from_model("grok-4") == "xai"

    def test_every_catalog_model_maps_back_to_its_provider(self):
        """Each auto-tier model must be attributable to the provider it is
        listed under — otherwise BYOK picks a key for the wrong provider."""
        from webagents.agents.skills.core.llm.models import (
            AUTO_MODEL_MAP,
            get_provider_from_model,
        )
        for tier, mapping in AUTO_MODEL_MAP.items():
            for provider, model in mapping.items():
                assert get_provider_from_model(model) == provider, f"{tier}:{model}"


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


# [removed: TestByokClaimParsing — see M4 triage note at top]
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


# [removed: TestLiteLLMKeyPriority — see M4 triage note at top]