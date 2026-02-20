"""
Inbound trust check tests.

Tests that the AuthSkill's validate_request hook correctly evaluates
accept_from rules against incoming agent tokens.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from webagents.trust import evaluate_trust_rules, extract_trust_labels


class TestInboundTrustEvaluation:
    """Test the trust evaluation logic as it would run in validate_request."""

    def test_everyone_allows_any_agent(self):
        accept_from = ["everyone"]
        assert evaluate_trust_rules("bob.bot", "alice.agent", accept_from) is True

    def test_everyone_allows_external_agent(self):
        accept_from = ["everyone"]
        assert evaluate_trust_rules("com.example.bot", "alice.agent", accept_from) is True

    def test_platform_denies_external_agent(self):
        accept_from = ["platform"]
        assert evaluate_trust_rules("com.example.bot", "alice.agent", accept_from) is False

    def test_platform_allows_platform_agent(self):
        accept_from = ["platform"]
        assert evaluate_trust_rules("bob.agent", "alice.agent", accept_from) is True

    def test_family_allows_same_namespace(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("alice.bot2", "alice.agent", accept_from) is True

    def test_family_denies_different_namespace(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("bob.bot", "alice.agent", accept_from) is False

    def test_family_allows_parent(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("alice", "alice.agent", accept_from) is True

    def test_family_allows_child(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("alice.agent.helper", "alice.agent", accept_from) is True

    def test_family_allows_deep_descendant(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("alice.bot.helper", "alice.agent", accept_from) is True

    def test_family_denies_external_same_tld(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("com.other.bot", "com.example.agent", accept_from) is False

    def test_family_allows_external_same_sld(self):
        accept_from = ["family"]
        assert evaluate_trust_rules("com.example.bot1", "com.example.bot2", accept_from) is True

    def test_nobody_blocks_all(self):
        accept_from = ["nobody"]
        assert evaluate_trust_rules("alice", "alice.agent", accept_from) is False
        assert evaluate_trust_rules("alice.bot", "alice.agent", accept_from) is False

    def test_deny_takes_precedence(self):
        accept_from = {"allow": ["everyone"], "deny": ["@spammer"]}
        assert evaluate_trust_rules("spammer", "alice.agent", accept_from) is False
        assert evaluate_trust_rules("bob", "alice.agent", accept_from) is True

    def test_deny_pattern(self):
        accept_from = {"allow": ["everyone"], "deny": ["@com.evil.**"]}
        assert evaluate_trust_rules("com.evil.bot", "alice.agent", accept_from) is False
        assert evaluate_trust_rules("com.good.bot", "alice.agent", accept_from) is True

    def test_verified_label_from_trusted_issuer(self):
        accept_from = ["#verified"]
        labels = extract_trust_labels(
            "read write trust:verified",
            issuer="https://robutler.ai",
        )
        assert evaluate_trust_rules("bob.agent", "alice.agent", accept_from, labels) is True

    def test_verified_label_from_untrusted_issuer(self):
        accept_from = ["#verified"]
        labels = extract_trust_labels(
            "read write trust:verified",
            issuer="https://evil.example.com",
        )
        assert evaluate_trust_rules("bob.agent", "alice.agent", accept_from, labels) is False

    def test_reputation_threshold(self):
        accept_from = ["#reputation:500"]
        labels = {"trust:reputation-750"}
        assert evaluate_trust_rules("bob", "alice.agent", accept_from, labels) is True

    def test_reputation_below_threshold(self):
        accept_from = ["#reputation:1000"]
        labels = {"trust:reputation-750"}
        assert evaluate_trust_rules("bob", "alice.agent", accept_from, labels) is False

    def test_mixed_rules_family_or_verified(self):
        accept_from = ["family", "#verified"]
        labels = {"trust:verified"}

        assert evaluate_trust_rules("alice.bot", "alice.agent", accept_from) is True
        assert evaluate_trust_rules("bob.bot", "alice.agent", accept_from, labels) is True
        assert evaluate_trust_rules("bob.bot", "alice.agent", accept_from) is False

    def test_no_rules_denies_all(self):
        assert evaluate_trust_rules("alice.bot", "alice.agent", []) is False

    def test_null_rules_handled(self):
        """When accept_from is None, the server should apply defaults (not tested here)."""
        pass

    def test_external_agent_self_issued_no_platform_labels(self):
        """External agent tokens may carry trust labels but they're ignored by default."""
        accept_from = ["#verified"]
        labels = extract_trust_labels(
            "read write trust:verified",
            issuer="https://example.com",
        )
        assert evaluate_trust_rules("com.example.bot", "alice.agent", accept_from, labels) is False

    def test_external_agent_with_domain_pattern(self):
        accept_from = ["@com.example.**"]
        assert evaluate_trust_rules("com.example.agents.bot1", "alice.agent", accept_from) is True
        assert evaluate_trust_rules("com.other.bot", "alice.agent", accept_from) is False
