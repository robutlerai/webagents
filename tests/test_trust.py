"""
Trust matching utility tests.

Tests for the webagents.trust module:
- Preset matching (everyone, nobody, family, platform)
- Glob pattern matching (@alice.*, @alice.**)
- Trust label matching (#verified, #reputation:N)
- Allow/deny rules with deny-takes-precedence
- Namespace derivation and external agent naming
"""

import pytest
from webagents.trust import (
    evaluate_trust_rules,
    extract_trust_labels,
    derive_namespace,
    is_same_namespace,
    is_reserved_tld,
    url_to_reversed_domain_name,
)


class TestPresets:
    def test_everyone_allows_any(self):
        assert evaluate_trust_rules("alice", "bob", ["everyone"]) is True
        assert evaluate_trust_rules("com.example.bot", "alice", ["everyone"]) is True

    def test_nobody_denies_all(self):
        assert evaluate_trust_rules("alice", "bob", ["nobody"]) is False
        assert evaluate_trust_rules("alice.bot", "alice", ["nobody"]) is False

    def test_family_same_namespace(self):
        assert evaluate_trust_rules("alice.bot1", "alice.bot2", ["family"]) is True
        assert evaluate_trust_rules("alice", "alice.bot", ["family"]) is True
        assert evaluate_trust_rules("alice.a.b", "alice.c", ["family"]) is True

    def test_family_different_namespace(self):
        assert evaluate_trust_rules("bob.bot", "alice.bot", ["family"]) is False
        assert evaluate_trust_rules("bob", "alice", ["family"]) is False

    def test_family_external_agents(self):
        assert evaluate_trust_rules(
            "com.example.bot1", "com.example.bot2", ["family"]
        ) is True
        assert evaluate_trust_rules(
            "com.example.bot", "com.other.bot", ["family"]
        ) is False

    def test_platform_preset(self):
        assert evaluate_trust_rules("alice.bot", "my-agent", ["platform"]) is True
        assert evaluate_trust_rules("bob", "my-agent", ["platform"]) is True
        # External agent (TLD root) is NOT platform
        assert evaluate_trust_rules("com.example.bot", "my-agent", ["platform"]) is False


class TestPatterns:
    def test_exact_match(self):
        assert evaluate_trust_rules("alice", "bob", ["@alice"]) is True
        assert evaluate_trust_rules("bob", "alice", ["@alice"]) is False

    def test_single_star_one_level(self):
        assert evaluate_trust_rules("alice.bot1", "my-agent", ["@alice.*"]) is True
        assert evaluate_trust_rules("alice.bot1.helper", "my-agent", ["@alice.*"]) is False

    def test_double_star_any_depth(self):
        assert evaluate_trust_rules("alice.bot1", "my-agent", ["@alice.**"]) is True
        assert evaluate_trust_rules("alice.bot1.helper", "my-agent", ["@alice.**"]) is True
        assert evaluate_trust_rules("alice.a.b.c", "my-agent", ["@alice.**"]) is True
        assert evaluate_trust_rules("bob.bot", "my-agent", ["@alice.**"]) is False

    def test_domain_patterns(self):
        assert evaluate_trust_rules("com.example.bot", "my-agent", ["@com.example.*"]) is True
        assert evaluate_trust_rules("com.example.a.b", "my-agent", ["@com.example.**"]) is True
        assert evaluate_trust_rules("com.other.bot", "my-agent", ["@com.example.*"]) is False


class TestTrustLabels:
    def test_verified_label(self):
        labels = {"trust:verified"}
        assert evaluate_trust_rules("anyone", "target", ["#verified"], labels) is True
        assert evaluate_trust_rules("anyone", "target", ["#verified"], set()) is False

    def test_reputation_threshold(self):
        labels = {"trust:reputation-500"}
        assert evaluate_trust_rules("anyone", "target", ["#reputation:100"], labels) is True
        assert evaluate_trust_rules("anyone", "target", ["#reputation:500"], labels) is True
        assert evaluate_trust_rules("anyone", "target", ["#reputation:501"], labels) is False

    def test_reputation_multiple_labels(self):
        labels = {"trust:reputation-200", "trust:reputation-500"}
        assert evaluate_trust_rules("anyone", "target", ["#reputation:300"], labels) is True

    def test_extract_labels_trusted_issuer(self):
        labels = extract_trust_labels(
            "read write trust:verified trust:reputation-100",
            issuer="https://robutler.ai",
        )
        assert labels == {"trust:verified", "trust:reputation-100"}

    def test_extract_labels_untrusted_issuer(self):
        labels = extract_trust_labels(
            "read write trust:verified",
            issuer="https://evil.example.com",
        )
        assert labels == set()


class TestAllowDenyRules:
    def test_allow_deny_deny_wins(self):
        rules = {"allow": ["everyone"], "deny": ["@spammer"]}
        assert evaluate_trust_rules("alice", "target", rules) is True
        assert evaluate_trust_rules("spammer", "target", rules) is False

    def test_allow_deny_pattern(self):
        rules = {"allow": ["everyone"], "deny": ["@com.evil.**"]}
        assert evaluate_trust_rules("com.evil.bot", "target", rules) is False
        assert evaluate_trust_rules("com.good.bot", "target", rules) is True

    def test_deny_with_label(self):
        rules = {"allow": ["everyone"], "deny": ["@spammer", "#reputation:50"]}
        labels = {"trust:reputation-100"}
        assert evaluate_trust_rules("alice", "target", rules, labels) is False

    def test_empty_allow(self):
        rules = {"allow": []}
        assert evaluate_trust_rules("alice", "target", rules) is False

    def test_empty_rules_list(self):
        assert evaluate_trust_rules("alice", "target", []) is False


class TestNamespaceDerivation:
    def test_platform_agent(self):
        assert derive_namespace("alice") == "alice"
        assert derive_namespace("alice.my-bot") == "alice"
        assert derive_namespace("alice.my-bot.helper") == "alice"

    def test_external_agent(self):
        assert derive_namespace("com.example.bot") == "com.example"
        assert derive_namespace("io.cool-bot") == "io.cool-bot"
        assert derive_namespace("org.foundation.agent.v2") == "org.foundation"

    def test_same_namespace(self):
        assert is_same_namespace("alice.bot1", "alice.bot2") is True
        assert is_same_namespace("alice", "alice.bot") is True
        assert is_same_namespace("alice.bot", "bob.bot") is False
        assert is_same_namespace("com.example.a", "com.example.b") is True
        assert is_same_namespace("com.example.a", "com.other.b") is False


class TestTLDReservation:
    def test_common_tlds(self):
        assert is_reserved_tld("com") is True
        assert is_reserved_tld("org") is True
        assert is_reserved_tld("io") is True
        assert is_reserved_tld("ai") is True

    def test_non_tld(self):
        assert is_reserved_tld("alice") is False
        assert is_reserved_tld("mybot") is False


class TestExternalAgentURL:
    def test_standard_url(self):
        name = url_to_reversed_domain_name("https://agents.example.com/my-bot")
        assert name == "com.example.agents.my-bot"

    def test_root_url(self):
        name = url_to_reversed_domain_name("https://cool-bot.io/")
        assert name == "io.cool-bot"

    def test_deep_path(self):
        name = url_to_reversed_domain_name("https://api.openai.com/v1/agents/gpt")
        assert name == "com.openai.api.v1.agents.gpt"


class TestMixedRules:
    def test_multiple_rules_any_match(self):
        rules = ["family", "#verified", "@bob"]
        assert evaluate_trust_rules("alice.bot", "alice.agent", rules) is True
        labels = {"trust:verified"}
        assert evaluate_trust_rules("random", "alice.agent", rules, labels) is True
        assert evaluate_trust_rules("bob", "alice.agent", rules) is True
        assert evaluate_trust_rules("charlie", "alice.agent", rules) is False
