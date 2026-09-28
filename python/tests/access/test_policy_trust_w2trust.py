"""
Trust-gated `access:` groups (plan item 2.5, 2026-09-26): the decision over the
platform's evidence, from the table both SDKs run
(`tests/fixtures/access/policy.json`, `trust_cases`; TypeScript
tests/unit/access/policy-trust-w2trust.test.ts). The platform is never dialled
here: `trust` IS what the access skill learned, or None when it could not.
"""

import json
import math
from pathlib import Path

import pytest

from webagents.access.policy import (
    TrustEvidence,
    TrustRequirement,
    decide,
    meets_trust,
    parse_access,
    trust_key,
    trust_requirements,
    verified_agent_of,
)

TABLE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "access" / "policy.json").read_text())
POLICIES = {name: parse_access(raw) for name, raw in TABLE["policies"].items()}


def _evidence(raw):
    if raw is None:
        return None
    return TrustEvidence(agent=raw["agent"], scores=dict(raw["scores"]))


@pytest.mark.parametrize("case", TABLE["trust_cases"], ids=lambda c: c["name"])
def test_the_decision_over_trust_evidence(case):
    decision = decide(POLICIES[case.get("policy", "open")], case["principals"], case.get("tier"), _evidence(case.get("trust")))
    assert decision.allow is case["expect"]["allow"]
    if decision.allow:
        assert list(decision.groups) == case["expect"]["groups"]


@pytest.mark.parametrize("name", list(TABLE["trust_requirements"]))
def test_what_a_block_asks_the_platform_for(name):
    expected = [TrustRequirement(min=r["min"], topic=r["topic"]) for r in TABLE["trust_requirements"][name]]
    assert trust_requirements(POLICIES[name]) == expected


def test_the_overall_score_is_filed_under_the_empty_key():
    assert trust_key(None) == ""
    assert trust_key("billing") == "billing"


def test_the_verified_agent_is_the_first_agent_principal():
    assert verified_agent_of(["key:x", "agent:https://a.example/x", "agent:https://b.example/y"]) == "https://a.example/x"
    assert verified_agent_of(["user:@alice", "key:x"]) is None


def test_meets_trust_fails_closed_on_every_missing_piece():
    requirement = TrustRequirement(min=0.5, topic=None)
    principals = ["agent:https://a.example/x"]
    assert meets_trust(requirement, principals, TrustEvidence("https://a.example/x", {"": 0.5})) is True
    assert meets_trust(requirement, principals, TrustEvidence("https://a.example/x", {"": 0.49})) is False
    assert meets_trust(requirement, principals, TrustEvidence("https://a.example/x", {"billing": 0.9})) is False
    assert meets_trust(requirement, principals, TrustEvidence("https://a.example/x", {"": math.nan})) is False
    assert meets_trust(requirement, principals, TrustEvidence("https://a.example/x", {"": True})) is False
    assert meets_trust(requirement, principals, TrustEvidence("https://other.example/x", {"": 1.0})) is False
    assert meets_trust(requirement, principals, None) is False
    assert meets_trust(requirement, ["user:@alice"], TrustEvidence("https://a.example/x", {"": 1.0})) is False


def test_a_block_with_no_trust_group_asks_for_nothing():
    assert trust_requirements(parse_access({})) == []
    assert trust_requirements(parse_access({"groups": {"a": {"members": ["user:@x"]}}})) == []
