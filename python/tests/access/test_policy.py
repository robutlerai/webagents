"""
The access block's decision and its refusals of a malformed block (ADR-0045), from
the table both SDKs run (`tests/fixtures/access/policy.json`; TypeScript
tests/unit/access/policy.test.ts).
"""

import json
from pathlib import Path

import pytest

from webagents.access.policy import AccessConfigError, decide, parse_access

TABLE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "access" / "policy.json").read_text())
POLICIES = {name: parse_access(raw) for name, raw in TABLE["policies"].items()}


@pytest.mark.parametrize("case", TABLE["cases"], ids=lambda c: c["name"])
def test_the_decision(case):
    decision = decide(POLICIES[case.get("policy", "open")], case["principals"], case.get("tier"))
    assert decision.allow is case["expect"]["allow"]
    if decision.allow:
        assert list(decision.groups) == case["expect"]["groups"]


@pytest.mark.parametrize("case", TABLE["bad"], ids=lambda c: c["error"][:60])
def test_a_malformed_block_is_refused_with_its_sentence(case):
    with pytest.raises(AccessConfigError) as raised:
        parse_access(case["access"])
    assert str(raised.value) == case["error"]


def test_an_empty_block_lets_everyone_in_as_everyone():
    assert decide(parse_access({}), []).groups == ("everyone",)
