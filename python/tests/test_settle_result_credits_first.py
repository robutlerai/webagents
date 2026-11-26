"""
Credit amounts are read under their `...Credits` names first, the Python twin of
`typescript/tests/unit/skills/payments/settle-result-credits-first.test.ts`. The platform answers
both `<name>Credits` and the older `<name>Dollars` (the same value); an older platform answers only
the second. The readers prefer the first, so the SDK no longer depends on the older names.
"""

from __future__ import annotations

from webagents.agents.skills.robutler.payments.settle_result import read_settle_result
from webagents.agents.skills.robutler.payments.skill import _locked_credits


def test_credits_only_answer_fills_both_names():
    r = read_settle_result({"success": True, "chargedCredits": 0.005, "remainingCredits": 1.5}, "t")
    assert r["chargedCredits"] == 0.005 and r["chargedDollars"] == 0.005
    assert r["remainingCredits"] == 1.5 and r["remainingDollars"] == 1.5


def test_older_dollars_only_answer_still_reads():
    r = read_settle_result({"success": True, "chargedDollars": 0.004}, "t")
    assert r["chargedCredits"] == 0.004 and r["chargedDollars"] == 0.004


def test_credits_name_wins_when_both_present():
    r = read_settle_result({"success": True, "chargedCredits": 0.003, "chargedDollars": 99}, "t")
    assert r["chargedCredits"] == 0.003 and r["chargedDollars"] == 0.003


def test_lock_reads_credits_first_then_dollars_then_fallback():
    assert _locked_credits({"lockedAmountCredits": 0.01, "lockedAmountDollars": 9}, 0.5) == 0.01
    assert _locked_credits({"lockedAmountDollars": 0.02}, 0.5) == 0.02
    assert _locked_credits({}, 0.5) == 0.5
    assert _locked_credits({"lockedAmountCredits": True}, 0.5) == 0.5
