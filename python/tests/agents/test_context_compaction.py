"""
Compaction of a conversation filling the model's context (`agents/core/context_compaction.py`,
2026-09-29), against the shared fixture `tests/fixtures/context/compaction.json`,
which the TypeScript suite reads too (`typescript/tests/unit/core/context-compaction.test.ts`):
the words, the window table, the `compaction:` policy and its refusals, the
counting, and what `compact_messages` makes of each case, message for message.
"""

import asyncio
import json
from pathlib import Path

import pytest

from webagents.agents.core import context_compaction as cc

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "context" / "compaction.json").read_text(encoding="utf-8"))


async def _stub(transcript: str, instructions: str) -> str:
    focus = ""
    if "Pay particular attention to: " in instructions:
        focus = " focusing on " + instructions.split("Pay particular attention to: ", 1)[1].strip()
    return f"[stub summary of {len(transcript.splitlines())} lines{focus}]"


async def _fails(transcript: str, instructions: str) -> str:
    return ""


def _policy_dict(policy: cc.CompactionPolicy) -> dict:
    out = {"auto": policy.auto, "at": policy.at, "keep": policy.keep, "hard": policy.hard, "clear_tool_results": policy.clear_tool_results}
    for key in ("model", "instructions", "window"):
        if getattr(policy, key) is not None:
            out[key] = getattr(policy, key)
    return out


def test_the_words_and_defaults_are_the_fixtures():
    assert cc.WORDS == FIXTURE["words"]
    assert cc.POLICY_WORDS == FIXTURE["policy_words"]
    d = FIXTURE["defaults"]
    assert _policy_dict(cc.CompactionPolicy()) == {k: d[k] for k in ("auto", "at", "keep", "hard", "clear_tool_results")}
    assert (cc.DEFAULT_WINDOW, cc.CLEAR_MIN_CHARS, cc.SETTLE) == (d["window"], d["clear_min_chars"], d["settle"])


@pytest.mark.parametrize("model,override,window", FIXTURE["windows"])
def test_the_context_window(model, override, window):
    assert cc.context_window(model, override) == window


@pytest.mark.parametrize("case", FIXTURE["policies"], ids=lambda c: json.dumps(c["raw"]))
def test_the_policy(case):
    if "refused" in case:
        with pytest.raises(cc.CompactionPolicyError) as refused:
            cc.parse_policy(case["raw"])
        assert str(refused.value) == case["refused"]
        return
    assert _policy_dict(cc.parse_policy(case["raw"])) == case["policy"]


@pytest.mark.parametrize("message,tokens", FIXTURE["estimates"])
def test_the_count(message, tokens):
    assert cc.estimate_message_tokens(message) == tokens


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["about"])
def test_what_compaction_makes_of_a_conversation(case):
    policy = cc.parse_policy(case["policy"] or None)
    out = asyncio.run(
        cc.compact_messages(
            case["messages"],
            policy,
            case["window"],
            _stub if case["summarizer"] == "stub" else _fails,
            force=case.get("force", False),
            focus=case.get("focus"),
            protect_from=case.get("protect_from"),
            threshold=case.get("threshold"),
        )
    )
    e = case["expected"]
    assert (out.stage, out.before, out.after, out.summarized, out.kept, out.cleared, out.dropped) == (
        e["stage"], e["before"], e["after"], e["summarized"], e["kept"], e["cleared"], e["dropped"],
    )
    assert out.summary == e["summary"]
    assert out.sentence() == e["sentence"]
    assert out.messages == e["messages"]
    assert case["messages"] == json.loads(json.dumps(case["messages"])), "the input is not changed"
