"""
What the chat keeps of a turn's tool calls, and how much goes back to the model
(2026-09-28), the same in both SDKs: `tests/fixtures/chat/turn_history.json`,
which the TypeScript suite runs too (`typescript/tests/unit/cli/turn-history-fixture.test.ts`).
"""

import copy
import json
from pathlib import Path

from webagents.cli.repl.render import ToolCall, ToolResult
from webagents.cli.turn_history import (
    TOOL_HISTORY_BUDGET_CHARS,
    TurnRecorder,
    history_for_model,
    left_out_result,
    spoken_count,
)

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "chat" / "turn_history.json").read_text())


def test_the_budget_and_the_words():
    assert TOOL_HISTORY_BUDGET_CHARS == FIXTURE["budget_chars"]
    assert left_out_result(FIXTURE["left_out"]["chars"]) == FIXTURE["left_out"]["text"]


def test_a_turn_keeps_its_answered_rounds_in_order():
    recorder = TurnRecorder()
    for e in FIXTURE["turn"]["events"]:
        if "call" in e:
            recorder.observe(ToolCall(**e["call"]))
        else:
            recorder.observe(ToolResult(id=e["result"]["id"], status="success", result=e["result"]["result"]))
    assert recorder.messages() == FIXTURE["turn"]["messages"]


def test_the_copy_sent_keeps_the_newest_results_and_leaves_the_rest_out():
    history = copy.deepcopy(FIXTURE["budget"]["history"])
    assert history_for_model(history, FIXTURE["budget"]["budget"]) == FIXTURE["budget"]["sent"]
    assert history == FIXTURE["budget"]["history"], "the conversation itself keeps everything"


def test_spoken_messages_are_what_n_messages_counts():
    assert spoken_count(FIXTURE["spoken"]["messages"]) == FIXTURE["spoken"]["count"]


def test_the_cli_preamble_words_and_when_it_applies():
    from webagents.cli.preamble import cli_preamble, with_cli_preamble

    p = FIXTURE["preamble"]
    assert cli_preamble(p["file"], p["folder"]) == p["text"]
    assert with_cli_preamble(p["instructions"], p["file"], ["filesystem"]) == p["with_it"]
    for case in p["cases"]:
        got = with_cli_preamble(p["instructions"], p["file"] if case["file"] else None, case["skills"])
        assert (got != p["instructions"]) == case["applies"], case
