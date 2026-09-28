"""
How a finished tool call is drawn: refusals as failures, and the sandbox's
hint on its own line (the ptypass-fixes lane, 2026-09-27; fixture
`tests/fixtures/cli/ptypass_fixes_tool_lines.json`). The TypeScript twin is
`typescript/tests/unit/cli/ptypass-fixes-tool-lines.test.ts`.

The real-terminal PTY pass saw a refused `.env` read drawn as a green
"Read 1 lines", "The owner declined..." in green, and a command's sandbox
hint hidden behind "(+N lines)" in the collapsed tool line, so the one
sentence naming the switch never reached the person.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from webagents.cli.repl.render import HINT_PREFIX, ToolSegment, split_hints, tool_failed, tool_lines, tool_result_summary
from webagents.cli.ui.theme import theme_for
from webagents.sandbox import REFUSAL_HINTS

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "ptypass_fixes_tool_lines.json").read_text())


def test_every_refusal_the_fixture_lists_is_a_failure_and_the_look_alikes_are_not():
    for text in FIXTURE["failed"]:
        assert tool_failed("success", text), text
    for text in FIXTURE["succeeded"]:
        assert not tool_failed("success", text), text
    for word in FIXTURE["failure_words"]["words"]:
        assert tool_failed("success", word[0].upper() + word[1:] + ": x"), word


def _drawn(tool: ToolSegment, width: int) -> list:
    theme = theme_for("dark")
    return tool_lines(theme, tool, tool.ended or 0.0, width)


def test_a_refused_read_is_red_and_not_read_1_lines():
    refused = FIXTURE["failed"][0]
    # Whole, never cut mid-word (2026-09-28, `cli/final_sdk_low_items.json`):
    # this pinned "... and the fil…".
    assert tool_result_summary("read_file", refused, "success") == refused
    tool = ToolSegment(key="c1", id="c1", name="read_file", arguments='{"path": ".env"}', status="success", result=refused, started=0.0, ended=0.1)
    lines = _drawn(tool, 120)
    text = "\n".join(line.plain for line in lines)
    assert "Read 1 lines" not in text and "Refused: .env is where" in text
    theme = theme_for("dark")
    assert lines[0].spans[0].style == theme.palette.error or str(lines[0].spans[0].style) == str(theme.palette.error)


def test_the_hint_comes_out_of_the_summary_and_is_drawn_whole_under_it():
    case = FIXTURE["hint"]["case"]
    assert HINT_PREFIX == FIXTURE["hint"]["prefix"]
    assert split_hints(case["result"])[1] == case["hints"]
    assert tool_result_summary("run_command", case["result"], "success") == case["summary"]
    assert case["hints"][0] in REFUSAL_HINTS.values()
    tool = ToolSegment(key="c1", id="c1", name="run_command", arguments='{"command": "curl -sS https://example.com"}', status="success", result=case["result"], started=0.0, ended=0.1)
    lines = [line.plain for line in _drawn(tool, 60)]
    flat = re.sub(r"\s+", " ", " ".join(lines))
    assert re.sub(r"\s+", " ", HINT_PREFIX + case["hints"][0]) in flat
    assert all(len(line) <= 60 for line in lines), lines
