"""
A question asked mid-turn stays on the screen, says what was decided, and
hands Ctrl+C back to the turn (the ptypass-fixes lane, 2026-09-27; fixture
`tests/fixtures/cli/ptypass_fixes_answers.json`). The TypeScript twin is
`typescript/tests/unit/cli/ptypass-fixes-answers.test.ts`.

The real-terminal PTY pass found two things in the Python chat:

  * Rich's live display, started again after the question, remembered how
    tall it last was and its first redraw moved up that many lines, over the
    question, the typed answer and the diff's last line (transcript `02`);
  * `_ask` took over the turn's SIGINT handler and then removed it, so a
    Ctrl+C later in the same turn ended it without the interrupt, and the
    chat said "The model returned no answer" (`14` against `15`).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.live import Live
from rich.text import Text

from webagents.cli.repl.chat_words import ANSWER_WORDS
from webagents.cli.repl.session import WebAgentsSession

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "ptypass_fixes_answers.json").read_text())


def test_the_answer_lines_are_the_fixtures():
    assert ANSWER_WORDS == {"control": FIXTURE["control"], "host": FIXTURE["host"]}


def test_the_live_display_draws_below_what_the_question_left():
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=80, color_system=None)
    live = Live(Text("LIVE LINE ONE\nLIVE LINE TWO\nLIVE LINE THREE"), console=console, auto_refresh=False, transient=True)
    live.start(refresh=True)
    fake = SimpleNamespace(_turn_live=live, _turn_keys=None, _turn_reenter_keys=None, _turn_renderer=None)
    with WebAgentsSession._turn_paused(fake):
        console.print(Text("  Make this change? [y/N] y"))
    after = out.getvalue().split("Make this change? [y/N] y", 1)[1]
    live.stop()
    # The first redraw after the question moves the cursor up over nothing:
    # before the fix it went up two lines, over the question and the answer.
    assert "\x1b[1A" not in after.split("LIVE LINE ONE", 1)[0]


def test_a_question_hands_ctrl_c_back_to_the_turn(monkeypatch):
    read_end, write_end = os.pipe()
    os.write(write_end, b"y\n")
    os.close(write_end)
    stdin = SimpleNamespace(fileno=lambda: read_end, readline=lambda: "")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    seen = []

    async def scenario():
        loop = asyncio.get_running_loop()

        def turn_interrupt():
            seen.append("turn interrupted")

        loop.add_signal_handler(signal.SIGINT, turn_interrupt)
        fake = SimpleNamespace(_turn_interrupt=turn_interrupt, console=Console(file=io.StringIO()))
        answer = await WebAgentsSession._ask(fake, "  Make this change? [y/N] ")
        # A Ctrl+C after the question reaches the turn's handler again.
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.2)
        loop.remove_signal_handler(signal.SIGINT)
        return answer

    try:
        assert asyncio.run(scenario()) == "y"
    finally:
        os.close(read_end)
    assert seen == ["turn interrupted"]


def test_the_path_in_the_question_is_the_folders_own(tmp_path):
    """Brief item 13: an absolute path wrapped mid-name across three lines."""
    from webagents.cli.repl.session import _control_path

    assert _control_path(str(tmp_path / "AGENT.md"), tmp_path) == "AGENT.md"
    assert _control_path("AGENT.md", tmp_path) == "AGENT.md"
    assert _control_path(str(tmp_path / ".agents" / "skills" / "x" / "SKILL.md"), tmp_path) == ".agents/skills/x/SKILL.md"
    assert _control_path("/etc/hosts", tmp_path) == "/etc/hosts"


def test_the_time_spent_answering_is_not_the_tools():
    """Brief item 13: a declined write read "1m 14s", the minute being the owner's."""
    from webagents.cli.repl.render import ToolCall, TurnRenderer

    renderer = TurnRenderer(Console(file=io.StringIO()))
    renderer.feed(ToolCall(id="c1", name="write_file", arguments="{}"))
    tool = next(segment for segment in renderer.segments if getattr(segment, "id", None) == "c1")
    before = tool.started
    renderer.shift_running_tools(60.0)
    assert tool.started == before + 60.0
