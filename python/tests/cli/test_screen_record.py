"""
The chat's record of the screen (2026-09-25), held to the fixture the
TypeScript chat's record reads too (`tests/fixtures/cli/screen_record.json`).

The `/` menu opens over the last rows of the conversation and draws them again
when it closes (`webagents/cli/ui/prompt_box.py`), so what the record says
those rows hold is what the person sees afterwards: a wrong row here is a
wrong row on screen, and an unknown row must say so rather than guess.
"""

import io
import json
import sys
from pathlib import Path

import pytest

from webagents.cli.ui.screen import ScreenRecord, record_screen

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "screen_record.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_the_screen_record_shared_fixture(case):
    size = {"columns": case["columns"], "rows": case["rows"]}
    screen = ScreenRecord(lambda: size["columns"], lambda: size["rows"])
    for step in case["steps"]:
        if "write" in step:
            screen.feed(step["write"])
        if "anchor" in step:
            screen.anchor(step["anchor"])
        if "scrolled" in step:
            screen.scrolled(step["scrolled"])
        if "resize" in step:
            size["columns"], size["rows"] = step["resize"]
    for check in case["checks"]:
        assert screen.plain_rows_above(check["above"]) == check["plain"], f"{check['above']} rows"
        if "rendered" in check:
            assert screen.rows_above(check["above"]) == check["rendered"]
        if check["plain"] is None:
            assert screen.rows_above(check["above"]) is None


class _Terminal(io.StringIO):
    """A text stream with a byte buffer beside it, as sys.stdout has."""

    def __init__(self):
        super().__init__()
        self.buffer = io.BytesIO()

    def isatty(self):
        return True


def test_recording_sees_both_streams_in_order_and_passes_everything_on(monkeypatch):
    out, err = _Terminal(), _Terminal()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    screen, stop = record_screen(lambda: 40, lambda: 10)
    print("one")
    sys.stderr.write("two\n")
    # prompt_toolkit writes bytes to the buffer; a character can be split across writes.
    data = "三\n".encode()
    sys.stdout.buffer.write(data[:2])
    sys.stdout.buffer.write(data[2:])
    assert screen.plain_rows_above(3) == ["one", "two", "三"]
    assert out.getvalue() == "one\n" and err.getvalue() == "two\n"
    assert out.buffer.getvalue() == data
    assert sys.stdout.isatty()
    stop()
    assert sys.stdout is out and sys.stderr is err
    print("four")
    assert screen.plain_rows_above(1) == ["三"]


def test_nothing_is_recorded_while_paused():
    # The input box draws itself; its rows are its own.
    screen = ScreenRecord(lambda: 40, lambda: 10)
    screen.feed("conversation\n")
    screen.paused = True
    screen.feed("╭──╮\n│ │\n╰──╯")
    screen.paused = False
    assert screen.plain_rows_above(1) == ["conversation"]
