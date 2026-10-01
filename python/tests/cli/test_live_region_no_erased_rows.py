"""
No rows left erased under the prompt after a turn (2026-09-29).

The owner's chat showed blank rows under the prompt box after a reply with a
code block ("why there are 4 \\n after the last commands"). rich's `Live`
redraws its region inside every print made while it runs, from the view it
built at its LAST refresh (`Live.process_renderables`), so a block printed as
it finished was followed by the region as it had last looked: the block's last
twelve lines once more, under it. The next refresh drew the region at its real
height, and the rows between stayed erased. At the bottom of the terminal
nothing ever wrote them again, and the next prompt box sat above them (six
rows in a PTY replay of a streamed code block). `TurnRenderer` now marks a
block printed before it prints and refreshes the region first
(`render.py` `flush`, `_print`).

What is measured: the lowest row the turn's output ever reached, less the row
the cursor ends on once the session has printed what follows a turn (a blank
line and the stats line, as `session.py` does). Anything above zero is a row a
terminal at its bottom edge shows blank under the next prompt. The sequences
counted are the ones rich's `Live` and `print` write: newline and cursor up.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Iterable, List

from rich.console import Console
from rich.live import Live

from webagents.cli.repl.render import TextDelta, TurnRenderer, events_from_chunk

#: The shape of the owner's reply: a line, a code block, a closing line.
CODE_REPLY = (
    "YO, here are the contents of the file:\n\nYO,\n\n```yaml\n"
    + "".join(f"key{i}: value {i}\n" for i in range(16))
    + "```\n\nYO, as you can see, that is all of it."
)


def rows_left_erased(stream: str) -> int:
    """How far below its final position the cursor had been: the rows the turn
    reached and then left blank."""
    row = lowest = 0
    for match in re.finditer(r"\x1b\[(\d*)A|\n", stream):
        if match.group(0) == "\n":
            row += 1
        else:
            row -= int(match.group(1) or 1)
        lowest = max(lowest, row)
    return lowest - row


def stream_turn(chunks: Iterable[object], *, refresh_before_print: bool = True) -> str:
    """Stream `chunks` (text or tool-call chunk dicts) through a live turn the
    way `session.py` does, the region redrawn after every chunk as its timer
    would, then print what the session prints after a turn."""
    out = StringIO()
    console = Console(file=out, width=80, height=40, force_terminal=True, force_interactive=True, color_system=None)
    renderer = TurnRenderer(console)
    with Live(console=console, auto_refresh=False, transient=True, get_renderable=renderer.live_view) as live:
        renderer.live = live if refresh_before_print else None
        for chunk in chunks:
            events = [TextDelta(chunk)] if isinstance(chunk, str) else events_from_chunk(chunk)
            for event in events:
                renderer.feed(event)
            renderer.flush()
            live.refresh()
        renderer.flush(final=True)
        renderer.live = None
    stats = renderer.stats()
    if stats is not None:
        console.print()
        console.print(stats)
    return out.getvalue()


def words(text: str) -> List[str]:
    return [word + " " for word in text.split(" ")]


def test_a_streamed_code_block_leaves_no_erased_rows():
    assert rows_left_erased(stream_turn(words(CODE_REPLY))) == 0


def test_a_tool_call_between_replies_leaves_no_erased_rows():
    chunks: List[object] = words("Let me read the file first.\n\n")
    chunks.append({"type": "tool_call", "call_id": "c1", "name": "read_file", "arguments": '{"path": "AGENT.md"}'})
    chunks.append({"type": "tool_result", "id": "c1", "status": "success", "result": "x\n" * 30})
    chunks.extend(words(CODE_REPLY))
    assert rows_left_erased(stream_turn(chunks)) == 0


def test_a_long_plain_reply_leaves_no_erased_rows():
    assert rows_left_erased(stream_turn(words(" ".join(f"line{i}\n\n" for i in range(40))))) == 0


def test_the_measure_sees_the_old_redraw():
    """The region redrawn stale, as before the fix (no refresh before a block
    prints): the code block's rows are left erased, so the measure is not
    blind to what it guards."""
    assert rows_left_erased(stream_turn(words(CODE_REPLY), refresh_before_print=False)) > 0
