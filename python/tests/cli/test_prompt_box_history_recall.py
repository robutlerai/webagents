"""
A line from history keeps the command menu closed (2026-09-29, the owner:
"up/down history stops when there is /command because menu grabs the focus").
↑ that brought back a `/mcp` opened the menu, and the next ↑ moved in the
menu, so the history walk stopped there. Driven here through the box's real
application, keys from a pipe, the box's own state read between them. The
TypeScript twin is `typescript/tests/unit/cli/input-history-recall.test.ts`.
"""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import Callable, List, Optional, Union

from prompt_toolkit.application import create_app_session, get_app
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from webagents.cli.ui.prompt_box import PromptBox
from webagents.cli.ui.theme import theme_for

UP, DOWN, LEFT, ESC, ENTER = "\x1b[A", "\x1b[B", "\x1b[D", "\x1b", "\r"
#: ctrl+u then ctrl+k: the line cleared on both sides of the cursor.
CLEAR = "\x15\x0b"
Step = Union[str, Callable[[PromptBox, str], None]]


def drive(history_lines: List[str], steps: List[Step]) -> Optional[str]:
    """Run the box: a string step is typed, a callable one is called with the
    box and the text it holds at that moment."""
    history = InMemoryHistory()
    for line in history_lines:
        history.append_string(line)
    box = PromptBox(
        theme_for(Console(file=StringIO(), width=100, force_terminal=False, color_system=None)),
        commands=[("/mcp", "The MCP servers this agent uses"), ("/help", "Show the commands and keys")],
        footer=lambda: [],
        history=history,
    )

    async def run() -> Optional[str]:
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                task = asyncio.ensure_future(box.ask("Message"))
                await asyncio.sleep(0.3)
                for step in steps:
                    if callable(step):
                        step(box, get_app().current_buffer.text)
                    else:
                        pipe.send_text(step)
                        await asyncio.sleep(0.25)
                return await asyncio.wait_for(task, 5)

    return asyncio.run(run())


def menu_closed(box: PromptBox, text: str) -> None:
    assert box.menu_items(text) == [], text


def menu_open(box: PromptBox, text: str) -> None:
    assert box.menu_items(text), text


def test_up_walks_past_a_command_in_history():
    assert drive(["hello", "/mcp", "first"], [UP, UP, menu_closed, UP, ENTER]) == "hello"


def test_down_walks_back_past_it_too():
    assert drive(["hello", "/mcp", "first"], [UP, UP, UP, DOWN, menu_closed, DOWN, ENTER]) == "first"


def test_moving_the_cursor_opens_the_menu_again():
    steps: List[Step] = [UP, menu_closed, LEFT, menu_open, ESC, CLEAR, "done", ENTER]
    assert drive(["/mcp"], steps) == "done"


def test_typing_opens_it_again():
    steps: List[Step] = [UP, menu_closed, "l", menu_open, ESC, CLEAR, "done", ENTER]
    assert drive(["/he"], steps) == "done"
