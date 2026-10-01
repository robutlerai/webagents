"""
Tab takes the grey suggestion from history (2026-09-29, the owner: "tab
completion won't work", a box showing `what dog` with `breeds are there?` in
grey after it). With the command menu closed, tab was prompt_toolkit's
`menu-complete`, and the box has no completer, so it did nothing. Nor did →:
the keys that take a suggestion are `load_auto_suggest_bindings`, which
`PromptSession` loads and the box's bare `Application` never did, so no key
took one. Driven here through the box's real application, keys
arriving from a pipe with pauses between them, as a person types: the
suggestion is computed after the text changes, not at once.
"""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import List, Optional

from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from webagents.cli.ui.prompt_box import PromptBox
from webagents.cli.ui.theme import theme_for


def ask_with_keys(keys: List[str], history_lines: List[str]) -> Optional[str]:
    """What the box returns when `keys` are typed one group at a time, with
    `history_lines` sent before."""
    history = InMemoryHistory()
    for line in history_lines:
        history.append_string(line)
    box = PromptBox(
        theme_for(Console(file=StringIO(), width=100, force_terminal=False, color_system=None)),
        commands=[("/help", "Show the commands and keys"), ("/exit", "Leave")],
        footer=lambda: [],
        history=history,
    )

    async def run() -> Optional[str]:
        with create_pipe_input() as pipe:
            with create_app_session(input=pipe, output=DummyOutput()):
                task = asyncio.ensure_future(box.ask("Message"))
                await asyncio.sleep(0.3)
                for group in keys:
                    pipe.send_text(group)
                    await asyncio.sleep(0.3)
                return await asyncio.wait_for(task, 5)

    return asyncio.run(run())


def test_tab_takes_the_suggestion_from_history():
    sent = ask_with_keys(["what dog", "\t", "\r"], ["what dog breeds are there?"])
    assert sent == "what dog breeds are there?"


def test_tab_with_no_suggestion_changes_nothing():
    assert ask_with_keys(["hello", "\t", "\r"], []) == "hello"


def test_the_right_arrow_takes_it():
    sent = ask_with_keys(["what dog", "\x1b[C", "\r"], ["what dog breeds are there?"])
    assert sent == "what dog breeds are there?"


def test_ctrl_e_takes_it_too():
    sent = ask_with_keys(["what dog", "\x05", "\r"], ["what dog breeds are there?"])
    assert sent == "what dog breeds are there?"
