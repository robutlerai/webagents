"""
The line-by-line chat's input box (2026-09-24).

The chat asked with a `PromptSession` and a coral bar, which cannot draw a
closed box: its layout has no place for a right edge beside wrapped input.
This is a small inline prompt_toolkit application instead, drawn the way the
TypeScript chat draws its box (`typescript/src/cli/ui/input.ts`):

  * a rounded box whose border runs the brand gradient, `❯` in coral, a
    placeholder while it is empty, and for its first seconds a faint starfield
    twinkling in the empty part (Codex's idle `sparkle_field`);
  * under it, a footer (agent, model, tokens, folder on the left; the keys
    that matter now on the right), or, while a `/` command is being typed, the
    command menu: ↑/↓ choose, tab completes, enter runs, esc closes;
  * the editing the chat already had: prompt_toolkit's emacs keys, history
    (↑ searches it by what is typed), suggestions from history (→ or tab accepts),
    alt+enter or ctrl+j for a new line, and the caller's own chords (Ctrl+T).

Ctrl+C clears the box, and on an empty box a second one within two seconds
leaves; Ctrl+D on an empty box leaves; esc twice clears. The box is erased
when it is done, and the caller prints the sent message in its place.

THE MENU NEVER SCROLLS THE TERMINAL (2026-09-25). A box at the bottom of the
terminal has no room under it for the menu. Growing into it scrolled the
terminal, which pushed conversation lines into the scrollback, and nothing
brings a line back down from there. The box used to scroll the screen back
down when the menu closed, so that it sat at the bottom again, and that left
blank rows in the history where those lines had been (the owner's "gap in
history after the command menu disappears"). So the menu opens over the last
rows of the conversation, above the box, when those rows are on screen and the
chat's record of them (`screen.py`) is sure of them. prompt_toolkit draws only
from where the application starts down, so these rows are drawn beside it,
cursor saved and restored, and drawn again as they were when the menu closes.
The box never moves and the scrollback is never touched.

THE MENU ALWAYS OPENS UPWARD WHEN IT CAN (2026-09-28). Until then it opened
under the box whenever it fitted there, so the room left under the box decided
the direction: at the bottom of the terminal the `/` menu opened above while
`/agent `'s four values, which fitted, opened below, and a box half way down
the screen opened everything below (the owner: "completion menu should always
open up"). Now it opens above whenever at least MIN_MENU_REACH known rows are
there to cover, with fewer items (and "N more") when there are fewer than it
wants. It opens under the box only when it cannot: the rows above are not
known, or there are too few of them (the top of a cleared screen). The
terminal may scroll then, and prompt_toolkit keeps the box where that leaves it
until the next message. That is a box a little higher, never a gap.
prompt_toolkit's own cursor position report also checks the record against
the screen. The TypeScript box does the same (`typescript/src/cli/ui/input.ts`).
Esc closes the menu at once: prompt_toolkit waited its default second to see
whether an enter followed (alt+enter is esc, enter).

A LINE FROM HISTORY KEEPS THE MENU CLOSED (2026-09-29). ↑ that brought back a
`/mcp` opened the command menu, and the next ↑ moved in the menu instead of
going further back, so history stopped at the first command in it (the owner:
"up/down history stops when there is /command because menu grabs the focus").
A line ↑, ↓, ctrl+p or ctrl+n brings back is `recalled`: the menu stays closed
and the arrows keep walking the history. Typing, or moving the cursor (←, →,
home, end), opens it again. The TypeScript box does the same
(`typescript/src/cli/ui/input.ts`, `recalled`).

SEARCH, AND PICKERS (2026-09-30, the owner: "/resume and other commands and
subcommands should have search/filter on typing and up/down arrow selection").
A command's values used to be matched on their names alone, one word at a
time, and `/resume` offered only `delete`, so a conversation could not be
chosen from the menu at all. Now:

  * A completer is told everything typed after the command and answers a
    `Slot`: the rows for the argument being typed, and the typed text they are
    matched against (`query`, a suffix of the line), which may be several
    words: `/resume launch plan`.
  * `rank_rows` keeps the rows the query finds: the names it starts, then the
    names it is inside, then the rows where every word of it starts a word of
    the name, the description or the row's `search` text (a conversation's
    whole first message). Commands match descriptions from two characters on,
    so `/h` still lists what starts with h. Fixture:
    `tests/fixtures/cli/chat_commands.json` `menu_search`.
  * A row that completes the command (`runs`: a conversation, a snapshot, a
    model, a command to explain) is chosen with enter: the line is sent. A row
    the line goes on after (a verb, one of several skills), or one whose
    command acts without asking (`/mcp remove`), is inserted, as before; tab
    always inserts. A row may insert other text than it shows (`insert`: a
    conversation's id for its number, so the line means the conversation
    highlighted whatever the list is when it runs).
  * `/resume` and `/rewind` (`pickers`): enter on the command in the menu
    opens its list instead of printing it, when there is something to choose.
The TypeScript box does the same (`ui/input.ts`, `Slot`, `rankRows`).
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.renderer import HeightIsUnknownError
from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, ConditionalAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.bindings.auto_suggest import load_auto_suggest_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import AppendAutoSuggestion, Processor, Transformation, TransformationInput
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.text import Text

from .motion import SPARKLE_SECONDS, spark_at
from .screen import ScreenRecord
from .theme import ChatTheme

#: How long a first ctrl+c (leave) or esc (clear) waits for its second.
CONFIRM_SECONDS = 2.0
MAX_MENU_ITEMS = 6
MAX_TEXT_ROWS = 10
#: The most rows the menu covers above the box: a blank row, the commands, and "N more".
MENU_REACH = MAX_MENU_ITEMS + 2
#: The fewest it opens above with: a blank row and two of the menu's (module docstring).
MIN_MENU_REACH = 3


def _box_width() -> int:
    """The box's width: one column short of the terminal's, as in the
    TypeScript box (`layoutPrompt`), which leaves the last column empty so a
    full-width row never leaves the cursor waiting to wrap."""
    return max(24, get_app().output.get_size().columns - 1)


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


#: The fewest characters a query needs before it matches command descriptions
#: (module docstring, "SEARCH"): below it, `/h` would match every command whose
#: description has an h-word.
COMMAND_SEARCH_MIN = 2


class MenuRow(NamedTuple):
    """One row of the menu: a command (`/name`), or a value for the argument being typed."""

    value: str
    description: str = ""
    #: More text a query matches, not shown (a conversation's whole first message).
    search: str = ""
    #: Choosing it completes the command: enter sends the line (module docstring).
    runs: bool = False
    #: What goes in the box instead of `value` (a conversation's id, for its number).
    insert: Optional[str] = None


class Slot(NamedTuple):
    """What a command offers for the argument being typed: the rows, and the
    typed text they are matched against, a suffix of the line."""

    rows: Sequence[Any]
    query: str


#: A command's completer: given everything typed after `/<command>`, its slot, or None.
Completer = Callable[[str], Optional[Slot]]


def argument_words(args: str) -> Tuple[List[str], str]:
    """The complete words before the one being typed, and that word ("" after a space)."""
    partial = "" if not args or args[-1].isspace() else args.split()[-1]
    return args[: len(args) - len(partial)].split(), partial


def rest_after(args: str, words: int) -> str:
    """The text typed after the first `words` words, as typed: a search's query."""
    rest = args.lstrip()
    for _ in range(words):
        first = rest.split(maxsplit=1)
        rest = rest[len(first[0]):].lstrip() if first else ""
    return rest


def _as_row(row: Any) -> MenuRow:
    return row if isinstance(row, MenuRow) else MenuRow(*row)


def _tokens(text: str) -> List[str]:
    """The words a query word may start: split at spaces and slashes, and into letters and digits."""
    lowered = text.lower()
    return [t for t in re.split(r"[\s/]+", lowered) if t] + re.findall(r"[a-z0-9]+", lowered)


def rank_rows(rows: Sequence[Any], query: str, commands: bool = False) -> List[MenuRow]:
    """The rows `query` finds, best first (module docstring, "SEARCH"): the
    names it starts, then the names it is inside, then the rows where every
    word of it starts a word of the name, the description or `search`. A
    command's name is matched without its `/`, and its description only from
    `COMMAND_SEARCH_MIN` characters. Order is kept within each group."""
    items = [_as_row(r) for r in rows]
    q = query.strip().lower()
    if not q:
        return items
    words = q.split()

    def name(row: MenuRow) -> str:
        value = row.value.lower()
        return value[1:] if commands and value.startswith("/") else value

    starts: List[MenuRow] = []
    inside: List[MenuRow] = []
    found: List[MenuRow] = []
    for row in items:
        n = name(row)
        if n.startswith(q):
            starts.append(row)
        elif q in n:
            inside.append(row)
        elif not commands or len(q) >= COMMAND_SEARCH_MIN:
            tokens = _tokens(" ".join((n, row.description, row.search)))
            if all(any(t.startswith(w) for t in tokens) for w in words):
                found.append(row)
    return starts + inside + found


def _menu_window(items: List[MenuRow], menu_index: int, max_rows: int):
    """What the menu shows in at most `max_rows` rows: every match when they
    fit, else as many as fit with "N more" under them, the window following
    the highlighted row. `(selected, offset, window, name_width, more)`; the
    TypeScript box's `menuLines` counts the same way."""
    selected = min(menu_index, len(items) - 1)
    count = len(items) if len(items) <= min(MAX_MENU_ITEMS, max_rows) else max(1, min(MAX_MENU_ITEMS, max_rows - 1))
    offset = max(0, min(selected - count + 1, len(items) - count))
    window = items[offset: offset + count]
    name_width = max(len(c[0]) for c in items[:count] + window) + 2
    return selected, offset, window, name_width, len(items) - count


class _Placeholder(Processor):
    """The placeholder and the starfield, on the first line of an empty box."""

    def __init__(self, box: "PromptBox") -> None:
        self.box = box

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        if ti.lineno != 0 or ti.document.text:
            return Transformation(ti.fragments)
        box = self.box
        room = max(8, ti.width - 3)
        hint = _clip(box.placeholder, room)
        fragments = list(ti.fragments) + [("class:wa-placeholder", hint)]
        theme = box.theme
        now = time.monotonic()
        if theme.animate and theme.rich_colour and now - box.shown_at < SPARKLE_SECONDS:
            start = len(hint) + 3
            fragments.append(("", "   "))
            for column in range(start, room):
                spark = spark_at(theme, column, now - box.shown_at)
                fragments.append((f"fg:{spark[1]}", spark[0]) if spark else ("", " "))
        return Transformation(fragments)


class PromptBox:
    def __init__(
        self,
        theme: ChatTheme,
        commands: Sequence[Tuple[str, str]],
        footer: Callable[[], List[str]],
        history: Optional[History] = None,
        extra_lines: Callable[[], List[str]] = lambda: [],
        key_bindings: Optional[KeyBindings] = None,
        screen: Optional[ScreenRecord] = None,
        console: Optional[Console] = None,
        completers: Optional[Dict[str, Completer]] = None,
        pickers: Callable[[str], bool] = lambda _name: False,
    ) -> None:
        self.theme = theme
        #: The chat's record of the screen, and the console that draws the menu
        #: when it opens over the conversation (module docstring). Without
        #: both, the menu always opens under the box.
        self.screen = screen
        self.console = console
        #: The commands, display like "/help" or "/agent list".
        self.commands: List[MenuRow] = [_as_row(c) for c in commands]
        #: By command name, what the box offers after `/<command> `
        #: (2026-09-26, interactive-mode spec 3.8; a `Slot` since 2026-09-30,
        #: module docstring "SEARCH"): given everything typed after the
        #: command, the rows for the argument being typed and the text they are
        #: matched against. With rows, the menu stays open after the command;
        #: with none, it closes and enter sends the line. The TypeScript box
        #: does the same (`ui/input.ts`, `Command.complete`).
        self.completers: Dict[str, Completer] = dict(completers or {})
        #: Whether enter on a command in the menu opens its list instead of
        #: running it (`/resume`, `/rewind`), when the list has a row to choose.
        self.pickers = pickers
        self.footer = footer
        self.extra_lines = extra_lines
        self.history = history or InMemoryHistory()
        self.extra_bindings = key_bindings
        self.placeholder = ""
        self.shown_at = time.monotonic()
        self.menu_index = 0
        self.menu_dismissed = False
        #: The text is a line ↑ or ↓ brought back from history (module
        #: docstring, "A LINE FROM HISTORY"): the menu stays closed until the
        #: line is edited or the cursor moves.
        self.recalled = False
        self._navigating = False
        self.exit_armed_at = 0.0
        self.esc_armed_at = 0.0
        self._reset_placement()

    def _reset_placement(self) -> None:
        #: Where the open menu went: "below" the box or "above" it; None while closed.
        self._placement: Optional[str] = None
        #: The rows the menu may cover above the box, as they were, oldest first.
        self._saved: Optional[List[str]] = None
        #: What is drawn over those rows now (None: the rows themselves).
        self._overlay: Optional[List[str]] = None
        #: The screen row (1-based) the box started on, once the terminal said,
        #: and how many rows sit above the application now.
        self._start_row: Optional[int] = None
        self._last_above: Optional[int] = None

    # -- state ----------------------------------------------------------------

    def menu_items(self, text: str) -> List[MenuRow]:
        """The rows the menu offers for `text`; empty when it is closed.

        While the command name is being typed, the commands it finds
        (`rank_rows`); after `/<command> `, the rows that command's completer
        offers, found by what is typed for them (module docstring, "SEARCH").
        An argument row never starts with `/`.
        """
        return self._menu(text)[0]

    def _menu(self, text: str) -> Tuple[List[MenuRow], Optional[str]]:
        """The menu's rows, and the query they answer: None for the command
        list, else the text typed for the argument (a suffix of `text`)."""
        if self.menu_dismissed or self.recalled or not text.startswith("/") or "\n" in text:
            return [], None
        if not re.search(r"\s", text):
            return rank_rows(self.commands, text[1:], commands=True), None
        command = re.split(r"\s", text[1:], maxsplit=1)[0]
        completer = self.completers.get(command)
        if completer is None:
            # No completer: a multi-word display name (`/agent list`) still narrows by prefix.
            return [c for c in self.commands if c.value.lower().startswith(text.lower())], None
        slot = completer(text[1 + len(command):])
        if slot is None or not slot.rows:
            return [], None
        return rank_rows(slot.rows, slot.query), slot.query

    def _opens_picker(self, command: str) -> bool:
        """Enter on `/<command>` opens its list: a picker with a row to choose."""
        completer = self.completers.get(command)
        if completer is None or not self.pickers(command):
            return False
        slot = completer(" ")
        return slot is not None and any(_as_row(r).runs for r in slot.rows)

    def enter_choice(self, text: str) -> Tuple[str, str]:
        """What enter does while the menu is open: `("send", line)`, `("insert",
        value)` for the argument being typed, or `("open", "/command ")` for a
        picker (module docstring, "SEARCH").

        A row the line goes on after is inserted, never run: the person sends
        the line once the menu has nothing more to offer. A row that completes
        the command sends the line with it. A row that IS the text already
        typed sends the line as typed: a fully typed `/agent edit` still
        offered `edit`, and enter put a space after it instead of sending
        (2026-09-27). The TypeScript prompt decides the same way.
        """
        items, query = self._menu(text)
        chosen = items[min(self.menu_index, len(items) - 1)]
        if query is None:
            if self._opens_picker(chosen.value[1:]):
                return "open", chosen.value + " "
            return "send", chosen.value
        typed = query.strip().lower()
        value = chosen.insert or chosen.value
        if typed and typed in (chosen.value.lower(), value.lower()):
            return "send", text
        if chosen.runs:
            return "send", text[: len(text) - len(query)] + value
        return "insert", value

    def _insert_argument(self, buffer: Buffer, value: str) -> None:
        """Put an offered value in place of what is typed for it, and a space after it."""
        _items, query = self._menu(buffer.text)
        head = buffer.text[: len(buffer.text) - len(query or "")]
        buffer.text = f"{head}{value} "
        buffer.cursor_position = len(buffer.text)

    def history_suggestions(self, text: Callable[[], str]) -> AutoSuggest:
        """Suggestions from history (→ or tab accepts), but none while the menu is open.

        With both, typing `/` showed the ghost of the last command sent
        (`/exit`) in the box while the menu's highlighted row, the one enter
        runs, was `/help`.
        """
        return ConditionalAutoSuggest(AutoSuggestFromHistory(), Condition(lambda: not self.menu_items(text())))

    def exit_armed(self, now: float) -> bool:
        return self.exit_armed_at > 0 and now - self.exit_armed_at < CONFIRM_SECONDS

    def esc_armed(self, now: float) -> bool:
        return self.esc_armed_at > 0 and now - self.esc_armed_at < CONFIRM_SECONDS

    # -- drawing --------------------------------------------------------------

    def _edge(self, t: float) -> str:
        """The box's edge: the plain border colour the whole way round
        ("Signal", `theme.py`); it used to run through the brand gradient."""
        return self.theme.palette.border

    def _border(self, left: str, right: str):
        width = _box_width()
        if not self.theme.rich_colour:
            return [("class:wa-edge", left + "─" * max(0, width - 2) + right)]
        out = []
        for i in range(width):
            ch = left if i == 0 else right if i == width - 1 else "─"
            out.append((f"fg:{self._edge(i / max(1, width - 1))}", ch))
        return out

    def _below(self, buffer: Buffer):
        """Under the box: the command menu while a command is being typed, else the footer."""
        width = _box_width()
        now = time.monotonic()
        # Drawn above the box, the menu is not part of the application.
        items = [] if self._placement == "above" else self.menu_items(buffer.text)
        out: List[Tuple[str, str]] = []
        if items:
            selected, offset, window, name_width, more = _menu_window(items, self.menu_index, MAX_MENU_ITEMS + 1)
            for index, (name, description, *_rest) in enumerate(window):
                active = offset + index == selected
                room = max(10, width - name_width - 5)
                if active:
                    out += [("class:wa-menu-marker", " ❯ "), ("class:wa-menu-name-active", name.ljust(name_width)),
                            ("class:wa-menu-desc-active", _clip(description, room))]
                else:
                    out += [("", "   "), ("class:wa-menu-name", name.ljust(name_width)),
                            ("class:wa-menu-desc", _clip(description, room))]
                out.append(("", "\n"))
            if more > 0:
                out.append(("class:wa-menu-desc", f"   {more} more, keep typing to narrow\n"))
        else:
            out += self._footer_line(buffer, width, now)
            out.append(("", "\n"))
        for line in self.extra_lines():
            out.append(("class:wa-footer", f" {line}\n"))
        if out and out[-1][1].endswith("\n"):
            out[-1] = (out[-1][0], out[-1][1][:-1])
        return out

    def _footer_line(self, buffer: Buffer, width: int, now: float):
        def hint(key: str, what: str):
            return [("class:wa-footer-key", key), ("class:wa-footer", f" {what}")]

        sep = [("class:wa-footer", " · ")]
        if self.exit_armed(now):
            variants = [[("class:wa-footer-warn", "press ctrl+c again to exit")]]
        elif self.esc_armed(now):
            variants = [[("class:wa-footer-warn", "press esc again to clear")]]
        else:
            hints = (
                [hint("enter", "send"), hint("alt+enter", "new line")]
                if buffer.text
                else [hint("/", "commands"), hint("↑", "history"), hint("ctrl+c", "exit")]
            )
            full: List[Tuple[str, str]] = []
            for index, h in enumerate(hints):
                full += (sep if index else []) + h
            variants = [full, hints[0], []]

        def length(fragments) -> int:
            return sum(len(t) for _, t in fragments)

        parts = list(self.footer())
        for right in variants:
            room = width - 2 - length(right) - (2 if right else 0)
            shown = list(parts)
            while len(shown) > 1 and len(" · ".join(shown)) > room:
                shown.pop()
            if shown and len(shown[0]) > room and right is not variants[-1]:
                continue
            left = _clip(" · ".join(shown), room) if room > 3 and shown else ""
            gap = max(1, width - 1 - len(left) - length(right))
            return [("class:wa-footer", " " + left + " " * gap)] + right
        return []

    # -- the application --------------------------------------------------------

    async def ask(self, placeholder: str) -> Optional[str]:
        """What was sent, or None when the person leaves."""
        self.placeholder = placeholder
        self.shown_at = time.monotonic()
        self.menu_index = 0
        self.menu_dismissed = False
        self.recalled = self._navigating = False
        self.exit_armed_at = self.esc_armed_at = 0.0
        self._reset_placement()
        buffer = Buffer(
            history=self.history,
            auto_suggest=self.history_suggestions(lambda: buffer.text),
            multiline=True,
            enable_history_search=True,
        )

        def changed(_buffer) -> None:
            self.menu_index = 0
            if self._navigating:
                self.recalled = True
                return
            self.recalled = False
            self.menu_dismissed = False

        def moved(_buffer) -> None:
            if not self._navigating:
                self.recalled = False

        buffer.on_text_changed += changed
        buffer.on_cursor_position_changed += moved
        p = self.theme.palette
        menu_open = Condition(lambda: bool(self.menu_items(buffer.text)))
        has_text = Condition(lambda: bool(buffer.text))
        kb = KeyBindings()

        def submit(event, text: str) -> None:
            buffer.append_to_history()
            event.app.exit(result=text)

        def navigate(move) -> None:
            self._navigating = True
            try:
                move()
            finally:
                self._navigating = False

        # ↑ and ↓ with the menu closed walk the history (prompt_toolkit's own
        # moves, under the guard that marks a line as recalled), and so do
        # ctrl+p and ctrl+n.
        @kb.add("up", filter=~menu_open)
        def _(event) -> None:
            navigate(lambda: buffer.auto_up(count=event.arg))

        @kb.add("down", filter=~menu_open)
        def _(event) -> None:
            navigate(lambda: buffer.auto_down(count=event.arg))

        @kb.add("c-p", filter=~menu_open)
        def _(event) -> None:
            navigate(lambda: buffer.history_backward(count=event.arg))

        @kb.add("c-n", filter=~menu_open)
        def _(event) -> None:
            navigate(lambda: buffer.history_forward(count=event.arg))

        @kb.add("up", filter=menu_open)
        def _(event) -> None:
            self.menu_index = (self.menu_index - 1) % len(self.menu_items(buffer.text))

        @kb.add("down", filter=menu_open)
        def _(event) -> None:
            self.menu_index = (self.menu_index + 1) % len(self.menu_items(buffer.text))

        @kb.add("tab", filter=menu_open)
        def _(event) -> None:
            items, query = self._menu(buffer.text)
            chosen = items[min(self.menu_index, len(items) - 1)]
            if query is not None:
                self._insert_argument(buffer, chosen.insert or chosen.value)
                return
            buffer.text = chosen.value + " "
            buffer.cursor_position = len(buffer.text)

        # TAB TAKES THE GREY SUGGESTION, as → does (2026-09-29, the owner: "tab
        # completion won't work"). With the menu closed, tab was prompt_toolkit's
        # `menu-complete`, and the box has no completer, so it did nothing while
        # a suggestion from history stood right after the cursor.
        suggestion_shown = Condition(
            lambda: buffer.suggestion is not None
            and bool(buffer.suggestion.text)
            and buffer.document.is_cursor_at_the_end
        )

        @kb.add("tab", filter=~menu_open & suggestion_shown)
        def _(event) -> None:
            buffer.insert_text(buffer.suggestion.text)

        @kb.add("enter", filter=menu_open)
        def _(event) -> None:
            action, value = self.enter_choice(buffer.text)
            if action == "insert":
                self._insert_argument(buffer, value)
                return
            if action == "open":
                buffer.text = value
                buffer.cursor_position = len(buffer.text)
                return
            submit(event, value)

        @kb.add("escape", filter=menu_open)
        def _(event) -> None:
            self.menu_dismissed = True

        @kb.add("enter", filter=~menu_open)
        def _(event) -> None:
            text = buffer.text
            if text.endswith("\\") and buffer.cursor_position == len(text):
                # A line ending in a backslash continues, as in a shell.
                buffer.text = text[:-1] + "\n"
                buffer.cursor_position = len(buffer.text)
                return
            submit(event, text)

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _(event) -> None:
            buffer.insert_text("\n")

        @kb.add("escape", "/")
        def _(event) -> None:
            # An esc and a `/` typed within the esc timeouts (0.2 s) arrive as
            # alt+/, which prompt_toolkit's emacs keys take for "complete" (the
            # box has no completer), so the `/` was lost. It is what was meant,
            # and it opens the menu (2026-09-28). The TypeScript box does the same.
            buffer.insert_text("/")

        @kb.add("escape", filter=~menu_open & has_text)
        def _(event) -> None:
            now = time.monotonic()
            if self.esc_armed(now):
                buffer.reset()
                self.esc_armed_at = 0.0
            else:
                self.esc_armed_at = now
                event.app.create_background_task(self._fade(event.app))

        @kb.add("c-c")
        def _(event) -> None:
            now = time.monotonic()
            if buffer.text:
                buffer.reset()
                self.exit_armed_at = 0.0
                return
            if self.exit_armed(now):
                event.app.exit(result=None)
                return
            self.exit_armed_at = now
            event.app.create_background_task(self._fade(event.app))

        @kb.add("c-d", filter=~has_text)
        def _(event) -> None:
            event.app.exit(result=None)

        @kb.add("c-l")
        def _(event) -> None:
            # prompt_toolkit's own clear, told to the record: the box is at the
            # top of a blank screen now, and nothing above it is covered.
            event.app.renderer.clear()
            if self.screen is not None:
                self.screen.clear_screen()
            self._reset_placement()
            self._start_row = 1

        edge_left = self._edge(0) if self.theme.rich_colour else p.border
        edge_right = self._edge(1) if self.theme.rich_colour else p.border
        input_window = Window(
            BufferControl(buffer=buffer, input_processors=[_Placeholder(self), AppendAutoSuggestion()]),
            wrap_lines=True,
            get_line_prefix=lambda line, wrap: [("class:wa-prompt", "❯ ")] if line == 0 and wrap == 0 else [("", "  ")],
            height=Dimension(min=1, max=MAX_TEXT_ROWS),
            dont_extend_height=True,
        )
        # The edges take the split's height (the input's rows). With
        # `dont_extend_height` they collapsed to no rows at all and the box
        # had no sides.
        body = VSplit([
            Window(char="│", width=1, style=f"fg:{edge_left}"),
            Window(char=" ", width=1),
            input_window,
            Window(char=" ", width=1),
            Window(char="│", width=1, style=f"fg:{edge_right}"),
            Window(width=1),  # the terminal's last column, left empty (`_box_width`)
        ])
        box = HSplit([
            Window(FormattedTextControl(lambda: self._border("╭", "╮")), height=1),
            body,
            Window(FormattedTextControl(lambda: self._border("╰", "╯")), height=1),
            Window(FormattedTextControl(lambda: self._below(buffer)), dont_extend_height=True),
        ])
        layout = Layout(box, focused_element=input_window)
        # Own names (wa-*): prompt_toolkit's built-in `menu` class, among
        # others, carries a grey background that dotted names would inherit.
        style = Style.from_dict({
            "wa-prompt": f"bold {p.accent}",
            "wa-placeholder": f"italic {p.faint}",
            "auto-suggestion": p.faint,
            "wa-edge": p.border,
            "wa-footer": p.faint,
            "wa-footer-key": p.muted,
            "wa-footer-warn": p.warning,
            "wa-menu-marker": p.accent,
            "wa-menu-name": p.muted,
            "wa-menu-name-active": f"bold {p.accent}",
            "wa-menu-desc": p.faint,
            "wa-menu-desc-active": p.text,
        })
        # The keys that take a suggestion (→, ctrl+e, ctrl+f, alt+f for one
        # word): `PromptSession` loads them, a bare `Application` does not, so
        # until 2026-09-29 the box showed history suggestions that no key could
        # take, whatever this module's docstring said. Tab is bound above.
        bindings = [load_key_bindings(), load_auto_suggest_bindings(), kb]
        if self.extra_bindings is not None:
            bindings.append(self.extra_bindings)
        app: Application = Application(
            layout=layout,
            key_bindings=merge_key_bindings(bindings),
            style=style,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
            before_render=lambda app: self._place(app, buffer),
            after_render=lambda app: self._after_frame(app, buffer),
        )
        # Esc on its own, sooner than the default half second; and as a key of
        # its own (it also begins alt+enter) at once, not after a second.
        app.ttimeoutlen = 0.1
        app.timeoutlen = 0.1
        if self.theme.animate and self.theme.rich_colour:
            app.pre_run_callables.append(lambda: app.create_background_task(self._twinkle(app, buffer)))
        # The box's own drawing is not the conversation: the record waits at the
        # box's first row until the box is gone.
        if self.screen is not None:
            self.screen.paused = True
        try:
            return await app.run_async()
        finally:
            self._finish(app)

    def _finish(self, app: Application) -> None:
        """The box is gone (prompt_toolkit erased it): put back what the menu
        covered, and hand the screen back to the record, told how far the box's
        growth scrolled the terminal."""
        if self._overlay is not None and self._saved is not None:
            self._draw_over(app, self._saved)
        if self.screen is not None:
            if self._start_row is not None and self._last_above is not None:
                self.screen.scrolled(self._start_row - (self._last_above + 1))
            self.screen.paused = False
        self._reset_placement()

    def _rows_above(self, app: Application) -> Optional[int]:
        try:
            return app.renderer.rows_above_layout
        except HeightIsUnknownError:
            return None

    def _place(self, app: Application, buffer: Buffer) -> None:
        """Before each frame: where the menu goes as it opens (module docstring):
        above the box whenever the rows it covers are on screen and known, even
        when it would fit under the box; else under it."""
        if not self.menu_items(buffer.text):
            self._placement = None
            return
        if self._placement is not None:
            return
        self._placement = "below"
        above = self._rows_above(app)
        if above is None or self.screen is None or self.console is None:
            return
        # As many rows as the menu may cover and the record knows, at least MIN_MENU_REACH.
        for count in range(min(MENU_REACH, above), MIN_MENU_REACH - 1, -1):
            saved = self.screen.rows_above(count)
            if saved is not None:
                self._saved = saved
                self._placement = "above"
                return

    def _after_frame(self, app: Application, buffer: Buffer) -> None:
        """After each frame: tie the record to the screen once, and draw over
        the rows above the box, or put them back."""
        above = self._rows_above(app)
        if above is not None:
            self._last_above = above
        if self.screen is not None and self._start_row is None:
            # prompt_toolkit asked where the box starts; the rows below the
            # cursor then are its minimum height (renderer.py), so the row
            # follows from the terminal's height.
            below = getattr(app.renderer, "_min_available_height", 0)
            if below > 0:
                self._start_row = app.output.get_size().rows - below + 1
                self.screen.anchor(self._start_row)
        if self._saved is None:
            return
        if self._placement == "above":
            # The menu has the rows it covers less the blank one over it.
            rows = self._menu_rows(buffer, max(24, app.output.get_size().columns - 1), len(self._saved) - 1)
            cover = ([""] + rows)[-len(self._saved):]
            self._draw_over(app, self._saved[: len(self._saved) - len(cover)] + cover)
            return
        # The menu closed: the conversation's rows go back.
        if self._overlay is not None:
            self._draw_over(app, self._saved)
        self._overlay = None
        self._saved = None

    def _draw_over(self, app: Application, rows: List[str]) -> None:
        """Draw `rows` on the screen rows just above the application, cursor
        saved and restored: prompt_toolkit's frame is not touched."""
        if rows == self._overlay or self._last_above is None:
            return
        first = self._last_above - len(rows) + 1
        out = "\x1b7"
        for index, row in enumerate(rows):
            out += f"\x1b[{first + index};1H\x1b[2K{row}"
        app.output.write_raw(out + "\x1b8")
        app.output.flush()
        self._overlay = list(rows)

    def _menu_rows(self, buffer: Buffer, width: int, max_rows: int = MAX_MENU_ITEMS + 1) -> List[str]:
        """The menu as the rows drawn over the conversation, at most `max_rows`:
        what `_below` draws under the box, in the same colours."""
        items = self.menu_items(buffer.text)
        if not items or self.console is None:
            return []
        p = self.theme.palette
        selected, offset, window, name_width, more = _menu_window(items, self.menu_index, max_rows)
        lines: List[Text] = []
        for index, (name, description, *_rest) in enumerate(window):
            room = max(10, width - name_width - 5)
            if offset + index == selected:
                lines.append(Text.assemble((" ❯ ", p.accent), (name.ljust(name_width), f"bold {p.accent}"),
                                           (_clip(description, room), p.text)))
            else:
                lines.append(Text.assemble("   ", (name.ljust(name_width), p.muted), (_clip(description, room), p.faint)))
        if more > 0:
            lines.append(Text(f"   {more} more, keep typing to narrow", style=p.faint))
        rows = []
        for line in lines:
            with self.console.capture() as capture:
                self.console.print(line, end="", soft_wrap=True)
            rows.append(capture.get())
        return rows

    async def _twinkle(self, app: Application, buffer: Buffer) -> None:
        """Redraws for the starfield, for as long as it lasts and the box is empty."""
        while time.monotonic() - self.shown_at < SPARKLE_SECONDS + 0.2:
            await asyncio.sleep(0.15)
            if not buffer.text:
                app.invalidate()
        app.invalidate()

    async def _fade(self, app: Application) -> None:
        """The "press again" hint goes away on its own."""
        await asyncio.sleep(CONFIRM_SECONDS + 0.05)
        app.invalidate()


def sent_message(theme: ChatTheme, width: int, text: str) -> List[Text]:
    """A sent message as it stays in the conversation: `❯ text` on a shaded band."""
    p = theme.palette
    width = max(24, width - 1)
    rows: List[str] = []
    for line in text.split("\n"):
        while len(line) > width - 4:
            rows.append(line[: width - 4])
            line = line[width - 4:]
        rows.append(line)
    out = []
    for index, row in enumerate(rows):
        prefix = Text("❯ ", style=f"bold {p.accent}") if index == 0 else Text("  ")
        line = Text(" ") + prefix + Text(row, style=p.text)
        if theme.rich_colour:
            line.append(" " * max(0, width - line.cell_len))
            line.stylize(f"on {p.surface}")
        out.append(line)
    return out
