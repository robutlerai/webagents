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
    (↑ searches it by what is typed), suggestions from history (→ accepts),
    alt+enter or ctrl+j for a new line, and the caller's own chords (Ctrl+T).

Ctrl+C clears the box, and on an empty box a second one within two seconds
leaves; Ctrl+D on an empty box leaves; esc twice clears. The box is erased
when it is done, and the caller prints the sent message in its place.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, List, Optional, Sequence, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, ConditionalAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import AppendAutoSuggestion, Processor, Transformation, TransformationInput
from prompt_toolkit.styles import Style
from rich.text import Text

from .motion import SPARKLE_SECONDS, spark_at
from .theme import ChatTheme

#: How long a first ctrl+c (leave) or esc (clear) waits for its second.
CONFIRM_SECONDS = 2.0
MAX_MENU_ITEMS = 6
MAX_TEXT_ROWS = 10


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


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
    ) -> None:
        self.theme = theme
        #: `(display, description)`, display like "/help" or "/agent list".
        self.commands = list(commands)
        self.footer = footer
        self.extra_lines = extra_lines
        self.history = history or InMemoryHistory()
        self.extra_bindings = key_bindings
        self.placeholder = ""
        self.shown_at = time.monotonic()
        self.menu_index = 0
        self.menu_dismissed = False
        self.exit_armed_at = 0.0
        self.esc_armed_at = 0.0

    # -- state ----------------------------------------------------------------

    def menu_items(self, text: str) -> List[Tuple[str, str]]:
        """The commands the menu offers for `text`; empty when it is closed."""
        if self.menu_dismissed or not text.startswith("/") or "\n" in text:
            return []
        query = text.lower()
        prefix = [c for c in self.commands if c[0].lower().startswith(query)]
        if " " in text:
            return prefix
        inside = [c for c in self.commands if c not in prefix and query[1:] in c[0].lower()]
        return prefix + inside

    def history_suggestions(self, text: Callable[[], str]) -> AutoSuggest:
        """Suggestions from history (→ accepts), but none while the menu is open.

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
        width = get_app().output.get_size().columns
        if not self.theme.rich_colour:
            return [("class:wa-edge", left + "─" * max(0, width - 2) + right)]
        out = []
        for i in range(width):
            ch = left if i == 0 else right if i == width - 1 else "─"
            out.append((f"fg:{self._edge(i / max(1, width - 1))}", ch))
        return out

    def _below(self, buffer: Buffer):
        """Under the box: the command menu while a command is being typed, else the footer."""
        width = get_app().output.get_size().columns
        now = time.monotonic()
        items = self.menu_items(buffer.text)
        out: List[Tuple[str, str]] = []
        if items:
            selected = min(self.menu_index, len(items) - 1)
            offset = max(0, min(selected - MAX_MENU_ITEMS + 1, len(items) - MAX_MENU_ITEMS))
            window = items[offset: offset + MAX_MENU_ITEMS]
            name_width = max(len(c[0]) for c in items[:MAX_MENU_ITEMS] + window) + 2
            for index, (name, description) in enumerate(window):
                active = offset + index == selected
                room = max(10, width - name_width - 5)
                if active:
                    out += [("class:wa-menu-marker", " ❯ "), ("class:wa-menu-name-active", name.ljust(name_width)),
                            ("class:wa-menu-desc-active", _clip(description, room))]
                else:
                    out += [("", "   "), ("class:wa-menu-name", name.ljust(name_width)),
                            ("class:wa-menu-desc", _clip(description, room))]
                out.append(("", "\n"))
            if len(items) > MAX_MENU_ITEMS:
                out.append(("class:wa-menu-desc", f"   {len(items) - MAX_MENU_ITEMS} more, keep typing to narrow\n"))
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
        self.exit_armed_at = self.esc_armed_at = 0.0
        buffer = Buffer(
            history=self.history,
            auto_suggest=self.history_suggestions(lambda: buffer.text),
            multiline=True,
            enable_history_search=True,
        )

        def changed(_buffer) -> None:
            self.menu_index = 0
            self.menu_dismissed = False

        buffer.on_text_changed += changed
        p = self.theme.palette
        menu_open = Condition(lambda: bool(self.menu_items(buffer.text)))
        has_text = Condition(lambda: bool(buffer.text))
        kb = KeyBindings()

        def submit(event, text: str) -> None:
            buffer.append_to_history()
            event.app.exit(result=text)

        @kb.add("up", filter=menu_open)
        def _(event) -> None:
            self.menu_index = (self.menu_index - 1) % len(self.menu_items(buffer.text))

        @kb.add("down", filter=menu_open)
        def _(event) -> None:
            self.menu_index = (self.menu_index + 1) % len(self.menu_items(buffer.text))

        @kb.add("tab", filter=menu_open)
        def _(event) -> None:
            items = self.menu_items(buffer.text)
            chosen = items[min(self.menu_index, len(items) - 1)][0]
            buffer.text = chosen + " "
            buffer.cursor_position = len(buffer.text)

        @kb.add("enter", filter=menu_open)
        def _(event) -> None:
            items = self.menu_items(buffer.text)
            submit(event, items[min(self.menu_index, len(items) - 1)][0])

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
        ])
        layout = Layout(
            HSplit([
                Window(FormattedTextControl(lambda: self._border("╭", "╮")), height=1),
                body,
                Window(FormattedTextControl(lambda: self._border("╰", "╯")), height=1),
                Window(FormattedTextControl(lambda: self._below(buffer)), dont_extend_height=True),
            ]),
            focused_element=input_window,
        )
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
        bindings = [load_key_bindings(), kb]
        if self.extra_bindings is not None:
            bindings.append(self.extra_bindings)
        app: Application = Application(
            layout=layout,
            key_bindings=merge_key_bindings(bindings),
            style=style,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
        )
        # Esc on its own, sooner than the default half second.
        app.ttimeoutlen = 0.1
        if self.theme.animate and self.theme.rich_colour:
            app.pre_run_callables.append(lambda: app.create_background_task(self._twinkle(app, buffer)))
        return await app.run_async()

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
