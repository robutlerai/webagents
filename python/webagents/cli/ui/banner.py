"""
The chat's opening screen: the WEBAGENTS wordmark and a card saying which
agent this is, which model it runs and what it is made of (2026-09-24).

The wordmark is the one `splash.py` always printed, drawn the way the
TypeScript chat draws it (`typescript/src/cli/ui/banner.ts`): its letters in
the palette's `wordmark_face` and its shadow strokes in `wordmark_shadow`
(`theme.py`, "Signal"), and on a terminal that animates, a band of the agent's
lime crossing it once as it appears (about half a second, never again in the
session).

ONE COLUMN IN, 79 WIDE (2026-09-25). It was indented two columns, so at 80
columns the art ran into the last column and the full wordmark was kept for 82
and up; a standard 80-column terminal got the three-line one. One column of
margin each side centres the 78-column art at 80 columns and never writes the
last column. Narrower terminals get the three-line wordmark, all in the
letters' colour (it has no shadow strokes).
"""

from __future__ import annotations

import math
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console, Group
from rich.text import Text

from .splash import WEBAGENTS_LOGO_BLOCK, WEBAGENTS_LOGO_DOUBLE
from .theme import ChatTheme, mix

_WORDMARK = WEBAGENTS_LOGO_BLOCK.strip("\n").split("\n")
_WORDMARK_SMALL = WEBAGENTS_LOGO_DOUBLE.strip("\n").split("\n")


#: The columns the full wordmark needs: one of margin, 78 of art, and the last
#: column left empty (module docstring, "One column in").
FULL_WORDMARK_COLUMNS = 80
_INDENT = " "


def _art(columns: int) -> List[str]:
    return _WORDMARK if columns >= FULL_WORDMARK_COLUMNS else _WORDMARK_SMALL


def wordmark(theme: ChatTheme, columns: int, sweep: float = -1.0) -> List[Text]:
    """The wordmark's lines; `sweep` is the band of light's position, -1 for none.

    The full art's letters are its `█` cells and the rest its shadow; the
    three-line art has no shadow strokes, so it is all letters."""
    art = _art(columns)
    p = theme.palette
    all_face = art is _WORDMARK_SMALL
    lines = []
    for row, line in enumerate(art):
        text = Text(_INDENT)
        for col, ch in enumerate(line):
            if ch == " ":
                text.append(" ")
                continue
            position = col + row * 2
            colour = p.wordmark_face if all_face or ch == "█" else p.wordmark_shadow
            if sweep >= 0:
                distance = abs(position - sweep)
                if distance < 6:
                    colour = mix(colour, p.agent, ((math.cos(distance / 6 * math.pi) + 1) / 2) * 0.85)
            text.append(ch, style=colour)
        lines.append(text)
    return lines


def play_wordmark(console: Console, theme: ChatTheme) -> None:
    """Draws the wordmark; with animation on, the light crosses it once first."""
    columns = console.width
    console.print()
    if not theme.animate or not theme.rich_colour:
        console.print(Group(*wordmark(theme, columns)))
        return
    from rich.live import Live

    art = _art(columns)
    span = max(len(line) for line in art) + len(art) * 2
    frames = 18
    with Live(Group(*wordmark(theme, columns, 0)), console=console, auto_refresh=False, transient=False) as live:
        for frame in range(frames + 1):
            sweep = -1.0 if frame == frames else -6 + frame / (frames - 1) * (span + 12)
            live.update(Group(*wordmark(theme, columns, sweep)), refresh=True)
            time.sleep(0.028)


@dataclass
class WelcomeInfo:
    agent: str
    description: str = ""
    model: str = ""
    #: The agent's tools by name, as the TypeScript card lists them.
    tools: List[str] = field(default_factory=list)
    folder: str = ""
    #: Shown in the card in amber: a missing key, a skill that failed to load.
    warnings: List[str] = field(default_factory=list)
    version: Optional[str] = None


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _truncate_start(text: str, width: int) -> str:
    return text if len(text) <= width else "…" + text[-max(0, width - 1):]


def welcome_card(theme: ChatTheme, columns: int, info: WelcomeInfo) -> List[Text]:
    """The card under the wordmark, and the line of keys under the card."""
    p = theme.palette
    width = max(40, min(columns - 1, 80))
    inner = width - 4

    title = Text.assemble(" ", ("✦", p.agent), " ", (info.agent, f"bold {p.text}"), " ")
    version = Text(f" webagents {info.version} ", style=p.faint) if info.version and info.version != "?" else Text("")
    top = Text("╭─", style=p.border)
    top.append_text(title)
    top.append("─" * max(0, width - 4 - title.cell_len - version.cell_len), style=p.border)
    top.append_text(version)
    top.append("─╮", style=p.border)

    def row(content: Optional[Text] = None) -> Text:
        content = content or Text("")
        line = Text("│ ", style=p.border)
        line.append_text(content)
        line.append(" " * max(0, inner - content.cell_len))
        line.append(" │", style=p.border)
        return line

    def field_row(name: str, value: str, keep_end: bool = False) -> Text:
        shown = (_truncate_start if keep_end else _truncate)(value, inner - 7)
        return row(Text.assemble((name.ljust(7), p.faint), (shown, p.text)))

    lines = [top]
    if info.description:
        lines.append(row(Text(_truncate(info.description, inner), style=p.muted)))
    lines.append(row())
    if info.model:
        lines.append(field_row("model", info.model))
    if info.tools:
        shown: List[str] = []
        used = 0
        for tool in info.tools:
            if used + len(tool) + 2 > inner - 7 - 10:
                break
            shown.append(tool)
            used += len(tool) + 2
        more = len(info.tools) - len(shown)
        lines.append(field_row("tools", ", ".join(shown) + (f"  +{more} more" if more > 0 else "")))
    if info.folder:
        lines.append(field_row("folder", info.folder, keep_end=True))
    for warning in info.warnings:
        # Wrapped, not cut: the advice is usually the end of the sentence.
        lines.append(row())
        for index, part in enumerate(textwrap.wrap(warning, max(10, inner - 2)) or [""]):
            lines.append(row(Text.assemble(("▲ " if index == 0 else "  ", p.warning), (part, p.warning))))
    lines.append(Text(f"╰{'─' * (width - 2)}╯", style=p.border))

    hints = Text(" ")
    for index, (key, what) in enumerate((("enter", "send"), ("/", "commands"), ("↑", "history"), ("esc", "stop a reply"))):
        if index:
            hints.append("  ·  ", style=p.faint)
        hints.append(key, style=p.muted)
        hints.append(f" {what}", style=p.faint)
    lines.append(hints)
    return lines
