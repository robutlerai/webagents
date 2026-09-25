"""
The chat's colours, in one place (2026-09-24).

The same palette as the TypeScript chat (`typescript/src/cli/ui/theme.ts`),
value for value, both held against `tests/fixtures/cli/theme.json`, so the two
SDKs look like one product.

SIGNAL (2026-09-25, the operator's choice): Robutler's black and white, and one
lime that means the agent. The WEBAGENTS wordmark is white letters with a grey
shadow (black and grey on a light terminal); lime is the agent's mark, its H1
headings, the spinner, the idle sparkles and the light that crosses the
wordmark once. Until that day the wordmark, H1, the input box border and a
hue-cycling spinner ran through a cyan-to-pink gradient, which with the
gradient-edged input box read as Gemini CLI's look. The person's side is plain
white, the input box border the plain border. Green (emerald, so it never
reads as the lime) is a tool that worked, red one that failed, amber a
warning, blue a link. Two variants, because a colour that reads on a dark
terminal is too pale on a light one.

Colour depth is Rich's business: every colour here is `#rrggbb` and Rich
lowers it to 256 or 16 colours, or drops it for NO_COLOR, on its own. What
Rich cannot know is the terminal's BACKGROUND, which the shaded bands (a sent
message, a code block) are mixed from; `ui/terminal.py` asks for it, the way
Gemini CLI and Codex do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional


def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    clamp = lambda v: max(0, min(255, int(round(v))))  # noqa: E731
    return "#{:02x}{:02x}{:02x}".format(clamp(r), clamp(g), clamp(b))


def mix(a: str, b: str, t: float) -> str:
    """Linear mix of two colours, `t` from 0 (a) to 1 (b)."""
    t = max(0.0, min(1.0, t))
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return rgb_to_hex(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)


def gradient_at(stops: List[str], t: float) -> str:
    """The colour at position `t` (0..1) along a gradient through `stops`."""
    if len(stops) == 1:
        return stops[0]
    k = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(len(stops) - 2, int(k))
    return mix(stops[i], stops[i + 1], k - i)


def luminance(value: str) -> float:
    r, g, b = hex_to_rgb(value)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


@dataclass(frozen=True)
class Palette:
    #: The wordmark's letters (the `█` cells) and its shadow (the box-drawing strokes).
    wordmark_face: str
    wordmark_shadow: str
    #: The person: prompt character, their messages, a running tool's dots.
    accent: str
    #: The agent, and the one colour of the brand: reply marker, bullets, H1, the spinner.
    agent: str
    success: str
    error: str
    warning: str
    info: str
    text: str
    muted: str
    faint: str
    border: str
    surface: str
    code_surface: str
    inline_code: str
    code: Dict[str, str] = field(default_factory=dict)


DARK = Palette(
    wordmark_face="#f4f4f5",
    wordmark_shadow="#52525b",
    accent="#fafafa",
    agent="#bef264",
    success="#34d399",
    error="#f87171",
    warning="#fbbf24",
    info="#60a5fa",
    text="#e4e4e7",
    muted="#a1a1aa",
    faint="#71717a",
    border="#52525b",
    surface="#26272b",
    code_surface="#1f2023",
    inline_code="#fdba74",
    code={
        "keyword": "#c084fc",
        "string": "#86efac",
        "number": "#fdba74",
        "comment": "#6b7280",
        "func": "#93c5fd",
        "type": "#67e8f9",
        "property": "#f9a8d4",
        "punctuation": "#9ca3af",
        "added": "#4ade80",
        "removed": "#f87171",
    },
)

LIGHT = Palette(
    wordmark_face="#18181b",
    wordmark_shadow="#a1a1aa",
    accent="#18181b",
    agent="#4d7c0f",
    success="#047857",
    error="#dc2626",
    warning="#b45309",
    info="#2563eb",
    text="#18181b",
    muted="#52525b",
    faint="#71717a",
    border="#a1a1aa",
    surface="#f1f1f2",
    code_surface="#f6f6f7",
    inline_code="#c2410c",
    code={
        "keyword": "#7c3aed",
        "string": "#15803d",
        "number": "#c2410c",
        "comment": "#6b7280",
        "func": "#1d4ed8",
        "type": "#0e7490",
        "property": "#be185d",
        "punctuation": "#4b5563",
        "added": "#15803d",
        "removed": "#b91c1c",
    },
)


@dataclass(frozen=True)
class ChatTheme:
    palette: Palette
    #: The terminal's background: asked for, or assumed from `light`.
    background: str
    light: bool
    #: Animations may run: a terminal, not CI, not switched off.
    animate: bool
    #: At least 256 colours, so shaded bands and the shimmer can be drawn.
    rich_colour: bool


def _light_from_env(env) -> bool:
    """COLORFGBG ("fg;bg", bg 7 or 15 is light), set by xterm, rxvt, iTerm, Konsole."""
    value = env.get("COLORFGBG", "")
    if not value:
        return False
    try:
        return int(value.split(";")[-1]) in (7, 15)
    except ValueError:
        return False


def theme_for(console=None, background: Optional[str] = None, env=None, animate: Optional[bool] = None) -> ChatTheme:
    """The theme for this terminal.

    With the terminal's own `background`, light or dark is read from it rather
    than guessed, and the bands are mixed from it: a lift toward white on a
    dark terminal, toward black on a light one.
    """
    env = os.environ if env is None else env
    light = luminance(background) > 0.55 if background else _light_from_env(env)
    palette = LIGHT if light else DARK
    if background:
        palette = replace(
            palette,
            surface=mix(background, "#000000", 0.06) if light else mix(background, "#ffffff", 0.12),
            code_surface=mix(background, "#000000", 0.035) if light else mix(background, "#ffffff", 0.065),
        )
    interactive = bool(console is not None and getattr(console, "is_terminal", False))
    colour_system = getattr(console, "color_system", None) if console is not None else "truecolor"
    if animate is None:
        animate = interactive and not env.get("CI") and "WEBAGENTS_NO_ANIMATION" not in env and env.get("TERM") != "dumb"
    return ChatTheme(
        palette=palette,
        background=background or ("#ffffff" if light else "#16181d"),
        light=light,
        animate=bool(animate),
        rich_colour=colour_system in ("truecolor", "256"),
    )


def markdown_styles(theme: ChatTheme) -> Dict[str, str]:
    """Rich's markdown style names, in this palette (see `ui/markdown.py` for the elements)."""
    p = theme.palette
    return {
        # H1 in the agent's lime, H2 in the text colour ("Signal", above).
        "markdown.h1": f"bold {p.agent}",
        "markdown.h2": f"bold {p.text}",
        "markdown.h3": f"bold {p.text}",
        "markdown.h4": f"bold {p.muted}",
        "markdown.h5": f"bold {p.muted}",
        "markdown.h6": f"italic {p.muted}",
        "markdown.code": p.inline_code,
        "markdown.link": f"underline {p.info}",
        "markdown.link_url": f"underline {p.info}",
        "markdown.item.bullet": p.agent,
        "markdown.item.number": p.agent,
        "markdown.hr": p.border,
        "markdown.block_quote": f"italic {p.muted}",
        "markdown.table.header": f"bold {p.agent}",
        "markdown.table.border": p.border,
        "markdown.s": f"strike {p.faint}",
        "markdown.em": "italic",
        "markdown.strong": "bold",
    }


def pygments_style(theme: ChatTheme):
    """A Pygments style in this palette, for code blocks (Rich's `Syntax`)."""
    from pygments.style import Style
    from pygments.token import (
        Comment, Generic, Keyword, Name, Number, Operator, Punctuation, String, Text, Token,
    )

    c = theme.palette.code
    text = theme.palette.text

    class ChatStyle(Style):
        background_color = theme.palette.code_surface
        styles = {
            Token: text,
            Text: text,
            Comment: f"italic {c['comment']}",
            Keyword: c["keyword"],
            Keyword.Constant: c["keyword"],
            Keyword.Type: c["type"],
            Name.Builtin: c["type"],
            Name.Class: c["type"],
            Name.Function: c["func"],
            Name.Decorator: c["func"],
            Name.Tag: c["keyword"],
            Name.Attribute: c["property"],
            Name.Variable: c["property"],
            String: c["string"],
            Number: c["number"],
            Operator: c["punctuation"],
            Punctuation: c["punctuation"],
            Generic.Inserted: c["added"],
            Generic.Deleted: c["removed"],
            Generic.Heading: f"bold {c['type']}",
            Generic.Subheading: c["type"],
        }

    return ChatStyle
