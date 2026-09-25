"""
The moving parts of the chat: the spinner star, the shimmer on the status
verb, the idle starfield in the empty input box (2026-09-24).

Pure functions of time, like their TypeScript twins (`ui/motion.ts`), so a
redraw at any moment shows the right frame and a test can ask for frame N
without waiting. The star is Claude Code's, the shimmer Codex's formula
(`codex-rs/tui/src/summary_shimmer.rs`), the starfield Codex's
`sparkle_field.rs`.
"""

from __future__ import annotations

import math

from rich.text import Text

from .theme import ChatTheme, mix

_STAR = ["·", "✢", "✳", "✶", "✻", "✽"]
#: A star that grows, then shrinks back: played forward then back.
STAR_FRAMES = _STAR + list(reversed(_STAR[1:-1]))
STAR_INTERVAL = 0.12

#: Braille dots, for a tool that is running.
DOT_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
DOT_INTERVAL = 0.08

SHIMMER_PERIOD = 2.0
SPARKS = ["⠁", "⠂", "⠄", "⠈", "⠐", "⠠", "⡀", "⢀"]
#: How long the starfield lasts after the box appears.
SPARKLE_SECONDS = 15.0


def frame_at(frames, interval: float, now: float) -> str:
    return frames[int(now / interval) % len(frames)]


def star_colour(theme: ChatTheme, now: float) -> str:
    """The star's colour: the agent's lime, breathing (full strength to 45%
    toward the background and back every few seconds). One colour, pulsing:
    until 2026-09-25 it cycled through a cyan-to-pink gradient, as Gemini
    CLI's spinner does (`theme.py`, "Signal")."""
    return mix(theme.palette.agent, theme.background, 0.45 * (1 - (math.sin(now / 0.65) + 1) / 2))


def shimmer(theme: ChatTheme, text: str, colour: str, now: float) -> Text:
    """`text` with a band of light sweeping across it every two seconds.

    Codex's formula: the text rests half-faded toward the background and a
    full-strength band, max(10% of the width, 3) characters either side of its
    centre, crosses it. The peak also lifts a little toward white, so the band
    reads on violet. Below 256 colours, or with animation off, plain colour.
    """
    if not theme.animate or not theme.rich_colour:
        return Text(text, style=colour)
    half = max(0.1 * len(text), 3.0)
    position = ((now % SHIMMER_PERIOD) / SHIMMER_PERIOD) * (len(text) + 2 * half) - half
    out = Text()
    for i, ch in enumerate(text):
        distance = min(abs(i + 0.5 - position) / half, 1.0)
        intensity = 0.5 * (1 + math.cos(math.pi * distance))
        lit = mix(colour, "#ffffff", 0.22 * intensity)
        out.append(ch, style=mix(theme.background, lit, 0.5 + 0.5 * intensity))
    return out


def _hash(n: int) -> float:
    x = ((n + 1) * 2654435761) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 2246822519) & 0xFFFFFFFF
    x ^= x >> 13
    return (x & 0xFFFFFFFF) / 4294967296


def spark_at(theme: ChatTheme, column: int, now: float):
    """One cell of the idle starfield: `(glyph, colour)`, or None for an empty cell.

    One cell in five holds a braille dot that twinkles on its own 4 to 7
    second cycle (brightness sin^12, at most 0.55), in the agent's lime.
    Deterministic per column, so a redraw never makes the field jump.
    """
    if _hash(column) >= 0.2:
        return None
    period = 4.0 + _hash(column + 101) * 3.0
    phase = ((now + _hash(column + 211) * period) % period) / period
    brightness = math.sin(math.pi * phase) ** 12 * 0.55
    if brightness < 0.04:
        return None
    colour = mix(theme.background, theme.palette.agent, brightness)
    return SPARKS[int(_hash(column + 307) * len(SPARKS))], colour
