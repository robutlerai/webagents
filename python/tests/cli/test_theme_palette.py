"""The chat's palette and wordmark, the same in both CLIs (2026-09-25, "Signal").

The palette is held value for value against `tests/fixtures/cli/theme.json`,
which the TypeScript chat is checked against too
(`typescript/tests/unit/cli/theme-palette.test.ts`), so the two cannot drift.
The wordmark is indented one column and needs 80: at 80 columns the full art
is centred and never writes the last column.
"""

import dataclasses
import json
import math
from pathlib import Path

from rich.console import Console

from webagents.cli.ui.banner import FULL_WORDMARK_COLUMNS, wordmark
from webagents.cli.ui.motion import star_colour
from webagents.cli.ui.theme import DARK, LIGHT, mix, theme_for

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "cli" / "theme.json").read_text())


def _theme():
    """The dark theme on a truecolour terminal whose background was not reported."""
    return theme_for(Console(force_terminal=True, color_system="truecolor", width=100), env={})


def test_both_palettes_are_the_shared_ones():
    assert dataclasses.asdict(DARK) == FIXTURE["dark"]
    assert dataclasses.asdict(LIGHT) == FIXTURE["light"]


def test_the_full_wordmark_is_one_column_in_and_fits_80_columns():
    theme = _theme()
    assert FULL_WORDMARK_COLUMNS == 80
    lines = [line.plain for line in wordmark(theme, 80)]
    assert len(lines) == 6
    assert all(line.startswith(" ") and not line.startswith("  ") for line in lines[:5])
    assert max(len(line.rstrip()) for line in lines) == 79
    assert len(wordmark(theme, 79)) == 3


def test_letters_and_shadow_take_their_own_colours():
    theme = _theme()
    first = wordmark(theme, 80)[0]
    styles = {first.plain[span.start]: str(span.style) for span in first.spans}
    assert styles["█"] == DARK.wordmark_face
    assert styles["╗"] == DARK.wordmark_shadow
    # The three-line art has no shadow strokes: all letters.
    small = wordmark(theme, 60)[0]
    assert {str(span.style) for span in small.spans} == {DARK.wordmark_face}


def test_the_spinner_breathes_in_the_agents_lime():
    """One colour, pulsing: full lime at the top of the breath, 45% toward the
    background at the bottom. It used to cycle through the brand gradient."""
    theme = _theme()
    top, bottom = 0.65 * math.pi / 2, 0.65 * 3 * math.pi / 2
    assert star_colour(theme, top) == DARK.agent
    assert star_colour(theme, bottom) == mix(DARK.agent, theme.background, 0.45)
