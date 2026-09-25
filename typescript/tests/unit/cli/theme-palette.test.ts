/**
 * The chat's palette and wordmark, the same in both CLIs (2026-09-25, "Signal").
 *
 * The palette is held value for value against
 * `python/tests/fixtures/cli/theme.json`, which the Python chat is checked
 * against too (`python/tests/cli/test_theme_palette.py`), so the two cannot
 * drift. The wordmark is indented one column and needs 80: at 80 columns the
 * full art is centred and never writes the last column.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { mix, stripAnsi } from '../../../src/cli/ui/ansi';
import { FULL_WORDMARK_COLUMNS, wordmark } from '../../../src/cli/ui/banner';
import { layoutPrompt, InputEditor } from '../../../src/cli/ui/input';
import { starColour } from '../../../src/cli/ui/motion';
import { DARK, LIGHT, themeFor, type Palette } from '../../../src/cli/ui/theme';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/theme.json'), 'utf8'),
) as { dark: Record<string, unknown>; light: Record<string, unknown> };

/** The palette under the Python field names the fixture uses (`wordmarkFace` is `wordmark_face`). */
function asFixture(palette: Palette): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(palette).map(([key, value]) => [key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`), value]),
  );
}

const colour = themeFor({ isTTY: true }, {}, { depth: 16777216, animate: false });

/** The 24-bit colour an SGR sequence sets, as #rrggbb. */
function sgrColour(sgr: string): string {
  const [r, g, b] = sgr.split(';').slice(2).map(Number);
  return `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

describe('the chat palette (2026-09-25, Signal)', () => {
  it('is the shared palette, dark and light', () => {
    expect(asFixture(DARK)).toEqual(FIXTURE.dark);
    expect(asFixture(LIGHT)).toEqual(FIXTURE.light);
  });

  it('draws the full wordmark one column in, 79 wide, from 80 columns', () => {
    expect(FULL_WORDMARK_COLUMNS).toBe(80);
    const lines = wordmark(colour, 80).map(stripAnsi);
    expect(lines).toHaveLength(6);
    for (const line of lines.slice(0, 5)) expect(line.startsWith(' ') && !line.startsWith('  ')).toBe(true);
    expect(Math.max(...lines.map((line) => line.trimEnd().length))).toBe(79);
    expect(wordmark(colour, 79)).toHaveLength(3);
  });

  it('gives the letters and the shadow their own colours', () => {
    const first = wordmark(colour, 80)[0];
    const cells = [...first.matchAll(/\x1b\[(38;2;\d+;\d+;\d+)m(.)/gu)].map((m) => [m[2], sgrColour(m[1])]);
    expect(cells.find(([ch]) => ch === '█')?.[1]).toBe(DARK.wordmarkFace);
    expect(cells.find(([ch]) => ch === '╗')?.[1]).toBe(DARK.wordmarkShadow);
    // The three-line art has no shadow strokes: all letters.
    const small = [...wordmark(colour, 60)[0].matchAll(/\x1b\[(38;2;\d+;\d+;\d+)m/g)].map((m) => sgrColour(m[1]));
    expect(new Set(small)).toEqual(new Set([DARK.wordmarkFace]));
  });

  it('breathes the spinner in the agent lime', () => {
    const top = 650 * (Math.PI / 2);
    const bottom = 650 * ((3 * Math.PI) / 2);
    expect(starColour(colour, top)).toBe(DARK.agent);
    expect(starColour(colour, bottom)).toBe(mix(DARK.agent, colour.background, 0.45));
  });

  it('draws the input box in the plain border colour', () => {
    const frame = layoutPrompt(colour, new InputEditor([], []), 60, { left: [] }, 'Message', 0);
    const edges = new Set([...frame.lines[0].matchAll(/\x1b\[(38;2;\d+;\d+;\d+)m/g)].map((m) => sgrColour(m[1])));
    expect(edges).toEqual(new Set([DARK.border]));
  });
});
