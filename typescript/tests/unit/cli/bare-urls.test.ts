/**
 * Which URLs in an answer are drawn as links, the same in both CLIs
 * (2026-09-25). The cases are `python/tests/fixtures/cli/bare_urls.json`,
 * which the Python chat runs through its markdown (`tests/cli/test_bare_urls.py`).
 * A bare URL stops at a quote now: a URL inside a JSON answer used to run to
 * the next space.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { inline } from '../../../src/cli/render';
import { themeFor } from '../../../src/cli/ui/theme';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/bare_urls.json'), 'utf8'),
) as { cases: { line: string; links: string[]; block?: string }[] };

const colour = themeFor({ isTTY: true }, {}, { depth: 16777216, animate: false });

/** The targets of the OSC 8 hyperlinks in `out`, in order. */
function links(out: string): string[] {
  return [...out.matchAll(/\x1b\]8;[^;]*;([^\x1b\x07]+)/g)].map((m) => m[1]);
}

describe('bare URLs in an answer', () => {
  it.each(FIXTURE.cases)('$line', ({ line, links: expected }) => {
    expect(links(inline(colour, line))).toEqual(expected);
  });
});
