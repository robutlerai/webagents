/**
 * How a finished tool call is drawn: refusals as failures, and the sandbox's
 * hint on its own line (the ptypass-fixes lane, 2026-09-27; fixture
 * `python/tests/fixtures/cli/ptypass_fixes_tool_lines.json`). The Python twin
 * is `python/tests/cli/test_ptypass_fixes_tool_lines.py`.
 *
 * WHY. The real-terminal PTY pass saw a refused `.env` read drawn as a green
 * "Read 1 lines", "The owner declined..." in green, and a command's sandbox
 * hint hidden behind "(+N lines)" in the collapsed tool line, so the one
 * sentence naming the switch never reached the person.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { HINT_PREFIX, TurnPrinter, splitHints, toolFailed, toolResultSummary } from '../../../src/cli/render';
import { REFUSAL_HINTS } from '../../../src/sandbox/index';
import type { StreamChunk } from '../../../src/core/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/ptypass_fixes_tool_lines.json'), 'utf8'));

function draw(chunks: StreamChunk[], columns: number, live = true): string {
  let written = '';
  const out = { isTTY: true, columns, rows: 24, write: (chunk: string) => ((written += chunk), true) } as unknown as NodeJS.WriteStream;
  const printer = new TurnPrinter({ out, color: false, live });
  for (const chunk of chunks) printer.feed(chunk);
  printer.finish();
  return written;
}

const call = (id: string, name: string, args: string): StreamChunk => ({ type: 'tool_call', tool_call: { id, name, arguments: args } });
const result = (id: string, text: string): StreamChunk => ({ type: 'tool_result', tool_result: { call_id: id, result: text, is_error: false } });

describe('refusals are failures', () => {
  it('for every sentence the fixture lists, and not for the look-alikes', () => {
    for (const text of FIXTURE.failed as string[]) expect(toolFailed(false, text), text).toBe(true);
    for (const text of FIXTURE.succeeded as string[]) expect(toolFailed(false, text), text).toBe(false);
    for (const word of FIXTURE.failure_words.words as string[]) expect(toolFailed(false, `${word[0].toUpperCase()}${word.slice(1)}: x`), word).toBe(true);
  });

  it('a refused read is not "Read 1 lines"', () => {
    const refused = (FIXTURE.failed as string[])[0];
    // Whole, never cut mid-word (2026-09-28, `cli/final_sdk_low_items.json`):
    // this pinned "... and the fil…".
    expect(toolResultSummary('read_file', refused, toolFailed(false, refused))).toBe(refused);
    const output = draw([call('c1', 'read_file', '{"path": ".env"}'), result('c1', refused)], 120);
    expect(output).not.toContain('Read 1 lines');
    expect(output).toContain('Refused: .env is where');
  });
});

describe('the sandbox hint', () => {
  it('comes out of the summary and is kept whole', () => {
    const c = FIXTURE.hint.case as { result: string; summary: string; hints: string[] };
    expect(HINT_PREFIX).toBe(FIXTURE.hint.prefix);
    expect(splitHints(c.result).hints).toEqual(c.hints);
    expect(toolResultSummary('runCommand', c.result, toolFailed(false, c.result))).toBe(c.summary);
    expect(Object.values(REFUSAL_HINTS)).toContain(c.hints[0]);
  });

  it('is drawn on its own lines under the tool, wrapped at words, in the chat and in plain output', () => {
    const c = FIXTURE.hint.case as { result: string; hints: string[] };
    for (const live of [true, false]) {
      const output = draw([call('c1', 'runCommand', '{"command": "curl -sS https://example.com"}'), result('c1', c.result)], 60, live);
      const flat = output.replace(/\s+/g, ' ');
      expect(flat, `live ${live}`).toContain(`${HINT_PREFIX}${c.hints[0]}`.replace(/\s+/g, ' '));
      for (const line of output.split('\n')) expect(line.length, line).toBeLessThanOrEqual(live ? 60 : 200);
    }
  });
});
