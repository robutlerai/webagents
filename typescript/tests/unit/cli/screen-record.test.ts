/**
 * The chat's record of the screen (2026-09-25), held to the fixture the
 * Python chat's record reads too (`python/tests/fixtures/cli/screen_record.json`).
 *
 * The `/` menu opens over the last rows of the conversation and draws them
 * again when it closes (`src/cli/ui/input.ts`), so what the record says those
 * rows hold is what the person sees afterwards: a wrong row here is a wrong
 * row on screen, and an unknown row must say so rather than guess.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ScreenRecord, recordScreen } from '../../../src/cli/ui/screen';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/screen_record.json'), 'utf8'),
) as {
  cases: Array<{
    name: string;
    columns: number;
    rows: number;
    steps: Array<{ write?: string; anchor?: number; scrolled?: number; resize?: [number, number] }>;
    checks: Array<{ above: number; plain: string[] | null; rendered?: string[] }>;
  }>;
};

describe('the screen record (shared fixture)', () => {
  for (const c of FIXTURE.cases) {
    it(c.name, () => {
      let columns = c.columns;
      let rows = c.rows;
      const screen = new ScreenRecord(
        () => columns,
        () => rows,
      );
      for (const step of c.steps) {
        if (step.write !== undefined) screen.feed(step.write);
        if (step.anchor !== undefined) screen.anchor(step.anchor);
        if (step.scrolled !== undefined) screen.scrolled(step.scrolled);
        if (step.resize) [columns, rows] = step.resize;
      }
      for (const check of c.checks) {
        expect(screen.plainRowsAbove(check.above), `${check.above} rows`).toEqual(check.plain);
        if (check.rendered) expect(screen.rowsAbove(check.above)).toEqual(check.rendered);
        if (check.plain === null) expect(screen.rowsAbove(check.above)).toBeNull();
      }
    });
  }
});

describe('recording a stream', () => {
  function stream() {
    const written: string[] = [];
    return {
      written,
      stream: {
        write(chunk: string | Uint8Array) {
          written.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'));
          return true;
        },
      } as unknown as NodeJS.WriteStream,
    };
  }

  it('sees what both streams write, in order, and passes it on untouched', () => {
    const out = stream();
    const err = stream();
    const { screen, stop } = recordScreen([out.stream, err.stream], () => 40, () => 10);
    out.stream.write('one\n');
    err.stream.write('two\n');
    out.stream.write(Buffer.from('thr'));
    out.stream.write(Buffer.from('ee\n'));
    expect(screen.plainRowsAbove(3)).toEqual(['one', 'two', 'three']);
    expect(out.written.join('')).toBe('one\nthree\n');
    expect(err.written).toEqual(['two\n']);
    stop();
    out.stream.write('four\n');
    expect(screen.plainRowsAbove(1)).toEqual(['three']);
  });

  it('keeps a multi-byte character split across two writes whole', () => {
    const out = stream();
    const { screen } = recordScreen([out.stream], () => 40, () => 10);
    const bytes = Buffer.from('日本\n');
    out.stream.write(bytes.subarray(0, 4));
    out.stream.write(bytes.subarray(4));
    expect(screen.plainRowsAbove(1)).toEqual(['日本']);
  });

  it('records nothing while paused (the input box draws itself)', () => {
    const screen = new ScreenRecord(() => 40, () => 10);
    screen.feed('conversation\n');
    screen.paused = true;
    screen.feed('╭──╮\n│ │\n╰──╯');
    screen.paused = false;
    expect(screen.plainRowsAbove(1)).toEqual(['conversation']);
  });
});
