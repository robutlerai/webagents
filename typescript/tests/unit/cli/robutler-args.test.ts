/**
 * The `robutler` command, the same in both CLIs (2026-09-25): its help, and
 * the `webagents` command line its options become. The cases are
 * `python/tests/fixtures/cli/robutler.json`, which the Python package's
 * `robutler` runs too (`tests/cli/test_robutler_command.py`).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { USAGE, robutlerCommand } from '../../../src/cli/robutler-args';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/robutler.json'), 'utf8'),
) as { usage: string; cases: { argv: string[]; run?: string[]; help?: boolean; error?: string }[] };

describe('robutler', () => {
  it('prints the shared help', () => {
    expect(USAGE).toBe(FIXTURE.usage);
  });

  it.each(FIXTURE.cases)('robutler $argv', ({ argv, run, help, error }) => {
    const command = robutlerCommand(argv);
    if (help) expect(command).toEqual({ kind: 'help' });
    else if (error) expect(command).toEqual({ kind: 'error', message: error });
    else expect(command).toEqual({ kind: 'run', argv: run });
  });
});
