/**
 * A mistyped command, said the way commander and the Python CLI say it
 * (2026-09-24, `src/cli/suggest.ts`).
 */

import { describe, expect, it } from 'vitest';

import { firstOperandIndex, suggestSimilar } from '../../../src/cli/suggest';

const COMMANDS = ['chat', 'connect', 'serve', 'daemon', 'login', 'logout', 'whoami', 'link', 'unlink', 'publish', 'help'];

describe('suggestSimilar (commander 14)', () => {
  it('suggests the closest command', () => {
    expect(suggestSimilar('publsh', COMMANDS)).toBe('\n(Did you mean publish?)');
  });

  it('picks the nearest, and lists every one at that distance', () => {
    expect(suggestSimilar('logn', COMMANDS)).toBe('\n(Did you mean login?)');
    expect(suggestSimilar('ab', ['abc', 'abd'])).toBe('\n(Did you mean one of abc, abd?)');
  });

  it('suggests nothing for a word that is nothing like one', () => {
    expect(suggestSimilar('xyzzy', COMMANDS)).toBe('');
  });

  it('matches long options by their names', () => {
    expect(suggestSimilar('--modle', ['--model', '--agent', '--help'])).toBe('\n(Did you mean --model?)');
  });
});

describe('firstOperandIndex', () => {
  const known = new Set(['-p', '--prompt', '--json', '--profile', '-h']);
  const takesValue = new Set(['-p', '--prompt', '--profile']);

  it('skips options and their values', () => {
    expect(firstOperandIndex(['--profile', 'local', 'publsh'], known, takesValue)).toBe(2);
    expect(firstOperandIndex(['-p', 'hello there'], known, takesValue)).toBe(-1);
    expect(firstOperandIndex(['--prompt=hi', 'extra'], known, takesValue)).toBe(1);
  });

  it('leaves an unknown flag to commander, which suggests the option', () => {
    expect(firstOperandIndex(['--modle', 'x'], known, takesValue)).toBe(-1);
  });
});
