/**
 * A hint names the command as the user must type it here (2026-09-25).
 *
 * Under `webagents --profile local`, "Run `webagents login`" signed the
 * DEFAULT profile in and left the local one signed out. Every hint now goes
 * through `cliCommand`, which adds `--profile` while one is active; the cases
 * are shared with the Python CLI (`python/tests/fixtures/cli/cli_command.json`,
 * run there by `tests/cli/test_cli_command.py`).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { cliCommand } from '../../../src/cli/config-store';
import { presentFailure } from '../../../src/cli/failures';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/cli_command.json'), 'utf8'),
) as {
  cases: { profile: string | null; rest: string; expected: string }[];
  hints: { profile: string; sign_in: string; credits: string };
};

vi.mock('../../../src/cli/credentials.js', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getToken: async () => undefined,
}));

const saved = process.env.WEBAGENTS_PROFILE;
afterEach(() => {
  if (saved === undefined) delete process.env.WEBAGENTS_PROFILE;
  else process.env.WEBAGENTS_PROFILE = saved;
});

function useProfile(profile: string | null): void {
  if (profile === null) delete process.env.WEBAGENTS_PROFILE;
  else process.env.WEBAGENTS_PROFILE = profile;
}

describe('cliCommand', () => {
  it.each(FIXTURE.cases)('$profile: $rest', ({ profile, rest, expected }) => {
    useProfile(profile);
    expect(cliCommand(rest)).toBe(expected);
  });

  it('puts the profile into the failure hints', () => {
    useProfile(FIXTURE.hints.profile);
    const proxyUrl = 'wss://robutler.example/llm';
    expect(presentFailure('does not run models for a CLI sign-in', { proxyUrl }).hint).toBe(FIXTURE.hints.sign_in);
    expect(presentFailure('insufficient credits', { proxyUrl }).hint).toBe(FIXTURE.hints.credits);
  });

  it("puts the profile into whoami's fix", async () => {
    useProfile('local');
    const { whoAmI } = await import('../../../src/cli/account');
    expect((await whoAmI()).fix).toBe('Run `webagents --profile local login`.');
  });
});
