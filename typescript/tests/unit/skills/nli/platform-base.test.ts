/**
 * `@name` is an agent on the platform (2026-09-25): `{baseUrl}/agents/{name}`,
 * with `baseUrl` from the one platform lookup (`src/skills/platform-url.ts`).
 * It defaulted to https://portal.webagents.ai, which does not resolve. The
 * Python skill is held to the same default (`TestTheDefaultBase` in
 * `python/tests/test_nli_skill.py`).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NLISkill } from '../../../../src/skills/nli/skill.js';

/** What the CLI's `resolvePlatformUrl` answers, per test (see `discovery/search.test.ts`). */
const cliPlatform = vi.hoisted(() => ({ answer: undefined as undefined | (() => [string, string]) }));
vi.mock('../../../../src/cli/config-store.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../../src/cli/config-store.js')>();
  return {
    ...actual,
    resolvePlatformUrl: (...args: Parameters<typeof actual.resolvePlatformUrl>) =>
      cliPlatform.answer ? cliPlatform.answer() : actual.resolvePlatformUrl(...args),
  };
});

describe('where @name goes', () => {
  const saved = { ...process.env };
  beforeEach(() => {
    for (const name of ['ROBUTLER_API_URL', 'ROBUTLER_INTERNAL_API_URL', 'WEBAGENTS_PROFILE']) delete process.env[name];
  });
  afterEach(() => {
    process.env = { ...saved };
    cliPlatform.answer = undefined;
  });

  it('is the configured base first, then ROBUTLER_API_URL, then ROBUTLER_INTERNAL_API_URL', () => {
    process.env.ROBUTLER_API_URL = 'https://api.example.com/';
    process.env.ROBUTLER_INTERNAL_API_URL = 'http://internal:3000';
    expect(new NLISkill({ baseUrl: 'https://mine.example.com/' }).normalizeUrl('@bob')).toBe('https://mine.example.com/agents/bob');
    expect(new NLISkill().normalizeUrl('@bob')).toBe('https://api.example.com/agents/bob');
    delete process.env.ROBUTLER_API_URL;
    expect(new NLISkill().normalizeUrl('@bob')).toBe('http://internal:3000/agents/bob');
  });

  it('is the portal the CLI is pointed at once initialized, then https://robutler.ai', async () => {
    cliPlatform.answer = () => ['https://macbook.example.ts.net/', 'global'];
    const skill = new NLISkill();
    await skill.initialize();
    expect(skill.normalizeUrl('@bob')).toBe('https://macbook.example.ts.net/agents/bob');

    cliPlatform.answer = () => {
      throw new Error('no CLI configuration here');
    };
    const bare = new NLISkill();
    await bare.initialize();
    expect(bare.normalizeUrl('@bob')).toBe('https://robutler.ai/agents/bob');
  });

  it('is never the old host', () => {
    expect(new NLISkill().normalizeUrl('@bob')).toBe('https://robutler.ai/agents/bob');
  });
});
