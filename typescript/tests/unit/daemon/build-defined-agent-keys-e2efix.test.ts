/**
 * The daemon and `webagents cron run` build their agents on the keys kept
 * with `webagents secrets set` (2026-09-26, the new-developer e2e run): an
 * agent that answered in the chat on a stored OPENAI_API_KEY failed under
 * the daemon and `cron run` with "OpenAI API key not configured", because
 * `buildDefinedAgent` resolved its skills with no `apiKeys` while the chat,
 * `serve`, `acp` and `mcp serve` all passed the store's keys. One helper now
 * feeds every server-side builder (`storedApiKeys`, `cli/provider-keys.ts`).
 *
 * Scratch HOME, the file secrets backend, never the keychain; the key is a
 * dummy and no request is made.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { buildDefinedAgent } from '../../../src/daemon/server';
import { storeProviderKey, storedApiKeys } from '../../../src/cli/provider-keys';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'FIREWORKS_API_KEY'];
const saved: Record<string, string | undefined> = {};
const DUMMY = 'sk-dummy-stored-for-the-daemon-test';

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-daemon-keys-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
});

afterEach(() => {
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
});

function definition(dir: string) {
  const filePath = path.join(dir, 'AGENT.md');
  const content = '---\nname: reporter\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nReport.\n';
  fs.writeFileSync(filePath, content);
  return { name: 'reporter', model: 'openai/gpt-4o-mini', skills: ['openai'], skillEntries: ['openai'], filePath, content };
}

function openaiKeyOf(agent: unknown): string | undefined {
  const skills = (agent as { skills: Array<{ name?: string; modelConfig?: { apiKey?: string } }> }).skills;
  return skills.find((s) => s.name === 'openai')?.modelConfig?.apiKey;
}

describe('the daemon builds an agent on the stored provider keys', () => {
  it('a key kept with `secrets set` reaches the model client when the shell has none', async () => {
    await storeProviderKey('OPENAI_API_KEY', DUMMY);
    expect(await storedApiKeys()).toEqual({ openai: DUMMY });
    const agent = await buildDefinedAgent(definition(tempDir('wa-daemon-keys-project-')));
    expect(agent).not.toBeNull();
    expect(openaiKeyOf(agent)).toBe(DUMMY);
    await agent!.cleanup();
  });

  it('a key exported in the shell wins over the stored one, as everywhere else', async () => {
    await storeProviderKey('OPENAI_API_KEY', DUMMY);
    process.env.OPENAI_API_KEY = 'sk-from-the-shell';
    expect(await storedApiKeys()).toEqual({});
    const agent = await buildDefinedAgent(definition(tempDir('wa-daemon-keys-project-')));
    // The skill reads the shell's variable itself; nothing stored is handed in.
    expect(openaiKeyOf(agent)).toBeUndefined();
    await agent!.cleanup();
  });

  it('nothing stored and no credential of its own: not served, and the daemon says why', async () => {
    // It was served with a keyless model client that failed every turn. The
    // daemon runs its callers' turns, so without a key or the agent's own
    // platform credential there is no model (S-327, 2026-09-28), never the
    // owner's sign-in; the line names both ways out.
    expect(await storedApiKeys()).toEqual({});
    const errors: string[] = [];
    const spy = vi.spyOn(console, 'error').mockImplementation((line: unknown) => { errors.push(String(line)); });
    try {
      const agent = await buildDefinedAgent(definition(tempDir('wa-daemon-keys-project-')));
      expect(agent).toBeNull();
    } finally {
      spy.mockRestore();
    }
    expect(errors.join('\n')).toContain("No model for this agent's callers: OPENAI_API_KEY is not set.");
    expect(errors.join('\n')).toContain('The agent is not served.');
  });
});
