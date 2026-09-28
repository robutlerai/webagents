/**
 * `webagents doctor -a <name>` checks that agent, as the chat's `-a` picks it
 * (2026-09-26, the new-developer e2e run: it was "unknown option '-a'"). An
 * unknown name is the chat's refusal (`agent-files.ts`), thrown before
 * anything is built. The Python doctor is pinned the same way
 * (`tests/cli/test_doctor_agent_option_e2efix.py`).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

import { AgentNotFound } from '../../../src/cli/agent-files';
import { runChecks } from '../../../src/cli/doctor';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_TOKEN', 'ROBUTLER_API_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY'];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-doctor-a-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.OPENAI_API_KEY = 'sk-dummy';
  const project = tempDir('wa-doctor-a-project-');
  fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: bot\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nBody\n');
  fs.writeFileSync(path.join(project, 'AGENT-helper.md'), '---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nHelps\n');
  process.chdir(project);
  for (const method of ['log', 'warn', 'error', 'info'] as const) vi.spyOn(console, method).mockImplementation(() => {});
});

afterEach(() => {
  process.chdir(cwd);
  vi.restoreAllMocks();
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
});

describe('doctor -a', () => {
  it('checks the named agent, and the default file without it', async () => {
    const named = (await runChecks({ agent: 'helper' })).find((c) => c.name === 'agent')!;
    expect(named.detail).toBe('helper (AGENT-helper.md)');
    const bare = (await runChecks()).find((c) => c.name === 'agent')!;
    expect(bare.detail).toBe('bot (AGENT.md)');
  });

  it('refuses a name the folder does not have, with the chat’s sentence', async () => {
    await expect(runChecks({ agent: 'nosuch' })).rejects.toBeInstanceOf(AgentNotFound);
    await expect(runChecks({ agent: 'nosuch' })).rejects.toThrow('There is no agent called nosuch in this folder.');
  });
});
