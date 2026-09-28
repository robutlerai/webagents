/**
 * `serve()` says at startup when the agent has `memory` but nothing verifies
 * a caller (2026-09-26, the new-developer e2e run): every served caller is a
 * caller nothing verified, who reads shared notes and writes nothing, and the
 * tester found that out one refused tool call at a time. The sentence is
 * `memory_without_auth` in `python/tests/fixtures/cli/serve_startup.json`,
 * which the Python `serve` prints too (`tests/cli/test_serve_memory_warning_e2efix.py`).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { serve, type ServeHandle } from '../../../src/server/node';
import { MEMORY_WITHOUT_AUTH, memoryWithoutAuthLine } from '../../../src/server/startup-lines';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const STARTUP = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/serve_startup.json'), 'utf8')) as {
  memory_without_auth: string;
  memory_skill: string;
};
const tempDir = tempDirs();

describe('serve says when memory has no AuthSkill', () => {
  let warned: string[] = [];
  let handle: ServeHandle | null = null;
  const ISOLATED = ['WEBAGENTS_PUBLIC_URL', 'WEBAGENTS_AGENT_TOKEN', 'ROBUTLER_API_URL', 'ROBUTLER_INTERNAL_API_URL'];
  const saved: Record<string, string | undefined> = {};

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    warned = [];
    vi.spyOn(console, 'warn').mockImplementation((...args: unknown[]) => void warned.push(args.map(String).join(' ')));
    vi.spyOn(console, 'info').mockImplementation(() => {});
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(async () => {
    await handle?.close();
    handle = null;
    vi.restoreAllMocks();
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  });

  it('the sentence is the fixture’s', () => {
    expect(MEMORY_WITHOUT_AUTH).toBe(STARTUP.memory_without_auth);
    expect(memoryWithoutAuthLine('rememberer')).toBe(STARTUP.memory_without_auth.replace('{name}', 'rememberer'));
  });

  it('prints it after the AuthSkill line for an agent that names memory, and not for one that does not', async () => {
    const dir = tempDir('wa-memory-warn-');
    const { skills, unknown } = await resolveSkillsByName([STARTUP.memory_skill], { agentDir: dir });
    expect(unknown).toEqual([]);
    const withMemory = new BaseAgent({ name: 'rememberer', instructions: 'x', skills });
    handle = await serve(withMemory, { port: 0, hostname: '127.0.0.1', keysDir: mkdtempSync(path.join(os.tmpdir(), 'wa-memory-keys-')), heartbeat: false, logging: false } as Parameters<typeof serve>[1]);
    const authLine = warned.findIndex((l) => l.includes('rememberer has no AuthSkill'));
    const memoryLine = warned.indexOf(memoryWithoutAuthLine('rememberer'));
    expect(authLine).toBeGreaterThanOrEqual(0);
    expect(memoryLine).toBe(authLine + 1);
    await handle.close();
    handle = null;

    warned = [];
    const without = new BaseAgent({ name: 'plain', instructions: 'x', skills: [] });
    handle = await serve(without, { port: 0, hostname: '127.0.0.1', keysDir: mkdtempSync(path.join(os.tmpdir(), 'wa-memory-keys-')), heartbeat: false, logging: false } as Parameters<typeof serve>[1]);
    expect(warned.some((l) => l.includes('has memory but no AuthSkill'))).toBe(false);
  });
});
