/**
 * A stdio MCP server whose command is not there (2026-09-29), against the
 * shared fixture `python/tests/fixtures/mcp_tool/connect_errors.json`
 * (`command_missing`), which the Python suite reads too
 * (`tests/agents/skills/test_mcp_command_missing.py`).
 *
 * The row said `spawn uvx ENOENT`, with nothing about what to install. Pinned
 * here: the sentence and hints are the fixture's; the command is looked for on
 * the server's PATH or as a file; a real skill pointed at a command that is not
 * there reports the sentence with `needsCommand`, and `doctor`'s fix line says
 * what to install.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { MCP_CHECK_WORDS, mcpCheck } from '../../../src/cli/doctor';
import { COMMAND_HINTS, COMMAND_MISSING, commandHint, commandMissingSentence } from '../../../src/skills/mcp/connect-errors';
import { MCPSkill, commandFound, ownerReferenceSources } from '../../../src/skills/mcp/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const MISSING = (
  JSON.parse(fs.readFileSync(path.join(FIXTURES, 'mcp_tool/connect_errors.json'), 'utf8')) as {
    command_missing: { sentence: string; hints: Record<string, string>; cases: Array<{ command: string; says: string }> };
  }
).command_missing;
const DOCTOR_WORDS = (JSON.parse(fs.readFileSync(path.join(FIXTURES, 'cli/secrets.json'), 'utf8')) as { doctor: { words: Record<string, string> } })
  .doctor.words;

describe('the words (command_missing)', () => {
  it('are the fixture words', () => {
    expect(COMMAND_MISSING).toBe(MISSING.sentence);
    expect(COMMAND_HINTS).toEqual(MISSING.hints);
    expect(MCP_CHECK_WORDS.fixCommand).toBe(DOCTOR_WORDS.fixCommand);
  });

  it.each(MISSING.cases.map((c) => [c.command, c] as const))('say what to install for %s', (_command, c) => {
    expect(commandMissingSentence(c.command)).toBe(c.says);
  });
});

describe('looking for the command', () => {
  let dir: string;
  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wa-command-'));
    fs.mkdirSync(path.join(dir, 'bin'));
    fs.writeFileSync(path.join(dir, 'bin', 'my-server'), '#!/bin/sh\nexit 0\n', { mode: 0o755 });
  });
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('finds a bare name on PATH and a path as a file', async () => {
    expect(await commandFound('my-server', path.join(dir, 'bin'))).toBe(true);
    expect(await commandFound('not-there', path.join(dir, 'bin'))).toBe(false);
    expect(await commandFound('bin/my-server', undefined, dir)).toBe(true);
    expect(await commandFound('bin/not-there', undefined, dir)).toBe(false);
  });
});

describe('a real skill whose server command is not there', () => {
  const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_DIR'];
  const saved: Record<string, string | undefined> = {};
  const tempDir = tempDirs();

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-mcp-command-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    for (const method of ['log', 'warn', 'error', 'info', 'debug'] as const) vi.spyOn(console, method).mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
  });

  it('reports what to install, and doctor says it too', async () => {
    const command = 'webagents-test-no-such-command';
    const skill = new MCPSkill({ mcp: { tools: { command, args: [] } } as never, references: ownerReferenceSources() });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    await agent.initialize();
    try {
      const row = skill.serverReport()[0];
      expect(row).toMatchObject({ name: 'tools', connected: false, error: commandMissingSentence(command), needsCommand: command });
      expect(row.error).not.toContain('ENOENT');
      const check = mcpCheck([row]);
      expect(check.status).toBe('fail');
      expect(check.fix).toBe(MCP_CHECK_WORDS.fixCommand.replace('{command}', command).replace('{hint}', commandHint(command)));
    } finally {
      await skill.cleanup?.();
    }
  });
});
