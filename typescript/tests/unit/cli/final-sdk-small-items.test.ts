/**
 * The smaller items of the CLI e2e pass (2026-09-28), in the TypeScript CLI:
 *
 *  - B8: an MCP stdio server's stderr goes to the profile's `logs/` folder,
 *    never to the terminal the chat draws on (the Python twin writes the same
 *    file, `local/mcp/skill.py` `mcp_stderr_log`);
 *  - B10: `--json` typed after the subcommand is lifted to the front like
 *    `--profile` (the argv cases are the shared fixture
 *    `cli/sandbox_default_global_options.json`).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../src/core/agent';
import { MCPSkill, mcpStderrLog } from '../../../src/skills/mcp/skill';
import { HOISTED_OPTIONS, hoistGlobalOptions } from '../../../src/cli/sandbox-default-argv';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ECHO_SERVER = path.resolve(HERE, '../../fixtures/mcp-echo-server.mjs');
const tempDir = tempDirs();
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const name of ['HOME', 'WEBAGENTS_PROFILE']) saved[name] = process.env[name];
  delete process.env.WEBAGENTS_PROFILE;
  process.env.HOME = tempDir('wa-final-sdk-small-');
});

afterEach(() => {
  for (const [name, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  vi.restoreAllMocks();
});

describe("an MCP stdio server's stderr (B8)", () => {
  it('goes to the profile log folder', async () => {
    const fd = await mcpStderrLog('echo/server one');
    expect(fd).toBeTypeOf('number');
    fs.writeSync(fd!, 'a banner the chat never shows\n');
    fs.closeSync(fd!);
    const log = path.join(process.env.HOME!, '.webagents', 'logs', 'mcp-echo_server_one.log');
    expect(fs.readFileSync(log, 'utf8')).toBe('a banner the chat never shows\n');
  });

  it('a real server writing to stderr never reaches this terminal', async () => {
    // A wrapper that writes a line to stderr, then runs the echo server.
    const noisy = path.join(tempDir('wa-final-sdk-noisy-'), 'noisy.mjs');
    fs.writeFileSync(noisy, `process.stderr.write('noisy server starting\\n');\nawait import(${JSON.stringify(ECHO_SERVER)});\n`);
    const skill = new MCPSkill({ mcp: { noisy: { command: process.execPath, args: [noisy] } } as never });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    const written: string[] = [];
    const original = process.stderr.write.bind(process.stderr);
    vi.spyOn(process.stderr, 'write').mockImplementation(((chunk: unknown, ...rest: unknown[]) => {
      written.push(String(chunk));
      return (original as (...a: unknown[]) => boolean)(chunk, ...rest);
    }) as typeof process.stderr.write);
    await agent.initialize();
    await agent.cleanup();
    expect(written.join('')).not.toContain('noisy server starting');
    const log = path.join(process.env.HOME!, '.webagents', 'logs', 'mcp-noisy.log');
    expect(fs.readFileSync(log, 'utf8')).toContain('noisy server starting');
  });
});

describe('--json after the subcommand (B10)', () => {
  it('is lifted like --profile', () => {
    expect(HOISTED_OPTIONS).toContain('--json');
    expect(hoistGlobalOptions(['doctor', '--json'])).toEqual(['--json', 'doctor']);
    expect(hoistGlobalOptions(['sandbox', 'setup', '--json'])).toEqual(['--json', 'sandbox', 'setup']);
  });
});
