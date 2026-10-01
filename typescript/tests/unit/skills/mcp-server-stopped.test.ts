/**
 * A stdio MCP server that stops before it answers (2026-09-29, the owner's
 * yoyo agent), against the shared fixture
 * `python/tests/fixtures/mcp_tool/connect_errors.json` (`server_stopped`), which
 * the Python suite reads too (`tests/agents/skills/test_mcp_server_stopped.py`).
 *
 * `uvx mcp-server-sqlite` fetched the server's last release with `mcp` 2.2.0;
 * the server died at start, and `/mcp` said `not connected: MCP error -32000:
 * Connection closed`. The reason sat in the server's stderr log, which nothing
 * named. Pinned here:
 *  - the words and the error-line rule are the fixture's;
 *  - a real server that writes a traceback and exits gets a report row saying
 *    its own error line and where its output is, the log holding it all;
 *  - `list_mcp_servers` names the server that did not connect, with why.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { globalDir } from '../../../src/cli/config-store';
import {
  LINE_MAX,
  SERVER_STOPPED,
  SERVER_STOPPED_SILENT,
  SERVER_STOPPED_UNLOGGED,
  connectionClosed,
  displayPath,
  lastErrorLine,
  serverStoppedSentence,
} from '../../../src/skills/mcp/connect-errors';
import { MCPSkill, ownerReferenceSources } from '../../../src/skills/mcp/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const STOPPED = (JSON.parse(readFileSync(path.join(FIXTURES, 'mcp_tool/connect_errors.json'), 'utf8')) as {
  server_stopped: {
    sentence: string;
    silent: string;
    unlogged: string;
    line_max: number;
    lines: Array<{ about: string; stderr: string; line: string | null }>;
    paths: Array<{ home: string; log: string; shown: string }>;
    says: { stderr: string; home: string; log: string; sentence: string };
  };
}).server_stopped;

describe('the words and the error line (server_stopped)', () => {
  it('uses the fixture words', () => {
    expect(SERVER_STOPPED).toBe(STOPPED.sentence);
    expect(SERVER_STOPPED_SILENT).toBe(STOPPED.silent);
    expect(SERVER_STOPPED_UNLOGGED).toBe(STOPPED.unlogged);
    expect(LINE_MAX).toBe(STOPPED.line_max);
  });

  it.each(STOPPED.lines.map((c) => [c.about, c] as const))('%s', (_about, c) => {
    expect(lastErrorLine(c.stderr) ?? null).toBe(c.line);
  });

  it('cuts a long error line', () => {
    const line = lastErrorLine(`Error: ${'x'.repeat(400)}`);
    expect(line?.length).toBe(LINE_MAX);
    expect(line?.endsWith('…')).toBe(true);
  });

  it.each(STOPPED.paths.map((c) => [c.log, c] as const))('writes home as ~ in %s', (_log, c) => {
    expect(displayPath(c.log, c.home)).toBe(c.shown);
  });

  it('says the sentence, silent and unlogged', () => {
    const says = STOPPED.says;
    expect(serverStoppedSentence(says.stderr, says.log, says.home)).toBe(says.sentence);
    expect(serverStoppedSentence('', says.log, says.home)).toBe(STOPPED.silent);
    expect(serverStoppedSentence(undefined, undefined, says.home)).toBe(STOPPED.unlogged);
  });

  it('counts the connection closing, and nothing else', () => {
    const closed = Object.assign(new Error('MCP error -32000: Connection closed'), { code: -32000 });
    expect(connectionClosed(closed)).toBe(true);
    expect(connectionClosed(new Error('Connection closed'))).toBe(true);
    expect(connectionClosed(new Error('MCP error -32001: Request timed out'))).toBe(false);
    expect(connectionClosed(Object.assign(new Error('spawn uvx ENOENT'), { code: 'ENOENT' }))).toBe(false);
  });
});

describe('a real stdio server that crashes at start', () => {
  const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_DIR'];
  const saved: Record<string, string | undefined> = {};
  const tempDir = tempDirs();

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-mcp-stopped-home-');
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

  it('reports its own error line and where its output is, and the tool names it', async () => {
    const crash = `process.stderr.write(${JSON.stringify(STOPPED.says.stderr)}); process.exit(1);`;
    const skill = new MCPSkill({
      mcp: { sqlite: { command: process.execPath, args: ['-e', crash] } } as never,
      references: ownerReferenceSources(),
    });
    const agent = new BaseAgent({ name: 'client', instructions: 'x', skills: [skill] });
    await agent.initialize();
    try {
      const log = path.join(globalDir(), 'logs', 'mcp-sqlite.log');
      const expected = SERVER_STOPPED.replace('{line}', STOPPED.lines[0].line!).replace('{log}', displayPath(log, os.homedir()));
      const report = skill.serverReport();
      expect(report[0]).toMatchObject({ name: 'sqlite', connected: false, error: expected });
      expect(JSON.stringify(report)).not.toContain('Connection closed');
      expect(readFileSync(log, 'utf8')).toContain(STOPPED.says.stderr);

      const listed = (await skill.listServers({}, {} as never)) as { not_connected?: Array<{ name: string; reason: string }> };
      expect(listed.not_connected).toEqual([{ name: 'sqlite', reason: expected }]);
    } finally {
      await skill.cleanup?.();
    }
  });
});
