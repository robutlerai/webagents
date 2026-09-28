/**
 * The chat with a stdio MCP server (interactive-mode part 2, 2026-09-26), against
 * the probe server `tests/fixtures/mcp-probe-server-mcpsecrets.mjs`:
 *
 *  - `/mcp` lists the server from the skill's own report, its transport and
 *    its tools, and the way to serve the agent to an MCP client;
 *  - `/reload` and `/model` rebuild the agent with the server twice over and
 *    the chat goes on (the Python chat died here with a CancelledError);
 *  - `/exit` (the chat's `shutdown()`) closes the server so the process can
 *    end: the e2e run's `/exit` hung for good with the server's pipes open.
 *
 * The Python twin is `tests/cli/test_chat_mcp_reload_interactive2.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => 'y'),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROBE_SERVER = path.resolve(HERE, '../../fixtures/mcp-probe-server-mcpsecrets.mjs');
const PROBE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/mcp_tool/probe_server_mcpsecrets.json'), 'utf8')) as {
  qualified: string[];
};

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let project = '';
let printed: string[] = [];

const AGENT = `---
name: mcp-agent
skills:
  - openai
  - mcp:
      probe:
        command: ${process.execPath}
        args: ["${PROBE_SERVER}"]
---
Read your server's environment.
`;

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-mcpexit-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '160';
  project = tempDir('wa-mcpexit-project-');
  process.chdir(project);
  fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT);
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => printed.push(args.map(String).join(' ')));
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => printed.push(args.map(String).join(' ')));
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

interface Inside {
  handleInput(line: string): Promise<void>;
  shutdown(): Promise<void>;
  mcpSkill(): { serverReport(): Array<{ name: string; connected: boolean; tools: string[] }>; sessions: Map<string, unknown> } | undefined;
  agent: { cleanup(): Promise<void> } | null;
}

function text(): string {
  return printed.join('\n');
}

describe('the chat with a stdio MCP server', () => {
  it('/mcp lists the server, /reload and /model rebuild it, and shutdown closes it', async () => {
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as Inside;
    expect(inside.mcpSkill()?.serverReport()[0]).toMatchObject({ name: 'probe', connected: true, tools: PROBE.qualified });

    await inside.handleInput('/mcp');
    expect(text()).toContain('MCP servers (from AGENT.md)');
    expect(text()).toContain(`● probe  stdio  2 tools: ${PROBE.qualified.join(', ')}`);
    expect(text()).toContain('To use this agent from an MCP client: webagents mcp serve (stdio) or webagents mcp serve --http <port>.');

    // Twice over: each rebuild opens a new server and closes the old one.
    for (const change of ['Read more.', 'Read even more.']) {
      fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT.replace("Read your server's environment.", change));
      printed = [];
      await inside.handleInput('/reload');
      expect(text()).toContain('✓ Reloaded mcp-agent from AGENT.md.');
      expect(inside.mcpSkill()?.serverReport()[0].connected).toBe(true);
    }
    printed = [];
    await inside.handleInput('/model openai/gpt-4o');
    expect(text()).toContain('✓ Model set to openai/gpt-4o');
    expect(inside.mcpSkill()?.serverReport()[0].connected).toBe(true);

    // /exit: the server is closed, and nothing is left to hold the process.
    const skill = inside.mcpSkill()!;
    const cleanup = vi.spyOn(inside.agent!, 'cleanup');
    await inside.shutdown();
    expect(cleanup).toHaveBeenCalledTimes(1);
    expect(skill.sessions.size).toBe(0);
    expect(skill.serverReport()).toEqual([]);
  }, 60_000);
});
