/**
 * One way to read the agent a directory describes (2026-09-24).
 *
 * Found walking the TypeScript onboarding path as a first-time developer:
 * `init` wrote `agent.json` + `instructions.md`, `serve` never read the
 * instructions (the agent ran with none), and `publish` posted the file as-is,
 * which the portal refused (`skills` a list where it wants a mapping). `init`
 * now writes the documented `AGENT.md`; the old pair stays readable.
 */

import { describe, expect, it } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as path from 'node:path';

import { AgentProjectError, loadAgentProject, toPlatformPayload } from '../../../src/cli/agent-project';
import { loadAgentConfigFile } from '../../../src/cli/serve-action';
import { parseAgentMarkdown } from '../../../src/agents/index';
import { tempDirs, TSX_CLI } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const makeTempDir = tempDirs();

const CLI_SOURCE = path.resolve(__dirname, '../../../src/cli/index.ts');

function tempDir(): string {
  return makeTempDir('wa-project-');
}

const AGENT_MD = `---
name: helper
description: Helps
model: openai/gpt-4o-mini
skills:
  - openai
  - mcp:
      servers: [filesystem]
intents:
  - help with things
---

# helper

You are a helpful assistant.
`;

describe('loadAgentProject', () => {
  it('reads AGENT.md, instructions included', () => {
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'AGENT.md'), AGENT_MD);
    const project = loadAgentProject(dir)!;
    expect(project.name).toBe('helper');
    expect(project.model).toBe('openai/gpt-4o-mini');
    expect(project.skills).toEqual(['openai', 'mcp']);
    expect(project.instructions).toContain('You are a helpful assistant.');
  });

  it('reads the old agent.json AND the instructions.md beside it', () => {
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'agent.json'), JSON.stringify({ name: 'old', skills: ['openai'] }));
    fs.writeFileSync(path.join(dir, 'instructions.md'), '# old\n\nBe terse.\n');
    // THE BUG: these instructions never reached the served agent.
    expect(loadAgentProject(dir)!.instructions).toContain('Be terse.');
  });

  it('prefers AGENT.md when both exist', () => {
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'AGENT.md'), AGENT_MD);
    fs.writeFileSync(path.join(dir, 'agent.json'), JSON.stringify({ name: 'old' }));
    expect(loadAgentProject(dir)!.name).toBe('helper');
  });

  it('is null when there is nothing, and throws when a file is broken', () => {
    expect(loadAgentProject(path.join(tempDir(), 'missing'))).toBeNull();
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'agent.json'), '{ "name": "x", }');
    expect(() => loadAgentProject(dir)).toThrow(AgentProjectError);
    expect(() => loadAgentProject(dir)).toThrow(/not valid JSON/);
  });

  it('feeds serve the instructions too', () => {
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'AGENT.md'), AGENT_MD);
    const config = loadAgentConfigFile(dir);
    expect(config.instructions).toContain('You are a helpful assistant.');
    expect(config.skills).toEqual(['openai', 'mcp']);
  });
});

describe('toPlatformPayload', () => {
  it("converts skills to the portal's mapping and sends the instructions", () => {
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'AGENT.md'), AGENT_MD);
    const payload = toPlatformPayload(loadAgentProject(dir)!);
    // THE BUG: the file's list went out as-is and the portal answered 400.
    expect(payload.skills).toEqual({ openai: {}, mcp: { servers: ['filesystem'] } });
    expect(payload.instructions).toContain('You are a helpful assistant.');
    expect(payload.intents).toEqual(['help with things']);
  });

  it('does not send keys the portal does not model', () => {
    const dir = tempDir();
    fs.writeFileSync(path.join(dir, 'agent.json'), JSON.stringify({ name: 'x', template: 'chatbot', skills: ['openai'] }));
    expect(toPlatformPayload(loadAgentProject(dir)!)).not.toHaveProperty('template');
  });
});

function runCli(args: string[], cwd: string, env: Record<string, string> = {}): Promise<{ code: number | null; out: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, ...args], {
      cwd,
      env: { ...process.env, HOME: path.join(cwd, '.home'), WEBAGENTS_PROFILE: '', ...env },
    });
    let out = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (out += d));
    child.on('close', (code) => resolve({ code, out }));
  });
}

describe('init writes the documented AGENT.md', () => {
  it('produces an AGENT.md project, and no agent.json', async () => {
    const cwd = tempDir();
    const { code, out } = await runCli(['init', 'first-agent'], cwd);
    expect(code).toBe(0);

    const dir = path.join(cwd, 'first-agent');
    expect(fs.existsSync(path.join(dir, 'AGENT.md'))).toBe(true);
    expect(fs.existsSync(path.join(dir, 'agent.json'))).toBe(false);
    // AGENT.md is the whole project, as the Python CLI's `init` makes it.
    expect(fs.existsSync(path.join(dir, 'package.json'))).toBe(false);

    const parsed = parseAgentMarkdown(fs.readFileSync(path.join(dir, 'AGENT.md'), 'utf-8'));
    expect(parsed.name).toBe('first-agent');
    expect(parsed.skills).toEqual(['openai']);
    expect(parsed.instructions).toContain('You are a helpful assistant.');
    // Only keys the Python loader's strict schema accepts, so both SDKs run it.
    expect(Object.keys(parsed.extra)).toEqual([]);

    // The step every first run trips on is named up front.
    expect(out).toContain('OPENAI_API_KEY');
  }, 60_000);

  it('publishes that project in the shape the portal accepts', async () => {
    const cwd = tempDir();
    await runCli(['init', 'first-agent'], cwd);

    let body: Record<string, unknown> = {};
    const server = http.createServer((req, res) => {
      let raw = '';
      req.on('data', (c) => (raw += c));
      req.on('end', () => {
        body = JSON.parse(raw);
        res.statusCode = 201;
        res.setHeader('content-type', 'application/json');
        res.end(JSON.stringify({ agent: { id: 'a-1', username: 'owner.first-agent' } }));
      });
    });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
    const port = (server.address() as { port: number }).port;
    try {
      // --yes: a first publish creates the agent, and with no terminal to ask
      // on, `publish` refuses to create one without it (publish-link.test.ts).
      const { code } = await runCli(['publish', '.', '--yes'], path.join(cwd, 'first-agent'), {
        WEBAGENTS_TOKEN: 'stub-token',
        ROBUTLER_API_URL: `http://127.0.0.1:${port}`,
        WEBAGENTS_SECRETS_BACKEND: 'file',
      });
      expect(code).toBe(0);
      expect(body.name).toBe('first-agent');
      expect(body.skills).toEqual({ openai: {} });
      expect(String(body.instructions)).toContain('You are a helpful assistant.');
      expect(body).not.toHaveProperty('template');
    } finally {
      server.close();
    }
  }, 60_000);
});
