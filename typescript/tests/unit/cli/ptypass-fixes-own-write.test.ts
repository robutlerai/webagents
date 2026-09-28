/**
 * The chat's own `always` write to the agent file is not "a change during the
 * reply", and the sandbox state follows it at once (the ptypass-fixes lane,
 * 2026-09-27, brief item 11). The Python twin is
 * `python/tests/cli/test_ptypass_fixes_own_write.py`.
 *
 * WHY. The real-terminal PTY pass answered `always` to a refused host: the
 * chat wrote the host into AGENT.md, then its next prompt said "AGENT.md
 * changed during the last reply", and `/sandbox` kept saying "(default)"
 * until `/reload`, though the session already ran what the file said.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS', 'WEBAGENTS_NO_SANDBOX',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let project = '';
let printed: string[] = [];

interface Inside {
  chatting: boolean;
  handleInput(line: string): Promise<void>;
  sayFileChanged(): void;
  streamToTerminal(content: string): Promise<void>;
  commandSandbox(): Promise<void>;
  shellSkill(): { asker?: { allowHostAlways(host: string): Promise<void> }; sandboxStateLine(): string; policy: { networkDomains: string[] } } | undefined;
}

const AGENT = '---\nname: helper\nskills:\n  - openai\n  - shell\n---\nHelp.\n';

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-ownwrite-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '100';
  project = tempDir('wa-ownwrite-project-');
  process.chdir(project);
  fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT);
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => printed.push(args.map(String).join(' ')));
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

async function chat(): Promise<Inside> {
  const repl = new InteractiveREPL({ interactive: true }) as unknown as Inside & { initialize(): Promise<void> };
  // The interactive loop is what attaches the host asker.
  repl.chatting = true;
  await repl.initialize();
  return repl;
}

describe('the chat\'s own write', () => {
  it('after always, the next prompt says nothing changed and /sandbox says (agent file)', async () => {
    const repl = await chat();
    const shell = repl.shellSkill()!;
    expect(shell.sandboxStateLine()).toBe('development (default)');
    repl.streamToTerminal = async () => {
      await shell.asker!.allowHostAlways('example.com');
    };
    await repl.handleInput('fetch it');
    expect(fs.readFileSync(path.join(project, 'AGENT.md'), 'utf8')).toContain('example.com');
    printed = [];
    repl.sayFileChanged();
    expect(printed.join('\n')).not.toContain('changed');
    expect(shell.sandboxStateLine()).toBe('development (agent file)');
    expect(shell.policy.networkDomains).toContain('example.com');
    await repl.commandSandbox();
    expect(printed.join('\n')).toContain('development (agent file)');
  });

  it('a change someone else made during the reply is still said', async () => {
    const repl = await chat();
    const shell = repl.shellSkill()!;
    repl.streamToTerminal = async () => {
      fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT.replace('Help.', 'Help more.'));
      await shell.asker!.allowHostAlways('example.com');
    };
    await repl.handleInput('fetch it');
    printed = [];
    repl.sayFileChanged();
    expect(printed.join('\n')).toContain('AGENT.md changed during the last reply');
  });
});
