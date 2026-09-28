/**
 * The chat bugs the 2026-09-26 e2e run found, pinned (interactive-mode part 2):
 *
 *  - the change notice says WHEN the agent file changed: an edit made while
 *    the chat sat idle is "changed since the chat loaded it", only one made
 *    between a message and its reply is "changed during the last reply";
 *  - the goodbye line counts this chat's replies and tokens, not a resumed
 *    conversation's;
 *  - the /help footer wraps at words, never inside "webagents";
 *  - /sandbox says "Invalid" for a declaration that did not resolve;
 *  - /tools leaves the turn-scoped content tools out.
 *
 * The Python twin is `tests/cli/test_chat_notice_interactive2.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { saveSession, sessionsDir } from '../../../src/cli/sessions';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let project = '';
let printed: string[] = [];

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-notice-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '100';
  project = tempDir('wa-notice-project-');
  process.chdir(project);
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

interface Inside {
  handleInput(line: string): Promise<void>;
  sayFileChanged(): void;
  streamToTerminal(content: string): Promise<void>;
  goodbye(): void;
  shellSkill(): unknown;
  agent: { toolRegistry: Map<string, { name: string; description?: string }>; name: string } | null;
  turns: number;
  sessionTokens: number;
  inputTokens: number;
  outputTokens: number;
}

const AGENT = '---\nname: helper\nskills:\n  - openai\n---\nHelp.\n';

async function chat(): Promise<Inside> {
  const repl = new InteractiveREPL({ interactive: true });
  await repl.initialize();
  return repl as unknown as Inside;
}

function text(): string {
  return printed.join('\n');
}

describe('the change notice says when the file changed', () => {
  beforeEach(() => {
    process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
    fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT);
  });

  it('an edit made while idle, before a message, is "changed since the chat loaded it"', async () => {
    const repl = await chat();
    repl.streamToTerminal = async () => {};
    // The person edits the file at the prompt, then sends a message.
    fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT.replace('Help.', 'Help more.'));
    await repl.handleInput('hi');
    printed = [];
    repl.sayFileChanged();
    expect(text()).toContain('✦ AGENT.md changed since the chat loaded it. /reload uses the new version.');
    expect(text()).not.toContain('during the last reply');
  });

  it('a change made between the message and the reply is "changed during the last reply"', async () => {
    const repl = await chat();
    repl.streamToTerminal = async () => {
      fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT.replace('Help.', 'Rewritten by the agent.'));
    };
    await repl.handleInput('hi');
    printed = [];
    repl.sayFileChanged();
    expect(text()).toContain('▲ AGENT.md changed during the last reply. /reload shows what changed.');
  });
});

describe('the goodbye line', () => {
  it("counts this chat's replies and tokens, not a resumed conversation's", async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT);
    const repl = await chat();
    saveSession(sessionsDir(project, 'helper'), {
      session_id: '0b7f8e2a-3333-4c2d-9a3e-000000000003',
      agent_name: 'helper',
      created_at: '2026-09-24T10:00:00.000001Z',
      updated_at: '2026-09-24T10:05:00.123456Z',
      messages: [
        { role: 'user', content: 'earlier' },
        { role: 'assistant', content: 'ok' },
      ],
      metadata: { sdk: 'python' },
      input_tokens: 30,
      output_tokens: 6,
    });
    await repl.handleInput('/resume 1');
    expect(repl.inputTokens + repl.outputTokens).toBe(36);
    // Nothing said in THIS chat yet: no goodbye line at all.
    printed = [];
    repl.goodbye();
    expect(text().trim()).toBe('');
    // One reply of 18 tokens here.
    repl.turns = 1;
    repl.sessionTokens = 18;
    printed = [];
    repl.goodbye();
    expect(text()).toContain('✦ 1 reply · 18 tokens ·');
    expect(text()).not.toContain('54 tokens');
    // /status still counts the conversation's tokens, resumed ones included.
    printed = [];
    await repl.handleInput('/status');
    expect(text()).toContain('2 messages, 36 tokens');
  });
});

describe('/help', () => {
  it('wraps its footer at words, never inside "webagents"', async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT);
    const repl = await chat();
    await repl.handleInput('/help');
    const lines = text().split('\n');
    const footer = lines.filter((l) => l.startsWith('Outside the chat') || l.startsWith('agents daemon') || l.includes('webagents daemon'));
    expect(footer.length).toBeGreaterThan(0);
    for (const line of lines) {
      expect(line.length, line).toBeLessThanOrEqual(99);
      expect(line.endsWith('web')).toBe(false);
    }
    expect(lines.some((l) => l.includes('webagents daemon (every agent here, with schedules).'))).toBe(true);
  });
});

describe('/sandbox and /tools', () => {
  it('/sandbox says Invalid for a declaration that did not resolve (D6)', async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: boxed\nskills:\n  - shell\n---\nHelp.\n');
    const repl = await chat();
    repl.shellSkill = () => ({ policy: null, sandboxError: 'network must be a list of domains' });
    await repl.handleInput('/sandbox');
    expect(text()).toContain('▲ Sandbox: Invalid: network must be a list of domains');
    expect(text()).toContain('Every command is refused until the declaration is fixed.');
  });

  it('/tools leaves the turn-scoped content tools out, as the Python chat never lists them', async () => {
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: helper\nskills:\n  - todo\n---\nHelp.\n');
    const repl = await chat();
    // As a turn stopped with Ctrl+C leaves them.
    repl.agent!.toolRegistry.set('present', { name: 'present', description: 'Display a piece of content' });
    repl.agent!.toolRegistry.set('read_content', { name: 'read_content', description: 'Load media' });
    await repl.handleInput('/tools');
    expect(text()).toContain('todo_add');
    expect(text()).not.toContain('present');
    expect(text()).not.toContain('read_content');
  });
});
