/**
 * The chat's commands, driven the way a person types them (2026-09-24).
 *
 * The list and its wording live in `src/cli/chat-commands.ts`, the reference
 * the Python chat is checked against (`python/tests/cli/test_chat_command_parity.py`).
 * These tests type into the chat and read what it printed, under a throwaway
 * HOME with the FILE secrets backend, no provider key and no sign-in: nothing
 * here touches the machine's keychain or its real login.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const H = vi.hoisted(() => ({ secrets: [] as string[], lines: [] as Array<string | null> }));
vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => (H.lines.length ? H.lines.shift()! : null)),
  promptSecret: vi.fn(async () => H.secrets.shift() ?? ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { CHAT_COMMANDS } from '../../../src/cli/chat-commands';
import { saveSession, sessionsDir, slugFor } from '../../../src/cli/sessions';
import { tempDirs } from '../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
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
  process.env.HOME = tempDir('wa-chat-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '100';
  project = tempDir('wa-chat-project-');
  process.chdir(project);
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  H.secrets = [];
  H.lines = [];
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

function agent(text: string, name = 'AGENT.md'): string {
  const file = path.join(project, name);
  fs.writeFileSync(file, text);
  return file;
}

type Inside = {
  handleInput(line: string): Promise<void>;
  messages: Array<{ role: string; content: string }>;
  sessionId: string;
  saveConversation(): void;
  agent: { name: string } | null;
  modelProblem: string | undefined;
};

async function chat(): Promise<Inside> {
  const repl = new InteractiveREPL({});
  await repl.initialize();
  return repl as unknown as Inside;
}

async function say(repl: Inside, line: string): Promise<string> {
  const before = printed.length;
  await repl.handleInput(line);
  return printed.slice(before).join('\n');
}

describe('the commands', () => {
  it('/help lists every command with its usage, and the keys', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const out = await say(await chat(), '/help');
    for (const command of CHAT_COMMANDS) {
      expect(out).toContain(command.usage);
      expect(out).toContain(command.description);
    }
    expect(out).toContain('Keys');
  });

  it('/help <command> shows how to type it; an unknown one says how to find the right one', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const repl = await chat();
    expect(await say(repl, '/help keys')).toContain('/keys [set|unset NAME]');
    const unknown = await say(repl, '/nope');
    expect(unknown).toContain('Unknown command /nope.');
    expect(unknown).toContain('/help');
  });

  it('does not send a message it has no model for, and names the way out', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const repl = await chat();
    const out = await say(repl, 'hello');
    expect(out).toContain('webagents login');
    expect(out).toContain('/login');
    expect(repl.messages).toEqual([]);
  });

  it('/new starts a fresh conversation', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const repl = await chat();
    repl.messages = [{ role: 'user', content: 'hi' }];
    const before = repl.sessionId;
    expect(await say(repl, '/new')).toContain('Started a new conversation.');
    expect(repl.messages).toEqual([]);
    expect(repl.sessionId).not.toBe(before);
  });

  it('keeps conversations under the profile, never in the project, owner-only', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const repl = await chat();
    repl.messages = [{ role: 'user', content: 'remember this' }];
    repl.saveConversation();
    const expected = path.join(process.env.HOME!, '.webagents', 'sessions', slugFor(fs.realpathSync(project)), 'helper');
    const file = path.join(sessionsDir(fs.realpathSync(project), 'helper'), `${repl.sessionId}.json`);
    expect(fs.existsSync(file)).toBe(true);
    expect(path.dirname(file)).toBe(expected);
    expect(fs.existsSync(path.join(project, '.webagents', 'sessions'))).toBe(false);
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);
  });

  it('/resume lists earlier conversations and continues one', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const repl = await chat();
    expect(await say(repl, '/resume')).toContain('No earlier conversations with helper in this folder.');
    repl.messages = [
      { role: 'user', content: 'plan the launch' },
      { role: 'assistant', content: 'Step one.' },
    ];
    repl.saveConversation();
    await say(repl, '/new');
    const listing = await say(repl, '/resume');
    expect(listing).toContain('Earlier conversations');
    expect(listing).toContain('plan the launch');
    expect(await say(repl, '/resume 9')).toContain('There is no conversation 9.');
    const out = await say(repl, '/resume 1');
    expect(out).toContain('Continuing the conversation from');
    expect(out).toContain('(2 messages)');
    expect(out).toContain('── Earlier in this conversation ──');
    expect(repl.messages[0].content).toBe('plan the launch');
  });

  it('/resume reads a conversation the Python chat wrote', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const repl = await chat();
    saveSession(sessionsDir(process.cwd(), 'helper'), {
      session_id: '0b7f8e2a-2222-4c2d-9a3e-000000000002',
      agent_name: 'helper',
      created_at: '2026-09-24T10:00:00.000001Z',
      updated_at: '2026-09-24T10:05:00.123456Z',
      messages: [
        { role: 'user', content: 'from python' },
        { role: 'assistant', content: 'ok' },
      ],
      metadata: { sdk: 'python' },
      input_tokens: 7,
      output_tokens: 2,
    });
    expect(await say(repl, '/resume 1')).toContain('Continuing the conversation');
    expect(repl.messages[0].content).toBe('from python');
  });

  it("/agent lists this folder's agents and the built-in one, and switches", async () => {
    agent('---\nname: helper\ndescription: Helps.\n---\nHelp.\n');
    agent('---\nname: writer\ndescription: Writes.\n---\nWrite.\n', 'AGENT-writer.md');
    const repl = await chat();
    const listing = await say(repl, '/agent');
    for (const word of ['helper', 'writer', 'robutler', 'built in']) expect(listing).toContain(word);
    expect(await say(repl, '/agent writer')).toContain('Now talking to writer.');
    expect(repl.agent?.name).toBe('writer');
    expect(await say(repl, '/agent writer')).toContain('Already talking to writer.');
    expect(await say(repl, '/agent ghost')).toContain('There is no agent called ghost in this folder.');
  });

  it('/keys lists where each key comes from, and sets and removes a stored one', async () => {
    agent('---\nname: helper\nmodel: openai/gpt-4o-mini\n---\nHelp.\n');
    const repl = await chat();
    const listing = await say(repl, '/keys');
    expect(listing).toContain('Model provider keys');
    expect(listing).toContain('not set');
    H.secrets = ['sk-entered'];
    expect(await say(repl, '/keys set OPENAI_API_KEY')).toContain('Stored OPENAI_API_KEY (an owner-only file).');
    expect(await say(repl, '/keys')).toContain('stored in an owner-only file');
    expect(repl.modelProblem).toBeUndefined();
    // Never into the environment the shell skill hands to its commands.
    expect(process.env.OPENAI_API_KEY).toBeUndefined();
    expect(await say(repl, '/keys unset OPENAI_API_KEY')).toContain('Removed OPENAI_API_KEY.');
    expect(await say(repl, '/keys set NOT_A_KEY')).toContain('is not a model provider key');
  });

  it('/status says who, what and where', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    const out = await say(await chat(), '/status');
    expect(out).toContain('Not signed in. /login signs in.');
    expect(out).toContain('helper (AGENT.md)');
    expect(out).toContain('none (no provider key is set). /login, or /keys set <NAME>.');
    expect(out).toContain('0 messages');
  });

  it("/sandbox says what the agent's commands may do", async () => {
    agent('---\nname: helper\nskills:\n  - filesystem\n---\nHelp.\n');
    expect(await say(await chat(), '/sandbox')).toContain('Sandbox: Not needed: this agent cannot run commands.');
  });

  it('@path includes a file, and leaves anything that is not one as typed', async () => {
    agent('---\nname: helper\n---\nHelp.\n');
    fs.writeFileSync(path.join(project, 'notes.md'), 'the notes');
    const repl = (await chat()) as unknown as { expandFileReferences(text: string): string };
    const out = repl.expandFileReferences('read @notes.md and ask @someone');
    expect(out).toContain('<file path="notes.md">\nthe notes\n</file>');
    expect(out).toContain('ask @someone');
    expect(repl.expandFileReferences('mail me@example.com')).toBe('mail me@example.com');
  });

  it('/publish needs an agent file', async () => {
    const repl = await chat(); // the built-in agent
    expect(repl.agent?.name).toBe('robutler');
    const out = await say(repl, '/publish');
    expect(out).toContain('Publishing needs an AGENT.md in this folder.');
    expect(out).toContain('webagents init');
  });
});
