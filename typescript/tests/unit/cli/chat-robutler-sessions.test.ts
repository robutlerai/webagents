/**
 * The chat's conversations on Robutler (`session: {backend: robutler}`,
 * 2026-09-25), the Python chat's cases (`tests/cli/test_chat.py`) against a
 * stub of the portal's `/api/agents/{id}/conversations` routes: signed out,
 * `/resume` says why they stay here; signed in and published, a turn is
 * recorded into the person's chat with the agent and remembered, the next
 * sends only what it added, and `/resume` lists Robutler's conversations and
 * continues one started on the web. A failing platform is said once.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { sessionsDir } from '../../../src/cli/sessions';
import { tempDirs } from '../../helpers/cli';

const tempDir = tempDirs();
const AGENT_ID = '9b1f2c3d-4e5f-4a6b-8c7d-0e1f2a3b4c5d';
const ROBUTLER_AGENT = '---\nname: helper\nskills:\n  - session: {backend: robutler}\n---\nHelp.\n';

const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let project = '';
let printed: string[] = [];

/** The platform, as the stub answers it. */
const platform = {
  requests: [] as { method: string; url: string; auth: string | null; body: unknown }[],
  recorded: [] as Record<string, unknown>[],
  listing: [] as unknown[],
  chats: {} as Record<string, unknown[]>,
};

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-rs-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'https://robutler.example';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '100';
  project = tempDir('wa-rs-project-');
  process.chdir(project);
  printed = [];
  platform.requests = [];
  platform.recorded = [];
  platform.listing = [];
  platform.chats = {};
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = new URL(String(input));
    const auth = new Headers(init?.headers).get('authorization');
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    platform.requests.push({ method: init?.method ?? 'GET', url: url.pathname, auth, body });
    const json = (value: unknown, status = 200) =>
      new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
    if (auth !== 'Bearer person-token') return json({ error: 'Unauthorized' }, 401);
    const base = `/api/agents/${AGENT_ID}/conversations`;
    if (init?.method === 'POST' && url.pathname === base) {
      platform.recorded.push(body as Record<string, unknown>);
      return json({ chatId: (body as { chatId?: string }).chatId ?? 'c-terminal', created: true }, 201);
    }
    if (url.pathname === base) return json({ conversations: platform.listing });
    if (url.pathname.startsWith(`${base}/`)) {
      const chat = url.pathname.slice(base.length + 1);
      return json({ chatId: chat, messages: platform.chats[chat] ?? [] });
    }
    return json({ error: 'Agent not found' }, 404);
  });
});

afterEach(() => {
  process.chdir(cwd);
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
  vi.restoreAllMocks();
});

type Inside = {
  handleInput(line: string): Promise<void>;
  messages: Array<{ role: string; content: string }>;
  sessionId: string;
  platformChatId: string | undefined;
  recordedCount: number;
  recording: Promise<void>;
  saveConversation(): void;
  recordOnRobutler(): void;
  sayRecordingProblem(): void;
};

async function chat(): Promise<Inside> {
  fs.writeFileSync(path.join(project, 'AGENT.md'), ROBUTLER_AGENT);
  const repl = new InteractiveREPL({});
  await repl.initialize();
  return repl as unknown as Inside;
}

async function say(repl: Inside, line: string): Promise<string> {
  const before = printed.length;
  await repl.handleInput(line);
  return printed.slice(before).join('\n');
}

function signedInAndPublished(): void {
  process.env.WEBAGENTS_TOKEN = 'person-token';
  fs.mkdirSync(path.join(project, '.webagents'), { recursive: true });
  fs.writeFileSync(
    path.join(project, '.webagents', 'config.json'),
    JSON.stringify({ 'link.agentId': AGENT_ID, 'link.agentName': 'ada.helper' }),
  );
}

async function record(repl: Inside): Promise<void> {
  repl.recordOnRobutler();
  await repl.recording;
}

describe('conversations on Robutler', () => {
  it('signed out, /resume says the conversations stay here, and asks nothing of the platform', async () => {
    const repl = await chat();
    const out = await say(repl, '/resume');
    expect(out).toContain('Conversations stay on this machine: sign in with `webagents login` to keep them on Robutler too.');
    expect(platform.requests).toEqual([]);
  });

  it('records a turn into the chat with the agent, remembers it, and sends only what the next adds', async () => {
    signedInAndPublished();
    const repl = await chat();
    repl.messages = [
      { role: 'user', content: 'plan the launch' },
      { role: 'assistant', content: 'Step one.' },
    ];
    repl.saveConversation();
    await record(repl);

    expect(platform.recorded).toEqual([
      {
        sessionId: repl.sessionId,
        messages: [
          { role: 'user', content: 'plan the launch' },
          { role: 'assistant', content: 'Step one.' },
        ],
      },
    ]);
    expect(repl.platformChatId).toBe('c-terminal');
    expect(repl.recordedCount).toBe(2);
    const file = path.join(sessionsDir(project, 'helper'), `${repl.sessionId}.json`);
    const kept = JSON.parse(fs.readFileSync(file, 'utf-8'));
    expect(kept.metadata).toMatchObject({ robutler_chat_id: 'c-terminal', robutler_recorded: 2 });

    repl.messages.push({ role: 'user', content: 'and then?' }, { role: 'assistant', content: 'Step two.' });
    await record(repl);
    expect(platform.recorded.at(-1)).toEqual({
      sessionId: repl.sessionId,
      chatId: 'c-terminal',
      messages: [
        { role: 'user', content: 'and then?' },
        { role: 'assistant', content: 'Step two.' },
      ],
    });
  });

  it('/resume lists Robutler and continues a chat started on the web', async () => {
    signedInAndPublished();
    const repl = await chat();
    repl.messages = [
      { role: 'user', content: 'plan the launch' },
      { role: 'assistant', content: 'Step one.' },
    ];
    repl.saveConversation();
    const here = repl.sessionId;
    await say(repl, '/new');
    platform.listing = [
      { chatId: 'c-web', sessionId: null, updatedAt: '2099-01-01T00:00:00.000Z', messageCount: 2, preview: 'from the web' },
    ];
    platform.chats['c-web'] = [
      { role: 'user', content: 'from the web' },
      { role: 'assistant', content: 'Hello.' },
    ];

    const listing = await say(repl, '/resume');
    expect(listing).toContain('Robutler: from the web');
    expect(listing).toContain('plan the launch');

    const out = await say(repl, '/resume 1');
    expect(out).toContain('Continuing the conversation');
    expect(out).toContain('(2 messages)');
    expect(repl.messages[0].content).toBe('from the web');
    expect(repl.platformChatId).toBe('c-web');
    expect(repl.recordedCount).toBe(2);
    expect(repl.sessionId).not.toBe(here);
    expect(fs.existsSync(path.join(sessionsDir(project, 'helper'), `${repl.sessionId}.json`))).toBe(true);
  });

  it('a failing platform is said once, and the conversation goes on', async () => {
    signedInAndPublished();
    process.env.WEBAGENTS_TOKEN = 'expired';
    const repl = await chat();
    repl.messages = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ];
    await record(repl);
    const before = printed.length;
    repl.sayRecordingProblem();
    repl.sayRecordingProblem();
    const said = printed.slice(before).join('\n');
    expect(
      said.split('This conversation is not being kept on Robutler: your sign-in has expired: run `webagents login`.').length - 1,
    ).toBe(1);
    expect(repl.platformChatId).toBeUndefined();
  });
});
