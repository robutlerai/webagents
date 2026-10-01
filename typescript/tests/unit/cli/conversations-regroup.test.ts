/**
 * The chat's commands regrouped, and the conversation's life cycle
 * (2026-09-29), the same in the Python chat
 * (`python/tests/cli/test_conversations_regroup.py`).
 *
 * The owner asked for a better grouping of the commands, for one level of
 * subcommands rather than `/agent model`, and how conversations are started,
 * continued and deleted. Pinned here: `/resume delete <number>` removes one
 * earlier conversation after asking (never the current one, never a copy on
 * Robutler); the chat starts a new conversation and says, in one faint line,
 * when the last one here is less than a day old; `-c` and `-r [number]` open
 * the chat on an earlier one; `/keys remove` is the verb (`unset` still
 * taken); and `new` and `edit`, the words `/agent` keeps for itself, are
 * refused as names.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const H = vi.hoisted(() => ({ lines: [] as Array<string | null> }));
vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => (H.lines.length ? H.lines.shift()! : null)),
  promptSecret: vi.fn(async () => ''),
  promptSecretOrPipe: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { CHAT_WORDS, fill } from '../../../src/cli/chat-words';
import { RESERVED_AGENT_NAMES, RESERVED_NAME, reservedName } from '../../../src/cli/init-templates';
import { deleteSession, saveSession, sessionsDir } from '../../../src/cli/sessions';
import { CLI_ARGS, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/cli');
const GRAMMAR = (JSON.parse(fs.readFileSync(path.join(FIXTURES, 'init_templates.json'), 'utf8')) as {
  name_grammar: { reserved: string[]; reserved_sentence: string };
}).name_grammar;
const JSON_ERRORS = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'json_errors.json'), 'utf8')) as Record<string, { code: string; message: string; exit: number }>;

const tempDir = tempDirs();
const ISOLATED = [
  'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
  'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
];
const saved: Record<string, string | undefined> = {};
const cwd = process.cwd();
let project = '';
let printed: string[] = [];
const AGENT = '---\nname: helper\n---\nHelp.\n';

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  process.env.HOME = tempDir('wa-regroup-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '100';
  project = tempDir('wa-regroup-project-');
  process.chdir(project);
  fs.writeFileSync(path.join(project, 'AGENT.md'), AGENT);
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
  vi.spyOn(console, 'warn').mockImplementation(() => {});
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

type Inside = {
  handleInput(line: string): Promise<void>;
  startWhereAsked(): Promise<void>;
  messages: Array<{ role: string; content: string }>;
  saveConversation(): void;
  sessionId: string;
};

async function chat(resume?: string): Promise<Inside> {
  const repl = new InteractiveREPL({ interactive: true, ...(resume !== undefined ? { resume } : {}) });
  await repl.initialize();
  return repl as unknown as Inside;
}

async function said(action: () => Promise<void>): Promise<string> {
  const before = printed.length;
  await action();
  return printed.slice(before).join('\n');
}

/** A conversation as the chat saves it, last used `hoursAgo` hours ago. */
function savedConversation(id: string, text: string, hoursAgo: number, chatId?: string): void {
  const dir = sessionsDir(fs.realpathSync(project), 'helper');
  saveSession(dir, {
    session_id: id,
    agent_name: 'helper',
    created_at: '',
    updated_at: '',
    messages: [{ role: 'user', content: text }, { role: 'assistant', content: 'ok' }],
    metadata: chatId ? { robutler_chat_id: chatId } : {},
    input_tokens: 0,
    output_tokens: 0,
  });
  const file = path.join(dir, `${id}.json`);
  const data = JSON.parse(fs.readFileSync(file, 'utf8')) as Record<string, unknown>;
  data.updated_at = new Date(Date.now() - hoursAgo * 3_600_000).toISOString();
  fs.writeFileSync(file, JSON.stringify(data));
}

describe('deleting', () => {
  it('/resume delete removes an earlier conversation after asking', async () => {
    savedConversation('11111111-1111-4111-8111-111111111111', 'the older one', 5);
    savedConversation('22222222-2222-4222-8222-222222222222', 'the newer one', 1);
    const dir = sessionsDir(fs.realpathSync(project), 'helper');
    const repl = await chat();
    H.lines = ['n'];
    let out = await said(() => repl.handleInput('/resume delete 2'));
    expect(out).toContain(CHAT_WORDS.resumeNotDeleted);
    expect(fs.existsSync(path.join(dir, '11111111-1111-4111-8111-111111111111.json'))).toBe(true);
    H.lines = ['y'];
    out = await said(() => repl.handleInput('/resume delete 2'));
    expect(out).toContain(fill('resumeDeleted', { when: '5 h ago' }));
    expect(out).not.toContain(CHAT_WORDS.resumeDeletedRemote);
    expect(fs.existsSync(path.join(dir, '11111111-1111-4111-8111-111111111111.json'))).toBe(false);
    const listing = await said(() => repl.handleInput('/resume'));
    expect(listing).toContain('the newer one');
    expect(listing).not.toContain('the older one');
    expect(listing).toContain('/resume delete <number> deletes one');
  });

  it('says what it cannot do, and that a copy on Robutler stays', async () => {
    savedConversation('33333333-3333-4333-8333-333333333333', 'on robutler too', 2, 'chat-1');
    const repl = await chat();
    expect(await said(() => repl.handleInput('/resume delete'))).toContain(fill('usage', { usage: '/resume delete <number>' }));
    const none = await said(() => repl.handleInput('/resume delete 7'));
    expect(none).toContain(fill('resumeNoNumber', { pick: '7' }));
    expect(none).toContain(CHAT_WORDS.resumeNoNumberHint);
    H.lines = ['y'];
    const out = await said(() => repl.handleInput('/resume delete 1'));
    expect(out).toContain(fill('resumeDeleted', { when: '2 h ago' }));
    expect(out).toContain(CHAT_WORDS.resumeDeletedRemote);
  });

  it('never offers the current conversation', async () => {
    const repl = await chat();
    repl.messages = [{ role: 'user', content: 'the current one' }, { role: 'assistant', content: 'ok' }];
    repl.saveConversation();
    expect(await said(() => repl.handleInput('/resume delete 1'))).toContain('No earlier conversations with helper in this folder.');
    expect(fs.existsSync(path.join(sessionsDir(fs.realpathSync(project), 'helper'), `${repl.sessionId}.json`))).toBe(true);
  });

  it('deleteSession takes the latest pointer with it', () => {
    const dir = tempDir('wa-regroup-store-');
    const blank = { agent_name: 'x', created_at: '', updated_at: '', messages: [], metadata: {}, input_tokens: 0, output_tokens: 0 };
    saveSession(dir, { ...blank, session_id: 'a' });
    saveSession(dir, { ...blank, session_id: 'b' });
    expect(fs.readFileSync(path.join(dir, '.latest'), 'utf8')).toBe('b');
    expect(deleteSession(dir, 'a')).toBe(true);
    expect(fs.readFileSync(path.join(dir, '.latest'), 'utf8')).toBe('b');
    expect(deleteSession(dir, 'b')).toBe(true);
    expect(fs.existsSync(path.join(dir, '.latest'))).toBe(false);
    expect(deleteSession(dir, 'b')).toBe(false);
  });
});

describe('starting', () => {
  it('says when the last conversation here is recent, and starts a new one', async () => {
    savedConversation('44444444-4444-4444-8444-444444444444', "yesterday's plan", 3);
    const repl = await chat();
    expect(await said(() => repl.startWhereAsked())).toContain(fill('lastConversation', { when: '3 h ago', count: '2' }));
    expect(repl.messages).toEqual([]);
  });

  it('does not mention an old one', async () => {
    savedConversation('55555555-5555-4555-8555-555555555555', 'last week', 72);
    const repl = await chat();
    expect((await said(() => repl.startWhereAsked())).trim()).toBe('');
  });

  it('-c continues the last one; -r lists them; -r 2 continues the second', async () => {
    savedConversation('66666666-6666-4666-8666-666666666666', 'the older one', 5);
    savedConversation('77777777-7777-4777-8777-777777777777', 'the newer one', 1);
    let repl = await chat('1');
    expect(await said(() => repl.startWhereAsked())).toContain('Continuing the conversation');
    expect(repl.messages[0].content).toBe('the newer one');
    repl = await chat('');
    const listing = await said(() => repl.startWhereAsked());
    expect(listing).toContain('Earlier conversations');
    expect(listing).toContain('the older one');
    expect(repl.messages).toEqual([]);
    repl = await chat('2');
    await said(() => repl.startWhereAsked());
    expect(repl.messages[0].content).toBe('the older one');
  });
});

describe('the flags, through the real CLI', () => {
  function runCli(args: string[]): Promise<{ code: number | null; out: string; err: string }> {
    return new Promise((resolve) => {
      const child = spawn(process.execPath, [...CLI_ARGS, ...args], {
        cwd: project,
        env: { ...process.env, WEBAGENTS_PROFILE: '' },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let out = '';
      let err = '';
      child.stdout.on('data', (d) => (out += d));
      child.stderr.on('data', (d) => (err += d));
      child.on('close', (code) => resolve({ code, out, err }));
    });
  }

  it.skipIf(TSX_PROBLEM !== null)('-r lists the conversations at the start; -c with -r or -p is refused', async () => {
    savedConversation('88888888-8888-4888-8888-888888888888', 'from the flag test', 1);
    const listed = await runCli(['-r']);
    expect(listed.out).toContain('Earlier conversations');
    expect(listed.out).toContain('from the flag test');
    const both = await runCli(['-c', '-r']);
    expect(both.code).toBe(2);
    expect(both.err).toContain('Use --continue or --resume, not both.');
    const withPrompt = await runCli(['-c', '-p', 'hi']);
    expect(withPrompt.code).toBe(2);
    expect(withPrompt.err).toContain('--continue and --resume open the chat; they do not go with -p.');
  }, 60_000);

  it.skipIf(TSX_PROBLEM !== null)('init refuses the names /agent keeps for itself', async () => {
    const plain = await runCli(['init', 'new']);
    expect(plain.code).toBe(1);
    expect(plain.err).toContain(RESERVED_NAME.replace('{name}', 'new'));
    expect(fs.existsSync(path.join(project, 'new'))).toBe(false);
    const asJson = await runCli(['--json', 'init', 'edit']);
    expect(asJson.code).toBe(JSON_ERRORS.reserved_name.exit);
    expect((JSON.parse(asJson.out) as { error: { code: string } }).error.code).toBe('reserved_name');
  }, 60_000);
});

describe('the subcommand rule', () => {
  it('/keys remove is the verb, and unset is still taken', async () => {
    const repl = await chat();
    expect(await said(() => repl.handleInput('/keys'))).toContain('/keys remove <NAME> removes a stored one');
    expect(await said(() => repl.handleInput('/keys drop OPENAI_API_KEY'))).toContain('Usage: /keys [set|remove NAME]');
    expect(await said(() => repl.handleInput('/keys remove OPENAI_API_KEY'))).toContain('OPENAI_API_KEY was not stored.');
    expect(await said(() => repl.handleInput('/keys unset OPENAI_API_KEY'))).toContain('OPENAI_API_KEY was not stored.');
  });

  it('refuses new and edit as agent names', async () => {
    expect([...RESERVED_AGENT_NAMES]).toEqual(GRAMMAR.reserved);
    expect(RESERVED_NAME).toBe(GRAMMAR.reserved_sentence);
    expect(RESERVED_NAME).toBe(JSON_ERRORS.reserved_name.message);
    expect(reservedName('edit')).toBe(true);
    expect(reservedName('some/where/New')).toBe(true);
    expect(reservedName('editor')).toBe(false);
    const repl = await chat();
    expect(await said(() => repl.handleInput('/agent new edit'))).toContain(RESERVED_NAME.replace('{name}', 'edit'));
    expect(fs.existsSync(path.join(project, 'AGENT-edit.md'))).toBe(false);
  });
});
