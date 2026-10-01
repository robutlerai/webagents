/**
 * Compaction in the chat (2026-09-29), the same in the Python chat
 * (`python/tests/cli/test_chat_compaction.py`).
 *
 * The owner asked for the best strategy for compaction, with settings, a
 * command and an API. Pinned here, with the agent's model replaced by a stub:
 * `/compact` makes a summary of everything before the latest exchange, the
 * conversation the model is sent becomes that, and the whole conversation
 * stays in the file (`transcript`); before a message is sent past
 * `compaction.at` the chat compacts first and says so; `/context` says how
 * full the context is and when it compacts; the footer shows it past half;
 * and a run's safety stop leaves the turn in progress whole.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
  promptSecretOrPipe: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { CHAT_WORDS, fill } from '../../../src/cli/chat-words';
import { DEFAULT_POLICY, WORDS, type CompactMessage } from '../../../src/core/context-compaction';
import { listSessions, loadSession, sessionsDir } from '../../../src/cli/sessions';
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
  process.env.HOME = tempDir('wa-compact-home-');
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
  process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
  process.env.NO_COLOR = '1';
  process.env.COLUMNS = '120';
  project = tempDir('wa-compact-project-');
  process.chdir(project);
  fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: helper\nmodel: openai/gpt-4o-mini\nskills:\n  - openai\n---\nHelp.\n');
  printed = [];
  vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
    printed.push(args.map(String).join(' '));
  });
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

type Agent = {
  compactionPolicy: typeof DEFAULT_POLICY;
  lastCompaction?: { stage: string };
  run: unknown;
  compactIfFull(conversation: CompactMessage[]): Promise<void>;
};
type Inside = {
  handleInput(line: string): Promise<void>;
  compactBeforeSending(content: string): Promise<void>;
  footerParts(): string[];
  messages: CompactMessage[];
  transcript: CompactMessage[] | undefined;
  sessionId: string;
  agent: Agent;
};

async function chat(): Promise<{ repl: Inside; asked: string[] }> {
  const repl = new InteractiveREPL({ interactive: true }) as unknown as Inside & { initialize(): Promise<void> };
  await repl.initialize();
  const asked: string[] = [];
  repl.agent.run = async (messages: Array<{ content: string }>) => {
    const text = messages[0].content;
    asked.push(text);
    const transcript = text.split('\n\n').slice(1).join('\n\n');
    return { content: `[stub summary of ${transcript.split('\n').length} lines]` };
  };
  return { repl, asked };
}

async function said(action: () => Promise<void> | void): Promise<string> {
  const before = printed.length;
  await action();
  return printed.slice(before).join('\n');
}

function talk(n: number, size = 0): CompactMessage[] {
  const out: CompactMessage[] = [];
  for (let i = 0; i < n; i += 1) {
    out.push({ role: 'user', content: `question ${i} ${'x'.repeat(size)}`.trimEnd() });
    out.push({ role: 'assistant', content: `answer ${i} ${'y'.repeat(size)}`.trimEnd() });
  }
  return out;
}

describe('compaction in the chat', () => {
  it('/compact summarizes, and the file keeps the whole conversation', async () => {
    const { repl, asked } = await chat();
    const whole = talk(3);
    repl.messages = [...whole];
    const out = await said(() => repl.handleInput('/compact the budget'));
    expect(out).toContain(CHAT_WORDS.compacting);
    expect(out).toContain('Compacted the conversation: 4 earlier messages became a summary, the last 2 stay as they were.');
    expect(repl.messages[0]).toEqual({ role: 'system', content: `${WORDS.summaryPrefix}[stub summary of 4 lines]` });
    expect(repl.messages.slice(1)).toEqual(whole.slice(4));
    expect(asked[0]).toContain('Pay particular attention to: the budget');
    const kept = loadSession(sessionsDir(fs.realpathSync(project), 'helper'), repl.sessionId);
    expect(kept?.messages).toEqual(repl.messages);
    expect(kept?.transcript).toEqual(whole);
  });

  it('says when there is nothing to compact', async () => {
    const { repl } = await chat();
    expect(await said(() => repl.handleInput('/compact'))).toContain(CHAT_WORDS.compactEmpty);
    repl.messages = talk(1);
    expect(await said(() => repl.handleInput('/compact'))).toContain('Nothing to compact yet');
    expect(repl.transcript).toBeUndefined();
  });

  it('compacts before a message past the threshold, and not again under it', async () => {
    const { repl } = await chat();
    repl.agent.compactionPolicy = { ...DEFAULT_POLICY, at: 200, keep: 40, hard: 10000 };
    repl.messages = talk(4, 200);
    expect(await said(() => repl.compactBeforeSending('and now?'))).toContain('Compacted the conversation');
    expect(String(repl.messages[0].content).startsWith(WORDS.summaryPrefix)).toBe(true);
    expect(repl.transcript).toEqual(talk(4, 200));
    const kept = [...repl.messages];
    expect((await said(() => repl.compactBeforeSending('thanks'))).trim()).toBe('');
    expect(repl.messages).toEqual(kept);
  });

  it('does nothing on its own with auto off', async () => {
    const { repl } = await chat();
    repl.agent.compactionPolicy = { ...DEFAULT_POLICY, auto: false, at: 200, keep: 40, hard: 10000 };
    repl.messages = talk(4, 200);
    expect((await said(() => repl.compactBeforeSending('and now?'))).trim()).toBe('');
    expect(repl.transcript).toBeUndefined();
  });

  it('/context says how full it is; the footer shows it past half', async () => {
    const { repl } = await chat();
    repl.messages = talk(2);
    const out = await said(() => repl.handleInput('/context'));
    expect(out).toContain(fill('contextHeading', { model: 'openai/gpt-4o-mini', window: '128k' }));
    expect(out).toContain('The conversation: about');
    expect(out).toContain('in 4 messages.');
    expect(out).toContain(fill('contextAuto', { at: '80', tokens: '102k' }));
    expect(repl.footerParts().some((p) => p.startsWith('context '))).toBe(false);
    repl.agent.compactionPolicy = { ...DEFAULT_POLICY, window: 40 };
    expect(repl.footerParts().some((p) => p.startsWith('context ') && p.endsWith('%'))).toBe(true);
    repl.agent.compactionPolicy = { ...DEFAULT_POLICY, auto: false };
    expect(await said(() => repl.handleInput('/context'))).toContain(CHAT_WORDS.contextAutoOff);
    expect(await said(() => repl.handleInput('/context now'))).toContain('Usage: /context');
  });

  it('a resumed conversation keeps its transcript', async () => {
    const { repl } = await chat();
    repl.messages = talk(3);
    await said(() => repl.handleInput('/compact'));
    const compacted = [...repl.messages];
    const whole = [...(repl.transcript ?? [])];
    await said(() => repl.handleInput('/new'));
    expect(repl.transcript).toBeUndefined();
    const out = await said(() => repl.handleInput('/resume 1'));
    expect(repl.messages).toEqual(compacted);
    expect(repl.transcript).toEqual(whole);
    // "N messages" is the whole conversation's words, not the compacted view's.
    expect(out).toContain('(6 messages)');
  });

  it('a compacted conversation is listed whole', async () => {
    const { repl } = await chat();
    repl.messages = talk(3);
    await said(() => repl.handleInput('/compact'));
    const [kept] = listSessions(sessionsDir(fs.realpathSync(project), 'helper'));
    expect([kept.messageCount, kept.preview]).toEqual([6, 'question 0']);
  });

  it("the run's safety stop leaves the turn in progress whole", async () => {
    const { repl } = await chat();
    const agent = repl.agent;
    agent.compactionPolicy = { ...DEFAULT_POLICY, at: 100, keep: 20, hard: 300 };
    const turn: CompactMessage[] = [
      { role: 'user', content: 'read the big file' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: 'z'.repeat(900) },
    ];
    const conversation: CompactMessage[] = [{ role: 'system', content: 'You help.' }, ...talk(3, 300), ...turn];
    await agent.compactIfFull(conversation);
    expect(conversation[0]).toEqual({ role: 'system', content: 'You help.' });
    expect(String(conversation[1].content).startsWith(WORDS.summaryPrefix)).toBe(true);
    expect(conversation.slice(-3)).toEqual(turn);
    expect(agent.lastCompaction?.stage).toBe('summarized');
  });
});
