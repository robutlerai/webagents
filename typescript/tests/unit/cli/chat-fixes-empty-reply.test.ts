/**
 * What the chat says about a turn that produced no answer (2026-09-27, the
 * chat-fixes lane), against the shared fixture
 * `python/tests/fixtures/cli/chat_fixes_empty_reply.json`, which the Python
 * suite reads too (`test_chat_fixes_empty_reply.py`):
 *
 *  - the sentences (`presentEmptyReply`), the same in both chats;
 *  - the printer draws the line for a turn with no text and no error, and
 *    says when the model only produced thinking;
 *  - an empty turn is not counted as a reply by the goodbye line, and leaves
 *    no unanswered message in the conversation.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import * as fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/cli/prompt', () => ({
  promptLine: vi.fn(async () => null),
  promptSecret: vi.fn(async () => ''),
}));

import { InteractiveREPL } from '../../../src/cli/app';
import { EMPTY_REPLY_HINT, presentEmptyReply } from '../../../src/cli/failures';
import { TurnPrinter } from '../../../src/cli/render';
import { RETRY_FINISH_REASONS } from '../../../src/skills/llm/proxy/skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TABLE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_fixes_empty_reply.json'), 'utf8'),
) as {
  hint: string;
  retry_finish_reasons: string[];
  cases: Array<{ name: string; reason: string | null; blocked: boolean; retried: boolean; thinking: boolean; rounds?: number; tool?: string; headline: string }>;
};

describe('what an empty reply says (the table both SDKs run)', () => {
  it.each(TABLE.cases.map((c) => [c.name, c] as const))('%s', (_name, c) => {
    const explained = presentEmptyReply({ reason: c.reason, blocked: c.blocked, retried: c.retried, rounds: c.rounds, tool: c.tool }, { thinking: c.thinking });
    expect(explained).toEqual({ headline: c.headline, hint: TABLE.hint });
  });

  it('the hint and the retry list are the fixture\'s', () => {
    expect(EMPTY_REPLY_HINT).toBe(TABLE.hint);
    expect([...RETRY_FINISH_REASONS]).toEqual(TABLE.retry_finish_reasons);
  });
});

describe('the printer', () => {
  function draw(chunks: Array<Record<string, unknown>>): string {
    const writes: string[] = [];
    const out = { write: (s: string) => { writes.push(s); return true; }, isTTY: false };
    const printer = new TurnPrinter({ out: out as unknown as NodeJS.WriteStream, color: false, live: false });
    printer.start();
    for (const chunk of chunks) printer.feed(chunk as never);
    printer.finish();
    return writes.join('');
  }

  it('draws the truthful line for a turn with no text and no error', () => {
    const output = draw([{ type: 'done', response: { content: '', finish: { reason: 'MALFORMED_FUNCTION_CALL', retried: true } } }]);
    expect(output).toContain('Error: The model returned no answer: the provider reported MALFORMED_FUNCTION_CALL, twice.');
    expect(output).toContain(TABLE.hint);
  });

  it('says when the model only produced thinking', () => {
    const output = draw([
      { type: 'thinking', thinking: { content: 'Let me see.' } },
      { type: 'done', response: { content: '', finish: { reason: 'STOP' } } },
    ]);
    expect(output).toContain('The model returned no answer: it only produced thinking (the provider reported STOP).');
  });

  it('says nothing of the kind when the turn answered, failed or was stopped', () => {
    expect(draw([{ type: 'delta', delta: 'Two files.' }, { type: 'done', response: { content: 'Two files.' } }])).not.toContain('returned no answer');
    expect(draw([{ type: 'error', error: new Error('boom') }])).not.toContain('returned no answer');
    const writes: string[] = [];
    const out = { write: (s: string) => { writes.push(s); return true; }, isTTY: false };
    const printer = new TurnPrinter({ out: out as unknown as NodeJS.WriteStream, color: false, live: false });
    printer.finish({ interrupted: true });
    expect(writes.join('')).not.toContain('returned no answer');
  });
});

describe('the chat', () => {
  const tempDir = tempDirs();
  const ISOLATED = [
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'XAI_API_KEY', 'HOME', 'WEBAGENTS_SECRETS_BACKEND',
    'WEBAGENTS_TOKEN', 'WEBAGENTS_PROFILE', 'ROBUTLER_API_URL', 'ROBUTLER_LLM_PROXY_URL', 'NO_COLOR', 'COLUMNS',
  ];
  const saved: Record<string, string | undefined> = {};
  const cwd = process.cwd();
  let written: string[] = [];

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    process.env.HOME = tempDir('wa-empty-home-');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
    process.env.OPENAI_API_KEY = 'sk-test-not-a-real-key';
    process.env.NO_COLOR = '1';
    process.env.COLUMNS = '100';
    const project = tempDir('wa-empty-project-');
    process.chdir(project);
    fs.writeFileSync(path.join(project, 'AGENT.md'), '---\nname: helper\nskills:\n  - openai\n---\nHelp.\n');
    written = [];
    vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => written.push(args.map(String).join(' ')));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(process.stdout, 'write').mockImplementation(((s: string | Uint8Array) => { written.push(String(s)); return true; }) as never);
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
    streamToTerminal(content: string): Promise<void>;
    agent: { runStreaming: unknown } | null;
    messages: Array<{ role: string; content: string }>;
    turns: number;
    sessionTokens: number;
  }

  it('after an answer at the cap the chat asks, and yes goes on from the conversation (2026-09-28)', async () => {
    const prompt = await import('../../../src/cli/prompt');
    vi.mocked(prompt.promptLine).mockResolvedValueOnce('');
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as Inside;
    const sent: Array<Array<{ role: string; content: string }>> = [];
    inside.agent!.runStreaming = async function* (messages: Array<{ role: string; content: string }>) {
      sent.push(messages.map((m) => ({ role: m.role, content: m.content })));
      if (sent.length === 1) {
        yield { type: 'delta', delta: 'Two folders so far.' };
        yield { type: 'done', response: { content: 'Two folders so far.', finish: { reason: 'tool_round_limit', rounds: 3 } } };
      } else {
        yield { type: 'delta', delta: 'All done.' };
        yield { type: 'done', response: { content: 'All done.' } };
      }
    } as never;
    await inside.streamToTerminal('nice');
    expect(sent).toHaveLength(2);
    expect(sent[1].slice(-2)).toEqual([{ role: 'assistant', content: 'Two folders so far.' }, { role: 'user', content: 'Keep going.' }]);
    const asked = vi.mocked(prompt.promptLine).mock.calls.map((c) => String(c[0]));
    expect(asked.some((q) => q.includes('Used 3 tool rounds. Keep going? [Y/n]'))).toBe(true);
    expect(written.join('')).toContain('All done.');
  });

  it('/rounds shows, sets and refuses, and a rebuild keeps this chat\'s value (2026-09-28)', async () => {
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as { commandRounds(args: string): Promise<void>; agent: { maxToolIterations: number } };
    await inside.commandRounds('');
    await inside.commandRounds('12');
    await inside.commandRounds('abc');
    const out = written.join('\n');
    expect(out).toContain('Tool rounds: 50 per turn (the default).');
    expect(out).toContain('Tool rounds set to 12 per turn, for this chat.');
    expect(out).toContain('/rounds must be a whole number from 1 to 1000, not "abc".');
    expect(inside.agent.maxToolIterations).toBe(12);
    await repl.initialize();
    expect((repl as unknown as { agent: { maxToolIterations: number } }).agent.maxToolIterations).toBe(12);
  });

  it('an empty turn says so, is not a reply, and leaves no unanswered message', async () => {
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as Inside;
    inside.agent!.runStreaming = async function* () {
      yield { type: 'done', response: { content: '', usage: { input_tokens: 120, output_tokens: 0, total_tokens: 120 }, finish: { reason: 'MALFORMED_FUNCTION_CALL', retried: true } } };
    };
    await inside.streamToTerminal('list the folder');
    const out = written.join('');
    expect(out).toContain('The model returned no answer: the provider reported MALFORMED_FUNCTION_CALL, twice.');
    expect(out).toContain(TABLE.hint);
    expect(inside.turns).toBe(0);
    expect(inside.messages).toEqual([]);
    expect(inside.sessionTokens).toBe(120);
  });

  it('a turn that answered counts, and prints no such line', async () => {
    const repl = new InteractiveREPL({ interactive: true });
    await repl.initialize();
    const inside = repl as unknown as Inside;
    inside.agent!.runStreaming = async function* () {
      yield { type: 'delta', delta: 'Two files.' };
      yield { type: 'done', response: { content: 'Two files.', finish: { reason: 'STOP' } } };
    };
    await inside.streamToTerminal('hi');
    expect(written.join('')).not.toContain('returned no answer');
    expect(inside.turns).toBe(1);
    expect(inside.messages.at(-1)).toEqual({ role: 'assistant', content: 'Two files.' });
  });
});
