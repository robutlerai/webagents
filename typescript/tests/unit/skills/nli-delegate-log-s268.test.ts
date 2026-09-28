/**
 * The delegate tool logs lengths, not text (S-268, 2026-09-26; S-227's rule).
 *
 * `[nli/delegate] → ... first200=<message>` went to stdout with a bare
 * `console.log` on every delegation, which for a hosted agent is the portal's
 * pod log; the line beside it listed the conversation's content ids and
 * roles. Both go through the agent trace now, and carry content only under
 * `LOG_LOOP_DEBUG=1`.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

vi.mock('../../../src/uamp/client.js', () => ({
  UAMPClient: vi.fn().mockImplementation(() => ({ on: vi.fn(), connect: vi.fn(), sendEvents: vi.fn(), close: vi.fn() })),
}));

import { NLISkill } from '../../../src/skills/nli/skill.js';
import { setAgentTrace } from '../../../src/core/trace.js';
import type { Context, AgenticMessage } from '../../../src/core/types.js';

const SECRET = 'the patient is Jane Doe, born 1970-01-01, and her diagnosis is confidential';

function makeContext(overrides: Record<string, unknown> = {}): Context {
  const store = new Map<string, unknown>(Object.entries(overrides));
  return {
    get: <T>(key: string) => store.get(key) as T | undefined,
    set: (key: string, value: unknown) => { store.set(key, value); },
    delete: (key: string) => { store.delete(key); },
    signal: new AbortController().signal,
  } as Context;
}

async function delegateOnce(): Promise<{ logged: string[]; traced: string[] }> {
  const traced: string[] = [];
  setAgentTrace({ enabled: true, sink: (line) => traced.push(line) });
  const log = vi.spyOn(console, 'log').mockImplementation(() => {});
  try {
    const skill = new NLISkill({ baseUrl: 'https://example.com', transport: 'http' });
    (skill as unknown as { streamMessage: unknown }).streamMessage = vi.fn().mockImplementation(async function* () {
      yield 'Done';
    });
    const messages: AgenticMessage[] = [
      { role: 'user', content: 'hello', content_items: [{ type: 'image', image: { url: '/x' }, content_id: 'cid-1' } as never] },
    ];
    await skill.delegate({ agent: 'helper', message: SECRET }, makeContext({ _agentic_messages: messages }));
    return { logged: log.mock.calls.map((c) => c.map(String).join(' ')), traced };
  } finally {
    setAgentTrace({ enabled: true, sink: (line) => console.log(line) });
    log.mockRestore();
  }
}

afterEach(() => {
  delete process.env.LOG_LOOP_DEBUG;
});

describe('nli delegate trace (S-268)', () => {
  it('writes lengths and counts, never the message or the content ids', async () => {
    delete process.env.LOG_LOOP_DEBUG;
    const { logged, traced } = await delegateOnce();
    expect(logged.join('\n')).not.toContain(SECRET);
    expect(traced.join('\n')).not.toContain(SECRET);
    expect(traced.join('\n')).not.toContain('cid-1');
    const line = traced.find((l) => l.startsWith('[nli/delegate] → @helper '));
    expect(line).toBeDefined();
    expect(line).toContain(`message=${SECRET.length} chars`);
    expect(line).not.toContain('first200=');
    const map = traced.find((l) => l.startsWith('[nli/delegate] convContentMap: '));
    expect(map).toContain('1 entries, messages=1, msgContentItems=[1]');
    expect(map).not.toContain('keys=');
  });

  it('carries the text only under LOG_LOOP_DEBUG=1, and then through the trace, not console.log', async () => {
    process.env.LOG_LOOP_DEBUG = '1';
    const { logged, traced } = await delegateOnce();
    expect(logged.join('\n')).not.toContain(SECRET);
    const line = traced.find((l) => l.startsWith('[nli/delegate] → @helper '));
    expect(line).toContain(`first200=${SECRET.slice(0, 200)}`);
    expect(traced.find((l) => l.startsWith('[nli/delegate] convContentMap: '))).toContain('keys=[cid-1]');
  });

  it('no bare console.log in the skill carries the delegated message', () => {
    const HERE = path.dirname(fileURLToPath(import.meta.url));
    const source = fs.readFileSync(path.resolve(HERE, '../../../src/skills/nli/skill.ts'), 'utf-8');
    const offenders = source
      .split('\n')
      .map((line, i) => ({ line, n: i + 1 }))
      .filter(({ line }) => /console\.log\(/.test(line) && /message\.slice|first200|convContentMap\.keys/.test(line))
      .map(({ n }) => `nli/skill.ts:${n}`);
    expect(offenders).toEqual([]);
  });
});
