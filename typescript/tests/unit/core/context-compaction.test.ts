/**
 * Compaction of a conversation filling the model's context
 * (`src/core/context-compaction.ts`, 2026-09-29), against the shared fixture
 * `python/tests/fixtures/context/compaction.json`, which the Python suite
 * reads too (`tests/agents/test_context_compaction.py`): the words, the window
 * table, the `compaction:` policy and its refusals, the counting, and what
 * `compactMessages` makes of each case, message for message.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  CLEAR_MIN_CHARS,
  CompactionPolicyError,
  DEFAULT_POLICY,
  DEFAULT_WINDOW,
  POLICY_WORDS,
  SETTLE,
  WORDS,
  compactMessages,
  compactionSentence,
  contextWindow,
  estimateMessageTokens,
  parsePolicy,
  type CompactMessage,
  type CompactionPolicy,
} from '../../../src/core/context-compaction';

const HERE = path.dirname(fileURLToPath(import.meta.url));
type Case = {
  about: string;
  window: number;
  policy: Record<string, unknown>;
  summarizer: 'stub' | 'fails';
  force?: boolean;
  focus?: string;
  protect_from?: number;
  threshold?: number;
  messages: CompactMessage[];
  expected: {
    stage: string;
    before: number;
    after: number;
    summarized: number;
    kept: number;
    cleared: number;
    dropped: number;
    summary: string | null;
    sentence: string;
    messages: CompactMessage[];
  };
};
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/context/compaction.json'), 'utf8')) as {
  words: Record<string, string>;
  policy_words: Record<string, string>;
  defaults: Record<string, number | boolean>;
  windows: Array<[string | null, number | null, number]>;
  policies: Array<{ raw: unknown; policy?: Record<string, unknown>; refused?: string }>;
  estimates: Array<[CompactMessage, number]>;
  cases: Case[];
};

async function stub(transcript: string, instructions: string): Promise<string> {
  const marker = 'Pay particular attention to: ';
  const focus = instructions.includes(marker) ? ` focusing on ${instructions.split(marker)[1].trim()}` : '';
  return `[stub summary of ${transcript.split('\n').length} lines${focus}]`;
}

function policyDict(p: CompactionPolicy): Record<string, unknown> {
  const out: Record<string, unknown> = { auto: p.auto, at: p.at, keep: p.keep, hard: p.hard, clear_tool_results: p.clearToolResults };
  for (const key of ['model', 'instructions', 'window'] as const) if (p[key] !== undefined) out[key] = p[key];
  return out;
}

describe('the fixture', () => {
  it('has the words and defaults', () => {
    expect(WORDS).toEqual(FIXTURE.words);
    expect(POLICY_WORDS).toEqual(FIXTURE.policy_words);
    const d = FIXTURE.defaults;
    expect(policyDict(DEFAULT_POLICY as CompactionPolicy)).toEqual({ auto: d.auto, at: d.at, keep: d.keep, hard: d.hard, clear_tool_results: d.clear_tool_results });
    expect([DEFAULT_WINDOW, CLEAR_MIN_CHARS, SETTLE]).toEqual([d.window, d.clear_min_chars, d.settle]);
  });

  it.each(FIXTURE.windows.map(([model, override, window]) => [String(model), model, override, window] as const))('window of %s', (_l, model, override, window) => {
    expect(contextWindow(model, override ?? undefined)).toBe(window);
  });

  it.each(FIXTURE.policies.map((c) => [JSON.stringify(c.raw), c] as const))('policy %s', (_l, c) => {
    if (c.refused !== undefined) {
      expect(() => parsePolicy(c.raw)).toThrow(new CompactionPolicyError(c.refused));
      return;
    }
    expect(policyDict(parsePolicy(c.raw))).toEqual(c.policy);
  });

  it.each(FIXTURE.estimates.map(([m, n], i) => [i, m, n] as const))('counts estimate %i', (_i, message, tokens) => {
    expect(estimateMessageTokens(message)).toBe(tokens);
  });

  it.each(FIXTURE.cases.map((c) => [c.about, c] as const))('%s', async (_about, c) => {
    const before = JSON.stringify(c.messages);
    const out = await compactMessages(c.messages, parsePolicy(Object.keys(c.policy).length ? c.policy : null), c.window, c.summarizer === 'stub' ? stub : async () => '', {
      ...(c.force ? { force: true } : {}),
      ...(c.focus ? { focus: c.focus } : {}),
      ...(c.protect_from !== undefined ? { protectFrom: c.protect_from } : {}),
      ...(c.threshold !== undefined ? { threshold: c.threshold } : {}),
    });
    const e = c.expected;
    expect([out.stage, out.before, out.after, out.summarized, out.kept, out.cleared, out.dropped]).toEqual([
      e.stage, e.before, e.after, e.summarized, e.kept, e.cleared, e.dropped,
    ]);
    expect(out.summary ?? null).toBe(e.summary);
    expect(compactionSentence(out)).toBe(e.sentence);
    expect(out.messages).toEqual(e.messages);
    expect(JSON.stringify(c.messages)).toBe(before);
  });
});
