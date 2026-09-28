/**
 * What the chat keeps of a turn's tool calls, and how much goes back to the
 * model (2026-09-28), the same in both SDKs:
 * `python/tests/fixtures/chat/turn_history.json`, which the Python suite runs
 * too (`tests/cli/test_turn_history_fixture.py`).
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { StreamChunk } from '../../../src/core/types';
import type { Message } from '../../../src/uamp/types';
import {
  TOOL_HISTORY_BUDGET_CHARS,
  TurnRecorder,
  historyForModel,
  leftOutResult,
  spokenCount,
} from '../../../src/cli/turn-history';
import { cliPreamble, withCliPreamble } from '../../../src/cli/preamble';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/chat/turn_history.json'), 'utf8'));

describe('turn history', () => {
  it('the budget and the words', () => {
    expect(TOOL_HISTORY_BUDGET_CHARS).toBe(FIXTURE.budget_chars);
    expect(leftOutResult(FIXTURE.left_out.chars)).toBe(FIXTURE.left_out.text);
  });

  it('a turn keeps its answered rounds, in order', () => {
    const recorder = new TurnRecorder();
    for (const e of FIXTURE.turn.events as Array<{ call?: { id: string; name: string; arguments: string }; result?: { id: string; result: string } }>) {
      const chunk: StreamChunk = e.call
        ? { type: 'tool_call', tool_call: e.call }
        : { type: 'tool_result', tool_result: { call_id: e.result!.id, result: e.result!.result } };
      recorder.observe(chunk);
    }
    expect(recorder.messages()).toEqual(FIXTURE.turn.messages);
  });

  it('the copy sent keeps the newest results whole and leaves the rest out', () => {
    const history = structuredClone(FIXTURE.budget.history) as Message[];
    expect(historyForModel(history, FIXTURE.budget.budget)).toEqual(FIXTURE.budget.sent);
    expect(history).toEqual(FIXTURE.budget.history); // the conversation itself keeps everything
  });

  it('"N messages" counts the spoken ones', () => {
    expect(spokenCount(FIXTURE.spoken.messages)).toBe(FIXTURE.spoken.count);
  });

  it('the CLI preamble, and when it applies', () => {
    const p = FIXTURE.preamble;
    expect(cliPreamble(p.file, p.folder)).toBe(p.text);
    expect(withCliPreamble(p.instructions, p.file, ['filesystem'])).toBe(p.with_it);
    for (const c of p.cases as Array<{ skills: unknown[]; file: boolean; applies: boolean }>) {
      const got = withCliPreamble(p.instructions, c.file ? p.file : undefined, c.skills);
      expect(got !== p.instructions, JSON.stringify(c)).toBe(c.applies);
    }
  });
});
