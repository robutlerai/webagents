/**
 * The charge the platform reports, as the TypeScript chat reads it (B4,
 * 2026-09-28), against the shared fixture `w2ops/final_sdk_total_cost.json`
 * the Python suite reads too. The platform never sent the charge, so the chat
 * showed a list-price estimate for Robutler's models; the final-billing lane
 * makes it send `usage.total_cost`, and the chat shows that number as it is,
 * summed over a turn's model calls.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, tool } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent, createResponseDeltaEvent, generateEventId } from '../../../src/uamp/events';
import { NO_COST, addTurnCost, costWords, reportedCostCredits } from '../../../src/skills/llm/pricing';
import { TurnPrinter } from '../../../src/cli/render';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/w2ops/final_sdk_total_cost.json'), 'utf8'),
) as {
  cases: Array<{ name: string; usage: Record<string, unknown>; reported: number | null }>;
  turn: { calls: Array<{ input_tokens: number; output_tokens: number; total_tokens: number; total_cost: number }>; total_cost: number };
  footer: { model: string; usage: { input_tokens: number; output_tokens: number; total_cost: number }; words: string };
};

describe('the reader', () => {
  for (const c of FIXTURE.cases) {
    it(c.name, () => {
      expect(reportedCostCredits(c.usage as never) ?? null).toBe(c.reported);
    });
  }

  it('the footer shows the charge with no tilde', () => {
    const cost = addTurnCost(NO_COST, FIXTURE.footer.model, FIXTURE.footer.usage);
    expect(cost.estimated).toBe(false);
    expect(costWords(cost.credits, cost.estimated)).toBe(FIXTURE.footer.words);
  });

  it('the printer carries the charge', () => {
    const printer = new TurnPrinter({ live: false, color: false, out: { write: () => true } as unknown as NodeJS.WriteStream });
    printer.feed({ type: 'done', response: { content: '', content_items: [], usage: FIXTURE.footer.usage as never } });
    expect(printer.usageCostCredits).toBe(FIXTURE.footer.usage.total_cost);
  });
});

/** A model that calls one tool, then answers: two calls, each with its own charge. */
class TwoCalls extends Skill {
  private call = 0;

  constructor() {
    super({ name: 'two-calls' });
  }

  @tool({ description: 'Echo', parameters: { type: 'object', properties: {} } })
  async echo(): Promise<string> {
    return 'echoed';
  }

  @handoff({ name: 'two-calls', priority: 10 })
  async *processUAMP(_events: ClientEvent[], _context: Context): AsyncGenerator<ServerEvent, void, unknown> {
    const usage = FIXTURE.turn.calls[this.call];
    const responseId = generateEventId();
    this.call += 1;
    if (this.call === 1) {
      const call = { id: 'call_1', name: 'echo', arguments: '{}' };
      yield createResponseDeltaEvent(responseId, { type: 'tool_call', tool_call: call });
      yield createResponseDoneEvent(responseId, [{ type: 'tool_call', tool_call: call } as never], 'completed', usage as never);
      return;
    }
    yield createResponseDeltaEvent(responseId, { type: 'text', text: 'done' });
    yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'done' }], 'completed', usage as never);
  }
}

describe("a turn's charge", () => {
  it('sums every model call of the turn', async () => {
    const agent = new BaseAgent({ name: 'priced', instructions: 'x', skills: [new TwoCalls()] });
    await agent.initialize();
    let last: { total_cost?: number } | undefined;
    for await (const chunk of agent.runStreaming([{ role: 'user', content: 'hi' }])) {
      if (chunk.type === 'done') last = chunk.response?.usage as { total_cost?: number } | undefined;
    }
    expect(last?.total_cost).toBeCloseTo(FIXTURE.turn.total_cost, 10);
  });
});
