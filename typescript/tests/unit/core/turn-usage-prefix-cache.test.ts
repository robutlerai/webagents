/**
 * A turn's usage sums every model call's cache legs, not only its tokens.
 *
 * The turn's `response.done` usage is the sum over its calls (one call's
 * figures used to be reported for a turn that used a tool). The cache
 * fields a platform reports beside `input_tokens` and `output_tokens`
 * (`cached_tokens`, and `cache_read_tokens`, `cache_write_tokens`,
 * `context_tokens` where a platform adds them) were not summed, so a fee
 * priced on them saw only the last call's. Pinned: two calls, each with
 * its own legs, add up; a call that reports no legs adds nothing; a turn
 * of one call reports that call's usage as it came.
 */

import { describe, it, expect } from 'vitest';
import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, tool } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDoneEvent, generateEventId } from '../../../src/uamp/events';
import type { ContentItem } from '../../../src/uamp/types';

class Adder extends Skill {
  @tool({ description: 'Add two numbers' })
  async add(params: { a: number; b: number }, _c: Context): Promise<number> {
    return params.a + params.b;
  }
}

/** Asks for `add` on the first call, answers on the second; each call reports its own usage. */
class TwoCalls extends Skill {
  calls = 0;
  constructor(private readonly usages: Array<Record<string, number>>) {
    super({ name: 'two-calls' });
  }

  @handoff({ name: 'two-calls' })
  async *processUAMP(_events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
    this.calls += 1;
    const responseId = generateEventId();
    const usage = this.usages[this.calls - 1];
    const output: ContentItem[] = this.calls === 1 && this.usages.length > 1
      ? [{ type: 'tool_call', tool_call: { id: 'c1', name: 'add', arguments: '{"a":1,"b":2}' } }]
      : [{ type: 'text', text: 'three' }];
    const done = createResponseDoneEvent(responseId, output) as unknown as { response: Record<string, unknown> };
    if (usage) done.response.usage = usage;
    yield done as unknown as ServerEvent;
  }
}

async function doneUsage(agent: BaseAgent): Promise<Record<string, unknown> | undefined> {
  const events: ServerEvent[] = [];
  for await (const e of agent.processUAMP([
    { type: 'session.create', event_id: generateEventId(), uamp_version: '1.0', session: { modalities: ['text'] } } as ClientEvent,
    { type: 'input.text', event_id: generateEventId(), text: 'add them' } as ClientEvent,
    { type: 'response.create', event_id: generateEventId() } as ClientEvent,
  ])) events.push(e);
  const done = events.find((e) => e.type === 'response.done') as unknown as { response: { usage?: Record<string, unknown> } } | undefined;
  return done?.response.usage;
}

describe('turn usage and the cache legs', () => {
  it('sums every leg over the calls of a turn', async () => {
    const model = new TwoCalls([
      { input_tokens: 1000, output_tokens: 20, total_tokens: 1020, cached_tokens: 900, cache_read_tokens: 0, cache_write_tokens: 900, context_tokens: 1900 },
      { input_tokens: 100, output_tokens: 30, total_tokens: 130, cached_tokens: 950, cache_read_tokens: 950, context_tokens: 1050 },
    ]);
    const usage = await doneUsage(new BaseAgent({ skills: [model, new Adder()] }));
    expect(model.calls).toBe(2);
    expect(usage).toMatchObject({
      input_tokens: 1100,
      output_tokens: 50,
      total_tokens: 1150,
      cached_tokens: 1850,
      cache_read_tokens: 950,
      cache_write_tokens: 900,
      context_tokens: 2950,
    });
  });

  it('a call with no legs adds nothing, and a turn of one call reports it as it came', async () => {
    const model = new TwoCalls([
      { input_tokens: 1000, output_tokens: 20, total_tokens: 1020, cache_read_tokens: 800 },
      { input_tokens: 100, output_tokens: 30, total_tokens: 130 },
    ]);
    const usage = await doneUsage(new BaseAgent({ skills: [model, new Adder()] }));
    expect(usage).toMatchObject({ input_tokens: 1100, output_tokens: 50, cache_read_tokens: 800 });
    expect(usage).not.toHaveProperty('cache_write_tokens');

    const single = new TwoCalls([{ input_tokens: 500, output_tokens: 10, total_tokens: 510, cached_tokens: 400, cache_read_tokens: 400 }]);
    const one = await doneUsage(new BaseAgent({ skills: [single, new Adder()] }));
    expect(one).toMatchObject({ input_tokens: 500, cached_tokens: 400, cache_read_tokens: 400 });
  });
});
