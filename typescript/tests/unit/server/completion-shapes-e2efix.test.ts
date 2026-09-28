/**
 * The shape of a served agent's chat completion (2026-09-26, the new-developer
 * e2e run): this server's streamed chunks carried `choices` alone, so an
 * OpenAI client could not tell which completion or model it was reading.
 * Pinned by `python/tests/fixtures/completions/response_shapes.json`, which
 * the Python suite runs too (`tests/server/test_completion_shapes_e2efix.py`):
 * every chunk carries `id`, `object`, `created` and `model`; the non-streamed
 * answer has a timestamp and sums every model call of a tool turn.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { BaseAgent } from '../../../src/core/agent';
import { Skill } from '../../../src/core/skill';
import { handoff, tool } from '../../../src/core/decorators';
import type { Context } from '../../../src/core/types';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events';
import { createResponseDeltaEvent, createResponseDoneEvent } from '../../../src/uamp/events';
import { createFetchHandler } from '../../../src/server/handler';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SHAPES = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/completions/response_shapes.json'), 'utf8')) as {
  completion: { object: string; top_level_keys: string[] };
  model_calls: { prompt_tokens: number; completion_tokens: number; total_tokens: number }[];
  turn_usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  stream: { object: string; chunk_keys: string[]; done: string };
};
const usageOf = (call: { prompt_tokens: number; completion_tokens: number; total_tokens: number }) => ({
  input_tokens: call.prompt_tokens,
  output_tokens: call.completion_tokens,
  total_tokens: call.total_tokens,
});

/** A model that calls `ping` on the first call of a turn and answers text on the second, each with the fixture's usage. */
class ScriptedLLM extends Skill {
  private calls = 0;

  @handoff({ name: 'scripted' })
  async *processUAMP(_events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
    // The first call of the turn asks for the tool; the second answers (the
    // loop hands the conversation over on the context, as `agentic-loop.test.ts` reads it).
    this.calls += 1;
    const second = this.calls > 1;
    if (!second) {
      yield createResponseDoneEvent('r1', [{ type: 'tool_call', tool_call: { id: 'call_1', name: 'ping', arguments: '{}' } }], 'completed', usageOf(SHAPES.model_calls[0]));
      return;
    }
    yield createResponseDeltaEvent('r2', { type: 'text', text: 'Tool said: ' });
    yield createResponseDeltaEvent('r2', { type: 'text', text: 'pong' });
    yield createResponseDoneEvent('r2', [{ type: 'text', text: 'Tool said: pong' }], 'completed', usageOf(SHAPES.model_calls[1]));
  }
}

class Tools extends Skill {
  @tool({ name: 'ping', description: 'Answers pong.' })
  async ping(_params: Record<string, never>, _ctx: Context): Promise<string> {
    return 'pong';
  }
}

async function served(): Promise<(request: Request) => Promise<Response>> {
  const agent = new BaseAgent({ name: 'shaped', instructions: 'x', skills: [new ScriptedLLM(), new Tools()] });
  await agent.initialize();
  return createFetchHandler(agent, { basePath: '' });
}

function post(body: Record<string, unknown>): Request {
  return new Request('http://127.0.0.1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer test' },
    body: JSON.stringify(body),
  });
}

describe('the completion shape', () => {
  it('a tool turn sums every model call, and created is a timestamp', async () => {
    const handler = await served();
    const response = await handler(post({ messages: [{ role: 'user', content: 'TOOL ping' }] }));
    expect(response.status).toBe(200);
    const body = (await response.json()) as Record<string, unknown>;
    expect(Object.keys(body)).toEqual(SHAPES.completion.top_level_keys);
    expect(body.object).toBe(SHAPES.completion.object);
    expect(typeof body.created === 'number' && body.created > 0).toBe(true);
    expect(body.usage).toEqual(SHAPES.turn_usage);
  });

  it('every streamed chunk carries the id, object, created and model', async () => {
    const handler = await served();
    const response = await handler(post({ messages: [{ role: 'user', content: 'TOOL ping' }], stream: true }));
    expect(response.status).toBe(200);
    const text = await response.text();
    const payloads = text.split('\n\n').map((l) => l.trim()).filter((l) => l.startsWith('data: ')).map((l) => l.slice('data: '.length));
    expect(payloads.at(-1)).toBe(SHAPES.stream.done);
    const chunks = payloads.slice(0, -1).map((p) => JSON.parse(p) as Record<string, unknown>);
    expect(chunks.length).toBeGreaterThan(1);
    const ids = new Set<string>();
    for (const chunk of chunks) {
      expect(Object.keys(chunk)).toEqual(SHAPES.stream.chunk_keys);
      expect(chunk.object).toBe(SHAPES.stream.object);
      expect(typeof chunk.created === 'number' && chunk.created > 0).toBe(true);
      expect(typeof chunk.model).toBe('string');
      ids.add(chunk.id as string);
    }
    // One completion, one id across its chunks.
    expect(ids.size).toBe(1);
    const content = chunks.map((c) => ((c.choices as { delta: { content?: string } }[])[0].delta.content ?? '')).join('');
    expect(content).toBe('Tool said: pong');
  });
});
