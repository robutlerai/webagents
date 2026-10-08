/**
 * Anthropic adapter: the capability table (Haiku 5 and later), system
 * blocks, explicit prompt-cache breakpoints and cumulative stream usage.
 *
 * The model's version, not a literal id, decides what goes on the wire:
 * Haiku 5.5 thinks by default and refuses a temperature, Haiku 4.5 does
 * neither. The request renders leading system messages as separate blocks
 * (a breakpoint on the first survives changes to the rest), keeps a
 * mid-conversation system message in place, and places at most four
 * `cache_control` markers, never on an empty block and never at the body
 * root. The stream parser takes the cumulative usage on `message_delta`
 * as the final word, so input a server tool added after `message_start`
 * is billed.
 */

import { describe, it, expect } from 'vitest';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import type { AdapterRequestParams, AdapterChunk, Message, ToolDefinition } from '../../../src/adapters/types.js';

function mockSSEResponse(chunks: unknown[]): Response {
  const lines = chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('');
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

async function collectChunks(gen: AsyncGenerator<AdapterChunk>): Promise<AdapterChunk[]> {
  const result: AdapterChunk[] = [];
  for await (const chunk of gen) result.push(chunk);
  return result;
}

function makeParams(overrides: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return {
    messages: [{ role: 'user', content: 'Hello' }],
    model: 'anthropic/claude-haiku-5-5',
    apiKey: 'test-key',
    ...overrides,
  };
}

const fn = (name: string): ToolDefinition => ({
  type: 'function',
  function: { name, description: `${name} tool`, parameters: { type: 'object', properties: {} } },
});

type Block = Record<string, unknown> & { cache_control?: { type: string } };
type WireMessage = { role: string; content: string | Block[] };

/** Every `cache_control` marker in the body, in render order (tools, system, messages). */
function markers(body: Record<string, unknown>): string[] {
  const out: string[] = [];
  for (const [i, t] of ((body.tools as Block[] | undefined) ?? []).entries()) if (t.cache_control) out.push(`tools[${i}]`);
  for (const [i, s] of ((body.system as Block[] | undefined) ?? []).entries()) if (s.cache_control) out.push(`system[${i}]`);
  for (const [i, m] of ((body.messages as WireMessage[]) ?? []).entries()) {
    if (Array.isArray(m.content)) {
      for (const [j, b] of m.content.entries()) if (b.cache_control) out.push(`messages[${i}].content[${j}]`);
    }
  }
  return out;
}

function build(overrides: Partial<AdapterRequestParams> = {}): Record<string, unknown> {
  return JSON.parse(anthropicAdapter.buildRequest(makeParams(overrides)).body);
}

describe('Haiku 5.5: thinking levels and sampling', () => {
  const cases: Array<{ level: 'off' | 'low' | 'medium' | 'high' | undefined; thinking: unknown; effort?: string }> = [
    { level: undefined, thinking: { type: 'adaptive' }, effort: 'medium' },
    { level: 'off', thinking: { type: 'disabled' }, effort: 'low' },
    { level: 'low', thinking: { type: 'adaptive' }, effort: 'low' },
    { level: 'medium', thinking: { type: 'adaptive' }, effort: 'medium' },
    { level: 'high', thinking: { type: 'adaptive' }, effort: 'high' },
  ];
  for (const { level, thinking, effort } of cases) {
    it(`claude-haiku-5-5: thinking=${level ?? 'undefined'} -> ${JSON.stringify(thinking)} at effort ${effort}`, () => {
      const body = build({ thinking: level, temperature: 0.7 });
      expect(body.thinking).toEqual(thinking);
      expect(body.output_config).toEqual({ effort });
      // Any temperature other than the default is a 400 on this model.
      expect(body.temperature).toBeUndefined();
    });
  }

  it('off keeps the non-thinking output ceiling; a thinking level gets the thinking one', () => {
    expect(build({ thinking: 'off' }).max_tokens).toBe(4096);
    expect(build({ thinking: 'low' }).max_tokens).toBe(16_000);
    expect(build().max_tokens).toBe(16_000);
  });

  it('the bare id and the provider-prefixed id resolve the same way', () => {
    expect(build({ model: 'claude-haiku-5-5', thinking: 'off' }).thinking).toEqual({ type: 'disabled' });
  });

  it('Haiku 4.5 stays a non-thinking model that takes a temperature', () => {
    const body = build({ model: 'anthropic/claude-haiku-4-5', thinking: 'off', temperature: 0.7 });
    expect(body.thinking).toBeUndefined();
    expect(body.output_config).toBeUndefined();
    expect(body.temperature).toBe(0.7);
    expect(body.max_tokens).toBe(4096);
  });

  it('never sends a temperature to a model that rejects sampling, thinking on or off', () => {
    for (const model of ['anthropic/claude-opus-4-7', 'anthropic/claude-opus-4-8', 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-5-5', 'anthropic/claude-opus-5-5', 'anthropic/claude-fable-5-1']) {
      for (const level of ['off', 'low', undefined] as const) {
        expect(build({ model, thinking: level, temperature: 0.7 }).temperature, `${model} ${level}`).toBeUndefined();
      }
    }
    // Models that still take one keep it with thinking off.
    expect(build({ model: 'anthropic/claude-sonnet-4-6', thinking: 'off', temperature: 0.7 }).temperature).toBe(0.7);
    expect(build({ model: 'anthropic/claude-opus-4-6', thinking: 'off', temperature: 0.7 }).temperature).toBe(0.7);
  });

  it('sends disabled only where the model thinks by default AND accepts it', () => {
    // Opus 5 and Sonnet 5: think by default, accept disabled.
    expect(build({ model: 'anthropic/claude-sonnet-5', thinking: 'off' }).thinking).toEqual({ type: 'disabled' });
    expect(build({ model: 'anthropic/claude-opus-5', thinking: 'off' }).thinking).toEqual({ type: 'disabled' });
    // The 5.5 pair and Fable reject disabled: off leaves their default in place.
    for (const model of ['anthropic/claude-sonnet-5-5', 'anthropic/claude-opus-5-5', 'anthropic/claude-fable-5-1']) {
      const body = build({ model, thinking: 'off' });
      expect(body.thinking, model).toBeUndefined();
      expect(body.output_config, model).toBeUndefined();
    }
    // Opus 4.7 is off unless asked: omitting the field is already off.
    expect(build({ model: 'anthropic/claude-opus-4-7', thinking: 'off' }).thinking).toBeUndefined();
  });
});

describe('system blocks', () => {
  it('renders each leading system message as its own block, in order, dropping empty ones', () => {
    const body = build({
      messages: [
        { role: 'system', content: 'Agent instructions.' },
        { role: 'system', content: '' },
        { role: 'system', content: 'File directory.' },
        { role: 'user', content: 'Hi' },
      ],
    });
    expect(body.system).toEqual([
      { type: 'text', text: 'Agent instructions.' },
      { type: 'text', text: 'File directory.' },
    ]);
  });

  it('omits system entirely when there is no system text', () => {
    expect(build({ messages: [{ role: 'user', content: 'Hi' }] }).system).toBeUndefined();
    expect(build({ messages: [{ role: 'system', content: '' }, { role: 'user', content: 'Hi' }] }).system).toBeUndefined();
  });

  it('keeps a mid-conversation system message in place: a text block after the tool_result blocks of the adjacent user turn', () => {
    const body = build({
      messages: [
        { role: 'system', content: 'Instructions.' },
        { role: 'user', content: 'Do the thing.' },
        { role: 'assistant', content: '', tool_calls: [{ id: 'tu_1', type: 'function', function: { name: 'guide', arguments: '{}' } }] },
        { role: 'tool', content: 'the guide', tool_call_id: 'tu_1' },
        { role: 'system', content: 'Tool-call budget: 2 calls left.' },
      ],
    });
    // Not hoisted: the prefix ahead of the history stays byte-identical.
    expect(body.system).toEqual([{ type: 'text', text: 'Instructions.' }]);
    const messages = body.messages as WireMessage[];
    const last = messages[messages.length - 1];
    expect(last.role).toBe('user');
    expect(last.content).toEqual([
      { type: 'tool_result', tool_use_id: 'tu_1', content: 'the guide' },
      { type: 'text', text: 'Tool-call budget: 2 calls left.' },
    ]);
  });

  it('becomes a user turn of its own after an assistant message, and joins a string user turn as blocks', () => {
    const afterAssistant = build({
      messages: [
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello.' },
        { role: 'system', content: 'Wrap up.' },
      ],
    });
    const m1 = afterAssistant.messages as WireMessage[];
    expect(m1[2]).toEqual({ role: 'user', content: [{ type: 'text', text: 'Wrap up.' }] });

    const afterUser = build({
      messages: [
        { role: 'user', content: 'Hi' },
        { role: 'system', content: 'Wrap up.' },
      ],
    });
    const m2 = afterUser.messages as WireMessage[];
    expect(m2).toHaveLength(1);
    expect(m2[0].content).toEqual([{ type: 'text', text: 'Hi' }, { type: 'text', text: 'Wrap up.' }]);
    expect(afterUser.system).toBeUndefined();
  });
});

describe('prompt cache breakpoints', () => {
  const agentShape = (extra: Partial<AdapterRequestParams> = {}): Partial<AdapterRequestParams> => ({
    messages: [
      { role: 'system', content: 'Agent instructions, byte-stable.', stable: true } as Message,
      { role: 'system', content: 'File directory, changes per turn.' },
      { role: 'user', content: 'Plan my event.' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'tu_1', type: 'function', function: { name: 'guide', arguments: '{}' } }] },
      { role: 'tool', content: 'the 12K guide', tool_call_id: 'tu_1' },
    ],
    tools: [fn('guide'), fn('run')],
    ...extra,
  });

  it('places nothing without promptCache', () => {
    expect(markers(build(agentShape()))).toEqual([]);
    expect(markers(build(agentShape({ promptCache: false })))).toEqual([]);
  });

  it('a typical agent step: the last tool, the stable system block, the last block of the final message', () => {
    const body = build(agentShape({ promptCache: true }));
    expect(markers(body)).toEqual(['tools[1]', 'system[0]', 'messages[2].content[0]']);
    expect(body.cache_control).toBeUndefined();
    const marker = (body.tools as Block[])[1].cache_control;
    expect(marker).toEqual({ type: 'ephemeral' });
    // The volatile second system block is left unmarked.
    expect((body.system as Block[])[1].cache_control).toBeUndefined();
  });

  it('BP2 follows the LAST leading system message marked stable; the first block without any marker', () => {
    const marked = build({
      promptCache: true,
      messages: [
        { role: 'system', content: 'A' },
        { role: 'system', content: 'B', stable: true } as Message,
        { role: 'system', content: 'C' },
        { role: 'user', content: 'Hi' },
      ],
    });
    expect(markers(marked)).toEqual(['system[1]']);
    const unmarked = build({
      promptCache: true,
      messages: [
        { role: 'system', content: 'A' },
        { role: 'system', content: 'B' },
        { role: 'user', content: 'Hi' },
      ],
    });
    expect(markers(unmarked)).toEqual(['system[0]']);
  });

  it('a one-shot request (no tools, no history) marks the system block only: no write premium on a tail nothing reads', () => {
    const body = build({
      promptCache: true,
      messages: [{ role: 'system', content: 'Instructions.' }, { role: 'user', content: 'One question.' }],
    });
    expect(markers(body)).toEqual(['system[0]']);
    expect((body.messages as WireMessage[])[0].content).toBe('One question.');
  });

  it('a single user turn WITH tools gets the tail breakpoint, converting string content to a text block', () => {
    const body = build({
      promptCache: true,
      tools: [fn('run')],
      messages: [{ role: 'user', content: 'One question.' }],
    });
    expect(markers(body)).toEqual(['tools[0]', 'messages[0].content[0]']);
    expect((body.messages as WireMessage[])[0].content).toEqual([{ type: 'text', text: 'One question.', cache_control: { type: 'ephemeral' } }]);
  });

  it('never marks an empty block', () => {
    const body = build({
      promptCache: true,
      tools: [fn('run')],
      messages: [
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: '', tool_calls: [{ id: 'tu_1', type: 'function', function: { name: 'run', arguments: '{}' } }] },
        { role: 'tool', content: '', tool_call_id: 'tu_1' },
      ],
    });
    // The final message's only block is an empty tool_result: no BP3.
    expect(markers(body)).toEqual(['tools[0]']);
  });

  it('skips BP1 when the last tool is a server tool, and BP2 still covers the tool list', () => {
    const body = build({
      promptCache: true,
      messages: [{ role: 'system', content: 'Instructions.' }, { role: 'user', content: 'Hi' }],
      tools: [fn('run'), { type: 'web_search_20250305', name: 'web_search' }],
    });
    expect(markers(body)).toEqual(['system[0]', 'messages[0].content[0]']);
  });

  it('marks an Anthropic-defined client tool when it is last', () => {
    const body = build({
      promptCache: true,
      messages: [{ role: 'user', content: 'Hi' }],
      tools: [fn('run'), { type: 'native', name: 'bash' }],
    });
    expect(markers(body)).toEqual(['tools[1]', 'messages[0].content[0]']);
  });

  it('anchors the last human user message when more than 15 positions follow it, never exceeding four markers', () => {
    const history: Message[] = [
      { role: 'system', content: 'Instructions.' },
      { role: 'user', content: 'Plan my event.' },
    ];
    for (let i = 0; i < 8; i++) {
      history.push({ role: 'assistant', content: '', tool_calls: [{ id: `tu_${i}`, type: 'function', function: { name: 'run', arguments: '{}' } }] });
      history.push({ role: 'tool', content: `result ${i}`, tool_call_id: `tu_${i}` });
    }
    const body = build({ promptCache: true, tools: [fn('run')], messages: history });
    const found = markers(body);
    expect(found).toHaveLength(4);
    expect(found).toEqual(['tools[0]', 'system[0]', 'messages[0].content[0]', 'messages[16].content[0]']);
    expect((body.messages as WireMessage[])[0].content).toEqual([{ type: 'text', text: 'Plan my event.', cache_control: { type: 'ephemeral' } }]);
  });

  it('places no anchor while the human message is within the lookback', () => {
    const history: Message[] = [
      { role: 'system', content: 'Instructions.' },
      { role: 'user', content: 'Plan my event.' },
    ];
    for (let i = 0; i < 7; i++) {
      history.push({ role: 'assistant', content: '', tool_calls: [{ id: `tu_${i}`, type: 'function', function: { name: 'run', arguments: '{}' } }] });
      history.push({ role: 'tool', content: `result ${i}`, tool_call_id: `tu_${i}` });
    }
    const body = build({ promptCache: true, tools: [fn('run')], messages: history });
    expect(markers(body)).toEqual(['tools[0]', 'system[0]', 'messages[14].content[0]']);
    expect((body.messages as WireMessage[])[0].content).toBe('Plan my event.');
  });

  it('uses the 5-minute TTL only', () => {
    const body = build(agentShape({ promptCache: true }));
    const all = JSON.stringify(body).match(/"cache_control":\{[^}]*\}/g) ?? [];
    expect(all.length).toBe(3);
    for (const m of all) expect(m).toBe('"cache_control":{"type":"ephemeral"}');
  });

  it('leaves the `stable` marker off the wire', () => {
    const body = build(agentShape({ promptCache: true }));
    expect(JSON.stringify(body)).not.toContain('"stable"');
  });
});

describe('parseStream: cumulative usage on message_delta', () => {
  it('takes the final input count from message_delta over message_start (a server tool added input mid-message)', async () => {
    const response = mockSSEResponse([
      { type: 'message_start', message: { usage: { input_tokens: 2679, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, output_tokens: 3 } } },
      { type: 'content_block_start', content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Found it.' } },
      { type: 'content_block_stop' },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 10682, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, output_tokens: 510, server_tool_use: { web_search_requests: 1 } } },
    ]);
    const usage = (await collectChunks(anthropicAdapter.parseStream(response))).find(c => c.type === 'usage') as Extract<AdapterChunk, { type: 'usage' }>;
    expect(usage.input).toBe(10682);
    expect(usage.output).toBe(510);
  });

  it('takes cache legs that appear or grow by the end of the stream', async () => {
    const response = mockSSEResponse([
      { type: 'message_start', message: { usage: { input_tokens: 100, cache_read_input_tokens: 2000, cache_creation_input_tokens: 0 } } },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { input_tokens: 120, cache_read_input_tokens: 2000, cache_creation_input_tokens: 7345, output_tokens: 9 } },
    ]);
    const usage = (await collectChunks(anthropicAdapter.parseStream(response))).find(c => c.type === 'usage') as Extract<AdapterChunk, { type: 'usage' }>;
    expect(usage).toMatchObject({ input: 120, output: 9, cache_read_input: 2000, cache_creation_input: 7345 });
  });

  it('keeps the message_start counts when the delta carries output only', async () => {
    const response = mockSSEResponse([
      { type: 'message_start', message: { usage: { input_tokens: 12, cache_read_input_tokens: 2140, cache_creation_input_tokens: 30 } } },
      { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 57 } },
    ]);
    const usage = (await collectChunks(anthropicAdapter.parseStream(response))).find(c => c.type === 'usage') as Extract<AdapterChunk, { type: 'usage' }>;
    expect(usage).toMatchObject({ input: 12, output: 57, cache_read_input: 2140, cache_creation_input: 30 });
  });
});
