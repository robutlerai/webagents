/**
 * Tool-call ids in a replayed history (`adapters/tool-ids.ts`), and the
 * same history through every adapter's request builder.
 *
 * The history under test is the shape a store with empty ids produces: an
 * assistant turn whose two calls carry `id: ''`, followed by the two output
 * rows, also id-less. Pinned here:
 *   - the derived ids are non-empty, stable across calls, and the output
 *     rows pair with their calls by order (name first);
 *   - a call nothing answers, with a message after it, is dropped and the
 *     assistant's words kept; so is an id-less output pairing with nothing;
 *     `unpaired: 'keep'` leaves both in;
 *   - a call in the LAST message is kept, answered or not (the loop is
 *     about to append its outputs), and an output with an id of its own is
 *     replayed as it came whether or not a call declared it: both as the
 *     other SDKs' adapters do;
 *   - a history whose ids are all present and paired comes back untouched
 *     (same array, same objects);
 *   - the Responses, Chat Completions, Anthropic, Gemini and Converse
 *     request bodies carry the same non-empty id on each call and its
 *     output, and no empty `call_id` / `tool_call_id` / `tool_use_id`.
 */
import { describe, it, expect } from 'vitest';
import { normalizeToolHistory } from '../../../src/adapters/tool-ids.js';
import { openaiAdapter } from '../../../src/adapters/responses.js';
import { fireworksAdapter } from '../../../src/adapters/completions.js';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import { googleAdapter } from '../../../src/adapters/google.js';
import { buildConverseBody } from '../../../src/adapters/bedrock-converse.js';
import type { Message } from '../../../src/adapters/types.js';

const ID_RE = /^[a-zA-Z0-9_-]+$/;

/** Two id-less calls and their two id-less outputs, after a user turn. */
function idless(): Message[] {
  return [
    { role: 'system', content: 'You plan days.' },
    { role: 'user', content: 'Help me plan Friday' },
    {
      role: 'assistant',
      content: 'Reading your day.',
      tool_calls: [
        { id: '', type: 'function', function: { name: 'get_day', arguments: '{"day":"2026-10-09"}' } },
        { id: '', type: 'function', function: { name: 'get_brief', arguments: '{}' } },
      ],
    },
    { role: 'tool', content: '{"rows":[]}', tool_call_id: '', name: 'get_day' },
    { role: 'tool', content: '{"isSet":true}', tool_call_id: '', name: 'get_brief' },
    { role: 'assistant', content: 'Your Friday is empty so far.' },
    { role: 'user', content: 'Find me robotics events' },
  ];
}

describe('normalizeToolHistory', () => {
  it('derives non-empty ids and pairs the outputs by order, the same bytes on every call', () => {
    const a = normalizeToolHistory(idless());
    const b = normalizeToolHistory(idless());
    const calls = a[2].tool_calls!;
    expect(calls).toHaveLength(2);
    for (const c of calls) expect(c.id).toMatch(ID_RE);
    expect(calls[0].id).not.toBe(calls[1].id);
    expect(a[3].tool_call_id).toBe(calls[0].id);
    expect(a[4].tool_call_id).toBe(calls[1].id);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it('pairs an id-less output by name before by order', () => {
    const history = idless();
    // The outputs arrive in the opposite order to the calls.
    [history[3], history[4]] = [history[4], history[3]];
    const out = normalizeToolHistory(history);
    const [first, second] = out[2].tool_calls!;
    expect(out[3].name).toBe('get_brief');
    expect(out[3].tool_call_id).toBe(second.id);
    expect(out[4].tool_call_id).toBe(first.id);
  });

  it('gives two identical calls different ids', () => {
    const history: Message[] = [
      { role: 'user', content: 'twice' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: '', function: { name: 'tick', arguments: '{}' } },
          { id: '', function: { name: 'tick', arguments: '{}' } },
        ],
      },
      { role: 'tool', content: '1', tool_call_id: '' },
      { role: 'tool', content: '2', tool_call_id: '' },
    ];
    const out = normalizeToolHistory(history);
    const ids = out[1].tool_calls!.map((c) => c.id);
    expect(new Set(ids).size).toBe(2);
    expect(out[2].tool_call_id).toBe(ids[0]);
    expect(out[3].tool_call_id).toBe(ids[1]);
  });

  it('drops a call nothing answers when a message follows it (keeping the words), and an id-less output pairing with nothing', () => {
    const history: Message[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: 'On it.', tool_calls: [{ id: '', function: { name: 'get_day', arguments: '{}' } }] },
      { role: 'user', content: 'still there?' },
      { role: 'tool', content: 'late, no id', tool_call_id: '' },
    ];
    const out = normalizeToolHistory(history);
    expect(out.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(out[1].content).toBe('On it.');
    expect(out[1].tool_calls).toBeUndefined();
  });

  it('replays an output with an id of its own as it came, declared or not', () => {
    const history: Message[] = [
      { role: 'user', content: 'go' },
      { role: 'tool', content: 'r', tool_call_id: 'fn_1' },
    ];
    expect(normalizeToolHistory(history)).toBe(history);
  });

  it('drops an assistant turn left with neither words nor calls', () => {
    const history: Message[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: '', tool_calls: [{ id: '', function: { name: 'get_day', arguments: '{}' } }] },
      { role: 'user', content: 'and?' },
    ];
    expect(normalizeToolHistory(history).map((m) => m.role)).toEqual(['user', 'user']);
  });

  it('keeps a trailing unanswered call untouched, its id derived when missing', () => {
    const withId: Message[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: 'On it.', tool_calls: [{ id: 'call_1', function: { name: 'get_day', arguments: '{}' } }] },
    ];
    expect(normalizeToolHistory(withId)).toBe(withId);
    const idless: Message[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: '', tool_calls: [{ id: '', function: { name: 'get_day', arguments: '{}' } }] },
    ];
    const out = normalizeToolHistory(idless);
    expect(out).toHaveLength(2);
    expect(out[1].tool_calls).toHaveLength(1);
    expect(out[1].tool_calls![0].id).toMatch(ID_RE);
  });

  it('keeps the unpaired items with `unpaired: keep`, ids filled in', () => {
    const history: Message[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: '', tool_calls: [{ id: '', function: { name: 'get_day', arguments: '{}' } }] },
      { role: 'user', content: 'and?' },
      { role: 'tool', content: 'late, no id', tool_call_id: '' },
    ];
    const out = normalizeToolHistory(history, { unpaired: 'keep' });
    expect(out).toHaveLength(4);
    expect(out[1].tool_calls![0].id).toMatch(ID_RE);
    expect(out[3].tool_call_id).toBe('');
  });

  it('returns a well-formed history untouched', () => {
    const history: Message[] = [
      { role: 'user', content: 'q' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', function: { name: 'f', arguments: '{}' } }] },
      { role: 'tool', content: 'r', tool_call_id: 'call_1' },
    ];
    const out = normalizeToolHistory(history);
    expect(out).toBe(history);
    expect(normalizeToolHistory([{ role: 'user', content: 'plain' }])).toHaveLength(1);
  });

  it('replays a second output for an already answered id as it came', () => {
    const history: Message[] = [
      { role: 'user', content: 'q' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'call_1', function: { name: 'f', arguments: '{}' } }] },
      { role: 'tool', content: 'r', tool_call_id: 'call_1' },
      { role: 'tool', content: 'again', tool_call_id: 'call_1' },
    ];
    expect(normalizeToolHistory(history)).toBe(history);
  });
});

describe('an id-less history on every wire', () => {
  const params = { model: 'm', apiKey: 'test-key', messages: idless() };

  it('Responses: every function_call and function_call_output carries the same non-empty call_id', () => {
    const body = JSON.parse(openaiAdapter.buildRequest({ ...params, model: 'gpt-5.5' }).body);
    const calls = body.input.filter((i: { type: string }) => i.type === 'function_call');
    const outputs = body.input.filter((i: { type: string }) => i.type === 'function_call_output');
    expect(calls).toHaveLength(2);
    expect(outputs).toHaveLength(2);
    for (const c of calls) expect(c.call_id).toMatch(ID_RE);
    expect(outputs.map((o: { call_id: string }) => o.call_id)).toEqual(calls.map((c: { call_id: string }) => c.call_id));
    expect(JSON.stringify(body)).not.toContain('"call_id":""');
  });

  it('Chat Completions: tool messages answer the assistant tool_calls by id', () => {
    const body = JSON.parse(fireworksAdapter.buildRequest({ ...params, model: 'deepseek-v3p2' }).body);
    const assistant = body.messages.find((m: { tool_calls?: unknown[] }) => Array.isArray(m.tool_calls));
    const tools = body.messages.filter((m: { role: string }) => m.role === 'tool');
    expect(assistant.tool_calls).toHaveLength(2);
    expect(tools).toHaveLength(2);
    for (const c of assistant.tool_calls) expect(c.id).toMatch(ID_RE);
    expect(tools.map((t: { tool_call_id: string }) => t.tool_call_id)).toEqual(assistant.tool_calls.map((c: { id: string }) => c.id));
  });

  it('Anthropic: the tool_use blocks are replayed, not dropped, and their tool_result blocks pair with them', () => {
    const body = JSON.parse(anthropicAdapter.buildRequest({ ...params, model: 'anthropic/claude-haiku-5-5' }).body);
    const uses: Array<{ id: string }> = [];
    const results: Array<{ tool_use_id: string }> = [];
    for (const m of body.messages) {
      if (!Array.isArray(m.content)) continue;
      for (const b of m.content) {
        if (b.type === 'tool_use') uses.push(b);
        if (b.type === 'tool_result') results.push(b);
      }
    }
    expect(uses).toHaveLength(2);
    expect(results).toHaveLength(2);
    for (const u of uses) expect(u.id).toMatch(ID_RE);
    expect(results.map((r) => r.tool_use_id)).toEqual(uses.map((u) => u.id));
  });

  it('Gemini: two functionCall parts are followed by two functionResponse parts with the same ids', () => {
    const body = JSON.parse(googleAdapter.buildRequest({ ...params, model: 'google/gemini-3.8-flash' }).body);
    const model = body.contents.find((c: { role: string; parts: Array<{ functionCall?: unknown }> }) => c.role === 'model' && c.parts.some((p) => p.functionCall));
    const responses = body.contents.filter((c: { parts: Array<{ functionResponse?: unknown }> }) => c.parts.some((p) => p.functionResponse));
    expect(model.parts.filter((p: { functionCall?: unknown }) => p.functionCall)).toHaveLength(2);
    expect(responses).toHaveLength(2);
    for (const r of responses) expect(r.parts[0].functionResponse.id).toMatch(ID_RE);
  });

  it('Converse: toolUse and toolResult share a toolUseId instead of refusing the request', () => {
    const body = buildConverseBody({ ...params, model: 'us.amazon.nova-pro-v1:0', tools: [{ type: 'function', function: { name: 'get_day', parameters: {} } }] }, { toolCalling: true });
    const messages = body.messages as Array<{ content: Array<{ toolUse?: { toolUseId: string }; toolResult?: { toolUseId: string } }> }>;
    const uses = messages.flatMap((m) => m.content.filter((b) => b.toolUse).map((b) => b.toolUse!.toolUseId));
    const results = messages.flatMap((m) => m.content.filter((b) => b.toolResult).map((b) => b.toolResult!.toolUseId));
    expect(uses).toHaveLength(2);
    expect(results.sort()).toEqual([...uses].sort());
  });
});
