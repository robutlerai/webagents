/**
 * Bedrock Converse as an SDK adapter (src/adapters/bedrock-converse.ts).
 *
 * Pinned: every row of the request mapping, every build-time refusal (each
 * one is an AWS 400 otherwise), the tool-call replay shape a tool-loop host
 * sends (arguments '' for a no-parameter call, a compacted
 * `<elided: N bytes>` value), the usage arithmetic (Converse excludes cached
 * tokens from `inputTokens`; Chat Completions includes them: 9 + 2,140 =
 * 2,149), the finish mapping the Chat Completions parser needs to release
 * tool calls, and the non-stream JSON.
 */
import { describe, it, expect } from 'vitest';
import { createHash } from 'node:crypto';
import {
  buildConverseBody,
  buildBedrockConverseRequest,
  converseToolUseId,
  converseHistoryToolName,
  sha256Hex,
  converseFinishReason,
  converseUsageToChat,
  converseStreamToChatCompletionsSSE,
  converseJsonToChatCompletion,
  parseBedrockConverseStream,
  createBedrockConverseAdapter,
  createInlineThinkingSplitter,
  type BedrockConverseOptions,
} from '../../../src/adapters/bedrock-converse.js';
import { bedrockEventMessage, bedrockExceptionMessage, BedrockRequestUnsupported, BedrockStreamError } from '../../../src/adapters/bedrock-eventstream.js';
import type { AdapterChunk, AdapterRequestParams, Message, ToolDefinition } from '../../../src/adapters/types.js';

const TOOLS: BedrockConverseOptions = { toolCalling: true };

function params(messages: Message[], extra: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return { model: 'bedrock/nova-pro', messages, apiKey: 'bedrock-key', ...extra };
}

const fn = (name: string, parameters: unknown = { type: 'object', properties: {} }, description = `${name} tool`): ToolDefinition =>
  ({ type: 'function', function: { name, description, parameters } });

function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream({ pull(c) { if (i < chunks.length) c.enqueue(chunks[i++]); else c.close(); } });
}
const eventStream = (frames: Uint8Array[]) =>
  new Response(streamOf(frames), { status: 200, headers: { 'content-type': 'application/vnd.amazon.eventstream' } });

async function collect(gen: AsyncGenerator<AdapterChunk>): Promise<AdapterChunk[]> {
  const out: AdapterChunk[] = [];
  for await (const c of gen) out.push(c);
  return out;
}

function refusal(f: () => unknown): string {
  try {
    f();
  } catch (err) {
    expect(err).toBeInstanceOf(BedrockRequestUnsupported);
    return (err as BedrockRequestUnsupported).reason;
  }
  throw new Error('expected a refusal');
}

describe('buildConverseBody: the mapping', () => {
  it('system messages become system blocks in order; blank ones and blank user text are skipped', () => {
    const body = buildConverseBody(params([
      { role: 'system', content: 'Be terse.' },
      { role: 'system', content: '   ' },
      { role: 'system', content: [{ type: 'text', text: 'Answer in French.' }] },
      { role: 'user', content: 'Hi' },
      { role: 'user', content: '  ' },
    ]));
    expect(body.system).toEqual([{ text: 'Be terse.' }, { text: 'Answer in French.' }]);
    expect(body.messages).toEqual([{ role: 'user', content: [{ text: 'Hi' }] }]);
    expect(body.toolConfig).toBeUndefined();
    expect(body.inferenceConfig).toBeUndefined();
  });

  it('UAMP items: text kept, media described as text, a file with resolved text inlined, the message text first', () => {
    const resolvedMedia = new Map([['/api/content/11111111-1111-1111-1111-111111111111', { kind: 'text' as const, mimeType: 'text/markdown', text: '# Notes' }]]);
    const body = buildConverseBody(params([{
      role: 'user',
      content: 'Look at these',
      content_items: [
        { type: 'image', image: '/api/content/22222222-2222-2222-2222-222222222222', content_id: '22222222-2222-2222-2222-222222222222' },
        { type: 'file', file: '/api/content/11111111-1111-1111-1111-111111111111', filename: 'notes.md' },
      ],
    }], { resolvedMedia }));
    const [turn] = body.messages as Array<{ content: Array<{ text: string }> }>;
    expect(turn.content[0]).toEqual({ text: 'Look at these' });
    expect(turn.content[1].text).toMatch(/^\[Available image: content_id=22222222/);
    expect(turn.content[1].text).toContain('NOT analysable by current model');
    expect(turn.content[2]).toEqual({ text: '<file name="notes.md" mime="text/markdown">\n# Notes\n</file>' });
  });

  it('assistant text and tool calls become text and toolUse blocks; tool results become toolResult blocks in a user turn', () => {
    const body = buildConverseBody(params([
      { role: 'user', content: 'What time is it?' },
      { role: 'assistant', content: 'Checking.', tool_calls: [{ id: 'tooluse_1', type: 'function', function: { name: 'get_time', arguments: '{"tz":"UTC"}' } }] },
      { role: 'tool', tool_call_id: 'tooluse_1', name: 'get_time', content: '12:00' },
    ], { tools: [fn('get_time')] }), TOOLS);
    expect(body.messages).toEqual([
      { role: 'user', content: [{ text: 'What time is it?' }] },
      { role: 'assistant', content: [{ text: 'Checking.' }, { toolUse: { toolUseId: 'tooluse_1', name: 'get_time', input: { tz: 'UTC' } } }] },
      { role: 'user', content: [{ toolResult: { toolUseId: 'tooluse_1', content: [{ text: '12:00' }] } }] },
    ]);
  });

  it('merges same-role neighbours and puts tool results first in a user turn', () => {
    const body = buildConverseBody(params([
      { role: 'user', content: 'a' },
      { role: 'user', content: 'b' },
      { role: 'assistant', tool_calls: [
        { id: 'c1', type: 'function', function: { name: 'one', arguments: '{}' } },
        { id: 'c2', type: 'function', function: { name: 'two', arguments: '{}' } },
      ], content: null },
      { role: 'tool', tool_call_id: 'c1', content: 'r1' },
      { role: 'user', content: 'and also', _inline_for_llm: true },
      { role: 'tool', tool_call_id: 'c2', content: '' },
    ], { tools: [fn('one'), fn('two')] }), TOOLS);
    const messages = body.messages as Array<{ role: string; content: Array<Record<string, unknown>> }>;
    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(messages[0].content).toEqual([{ text: 'a' }, { text: 'b' }]);
    expect(messages[2].content).toEqual([
      { toolResult: { toolUseId: 'c1', content: [{ text: 'r1' }] } },
      { toolResult: { toolUseId: 'c2', content: [{ text: '(no output)' }] } },
      { text: 'and also' },
    ]);
  });

  it('tools become toolConfig: function specs, top-level oneOf/anyOf/allOf dropped, no toolChoice, no blank description', () => {
    const body = buildConverseBody(params([{ role: 'user', content: 'go' }], {
      tools: [
        fn('search', { type: 'object', properties: { q: { type: 'string' } }, required: ['q'], anyOf: [{ required: ['q'] }] }),
        fn('noop', undefined, '  '),
      ],
    }), TOOLS);
    expect(body.toolConfig).toEqual({
      tools: [
        { toolSpec: { name: 'search', description: 'search tool', inputSchema: { json: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] } } } },
        { toolSpec: { name: 'noop', inputSchema: { json: { type: 'object', properties: {} } } } },
      ],
    });
  });

  it('maxTokens and temperature ride inferenceConfig; over the cap is clamped by default', () => {
    expect(buildConverseBody(params([{ role: 'user', content: 'x' }], { maxTokens: 800, temperature: 0.3 })).inferenceConfig)
      .toEqual({ maxTokens: 800, temperature: 0.3 });
    expect(buildConverseBody(params([{ role: 'user', content: 'x' }], { maxTokens: 32_768 }), { maxOutputTokens: 4096 }).inferenceConfig)
      .toEqual({ maxTokens: 4096 });
  });

  it('a thinking level is honoured only as the options allow: nothing is sent either way', () => {
    expect(buildConverseBody(params([{ role: 'user', content: 'x' }], { thinking: 'off' }))).not.toHaveProperty('additionalModelRequestFields');
    expect(buildConverseBody(params([{ role: 'user', content: 'x' }], { thinking: 'high' }), { reasoning: 'always' }).messages).toHaveLength(1);
  });
});

describe('the request', () => {
  it('posts to converse-stream (or converse) under the encoded model id, with the bearer key', () => {
    const p = params([{ role: 'user', content: 'hi' }]);
    const streamed = buildBedrockConverseRequest(p, { region: 'us-west-2', modelId: 'us.meta.llama3-3-70b-instruct-v1:0' });
    expect(streamed.url).toBe('https://bedrock-runtime.us-west-2.amazonaws.com/model/us.meta.llama3-3-70b-instruct-v1%3A0/converse-stream');
    expect(streamed.headers).toEqual({ 'content-type': 'application/json', accept: 'application/vnd.amazon.eventstream', authorization: 'Bearer bedrock-key' });
    const once = buildBedrockConverseRequest({ ...p, stream: false }, { baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com/', modelId: 'amazon.nova-pro-v1:0' });
    expect(once.url).toBe('https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-pro-v1%3A0/converse');
    expect(once.headers.accept).toBe('application/json');
    expect(JSON.parse(once.body)).toEqual({ messages: [{ role: 'user', content: [{ text: 'hi' }] }] });
  });

  it('defaults the model id to the last segment of the requested model', () => {
    expect(createBedrockConverseAdapter().buildRequest(params([{ role: 'user', content: 'hi' }])).url)
      .toBe('https://bedrock-runtime.us-east-1.amazonaws.com/model/nova-pro/converse-stream');
  });
});

describe('build-time refusals', () => {
  const user: Message = { role: 'user', content: 'x' };
  it('refuses each request Converse would reject', () => {
    expect(refusal(() => buildConverseBody(params([user], { tools: [{ type: 'web_search' } as ToolDefinition] }), TOOLS))).toBe('tool web_search');
    expect(refusal(() => buildConverseBody(params([user], { tools: [fn('get_time')] })))).toMatch(/without tool calling/);
    expect(refusal(() => buildConverseBody(params([user], { tools: [fn('a'.repeat(65))] }), TOOLS))).toMatch(/tool name/);
    expect(refusal(() => buildConverseBody(params([user], { tools: [fn('has.dot')] }), TOOLS))).toMatch(/tool name/);
    expect(refusal(() => buildConverseBody(params([]), TOOLS))).toBe('no messages');
    expect(refusal(() => buildConverseBody(params([{ role: 'system', content: 's' }, { role: 'assistant', content: 'hello' }, user])))).toMatch(/first message/);
    expect(refusal(() => buildConverseBody(params([user, { role: 'function', content: 'x' }])))).toMatch(/message role function/);
    expect(refusal(() => buildConverseBody(params([user], { thinking: 'off' }), { reasoning: 'always' }))).toMatch(/always reasons/);
    expect(refusal(() => buildConverseBody(params([user], { thinking: 'high' })))).toBe('thinking high');
    expect(refusal(() => buildConverseBody(params([user], { thinking: 'low' }), { reasoning: 'none' }))).toBe('thinking low');
    expect(refusal(() => buildConverseBody(params([user], { maxTokens: 9000 }), { maxOutputTokens: 8192, overCap: 'refuse' }))).toMatch(/max_tokens 9000/);
  });

  const call = (id: string, name = 'get_time') => ({ id, type: 'function', function: { name, arguments: '{}' } });
  const REFUSE: BedrockConverseOptions = { toolHistory: 'refuse' };
  it('refuses tool history the request cannot carry, and unpaired calls or results, under toolHistory: refuse', () => {
    const history: Message[] = [user, { role: 'assistant', tool_calls: [call('c1')], content: null }, { role: 'tool', tool_call_id: 'c1', content: 'r' }];
    expect(refusal(() => buildConverseBody(params(history), REFUSE))).toMatch(/tool history without tools/);
    expect(refusal(() => buildConverseBody(params([user, { role: 'assistant', tool_calls: [call('c1'), call('c2')], content: null }, { role: 'tool', tool_call_id: 'c1', content: 'r' }], { tools: [fn('get_time')] }), { ...TOOLS, ...REFUSE })))
      .toBe('a tool call without its result');
    expect(refusal(() => buildConverseBody(params([user, { role: 'assistant', tool_calls: [call('c1')], content: null }], { tools: [fn('get_time')] }), { ...TOOLS, ...REFUSE })))
      .toBe('a tool call without its result');
    expect(refusal(() => buildConverseBody(params([user, { role: 'assistant', content: 'hm' }, { role: 'tool', tool_call_id: 'zz', content: 'r' }], { tools: [fn('get_time')] }), { ...TOOLS, ...REFUSE })))
      .toBe('a tool result without its call');
  });

  // A tool loop itself produces both shapes. A round that calls a
  // server-side tool and a client tool together may be replayed with only the
  // server-side call answered, and an agent's wrap-up call sends its full
  // history with no tools. Under `adapt` (the default) the wrap-up is served
  // and the mixed round goes to AWS, which answers it with the same 400 other
  // providers return.
  it('writes tool history as text when the request has no tools, under toolHistory: adapt (the default)', () => {
    const body = buildConverseBody(params([
      user,
      { role: 'assistant', content: 'Checking.', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'functions.get time', arguments: '{"tz":"UTC"}' } }] },
      { role: 'tool', tool_call_id: 'c1', name: 'functions.get time', content: '12:00' },
      { role: 'assistant', tool_calls: [call('c2')], content: null },
      { role: 'user', content: 'Now wrap up.' },
    ]), TOOLS);
    expect(body.toolConfig).toBeUndefined();
    expect(body.messages).toEqual([
      { role: 'user', content: [{ text: 'x' }] },
      { role: 'assistant', content: [{ text: 'Checking.' }, { text: '[called functions.get time({"tz":"UTC"})]' }] },
      { role: 'user', content: [{ text: '[result of functions.get time]\n12:00' }] },
      { role: 'assistant', content: [{ text: '[called get_time({})]' }] },
      { role: 'user', content: [{ text: 'Now wrap up.' }] },
    ]);
    expect(JSON.stringify(body)).not.toMatch(/toolUse|toolResult/);
  });

  it('sends a mixed round (a server-side call answered, a client call not) as built, under toolHistory: adapt', () => {
    const body = buildConverseBody(params([
      user,
      { role: 'assistant', tool_calls: [call('platform_1', 'text_editor'), call('client_1', 'ask_user')], content: null },
      { role: 'tool', tool_call_id: 'platform_1', name: 'text_editor', content: 'saved' },
    ], { tools: [fn('text_editor'), fn('ask_user')] }), TOOLS);
    const messages = body.messages as Array<{ role: string; content: Array<Record<string, any>> }>;
    expect(messages[1].content.map((b) => b.toolUse.toolUseId)).toEqual(['platform_1', 'client_1']);
    expect(messages[2].content).toEqual([{ toolResult: { toolUseId: 'platform_1', content: [{ text: 'saved' }] } }]);
    expect((body.toolConfig as { tools: unknown[] }).tools).toHaveLength(2);
    // A result with no call goes as built too; AWS judges it.
    expect(() => buildConverseBody(params([user, { role: 'assistant', content: 'hm' }, { role: 'tool', tool_call_id: 'zz', content: 'r' }], { tools: [fn('get_time')] }), TOOLS)).not.toThrow();
  });

  // A past call's name the model produced outside AWS's pattern is made to
  // fit rather than refused: a tool loop re-sends the history after every
  // round, and a refusal there would end the turn halfway.
  it('replays a past call whose name falls outside AWS\'s pattern under a name that fits', () => {
    const body = buildConverseBody(params([
      user,
      { role: 'assistant', tool_calls: [call('c1', 'functions.get time'), call('c2', '\u00e9'.repeat(3)), call('c3', 'x'.repeat(70))], content: null },
      { role: 'tool', tool_call_id: 'c1', content: 'r1' },
      { role: 'tool', tool_call_id: 'c2', content: 'r2' },
      { role: 'tool', tool_call_id: 'c3', content: 'r3' },
    ], { tools: [fn('get_time')] }), TOOLS);
    const names = (body.messages as Array<{ content: Array<Record<string, any>> }>)[1].content.map((b) => b.toolUse.name);
    expect(names).toEqual(['functions_get_time', '___', 'x'.repeat(64)]);
    expect(converseHistoryToolName('get_time')).toBe('get_time');
    expect(converseHistoryToolName('')).toBe('tool');
  });

  it('refuses media a vision-capable provider could see when told to, and describes it otherwise', () => {
    const withImage: Message = { role: 'user', content: 'what is this', content_items: [{ type: 'image', image: 'data:image/png;base64,iVBORw0KGgo=' }] };
    expect(refusal(() => buildConverseBody(params([withImage]), { media: 'refuse' }))).toBe('media input');
    expect((buildConverseBody(params([withImage])).messages as Array<{ content: unknown[] }>)[0].content).toHaveLength(2);
    const openaiParts: Message = { role: 'user', content: [{ type: 'text', text: 'see' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }] };
    expect(refusal(() => buildConverseBody(params([openaiParts]), { media: 'refuse' }))).toBe('media input');
  });
});

describe('tool ids and the tool-loop replay shape', () => {
  it('passes an id AWS accepts and maps any other to t_ + 32 hex of its SHA-256, deterministically', () => {
    expect(converseToolUseId('tooluse_AbC-1.2:3')).toBe('tooluse_AbC-1.2:3');
    const odd = 'call/with spaces+plus';
    const mapped = converseToolUseId(odd);
    expect(mapped).toBe(`t_${createHash('sha256').update(odd).digest('hex').slice(0, 32)}`);
    expect(converseToolUseId(odd)).toBe(mapped);
    expect(converseToolUseId('x'.repeat(65))).toMatch(/^t_[0-9a-f]{32}$/);
  });

  it('computes SHA-256 like node:crypto', () => {
    for (const s of ['', 'abc', 'é and emoji \u{1F600}', 'a'.repeat(55), 'a'.repeat(56), 'a'.repeat(64), 'b'.repeat(1000)]) {
      expect(sha256Hex(s), JSON.stringify(s.slice(0, 10))).toBe(createHash('sha256').update(s).digest('hex'));
    }
  });

  it('round-trips a replayed tool round: no content key, arguments \'\', an elided value, an odd id', () => {
    const odd = 'functions.get_time:0 #1';
    const elided = JSON.stringify({ path: '/notes.md', content: '<elided: 5000 bytes>' });
    const body = buildConverseBody(params([
      { role: 'user', content: 'What time is it? Then save it.' },
      { role: 'assistant', tool_calls: [{ id: odd, type: 'function', function: { name: 'get_time', arguments: '' } }] } as Message,
      { role: 'tool', tool_call_id: odd, name: 'get_time', content: '{"time":"12:00 UTC"}' },
      { role: 'assistant', content: 'Saving.', tool_calls: [{ id: 'call_2', type: 'function', function: { name: 'text_editor', arguments: elided } }] },
      { role: 'tool', tool_call_id: 'call_2', name: 'text_editor', content: 'saved' },
    ], { tools: [fn('get_time'), fn('text_editor')] }), TOOLS);
    const messages = body.messages as Array<{ role: string; content: Array<Record<string, any>> }>;
    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user', 'assistant', 'user']);
    expect(messages[1].content).toEqual([{ toolUse: { toolUseId: converseToolUseId(odd), name: 'get_time', input: {} } }]);
    expect(messages[2].content[0].toolResult.toolUseId).toBe(converseToolUseId(odd));
    expect(messages[3].content[1].toolUse.input).toEqual({ path: '/notes.md', content: '<elided: 5000 bytes>' });
    expect(messages[4].content[0].toolResult).toEqual({ toolUseId: 'call_2', content: [{ text: 'saved' }] });
  });

  it('treats arguments that are not a JSON object as {}', () => {
    const body = buildConverseBody(params([
      { role: 'user', content: 'x' },
      { role: 'assistant', tool_calls: [
        { id: 'a', type: 'function', function: { name: 't', arguments: '{"broken' } },
        { id: 'b', type: 'function', function: { name: 't', arguments: '[1,2]' } },
      ], content: null },
      { role: 'tool', tool_call_id: 'a', content: '1' },
      { role: 'tool', tool_call_id: 'b', content: '2' },
    ], { tools: [fn('t')] }), TOOLS);
    const uses = (body.messages as Array<{ content: Array<Record<string, any>> }>)[1].content.map((b) => b.toolUse.input);
    expect(uses).toEqual([{}, {}]);
  });
});

describe('usage and finish', () => {
  it('adds the cache legs back into prompt_tokens: 9 input + 2,140 cache read = 2,149, cached 2,140', () => {
    expect(converseUsageToChat({ inputTokens: 9, outputTokens: 197, totalTokens: 2346, cacheReadInputTokens: 2140, cacheWriteInputTokens: 0 }))
      .toEqual({ prompt_tokens: 2149, completion_tokens: 197, total_tokens: 2346, prompt_tokens_details: { cached_tokens: 2140 } });
    expect(converseUsageToChat({ inputTokens: 10, outputTokens: 1, cacheWriteInputTokens: 500 }).prompt_tokens).toBe(510);
    expect(converseUsageToChat(undefined)).toEqual({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, prompt_tokens_details: { cached_tokens: 0 } });
  });

  it('maps every stop reason', () => {
    expect(converseFinishReason('end_turn')).toBe('stop');
    expect(converseFinishReason('stop_sequence')).toBe('stop');
    expect(converseFinishReason('tool_use')).toBe('tool_calls');
    expect(converseFinishReason('max_tokens')).toBe('length');
    expect(converseFinishReason('model_context_window_exceeded')).toBe('length');
    expect(converseFinishReason('guardrail_intervened')).toBe('content_filter');
    expect(converseFinishReason('content_filtered')).toBe('content_filter');
    expect(converseFinishReason('malformed_model_output')).toBe('malformed_model_output');
    expect(converseFinishReason('malformed_tool_use')).toBe('malformed_tool_use');
  });
});

/** A ConverseStream answer: reasoning, text, a tool call split in two, the stop and the usage, with AWS's `p` padding. */
function toolAnswer(): Uint8Array[] {
  return [
    bedrockEventMessage('messageStart', { role: 'assistant', p: 'abcd' }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { reasoningContent: { text: 'The user wants the time.' } } }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { reasoningContent: { signature: 'sig' } } }),
    bedrockEventMessage('contentBlockStop', { contentBlockIndex: 0 }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 1, delta: { text: 'Let me ' } }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 1, delta: { text: 'check.' } }),
    bedrockEventMessage('contentBlockStop', { contentBlockIndex: 1 }),
    bedrockEventMessage('contentBlockStart', { contentBlockIndex: 2, start: { toolUse: { toolUseId: 'tooluse_X1', name: 'get_time' } } }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 2, delta: { toolUse: { input: '{"tz":' } } }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 2, delta: { toolUse: { input: '"UTC"}' } } }),
    bedrockEventMessage('contentBlockStop', { contentBlockIndex: 2 }),
    bedrockEventMessage('someFutureEvent', { anything: true }),
    bedrockEventMessage('messageStop', { stopReason: 'tool_use' }),
    bedrockEventMessage('metadata', { usage: { inputTokens: 9, outputTokens: 197, totalTokens: 2346, cacheReadInputTokens: 2140, cacheWriteInputTokens: 0 }, metrics: { latencyMs: 812 } }),
  ];
}

describe('ConverseStream to Chat Completions SSE', () => {
  it('writes Chat Completions chunks, then [DONE]', async () => {
    const sse = await converseStreamToChatCompletionsSSE(eventStream(toolAnswer()), { model: 'us.meta.llama3-3-70b-instruct-v1:0', id: 'chatcmpl-bedrock-t', created: 1 });
    expect(sse.status).toBe(200);
    const events = (await sse.text()).split('\n\n').filter(Boolean);
    expect(events.at(-1)).toBe('data: [DONE]');
    const chunks = events.slice(0, -1).map((e) => JSON.parse(e.slice('data: '.length)));
    for (const c of chunks) expect(c).toMatchObject({ id: 'chatcmpl-bedrock-t', object: 'chat.completion.chunk', created: 1, model: 'us.meta.llama3-3-70b-instruct-v1:0' });
    expect(chunks.map((c) => c.choices[0]?.delta ?? null)).toEqual([
      { role: 'assistant' },
      { reasoning_content: 'The user wants the time.' },
      { content: 'Let me ' },
      { content: 'check.' },
      { tool_calls: [{ index: 0, id: 'tooluse_X1', type: 'function', function: { name: 'get_time', arguments: '' } }] },
      { tool_calls: [{ index: 0, function: { arguments: '{"tz":' } }] },
      { tool_calls: [{ index: 0, function: { arguments: '"UTC"}' } }] },
      {},
      null,
    ]);
    expect(chunks[7].choices[0].finish_reason).toBe('tool_calls');
    expect(chunks[8].usage).toEqual({ prompt_tokens: 2149, completion_tokens: 197, total_tokens: 2346, prompt_tokens_details: { cached_tokens: 2140 } });
  });

  it('parses through the SDK Chat Completions parser to thinking, text, the tool call, usage and finish', async () => {
    const chunks = await collect(parseBedrockConverseStream(eventStream(toolAnswer())));
    expect(chunks).toEqual([
      { type: 'thinking', text: 'The user wants the time.' },
      { type: 'text', text: 'Let me ' },
      { type: 'text', text: 'check.' },
      { type: 'tool_call_start', id: 'tooluse_X1', name: 'get_time' },
      { type: 'tool_call', id: 'tooluse_X1', name: 'get_time', arguments: '{"tz":"UTC"}' },
      { type: 'usage', input: 2149, output: 197, cache_read_input: 2140 },
      { type: 'finish', reason: 'tool_calls' },
    ]);
  });

  it('numbers two tool blocks 0 and 1 and releases both', async () => {
    const frames = [
      bedrockEventMessage('messageStart', { role: 'assistant' }),
      bedrockEventMessage('contentBlockStart', { contentBlockIndex: 0, start: { toolUse: { toolUseId: 'a', name: 'one' } } }),
      bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { toolUse: { input: '{}' } } }),
      bedrockEventMessage('contentBlockStart', { contentBlockIndex: 1, start: { toolUse: { toolUseId: 'b', name: 'two' } } }),
      bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 1, delta: { toolUse: { input: '{"x":1}' } } }),
      bedrockEventMessage('messageStop', { stopReason: 'tool_use' }),
      bedrockEventMessage('metadata', { usage: { inputTokens: 5, outputTokens: 6 } }),
    ];
    const calls = (await collect(parseBedrockConverseStream(eventStream(frames)))).filter((c) => c.type === 'tool_call');
    expect(calls).toEqual([
      { type: 'tool_call', id: 'a', name: 'one', arguments: '{}' },
      { type: 'tool_call', id: 'b', name: 'two', arguments: '{"x":1}' },
    ]);
  });

  it('marks a guardrail stop as a blocked finish', async () => {
    const chunks = await collect(parseBedrockConverseStream(eventStream([
      bedrockEventMessage('messageStart', { role: 'assistant' }),
      bedrockEventMessage('messageStop', { stopReason: 'guardrail_intervened' }),
      bedrockEventMessage('metadata', { usage: { inputTokens: 5, outputTokens: 0 } }),
    ])));
    expect(chunks.at(-1)).toEqual({ type: 'finish', reason: 'content_filter', blocked: true });
  });

  it('turns a first-frame exception into a refusal, and a later one into a stream error', async () => {
    const refused = await converseStreamToChatCompletionsSSE(eventStream([bedrockExceptionMessage('throttlingException', 'slow down')]), { model: 'm' });
    expect(refused.status).toBe(429);
    // `messageStart` says only `role: assistant`, so priming holds it: an
    // exception straight after it is still a refusal before any byte.
    const opened = await converseStreamToChatCompletionsSSE(eventStream([
      bedrockEventMessage('messageStart', { role: 'assistant' }),
      bedrockExceptionMessage('throttlingException', 'slow down'),
    ]), { model: 'm' });
    expect(opened.status).toBe(429);
    // So does `contentBlockStart`: it names a tool block, none of its input.
    const blockOpened = await converseStreamToChatCompletionsSSE(eventStream([
      bedrockEventMessage('messageStart', { role: 'assistant' }),
      bedrockEventMessage('contentBlockStart', { contentBlockIndex: 0, start: { toolUse: { toolUseId: 'a', name: 'one' } } }),
      bedrockExceptionMessage('modelStreamErrorException', 'model fault'),
    ]), { model: 'm' });
    expect(blockOpened.status).toBe(502);
    await expect(collect(parseBedrockConverseStream(eventStream([bedrockExceptionMessage('throttlingException', 'slow down')]))))
      .rejects.toBeInstanceOf(BedrockStreamError);
    await expect(collect(parseBedrockConverseStream(eventStream([
      bedrockEventMessage('messageStart', { role: 'assistant' }),
      bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { text: 'partial' } }),
      bedrockExceptionMessage('modelStreamErrorException', 'model fault'),
    ])))).rejects.toThrow(/modelStreamErrorException: model fault/);
  });

  it('parses an already transcoded body unchanged (the adapter is safe either side of the boundary)', async () => {
    const sse = await converseStreamToChatCompletionsSSE(eventStream(toolAnswer()), { model: 'm' });
    const adapter = createBedrockConverseAdapter({ toolCalling: true });
    const chunks = await collect(adapter.parseStream(sse));
    expect(chunks.find((c) => c.type === 'usage')).toEqual({ type: 'usage', input: 2149, output: 197, cache_read_input: 2140 });
  });
});

describe('Converse JSON to chat.completion', () => {
  it('maps text, reasoning, tool calls, finish and usage', () => {
    const out = converseJsonToChatCompletion({
      output: { message: { role: 'assistant', content: [
        { reasoningContent: { reasoningText: { text: 'thinking...', signature: 's' } } },
        { text: 'Let me check.' },
        { toolUse: { toolUseId: 'tooluse_abc', name: 'get_time', input: { tz: 'UTC' } } },
      ] } },
      stopReason: 'tool_use',
      usage: { inputTokens: 9, outputTokens: 197, totalTokens: 2346, cacheReadInputTokens: 2140, cacheWriteInputTokens: 0 },
      metrics: { latencyMs: 1234 },
    }, { model: 'amazon.nova-pro-v1:0', id: 'chatcmpl-bedrock-1', created: 7 });
    expect(out).toEqual({
      id: 'chatcmpl-bedrock-1',
      object: 'chat.completion',
      created: 7,
      model: 'amazon.nova-pro-v1:0',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: 'Let me check.',
          reasoning_content: 'thinking...',
          tool_calls: [{ id: 'tooluse_abc', type: 'function', function: { name: 'get_time', arguments: '{"tz":"UTC"}' } }],
        },
        finish_reason: 'tool_calls',
      }],
      usage: { prompt_tokens: 2149, completion_tokens: 197, total_tokens: 2346, prompt_tokens_details: { cached_tokens: 2140 } },
    });
  });

  it('gives null content for an answer with no text', () => {
    const out = converseJsonToChatCompletion({ output: { message: { content: [] } }, stopReason: 'max_tokens', usage: { inputTokens: 1, outputTokens: 0 } }, { model: 'm' });
    expect((out.choices as Array<{ message: { content: unknown }; finish_reason: string }>)[0]).toMatchObject({ message: { content: null }, finish_reason: 'length' });
  });
});

// Amazon Nova writes its reasoning inline, as <thinking>...</thinking> inside
// ordinary text. With `inlineThinkingTags` the span becomes reasoning and the
// reader sees only the answer; without it, text is passed through untouched.
describe('inline <thinking> spans', () => {
  it('splits a span that arrives in one piece, trimming the space it leaves', () => {
    const s = createInlineThinkingSplitter();
    expect(s.push('<thinking> The user asks two things. </thinking> I am the agent.')).toEqual({ content: 'I am the agent.', reasoning: ' The user asks two things. ' });
    expect(s.end()).toEqual({ content: '', reasoning: '' });
  });

  it('holds back a tag split across chunks until it is complete', () => {
    const s = createInlineThinkingSplitter();
    const parts = ['<thin', 'king>plan', ' it</thi', 'nking>\n\nAnswer', ' here. a < b'].map((p) => s.push(p));
    const end = s.end();
    expect(parts.map((p) => p.content).join('') + end.content).toBe('Answer here. a < b');
    expect(parts.map((p) => p.reasoning).join('') + end.reasoning).toBe('plan it');
  });

  it('leaves text with no span alone, and counts an unclosed span as reasoning', () => {
    const plain = createInlineThinkingSplitter();
    expect(plain.push('Hello <b>there</b>')).toEqual({ content: 'Hello <b>there</b>', reasoning: '' });
    const open = createInlineThinkingSplitter();
    expect(open.push('<thinking>never closed')).toEqual({ content: '', reasoning: 'never closed' });
    expect(open.end()).toEqual({ content: '', reasoning: '' });
  });

  const novaFrames = () => [
    bedrockEventMessage('messageStart', { role: 'assistant' }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { text: '<thinking> I should answer' } }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { text: ' directly. </thin' } }),
    bedrockEventMessage('contentBlockDelta', { contentBlockIndex: 0, delta: { text: 'king> 17 times 23 is 391.' } }),
    bedrockEventMessage('contentBlockStop', { contentBlockIndex: 0 }),
    bedrockEventMessage('messageStop', { stopReason: 'end_turn' }),
  ];

  it('streams the span as reasoning_content when asked, and as content when not', async () => {
    const on = await converseStreamToChatCompletionsSSE(eventStream(novaFrames()), { model: 'amazon.nova-pro-v1:0', inlineThinkingTags: true });
    const deltas = (await on.text()).split('\n\n').filter((e) => e && e !== 'data: [DONE]').map((e) => JSON.parse(e.slice(6)).choices[0]?.delta ?? {});
    expect(deltas.map((d) => d.content ?? '').join('')).toBe('17 times 23 is 391.');
    expect(deltas.map((d) => d.reasoning_content ?? '').join('')).toBe(' I should answer directly. ');
    expect(deltas.some((d) => typeof d.content === 'string' && d.content.includes('<thinking>'))).toBe(false);

    const off = await converseStreamToChatCompletionsSSE(eventStream(novaFrames()), { model: 'amazon.nova-pro-v1:0' });
    const raw = (await off.text()).split('\n\n').filter((e) => e && e !== 'data: [DONE]').map((e) => JSON.parse(e.slice(6)).choices[0]?.delta?.content ?? '').join('');
    expect(raw).toBe('<thinking> I should answer directly. </thinking> 17 times 23 is 391.');
  });

  it('parses a Nova stream to thinking then text when asked', async () => {
    const chunks = await collect(parseBedrockConverseStream(eventStream(novaFrames()), 'amazon.nova-pro-v1:0', { inlineThinkingTags: true }));
    const text = chunks.filter((c) => c.type === 'text').map((c) => (c as { text: string }).text).join('');
    const thinking = chunks.filter((c) => c.type === 'thinking').map((c) => (c as { text: string }).text).join('');
    expect(text).toBe('17 times 23 is 391.');
    expect(thinking).toBe(' I should answer directly. ');
  });

  it('splits a non-stream answer the same way', () => {
    const json = { output: { message: { role: 'assistant', content: [{ text: '<thinking>check the tool</thinking>Lisbon has about 545,000 people.' }] } }, stopReason: 'end_turn', usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 } };
    const on = converseJsonToChatCompletion(json, { model: 'amazon.nova-lite-v1:0', inlineThinkingTags: true }) as { choices: Array<{ message: { content: string; reasoning_content?: string } }> };
    expect(on.choices[0].message.content).toBe('Lisbon has about 545,000 people.');
    expect(on.choices[0].message.reasoning_content).toBe('check the tool');
    const off = converseJsonToChatCompletion(json, { model: 'amazon.nova-lite-v1:0' }) as { choices: Array<{ message: { content: string } }> };
    expect(off.choices[0].message.content).toContain('<thinking>');
  });
});
