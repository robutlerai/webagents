/**
 * Claude on Bedrock InvokeModel (src/adapters/bedrock-invoke.ts).
 *
 * The property that matters: an Invoke answer reaches every consumer as the
 * Anthropic API's own SSE. So the transcoded stream is parsed by the SDK's
 * Anthropic parser to EXACTLY the chunks an equivalent Messages SSE fixture
 * gives (usage legs with cache, thinking, tool_use, the finish reason), and
 * Bedrock's `amazon-bedrock-invocationMetrics` never leaves the transcoder.
 */
import { describe, it, expect } from 'vitest';
import {
  buildBedrockAnthropicInvokeRequest,
  invokeStreamToAnthropicSSE,
  parseBedrockAnthropicInvokeStream,
  createBedrockAnthropicInvokeAdapter,
  BEDROCK_ANTHROPIC_VERSION,
} from '../../../src/adapters/bedrock-invoke.js';
import { anthropicAdapter } from '../../../src/adapters/anthropic.js';
import { bedrockEventMessage, bedrockExceptionMessage, BedrockStreamError } from '../../../src/adapters/bedrock-eventstream.js';
import type { AdapterChunk, AdapterRequestParams, ToolDefinition } from '../../../src/adapters/types.js';

function params(extra: Partial<AdapterRequestParams> = {}): AdapterRequestParams {
  return {
    model: 'anthropic/claude-sonnet-4-6',
    messages: [{ role: 'system', content: 'Be terse.' }, { role: 'user', content: 'Hi' }],
    apiKey: 'bedrock-key',
    ...extra,
  };
}

function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream({ pull(c) { if (i < chunks.length) c.enqueue(chunks[i++]); else c.close(); } });
}
const eventStream = (frames: Uint8Array[]) =>
  new Response(streamOf(frames), { status: 200, headers: { 'content-type': 'application/vnd.amazon.eventstream' } });
const chunkFrame = (event: unknown) => bedrockEventMessage('chunk', { bytes: Buffer.from(JSON.stringify(event)).toString('base64'), p: 'abcdefghij' });

async function collect(gen: AsyncGenerator<AdapterChunk>): Promise<AdapterChunk[]> {
  const out: AdapterChunk[] = [];
  for await (const c of gen) out.push(c);
  return out;
}

/** One Claude answer: cache legs, a thinking block, text, a tool call, the stop, Bedrock's metrics on the last event. */
const EVENTS: Array<Record<string, unknown>> = [
  { type: 'message_start', message: { id: 'msg_1', type: 'message', role: 'assistant', content: [], usage: { input_tokens: 12, cache_read_input_tokens: 2140, cache_creation_input_tokens: 30, output_tokens: 1 } } },
  { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } },
  { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'They want the time.' } },
  { type: 'content_block_stop', index: 0 },
  { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } },
  { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Checking.' } },
  { type: 'content_block_stop', index: 1 },
  { type: 'content_block_start', index: 2, content_block: { type: 'tool_use', id: 'toolu_01', name: 'get_time', input: {} } },
  { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{"tz":' } },
  { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '"UTC"}' } },
  { type: 'content_block_stop', index: 2 },
  { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 57 } },
  { type: 'message_stop', 'amazon-bedrock-invocationMetrics': { inputTokenCount: 12, outputTokenCount: 57, invocationLatency: 900, firstByteLatency: 300 } },
];

function messagesSse(events: Array<Record<string, unknown>>): Response {
  const text = events.map((e) => {
    const { 'amazon-bedrock-invocationMetrics': _m, ...clean } = e;
    return `event: ${String(e.type)}\ndata: ${JSON.stringify(clean)}\n\n`;
  }).join('');
  return new Response(text, { headers: { 'content-type': 'text/event-stream' } });
}

describe('buildBedrockAnthropicInvokeRequest', () => {
  it('is the Anthropic body without model and stream, with anthropic_version and the betas in the body, on the encoded Invoke path', () => {
    const tools: ToolDefinition[] = [
      { type: 'function', function: { name: 'get_time', parameters: { type: 'object', properties: {} } } },
      { type: 'computer_20250124', name: 'computer', display_width_px: 1024, display_height_px: 768, beta: 'computer-use-2025-01-24' } as ToolDefinition,
    ];
    const direct = anthropicAdapter.buildRequest(params({ tools }));
    const req = buildBedrockAnthropicInvokeRequest(params({ tools }), {
      region: 'us-east-1',
      modelId: 'global.anthropic.claude-opus-4-6-v1:0',
      betas: ['context-management-2025-06-27'],
    });
    expect(req.url).toBe('https://bedrock-runtime.us-east-1.amazonaws.com/model/global.anthropic.claude-opus-4-6-v1%3A0/invoke-with-response-stream');
    expect(req.headers).toEqual({
      'content-type': 'application/json',
      accept: 'application/vnd.amazon.eventstream',
      'x-amzn-bedrock-accept': 'application/json',
      authorization: 'Bearer bedrock-key',
    });
    const body = JSON.parse(req.body);
    const { model: _model, stream: _stream, ...directBody } = JSON.parse(direct.body);
    expect(body).not.toHaveProperty('model');
    expect(body).not.toHaveProperty('stream');
    expect(body.anthropic_version).toBe(BEDROCK_ANTHROPIC_VERSION);
    expect(body.anthropic_beta).toEqual(['computer-use-2025-01-24', 'context-management-2025-06-27']);
    const { anthropic_version: _v, anthropic_beta: _b, ...rest } = body;
    expect(rest).toEqual(directBody);
  });

  it('uses /invoke and application/json for a non-stream call, and sends no anthropic_beta when there is none', () => {
    const req = createBedrockAnthropicInvokeAdapter({ baseUrl: 'https://bedrock-runtime.eu-west-1.amazonaws.com' }).buildRequest(params({ stream: false }));
    expect(req.url).toBe('https://bedrock-runtime.eu-west-1.amazonaws.com/model/claude-sonnet-4-6/invoke');
    expect(req.headers.accept).toBe('application/json');
    expect(JSON.parse(req.body)).not.toHaveProperty('anthropic_beta');
    expect(req.headers).not.toHaveProperty('x-api-key');
    expect(req.headers).not.toHaveProperty('anthropic-version');
  });
});

describe('the Invoke stream as Anthropic SSE', () => {
  it('parses to exactly the chunks the same answer gives on the Messages wire', async () => {
    const viaMessages = await collect(anthropicAdapter.parseStream(messagesSse(EVENTS)));
    const viaInvoke = await collect(parseBedrockAnthropicInvokeStream(eventStream(EVENTS.map(chunkFrame))));
    expect(viaInvoke).toEqual(viaMessages);
    expect(viaInvoke).toEqual([
      { type: 'thinking', text: 'They want the time.' },
      { type: 'text', text: 'Checking.' },
      { type: 'tool_call_start', id: 'toolu_01', name: 'get_time' },
      { type: 'tool_call', id: 'toolu_01', name: 'get_time', arguments: '{"tz":"UTC"}' },
      { type: 'usage', input: 12, output: 57, cache_read_input: 2140, cache_creation_input: 30 },
      { type: 'finish', reason: 'tool_use' },
    ]);
  });

  it('writes named SSE events and strips Bedrock\'s invocation metrics', async () => {
    const sse = await invokeStreamToAnthropicSSE(eventStream(EVENTS.map(chunkFrame)));
    expect(sse.status).toBe(200);
    expect(sse.headers.get('content-type')).toBe('text/event-stream');
    const text = await sse.text();
    expect(text).not.toContain('amazon-bedrock-invocationMetrics');
    expect(text).not.toContain('"p"');
    expect(text.startsWith('event: message_start\ndata: {"type":"message_start"')).toBe(true);
    expect(text.endsWith('event: message_stop\ndata: {"type":"message_stop"}\n\n')).toBe(true);
    expect(text.split('\n\n').filter(Boolean)).toHaveLength(EVENTS.length);
  });

  it('reports a refusal as a blocked finish, an answer and not an error', async () => {
    const chunks = await collect(parseBedrockAnthropicInvokeStream(eventStream([
      { type: 'message_start', message: { usage: { input_tokens: 40 } } },
      { type: 'message_delta', delta: { stop_reason: 'refusal' }, usage: { output_tokens: 0 } },
      { type: 'message_stop' },
    ].map(chunkFrame))));
    expect(chunks).toEqual([
      { type: 'usage', input: 40, output: 0 },
      { type: 'finish', reason: 'refusal', blocked: true },
    ]);
  });

  it('refuses before the first chunk (synthetic status) and errors after it like Anthropic does', async () => {
    const refused = await invokeStreamToAnthropicSSE(eventStream([bedrockExceptionMessage('throttlingException', 'Too many tokens')]));
    expect(refused.status).toBe(429);
    expect(await refused.json()).toEqual({ message: 'Too many tokens', type: 'throttlingException' });
    await expect(collect(parseBedrockAnthropicInvokeStream(eventStream([bedrockExceptionMessage('validationException', 'bad')]))))
      .rejects.toBeInstanceOf(BedrockStreamError);

    // `message_start`, `content_block_start` and `ping` open the answer or a
    // block of it without any of it, so priming holds them: an exception
    // straight after them is still a refusal before any byte.
    const opened = await invokeStreamToAnthropicSSE(eventStream([chunkFrame(EVENTS[0]), bedrockExceptionMessage('throttlingException', 'Too many tokens')]));
    expect(opened.status).toBe(429);
    const blockOpened = await invokeStreamToAnthropicSSE(eventStream([
      chunkFrame(EVENTS[0]), chunkFrame({ type: 'ping' }), chunkFrame(EVENTS[1]), bedrockExceptionMessage('modelStreamErrorException', 'model fault'),
    ]));
    expect(blockOpened.status).toBe(502);

    const begun = [chunkFrame(EVENTS[0]), chunkFrame(EVENTS[1]), chunkFrame(EVENTS[2]), bedrockExceptionMessage('modelStreamErrorException', 'model fault')];
    const late = await invokeStreamToAnthropicSSE(eventStream(begun));
    expect(late.status).toBe(200);
    const text = await late.text();
    expect(text.startsWith('event: message_start\n')).toBe(true);
    expect(text.endsWith('event: error\ndata: {"type":"error","error":{"type":"modelStreamErrorException","message":"model fault"}}\n\n')).toBe(true);
    await expect(collect(parseBedrockAnthropicInvokeStream(eventStream(begun))))
      .rejects.toThrow(/Anthropic stream error: modelStreamErrorException: model fault/);
  });

  it('ignores frames that are not chunks and refuses a chunk that is not an Anthropic event', async () => {
    const chunks = await collect(parseBedrockAnthropicInvokeStream(eventStream([
      bedrockEventMessage('somethingElse', { x: 1 }),
      ...EVENTS.map(chunkFrame),
    ])));
    expect(chunks.at(-1)).toEqual({ type: 'finish', reason: 'tool_use' });
    const bad = await invokeStreamToAnthropicSSE(eventStream([bedrockEventMessage('chunk', { bytes: Buffer.from('not json').toString('base64') })]));
    expect(bad.status).toBe(502);
  });
});
