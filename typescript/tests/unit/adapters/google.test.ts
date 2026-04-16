/**
 * Google Gemini Adapter Unit Tests
 *
 * Tests buildRequest for message conversion, model aliasing, media support,
 * and parseStream for thinking/text chunk emission.
 */

import { describe, it, expect } from 'vitest';
import { googleAdapter } from '../../../src/adapters/google.js';
import type { AdapterRequestParams, AdapterChunk } from '../../../src/adapters/types.js';

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
    model: 'gemini-2.5-flash',
    apiKey: 'test-key',
    ...overrides,
  };
}

describe('googleAdapter', () => {
  it('has name "google"', () => {
    expect(googleAdapter.name).toBe('google');
  });

  describe('mediaSupport', () => {
    it('supports all modalities as base64', () => {
      expect(googleAdapter.mediaSupport).toEqual({
        image: 'base64',
        audio: 'base64',
        video: 'base64',
        document: 'base64',
      });
    });
  });

  describe('buildRequest', () => {
    it('builds a valid streaming request', () => {
      const req = googleAdapter.buildRequest(makeParams());
      expect(req.url).toContain('generativelanguage.googleapis.com');
      expect(req.url).toContain('gemini-2.5-flash');
      expect(req.url).toContain('streamGenerateContent');
      expect(req.url).toContain('alt=sse');
      expect(req.url).toContain('key=test-key');
      expect(req.headers['Content-Type']).toBe('application/json');

      const body = JSON.parse(req.body);
      expect(body).toHaveProperty('contents');
    });

    it('strips provider prefix from model name', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'google/gemini-2.5-flash' }));
      expect(req.url).toContain('gemini-2.5-flash');
      expect(req.url).not.toContain('google/');
    });

    it('applies model aliases', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3-flash' }));
      expect(req.url).toContain('gemini-3-flash-preview');
    });

    it('includes temperature when provided', () => {
      const req = googleAdapter.buildRequest(makeParams({ temperature: 0.5 }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.temperature).toBe(0.5);
    });

    it('includes maxOutputTokens when maxTokens is set', () => {
      const req = googleAdapter.buildRequest(makeParams({ maxTokens: 1000 }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.maxOutputTokens).toBe(1000);
    });

    it('converts user message to Gemini contents format', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'assistant', content: 'Hello!' },
        ],
      }));
      const body = JSON.parse(req.body);
      expect(body.contents).toBeInstanceOf(Array);
      expect(body.contents.length).toBe(2);
      expect(body.contents[0].role).toBe('user');
      expect(body.contents[1].role).toBe('model');
    });

    it('converts tool_calls without thought signatures to native functionCall parts', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Search for cats' },
          {
            role: 'assistant',
            content: 'Let me search for that.',
            tool_calls: [{
              id: 'call_abc123',
              type: 'function' as const,
              function: { name: 'search', arguments: '{"query":"cats"}' },
            }],
          },
          { role: 'tool', content: '10 results', tool_call_id: 'call_abc123', name: 'search' },
          { role: 'user', content: 'Thanks' },
        ],
      }));
      const body = JSON.parse(req.body);
      const modelMsg = body.contents.find((c: { role: string }) => c.role === 'model');
      expect(modelMsg).toBeDefined();
      const fcPart = modelMsg.parts.find((p: { functionCall?: unknown }) => p.functionCall);
      expect(fcPart).toBeDefined();
      expect(fcPart.functionCall.name).toBe('search');
      expect(fcPart.functionCall.args).toEqual({ query: 'cats' });
      expect(fcPart.thought_signature).toBeUndefined();
      expect(fcPart.functionCall.thought_signature).toBeUndefined();
      const textParts = modelMsg.parts.filter((p: { text?: string }) => typeof p.text === 'string');
      for (const tp of textParts) {
        expect(tp.text).not.toContain('[Called tool');
      }
    });

    it('extracts system instruction from system role messages', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hi' },
        ],
      }));
      const body = JSON.parse(req.body);
      expect(body.system_instruction).toBeDefined();
      expect(body.contents.length).toBe(1);
    });

    it('converts tools to Google function declarations', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: { name: 'search', description: 'Search the web', parameters: { type: 'object', properties: {} } },
        }],
      }));
      const body = JSON.parse(req.body);
      expect(body.tools).toBeDefined();
    });

    it('sets responseModalities for image models', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3.1-flash-image' }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.responseModalities).toContain('IMAGE');
    });

    it('builds non-streaming request when stream is false', () => {
      const req = googleAdapter.buildRequest(makeParams({ stream: false }));
      expect(req.url).toContain('generateContent');
      expect(req.url).not.toContain('alt=sse');
    });

    it('includes thinkingConfig for gemini-2.5 models', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-2.5-flash' }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toEqual({ includeThoughts: true });
    });

    it('includes thinkingConfig for gemini-3 models (no thinkingLevel for non-lite)', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3-flash' }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toEqual({ includeThoughts: true });
    });

    it('includes thinkingConfig for gemini-3.1-pro (no thinkingLevel)', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3.1-pro' }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toEqual({ includeThoughts: true });
    });

    it('sets thinkingLevel=low for gemini-3.1-flash-lite when thinking enabled', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3.1-flash-lite' }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toEqual({ includeThoughts: true, thinkingLevel: 'low' });
    });

    it('sets thinkingLevel=minimal for gemini-3 when thinking=false', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3-flash', thinking: false }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toEqual({ thinkingLevel: 'minimal' });
    });

    it('sets thinkingBudget=0 for gemini-2.5 when thinking=false', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-2.5-flash', thinking: false }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toEqual({ thinkingBudget: 0 });
    });

    it('does NOT include thinkingConfig for gemini-1.5 models', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-1.5-flash' }));
      const body = JSON.parse(req.body);
      expect(body.generationConfig?.thinkingConfig).toBeUndefined();
    });

    it('aliases gemini-3-flash to gemini-3-flash-preview', () => {
      const req = googleAdapter.buildRequest(makeParams({ model: 'gemini-3-flash' }));
      expect(req.url).toContain('gemini-3-flash-preview');
    });

    it('attaches thought_signature to model-turn image parts from resolvedMedia', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, {
          mimeType: 'image/png',
          base64: 'iVBOR...',
          thoughtSignature: 'sig_abc_123',
        }],
      ]);
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'draw a car' },
          {
            role: 'assistant',
            content: 'Here is the car I generated',
            content_items: [{ type: 'image', image: `/api/content/${uuid}` }],
          },
          { role: 'user', content: 'make it green' },
        ],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      expect(modelTurn).toBeDefined();
      const imgPart = modelTurn.parts.find((p: Record<string, unknown>) => p.inlineData);
      expect(imgPart).toBeDefined();
      expect(imgPart.inlineData.mimeType).toBe('image/png');
      expect(imgPart.thought_signature).toBe('sig_abc_123');
    });

    it('does NOT attach thought_signature to user-turn image parts without thoughtSignature', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, {
          mimeType: 'image/png',
          base64: 'iVBOR...',
        }],
      ]);
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          {
            role: 'user',
            content: 'edit this',
            content_items: [{ type: 'image', image: `/api/content/${uuid}` }],
          },
        ],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const userTurn = body.contents[0];
      expect(userTurn.role).toBe('user');
      const imgPart = userTurn.parts.find((p: Record<string, unknown>) => p.inlineData);
      expect(imgPart).toBeDefined();
      expect(imgPart.thought_signature).toBeUndefined();
    });

    it('renders model-turn images without thought_signature as inlineData', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, {
          mimeType: 'image/png',
          base64: 'iVBOR...',
        }],
      ]);
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          {
            role: 'assistant',
            content: 'Generated image',
            content_items: [{ type: 'image', image: `/api/content/${uuid}` }],
          },
          { role: 'user', content: 'make it blue' },
        ],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      const imgPart = modelTurn.parts.find((p: Record<string, unknown>) => p.inlineData);
      expect(imgPart).toBeDefined();
      expect(imgPart.inlineData.mimeType).toBe('image/png');
      expect(imgPart.thought_signature).toBeUndefined();
    });
  });

  describe('parseStream', () => {
    it('yields thinking chunk for parts with thought=true', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: 'I need to think...', thought: true }] } }] },
        { candidates: [{ content: { parts: [{ text: 'The answer is 42.' }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      expect(chunks[0]).toEqual({ type: 'thinking', text: 'I need to think...' });
      expect(chunks[1]).toEqual({ type: 'text', text: 'The answer is 42.' });
    });

    it('does not double-emit thought parts as text', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: 'reasoning here', thought: true }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      expect(chunks).toHaveLength(1);
      expect(chunks[0].type).toBe('thinking');
    });

    it('yields regular text for parts without thought flag', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: 'Hello world' }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      expect(chunks).toHaveLength(1);
      expect(chunks[0]).toEqual({ type: 'text', text: 'Hello world' });
    });

    it('handles mixed thought and text parts in a single candidate', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [
          { text: 'Let me reason...', thought: true },
          { text: 'Here is the result.' },
        ] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      expect(chunks[0]).toEqual({ type: 'thinking', text: 'Let me reason...' });
      expect(chunks[1]).toEqual({ type: 'text', text: 'Here is the result.' });
    });

    it('yields thinking chunk even when thought part has no text field', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ thought: true }] } }] },
        { candidates: [{ content: { parts: [{ text: 'actual thought', thought: true }] } }] },
        { candidates: [{ content: { parts: [{ text: 'The answer.' }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      expect(chunks[0]).toEqual({ type: 'thinking', text: '' });
      expect(chunks[1]).toEqual({ type: 'thinking', text: 'actual thought' });
      expect(chunks[2]).toEqual({ type: 'text', text: 'The answer.' });
    });

    it('yields cache_read_input from usageMetadata.cachedContentTokenCount (camelCase)', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
        { usageMetadata: { promptTokenCount: 500, candidatesTokenCount: 20, cachedContentTokenCount: 300 } },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const usage = chunks.find(c => c.type === 'usage');
      expect(usage).toBeDefined();
      expect(usage!.input).toBe(500);
      expect(usage!.output).toBe(20);
      expect(usage!.cache_read_input).toBe(300);
    });

    it('yields cache_read_input from usageMetadata.cached_content_token_count (snake_case)', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
        { usageMetadata: { promptTokenCount: 400, candidatesTokenCount: 15, cached_content_token_count: 200 } },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const usage = chunks.find(c => c.type === 'usage');
      expect(usage).toBeDefined();
      expect(usage!.cache_read_input).toBe(200);
    });

    it('omits cache_read_input when cachedContentTokenCount is absent', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
        { usageMetadata: { promptTokenCount: 100, candidatesTokenCount: 10 } },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const usage = chunks.find(c => c.type === 'usage');
      expect(usage).toBeDefined();
      expect(usage!.cache_read_input).toBeUndefined();
    });

    it('yields thinking chunk when thought=true but text is empty string', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{ text: '', thought: true }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      expect(chunks).toHaveLength(1);
      expect(chunks[0].type).toBe('thinking');
      expect(chunks[0].text).toBe('');
    });

    it('captures thoughtSignature from Part level on functionCall (camelCase)', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{
          functionCall: { name: 'search', args: { query: 'cats' } },
          thoughtSignature: 'sig_abc',
        }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const tc = chunks.find(c => c.type === 'tool_call');
      expect(tc).toBeDefined();
      expect(tc!.id).toContain('|ts:sig_abc');
      expect(tc!.name).toBe('search');
    });

    it('captures thought_signature from Part level on functionCall (snake_case)', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{
          functionCall: { name: 'search', args: { query: 'cats' } },
          thought_signature: 'sig_def',
        }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const tc = chunks.find(c => c.type === 'tool_call');
      expect(tc).toBeDefined();
      expect(tc!.id).toContain('|ts:sig_def');
    });

    it('handles parallel function calls — only first has signature', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [
          {
            functionCall: { name: 'get_weather', args: { city: 'Paris' } },
            thoughtSignature: 'sig_parallel',
          },
          {
            functionCall: { name: 'get_weather', args: { city: 'London' } },
          },
        ] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const toolCalls = chunks.filter(c => c.type === 'tool_call');
      expect(toolCalls).toHaveLength(2);
      expect(toolCalls[0].id).toContain('|ts:sig_parallel');
      expect(toolCalls[1].id).not.toContain('|ts:');
    });

    it('produces tool_call without |ts: when no signature present', async () => {
      const response = mockSSEResponse([
        { candidates: [{ content: { parts: [{
          functionCall: { name: 'search', args: {} },
        }] } }] },
      ]);
      const chunks = await collectChunks(googleAdapter.parseStream(response));
      const tc = chunks.find(c => c.type === 'tool_call');
      expect(tc).toBeDefined();
      expect(tc!.id).not.toContain('|ts:');
    });
  });

  describe('convertMessages (thought_signature round-trip)', () => {
    it('outputs thought_signature as Part-level sibling of functionCall', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Search for cats' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{
              id: 'call_abc|ts:sig_roundtrip',
              type: 'function' as const,
              function: { name: 'search', arguments: '{"query":"cats"}' },
            }],
          },
          { role: 'tool', content: 'results', tool_call_id: 'call_abc|ts:sig_roundtrip', name: 'search' },
        ],
      }));
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      expect(modelTurn).toBeDefined();
      const fcPart = modelTurn.parts.find((p: Record<string, unknown>) => p.functionCall);
      expect(fcPart).toBeDefined();
      expect(fcPart.thought_signature).toBe('sig_roundtrip');
      expect((fcPart.functionCall as Record<string, unknown>).thought_signature).toBeUndefined();
    });

    it('omits thought_signature when tool call ID has no |ts: marker', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Search' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [{
              id: 'call_plain',
              type: 'function' as const,
              function: { name: 'search', arguments: '{}' },
            }],
          },
          { role: 'tool', content: 'ok', tool_call_id: 'call_plain', name: 'search' },
        ],
      }));
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      const fcPart = modelTurn.parts.find((p: Record<string, unknown>) => p.functionCall);
      expect(fcPart.thought_signature).toBeUndefined();
    });

    it('handles parallel tool calls — signature only on first', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Weather in Paris and London' },
          {
            role: 'assistant',
            content: '',
            tool_calls: [
              {
                id: 'call_1|ts:sig_first',
                type: 'function' as const,
                function: { name: 'get_weather', arguments: '{"city":"Paris"}' },
              },
              {
                id: 'call_2',
                type: 'function' as const,
                function: { name: 'get_weather', arguments: '{"city":"London"}' },
              },
            ],
          },
          { role: 'tool', content: '15C', tool_call_id: 'call_1|ts:sig_first', name: 'get_weather' },
          { role: 'tool', content: '12C', tool_call_id: 'call_2', name: 'get_weather' },
        ],
      }));
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      const fcParts = modelTurn.parts.filter((p: Record<string, unknown>) => p.functionCall);
      expect(fcParts).toHaveLength(2);
      expect(fcParts[0].thought_signature).toBe('sig_first');
      expect(fcParts[1].thought_signature).toBeUndefined();
    });

    it('replaces media content_items with text metadata for tool messages', () => {
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'user', content: 'Generate music' },
          {
            role: 'assistant',
            content: null,
            tool_calls: [{ id: 'tc_1', type: 'function' as const, function: { name: 'gen_audio', arguments: '{}' } }],
          },
          {
            role: 'tool',
            content: 'Generated audio',
            tool_call_id: 'tc_1',
            name: 'gen_audio',
            content_items: [
              { type: 'audio', content_id: 'aud-001', duration_ms: 30000, format: 'mp3' },
            ],
          } as any,
        ],
      }));

      const body = JSON.parse(req.body);
      const toolTurn = body.contents.find((c: { parts: any[] }) =>
        c.parts?.some((p: any) => p.functionResponse),
      );

      expect(toolTurn).toBeDefined();
      const frPart = toolTurn.parts.find((p: any) => p.functionResponse);
      expect(frPart.functionResponse.name).toBe('gen_audio');
      expect(frPart.functionResponse.id).toBe('tc_1');

      // No inlineData anywhere in the turn
      const inlineDataParts = toolTurn.parts.filter((p: any) => p.inlineData);
      expect(inlineDataParts).toHaveLength(0);

      // Text metadata present
      const textParts = toolTurn.parts.filter((p: any) => p.text && p.text.includes('[Available audio:'));
      expect(textParts).toHaveLength(1);
      expect(textParts[0].text).toContain('content_id=aud-001');
      expect(textParts[0].text).toContain('duration=0:30');
    });
  });
});
