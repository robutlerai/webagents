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
          { role: 'assistant', content: `![Generated image](/api/content/${uuid})` },
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

    it('does NOT attach thought_signature to user-turn image parts', () => {
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
          { role: 'user', content: `edit this ![photo](/api/content/${uuid})` },
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

    it('falls back to text placeholder for model-turn images without thought_signature', () => {
      const uuid = 'f485e424-14a1-482d-968e-5b03f6113331';
      const resolvedMedia = new Map([
        [`/api/content/${uuid}`, {
          mimeType: 'image/png',
          base64: 'iVBOR...',
        }],
      ]);
      const req = googleAdapter.buildRequest(makeParams({
        messages: [
          { role: 'assistant', content: `![Generated image](/api/content/${uuid})` },
          { role: 'user', content: 'make it blue' },
        ],
        resolvedMedia,
      }));
      const body = JSON.parse(req.body);
      const modelTurn = body.contents.find((c: { role: string }) => c.role === 'model');
      const imgPart = modelTurn.parts.find((p: Record<string, unknown>) => p.inlineData);
      expect(imgPart).toBeUndefined();
      const textFallback = modelTurn.parts.find(
        (p: Record<string, unknown>) => p.text === '[Previously generated image]',
      );
      expect(textFallback).toBeDefined();
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
  });
});
