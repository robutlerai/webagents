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

    it('strips Gemini-unsupported schema fields (additionalProperties, allOf, not, const, $ref)', () => {
      // Gemini's `parameters` Schema dialect rejects these fields with
      // HTTP 400; we keep them in the source schema for OpenAI/Anthropic
      // strict mode but must strip them on the wire. Regression test for
      // the 400 the live MEMORY_TOOL hit on
      // generativelanguage.googleapis.com/v1beta.
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 'memory',
            description: 'memory',
            parameters: {
              type: 'object',
              additionalProperties: false,
              properties: {
                command: { type: 'string', enum: ['view', 'create'] },
                path: { type: 'string', pattern: '^/memories(/.*)?$', maxLength: 256 },
                content: { type: 'string' },
                nested: {
                  type: 'object',
                  additionalProperties: true,
                  properties: { x: { type: 'string' } },
                },
              },
              required: ['command'],
              allOf: [{ type: 'string' }],
              not: { type: 'string' },
              $ref: '#/definitions/Foo',
              $defs: { Foo: { type: 'string' } },
              definitions: { Bar: { type: 'string' } },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const params = body.tools[0].function_declarations[0].parameters;

      for (const banned of ['additionalProperties', 'allOf', 'not', 'const', '$ref', '$defs', 'definitions']) {
        expect(params).not.toHaveProperty(banned);
      }
      expect(params.properties.nested).not.toHaveProperty('additionalProperties');

      expect(params.type).toBe('object');
      expect(params.required).toEqual(['command']);
      expect(params.properties.command.enum).toEqual(['view', 'create']);
      expect(params.properties.path.pattern).toBe('^/memories(/.*)?$');
      expect(params.properties.path.maxLength).toBe(256);
    });

    it('keeps anyOf and strips its siblings when no useful object/array shape is present', () => {
      // Per generativelanguage.googleapis.com/v1beta validator: when a
      // node has `anyOf` and NO useful object/array shape (no type:object,
      // no properties, no items), keep the union and strip cosmetic
      // siblings (description, title, etc).
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 'github_review',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                value: {
                  description: 'either a string or a number',
                  title: 'Value',
                  anyOf: [{ type: 'string' }, { type: 'number' }],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const value = body.tools[0].function_declarations[0].parameters.properties.value;

      expect(Object.keys(value)).toEqual(['anyOf']);
      expect(value.anyOf).toHaveLength(2);
      expect(value.anyOf[0]).toEqual({ type: 'string' });
    });

    it('drops the union (not the parent) when a node has BOTH a useful object shape AND a union', () => {
      // Regression for the path-less `INVALID_ARGUMENT` we hit on
      // MEMORY_TOOL: the top-level `parameters` declares
      //   type:'object', properties:{…}, required:['command'], oneOf:[…]
      // Stripping all siblings around `oneOf` would leave bare
      //   { anyOf: [...] }
      // with no top-level type — Gemini rejects that with no path. The
      // sanitizer must instead drop the union and keep the typed parent.
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 'memory',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                command: { type: 'string', enum: ['view', 'create'] },
                path: { type: 'string' },
                content: { type: 'string' },
              },
              required: ['command'],
              oneOf: [
                { properties: { command: { const: 'view' } } },
                { properties: { command: { const: 'create' } }, required: ['command', 'path', 'content'] },
              ],
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const params = body.tools[0].function_declarations[0].parameters;

      expect(params.type).toBe('object');
      expect(params).not.toHaveProperty('oneOf');
      expect(params).not.toHaveProperty('anyOf');
      expect(params.required).toEqual(['command']);
      expect(params.properties.command.enum).toEqual(['view', 'create']);
      expect(params.properties.path.type).toBe('string');
      expect(params.properties.content.type).toBe('string');
    });

    it('drops the union (not the parent) when a node has a useful array shape AND a union', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                comments: {
                  type: 'array',
                  description: 'List of review comments',
                  items: { type: 'string' },
                  anyOf: [
                    { type: 'string' },
                    { type: 'array', items: { type: 'string' } },
                  ],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const comments = body.tools[0].function_declarations[0].parameters.properties.comments;

      expect(comments.type).toBe('array');
      expect(comments.items).toEqual({ type: 'string' });
      expect(comments).not.toHaveProperty('anyOf');
      expect(comments).not.toHaveProperty('oneOf');
    });

    it('rewrites oneOf as anyOf (Gemini interprets them identically) and strips siblings', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                v: {
                  type: 'string',
                  description: 'union',
                  oneOf: [{ type: 'string' }, { type: 'integer' }],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const v = body.tools[0].function_declarations[0].parameters.properties.v;

      expect(Object.keys(v)).toEqual(['anyOf']);
      expect(v).not.toHaveProperty('oneOf');
      expect(v.anyOf).toHaveLength(2);
    });

    it('infers type:"object" on subschemas that have properties/required but no explicit type', () => {
      // Regression for the live 400 from generativelanguage.googleapis.com:
      //   "parameters.any_of[0].properties: only allowed for OBJECT type"
      // Gemini's validator rejects `properties` unless `type:object` is set.
      // Verifies type-inference inside an `anyOf` branch where the parent
      // node has no useful object shape, so the union itself is preserved.
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                payload: {
                  anyOf: [
                    { properties: { command: { type: 'string' } } },
                    { properties: { command: { type: 'string' }, path: { type: 'string' } }, required: ['command', 'path'] },
                  ],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const payload = body.tools[0].function_declarations[0].parameters.properties.payload;

      expect(payload.anyOf).toHaveLength(2);
      for (const branch of payload.anyOf) {
        expect(branch.type).toBe('object');
      }
      expect(payload.anyOf[1].required).toEqual(['command', 'path']);
    });

    it('drops `required` entries that are not declared in the same node\'s `properties`', () => {
      // Regression for the live 400 from generativelanguage.googleapis.com:
      //   "parameters.any_of[2].required[1]: property is not defined"
      // JSON Schema allows discriminated-union branches to reference
      // properties declared on the parent; Gemini's validator does not.
      // Wrap the union inside a property so the parent doesn't trigger
      // the "drop union to keep parent" rule, letting us inspect the
      // branch's `required`.
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                payload: {
                  anyOf: [
                    {
                      type: 'object',
                      properties: { command: { type: 'string' } },
                      required: ['command', 'path', 'content'],
                    },
                  ],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const branch = body.tools[0].function_declarations[0].parameters.properties.payload.anyOf[0];
      expect(branch.required).toEqual(['command']);
    });

    it('drops `required` entirely when no `properties` are declared at the same node', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                payload: {
                  anyOf: [{ required: ['x'] }],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const branch = body.tools[0].function_declarations[0].parameters.properties.payload.anyOf[0];
      expect(branch).not.toHaveProperty('required');
    });

    it('rewrites `const: X` to `enum: [X]` with `type: "string"` (Gemini does not support const)', () => {
      // Regression for the second path-less `INVALID_ARGUMENT` on
      // MEMORY_TOOL: a discriminator like `command: { const: 'view' }`
      // has its `const` stripped (rule 2) and would collapse to the
      // empty schema `command: {}` — making `command` look typeless and
      // tripping a generic INVALID_ARGUMENT. Rewriting to the equivalent
      // single-element string `enum` preserves the discriminator and
      // gives Gemini a concrete type.
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                command: { const: 'view' },
                code: { const: 42 },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const props = body.tools[0].function_declarations[0].parameters.properties;

      expect(props.command).toEqual({ enum: ['view'], type: 'string' });
      expect(props.command).not.toHaveProperty('const');
      expect(props.code.enum).toEqual(['42']);
      expect(props.code.type).toBe('string');
    });

    it('preserves an existing explicit `type` when rewriting `const`', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                v: { type: 'string', const: 'view' },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const v = body.tools[0].function_declarations[0].parameters.properties.v;
      expect(v).toEqual({ type: 'string', enum: ['view'] });
    });

    it('infers type:"array" on subschemas that have items but no explicit type', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                tags: { items: { type: 'string' } },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const tags = body.tools[0].function_declarations[0].parameters.properties.tags;
      expect(tags.type).toBe('array');
    });

    it('rewrites `const` to single-element `enum` inside an anyOf branch (const is unsupported there too)', () => {
      const req = googleAdapter.buildRequest(makeParams({
        tools: [{
          type: 'function',
          function: {
            name: 't',
            description: '',
            parameters: {
              type: 'object',
              properties: {
                cmd: {
                  anyOf: [
                    { type: 'string', const: 'view' },
                    { type: 'string', const: 'create' },
                  ],
                },
              },
            },
          },
        }],
      }));
      const body = JSON.parse(req.body);
      const cmd = body.tools[0].function_declarations[0].parameters.properties.cmd;

      expect(cmd.anyOf[0]).toEqual({ type: 'string', enum: ['view'] });
      expect(cmd.anyOf[1]).toEqual({ type: 'string', enum: ['create'] });
      for (const branch of cmd.anyOf) {
        expect(branch).not.toHaveProperty('const');
      }
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
          kind: 'binary' as const,
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
          kind: 'binary' as const,
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
          kind: 'binary' as const,
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
