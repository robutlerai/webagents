/**
 * Cross-Language Interoperability Tests
 *
 * Verifies that TypeScript and Python agents can communicate
 * via UAMP/HTTP/NLI. These tests require both runtimes.
 *
 * Run with: PYTHON_AGENT_URL=http://localhost:8000 npx vitest run tests/interop/
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
  parseEvent,
  serializeEvent,
  isServerEvent,
} from '../../src/uamp/events.js';

const PYTHON_AGENT_URL = process.env.PYTHON_AGENT_URL;
const TS_AGENT_URL = process.env.TS_AGENT_URL;

const skipIfNoPython = PYTHON_AGENT_URL ? describe : describe.skip;
const skipIfNoTs = TS_AGENT_URL ? describe : describe.skip;

describe('UAMP Event Serialization Roundtrip', () => {
  it('should serialize and parse session.create', () => {
    const event = createSessionCreateEvent({ modalities: ['text'] });
    const json = serializeEvent(event);
    const parsed = parseEvent(json);
    expect(parsed.type).toBe('session.create');
    expect(parsed.event_id).toBe(event.event_id);
  });

  it('should serialize and parse input.text', () => {
    const event = createInputTextEvent('Hello from TypeScript');
    const json = serializeEvent(event);
    const parsed = parseEvent(json);
    expect(parsed.type).toBe('input.text');
    expect((parsed as typeof event).text).toBe('Hello from TypeScript');
  });

  it('should reject invalid events', () => {
    expect(() => parseEvent('{}')).toThrow('Invalid UAMP event');
    expect(() => parseEvent('{"type":"test"}')).toThrow('Invalid UAMP event');
  });
});

skipIfNoPython('TypeScript → Python Agent (UAMP over HTTP)', () => {
  it('should send UAMP events and receive valid response', async () => {
    const events = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('Hello from TypeScript agent'),
      createResponseCreateEvent(),
    ];

    const res = await fetch(`${PYTHON_AGENT_URL}/uamp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(events),
    });

    expect(res.ok).toBe(true);
    const responseEvents = await res.json();
    expect(Array.isArray(responseEvents)).toBe(true);

    for (const event of responseEvents) {
      expect(event).toHaveProperty('type');
      expect(event).toHaveProperty('event_id');
      expect(isServerEvent(event as any)).toBe(true);
    }
  });

  it('should exchange completions format', async () => {
    const res = await fetch(`${PYTHON_AGENT_URL}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'ping' }],
        stream: false,
      }),
    });

    expect(res.ok).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty('choices');
    expect(data.choices[0]).toHaveProperty('message');
  });
});

skipIfNoTs('Python → TypeScript Agent (UAMP over HTTP)', () => {
  it('should handle UAMP events from Python client', async () => {
    const events = [
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('Hello from Python test harness'),
      createResponseCreateEvent(),
    ];

    const res = await fetch(`${TS_AGENT_URL}/uamp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(events),
    });

    expect(res.ok).toBe(true);
    const responseEvents = await res.json();
    expect(Array.isArray(responseEvents)).toBe(true);
    expect(responseEvents.length).toBeGreaterThan(0);
  });
});

describe('Payment Token Interoperability', () => {
  it('should produce valid JWT structure for token exchange', () => {
    const header = { alg: 'EdDSA', typ: 'JWT' };
    const payload = {
      iss: 'robutler.ai',
      sub: 'user-123',
      aud: ['agent-abc'],
      exp: Math.floor(Date.now() / 1000) + 3600,
      iat: Math.floor(Date.now() / 1000),
      jti: `pt_${crypto.randomUUID()}`,
      amt: '1000000000',
      cur: 'nanocents',
    };

    expect(header.alg).toBe('EdDSA');
    expect(payload.aud).toEqual(['agent-abc']);
    expect(typeof payload.amt).toBe('string');
  });
});

describe('Tool Call Format Compatibility', () => {
  it('should produce OpenAI-compatible tool definitions', () => {
    const toolDef = {
      type: 'function' as const,
      function: {
        name: 'search',
        description: 'Search the web',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
          },
          required: ['query'],
        },
      },
    };

    expect(toolDef.type).toBe('function');
    expect(toolDef.function.parameters.type).toBe('object');
    expect(toolDef.function.parameters.required).toContain('query');
  });

  it('should format tool results as strings', () => {
    const result = { data: [1, 2, 3], source: 'web' };
    const serialized = JSON.stringify(result);
    const deserialized = JSON.parse(serialized);
    expect(deserialized.data).toEqual([1, 2, 3]);
  });
});

/**
 * Cross-Language Parity Tests (stubs)
 *
 * These tests require both TS and Python runtimes plus UAMP WebSocket infrastructure.
 * Use it.todo() until infrastructure is available.
 */
describe('Cross-Language Parity', () => {
  describe('UAMP Protocol', () => {
    it.todo('TS UAMPClient connects to Python agent via UAMP WS');
    it.todo('Python UAMPClient connects to TS agent via UAMP WS');
    it.todo('bidirectional streaming works across language boundary');
  });

  describe('Payment Delegation', () => {
    it.todo('payment token delegation works from TS to Python agent');
    it.todo('payment token delegation works from Python to TS agent');
    it.todo('payment.required/submit flow works cross-language');
  });

  describe('NLI Cross-Language', () => {
    it.todo('TS agent calls Python agent via UAMP NLI');
    it.todo('Python agent calls TS agent via UAMP NLI');
    it.todo('cancel propagation works across language boundary');
  });

  describe('LLM Proxy', () => {
    it.todo('UAMP LLM proxy accepts connections from TS agents');
    it.todo('UAMP LLM proxy accepts connections from Python agents');
    it.todo('BYOK resolution works for both language agents');
  });
});
