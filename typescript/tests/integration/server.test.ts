/**
 * Server Integration Tests
 * 
 * Tests the HTTP/WebSocket server with agents.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { tool, handoff, http } from '../../src/core/decorators.js';
import { createFetchHandler } from '../../src/server/handler.js';
import { AgentIdentity } from '../../src/crypto/identity.js';
import type { Context } from '../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../src/uamp/events.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
  createSessionCreateEvent,
  createInputTextEvent,
  createResponseCreateEvent,
} from '../../src/uamp/events.js';

describe('Server Integration', () => {
  class EchoLLM extends Skill {
    @handoff({ name: 'echo-llm' })
    async *processUAMP(events: ClientEvent[], _ctx: Context): AsyncGenerator<ServerEvent> {
      const texts: string[] = [];
      for (const e of events) {
        if (e.type === 'input.text') {
          texts.push((e as { text: string }).text);
        }
      }
      
      const response = `Echo: ${texts.join(', ')}`;
      yield createResponseDeltaEvent('r1', { type: 'text', text: response });
      yield createResponseDoneEvent('r1', [{ type: 'text', text: response }]);
    }
  }

  class ToolsSkill extends Skill {
    @tool({ description: 'Add two numbers' })
    async add(params: { a: number; b: number }, _ctx: Context) {
      return params.a + params.b;
    }
  }

  class CustomAPISkill extends Skill {
    @http({ path: '/custom/endpoint', method: 'POST' })
    async customEndpoint(req: Request, _ctx: Context): Promise<Response> {
      const body = await req.json();
      return new Response(JSON.stringify({ custom: true, received: body }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }
  }

  let agent: BaseAgent;
  let handler: (req: Request) => Promise<Response>;

  beforeEach(() => {
    agent = new BaseAgent({
      name: 'test-server-agent',
      description: 'Agent for testing server',
      skills: [new EchoLLM(), new ToolsSkill(), new CustomAPISkill()],
    });
    handler = createFetchHandler(agent);
  });

  describe('health endpoint', () => {
    it('returns healthy status', async () => {
      const response = await handler(new Request('http://localhost/health'));
      
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.status).toBe('healthy');
    });
  });

  describe('info endpoint', () => {
    it('returns agent info', async () => {
      const response = await handler(new Request('http://localhost/info'));
      
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.name).toBe('test-server-agent');
      expect(body.description).toBe('Agent for testing server');
    });

    it('includes capabilities', async () => {
      const response = await handler(new Request('http://localhost/info'));
      const body = await response.json();
      
      expect(body.capabilities).toBeDefined();
      expect(body.capabilities.modalities).toContain('text');
    });
  });

  describe('UAMP endpoint', () => {
    it('processes UAMP events via POST', async () => {
      const events = [
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('Hello'),
        createResponseCreateEvent(),
      ];

      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(events),
      }));

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(Array.isArray(body)).toBe(true);
      expect(body.length).toBeGreaterThan(0);
    });

    it('returns UAMP response events', async () => {
      const events = [
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('Test message'),
        createResponseCreateEvent(),
      ];

      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(events),
      }));

      const body = await response.json();
      
      // Should have delta and done events
      const hasResponseDone = body.some(
        (e: { type: string }) => e.type === 'response.done'
      );
      expect(hasResponseDone).toBe(true);
    });

    it('handles invalid JSON gracefully', async () => {
      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not valid json',
      }));

      expect(response.status).toBe(500);
    });
  });

  describe('UAMP streaming endpoint', () => {
    it('streams UAMP events via SSE', async () => {
      const events = [
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('Stream test'),
        createResponseCreateEvent(),
      ];

      const response = await handler(new Request('http://localhost/uamp/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(events),
      }));

      expect(response.status).toBe(200);
      expect(response.headers.get('Content-Type')).toBe('text/event-stream');

      // Read the stream
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let result = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value);
      }

      expect(result).toContain('data:');
    });
  });

  describe('custom HTTP endpoints', () => {
    it('routes to skill HTTP handlers', async () => {
      const response = await handler(new Request('http://localhost/custom/endpoint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test: 'data' }),
      }));

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.custom).toBe(true);
      expect(body.received.test).toBe('data');
    });

    it('returns 404 for unknown endpoints', async () => {
      const response = await handler(new Request('http://localhost/unknown'));
      expect(response.status).toBe(404);
    });
  });

  describe('CORS handling', () => {
    it('handles OPTIONS preflight requests', async () => {
      const response = await handler(new Request('http://localhost/health', {
        method: 'OPTIONS',
      }));

      // OPTIONS should return a successful response with CORS headers
      expect(response.ok).toBe(true);
      expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
      expect(response.headers.get('Access-Control-Allow-Methods')).toContain('POST');
    });

    it('includes CORS headers on regular responses', async () => {
      const response = await handler(new Request('http://localhost/health'));
      
      expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
    });
  });

  describe('error handling', () => {
    it('returns 500 for invalid JSON', async () => {
      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not valid json',
      }));

      expect(response.status).toBe(500);
    });

    it('returns error details in response', async () => {
      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not valid json',
      }));

      const body = await response.json();
      expect(body.error).toBeDefined();
    });
  });

  describe('content negotiation', () => {
    it('returns JSON for info endpoint', async () => {
      const response = await handler(new Request('http://localhost/info'));
      expect(response.headers.get('Content-Type')).toBe('application/json');
    });

    it('returns SSE for streaming endpoint', async () => {
      const events = [
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('test'),
        createResponseCreateEvent(),
      ];

      const response = await handler(new Request('http://localhost/uamp/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ events }),
      }));

      expect(response.headers.get('Content-Type')).toBe('text/event-stream');
    });
  });

  describe('request routing', () => {
    it('routes GET /health', async () => {
      const response = await handler(new Request('http://localhost/health'));
      expect(response.status).toBe(200);
    });

    it('routes GET /info', async () => {
      const response = await handler(new Request('http://localhost/info'));
      expect(response.status).toBe(200);
    });

    it('routes POST /uamp', async () => {
      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([]),
      }));
      expect(response.status).toBe(200);
    });

    it('routes POST /uamp/stream', async () => {
      const response = await handler(new Request('http://localhost/uamp/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([]),
      }));
      expect(response.status).toBe(200);
    });
  });

  describe('.well-known/agent.json (A2A agent card)', () => {
    it('returns agent card with name and description', async () => {
      const response = await handler(new Request('http://localhost/.well-known/agent.json'));
      expect(response.status).toBe(200);

      const body = await response.json();
      expect(body.name).toBe('test-server-agent');
      expect(body.description).toBe('Agent for testing server');
      expect(body.url).toBe('http://localhost');
      expect(body.capabilities.streaming).toBe(true);
    });

    it('includes authentication schemes', async () => {
      const response = await handler(new Request('http://localhost/.well-known/agent.json'));
      const body = await response.json();
      expect(body.authentication).toBeDefined();
      expect(body.authentication.schemes).toContain('Bearer');
    });

    it('lists agent skills/tools', async () => {
      const response = await handler(new Request('http://localhost/.well-known/agent.json'));
      const body = await response.json();
      expect(Array.isArray(body.skills)).toBe(true);
      const addTool = body.skills.find((s: { id: string }) => s.id === 'add');
      expect(addTool).toBeDefined();
    });
  });

  describe('.well-known/jwks.json (AOAuth)', () => {
    it('returns 404 when identity not configured', async () => {
      const response = await handler(new Request('http://localhost/.well-known/jwks.json'));
      expect(response.status).toBe(404);
    });

    it('returns JWKS when identity is configured', async () => {
      const identity = new AgentIdentity({
        agentId: 'test-server-agent',
        issuer: 'http://localhost',
      });
      await identity.initialize();
      const handlerWithId = createFetchHandler(agent, { identity });
      const response = await handlerWithId(new Request('http://localhost/.well-known/jwks.json'));

      expect(response.status).toBe(200);
      expect(response.headers.get('Cache-Control')).toContain('public');

      const body = await response.json();
      expect(body.keys).toBeDefined();
      expect(body.keys.length).toBe(1);
      expect(body.keys[0].kty).toBe('OKP');
      expect(body.keys[0].crv).toBe('Ed25519');
      expect(body.keys[0].kid).toBe('test-server-agent');
    });
  });

  describe('.well-known/openid-configuration (AOAuth discovery)', () => {
    it('returns 404 when identity not configured', async () => {
      const response = await handler(new Request('http://localhost/.well-known/openid-configuration'));
      expect(response.status).toBe(404);
    });

    it('returns discovery document when configured', async () => {
      const identity = new AgentIdentity({
        agentId: 'test-server-agent',
        issuer: 'http://localhost',
      });
      await identity.initialize();
      const handlerWithId = createFetchHandler(agent, { identity });
      const response = await handlerWithId(new Request('http://localhost/.well-known/openid-configuration'));

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.issuer).toBe('http://localhost');
      expect(body.jwks_uri).toBe('http://localhost/.well-known/jwks.json');
      expect(body.grant_types_supported).toContain('client_credentials');
    });
  });

  describe('chat/completions endpoint', () => {
    it('returns non-streaming completion', async () => {
      const response = await handler(new Request('http://localhost/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'Hello' }],
          stream: false,
        }),
      }));

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices).toBeDefined();
      expect(body.choices[0].message.role).toBe('assistant');
      expect(body.choices[0].finish_reason).toBe('stop');
      expect(body.object).toBe('chat.completion');
    });

    it('also responds at /v1/chat/completions', async () => {
      const response = await handler(new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'Hello' }],
          stream: false,
        }),
      }));

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices).toBeDefined();
    });

    it('handles invalid JSON gracefully', async () => {
      const response = await handler(new Request('http://localhost/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not json',
      }));

      expect(response.status).toBe(500);
      const body = await response.json();
      expect(body.error).toBeDefined();
    });
  });
});
