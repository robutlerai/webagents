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
import { createAgentApp } from '../../src/server/node.js';
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

/**
 * What every real caller of a billable route presents.
 *
 * The credential floor (src/server/credential-floor.ts) refuses an anonymous
 * POST to `/uamp`, `/uamp/stream`, `/chat/completions` or
 * `/v1/chat/completions` from the request line alone, at the top of
 * `createFetchHandler` above route dispatch and above `await request.json()`.
 * The endpoint cases in this file are about what those routes DO once a
 * caller is let in, so they present a credential; they used to assert a world
 * where a billable model endpoint answers anyone who can reach the port, which
 * is the exact shape the floor was built to end.
 *
 * The floor checks presence, not validity (`hasCredential`), and the fixture
 * agent carries no AuthSkill, so a dummy bearer is the honest instrument: a
 * real signed token would assert nothing these cases are about and would
 * couple them to the crypto suite. The value is the one the Python suite uses
 * (python/tests/server/conftest.py, AUTHED_HEADERS), so the two suites do not
 * drift on what "authenticated enough" looks like. The floor's own behaviour
 * is asserted anonymously in the describe at the bottom of this file and
 * exhaustively in tests/unit/server/billable-routes.test.ts.
 */
const AUTHED_HEADERS = {
  'Content-Type': 'application/json',
  Authorization: 'Bearer test-service-token',
};

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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
        body: 'not valid json',
      }));

      expect(response.status).toBe(500);
    });

    it('returns error details in response', async () => {
      const response = await handler(new Request('http://localhost/uamp', {
        method: 'POST',
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
        body: JSON.stringify([]),
      }));
      expect(response.status).toBe(200);
    });

    it('routes POST /uamp/stream', async () => {
      const response = await handler(new Request('http://localhost/uamp/stream', {
        method: 'POST',
        headers: AUTHED_HEADERS,
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
      // resolveCardUrl (src/server/handler.ts): configured publicUrl, then
      // WEBAGENTS_PUBLIC_URL, then basePath as a RELATIVE reference, and the
      // basePath here is ''. The request origin was dropped on purpose: it is
      // a Host-header guess that is wrong behind a proxy or a tunnel, and the
      // Python SDK already answered relatively
      // (python/webagents/server/core/registration.py, resolve_public_base_url),
      // so the two SDKs disagreed until this changed.
      expect(body.url).toBe('/');
      expect(body.capabilities.streaming).toBe(true);
    });

    it('uses the configured publicUrl as the card url', async () => {
      // The tier a deployment actually uses.
      const configured = createFetchHandler(agent, { publicUrl: 'https://agent.example.com' });
      const response = await configured(new Request('http://localhost/.well-known/agent.json'));
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.url).toBe('https://agent.example.com');
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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
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
        headers: AUTHED_HEADERS,
        body: 'not json',
      }));

      expect(response.status).toBe(500);
      const body = await response.json();
      expect(body.error).toBeDefined();
    });
  });

  describe('transport skill endpoints', () => {
    it('httpRegistry endpoints from transport skills are mounted', async () => {
      const { CompletionsTransportSkill } = await import('../../src/skills/transport/completions/skill.js');
      const transportAgent = new BaseAgent({
        name: 'transport-test',
        skills: [new EchoLLM(), new CompletionsTransportSkill()],
      });

      const httpHandler = transportAgent.getHttpHandler('/v1/chat/completions', 'POST');
      expect(httpHandler).toBeDefined();

      const modelsHandler = transportAgent.getHttpHandler('/v1/models', 'GET');
      expect(modelsHandler).toBeDefined();
    });

    it('custom @http endpoints coexist with transport skill endpoints', async () => {
      const { CompletionsTransportSkill } = await import('../../src/skills/transport/completions/skill.js');
      const transportAgent = new BaseAgent({
        name: 'coexist-test',
        skills: [new EchoLLM(), new CustomAPISkill(), new CompletionsTransportSkill()],
      });

      expect(transportAgent.getHttpHandler('/custom/endpoint', 'POST')).toBeDefined();
      expect(transportAgent.getHttpHandler('/v1/chat/completions', 'POST')).toBeDefined();
    });

    it('agent with CompletionsTransportSkill responds to /v1/chat/completions', async () => {
      const { CompletionsTransportSkill } = await import('../../src/skills/transport/completions/skill.js');
      const transportAgent = new BaseAgent({
        name: 'completions-test',
        skills: [new EchoLLM(), new CompletionsTransportSkill()],
      });

      const httpHandler = transportAgent.getHttpHandler('/v1/chat/completions', 'POST');
      expect(httpHandler).toBeDefined();

      const { ContextImpl } = await import('../../src/core/context.js');
      const ctx = new ContextImpl();

      const req = new Request('http://localhost/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'test',
          messages: [{ role: 'user', content: 'Hello' }],
        }),
      });

      const response = await httpHandler!.handler(req, ctx);
      expect(response.status).toBe(200);

      const body = await response.json();
      expect(body.choices).toBeDefined();
      expect(body.choices[0].message.role).toBe('assistant');
    });

    it('agent with A2ATransportSkill responds to /a2a and /.well-known/agent.json', async () => {
      const { A2ATransportSkill } = await import('../../src/skills/transport/a2a/skill.js');
      const transportAgent = new BaseAgent({
        name: 'a2a-test',
        skills: [new EchoLLM(), new A2ATransportSkill()],
      });

      expect(transportAgent.getHttpHandler('/a2a', 'POST')).toBeDefined();
      expect(transportAgent.getHttpHandler('/.well-known/agent.json', 'GET')).toBeDefined();
    });
  });

  // Every billable case above now presents a credential, which on its own is
  // indistinguishable from the floor having been removed. This block is the
  // witness, through a whole server rather than the predicate: the same paths,
  // anonymous, must be refused, and the paths the floor must NOT touch must
  // stay open. It runs against both single-agent server classes,
  // `createFetchHandler` and `createAgentApp` (what `serve()` builds), because
  // each installs the floor at its own point and either can lose it alone.
  // WebAgentsServer, the third class, has the same block in
  // multi-server.test.ts. billable-routes.test.ts is the exhaustive walk over
  // the route tables; this is the integration suite's ability to notice a fifth
  // door on its own.
  describe('the credential floor, through the real handler', () => {
    const billable = ['/uamp', '/uamp/stream', '/chat/completions', '/v1/chat/completions'];
    const uampBody = JSON.stringify([
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('anonymous'),
      createResponseCreateEvent(),
    ]);
    const completionsBody = JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }] });
    const bodyFor = (path: string) => (path.startsWith('/uamp') ? uampBody : completionsBody);

    const servers: Array<[string, () => (req: Request) => Promise<Response>]> = [
      ['createFetchHandler', () => createFetchHandler(agent)],
      ['createAgentApp', () => {
        const { app } = createAgentApp(agent, { logging: false });
        return (req: Request) => Promise.resolve(app.fetch(req));
      }],
    ];

    for (const [label, build] of servers) {
      describe(label, () => {
        it('refuses an anonymous POST to every billable path with exactly 401', async () => {
          const fetchLike = build();
          for (const path of billable) {
            const response = await fetchLike(new Request(`http://localhost${path}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: bodyFor(path),
            }));
            expect(response.status, `POST ${path}`).toBe(401);
            const body = await response.json();
            expect(body.error.code, `POST ${path}`).toBe('unauthorized');
          }
        });

        it('treats a bare Bearer with nothing after it as no credential', async () => {
          const fetchLike = build();
          const response = await fetchLike(new Request('http://localhost/uamp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: 'Bearer' },
            body: uampBody,
          }));
          expect(response.status).toBe(401);
        });

        it('lets any of the three credential headers past the floor', async () => {
          // Presence only: the floor reads a header, it does not verify a token.
          // AuthSkill, when an agent has one, is what verifies.
          const fetchLike = build();
          for (const header of ['authorization', 'x-api-key', 'x-owner-assertion']) {
            const response = await fetchLike(new Request('http://localhost/uamp', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', [header]: 'dummy-credential' },
              body: uampBody,
            }));
            expect(response.status, header).not.toBe(401);
          }
        });

        it('refuses anonymous-plus-malformed with 401, never a parser 500', async () => {
          // The credentialed malformed cases above are the 500s. The floor
          // runs before the body is read, so an anonymous caller cannot make
          // the server parse arbitrary bytes and observes the same 401 on both
          // SDKs rather than a parser error on one of them.
          const fetchLike = build();
          for (const path of ['/uamp', '/chat/completions']) {
            const response = await fetchLike(new Request(`http://localhost${path}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: 'not valid json',
            }));
            expect(response.status, `POST ${path}`).toBe(401);
          }
        });

        it('is not blanket: a skill @http POST and the public GETs stay anonymous', async () => {
          const fetchLike = build();
          const custom = await fetchLike(new Request('http://localhost/custom/endpoint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ test: 'data' }),
          }));
          expect(custom.status).toBe(200);

          for (const path of ['/health', '/info']) {
            const response = await fetchLike(new Request(`http://localhost${path}`));
            expect(response.status, `GET ${path}`).toBe(200);
          }
        });
      });
    }
  });
});
