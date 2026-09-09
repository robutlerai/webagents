/**
 * WebAgentsServer (Multi-Agent) Integration Tests
 *
 * Tests the Hono-based multi-agent server's transport and AOAuth endpoints.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { WebAgentsServer } from '../../src/server/multi.js';
import { BaseAgent } from '../../src/core/agent.js';
import { Skill } from '../../src/core/skill.js';
import { handoff } from '../../src/core/decorators.js';
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
 * The credential floor (src/server/credential-floor.ts) is installed on
 * `WebAgentsServer` as a Hono middleware in `createApp` (src/server/multi.ts),
 * upstream of `routeToAgent`, so an anonymous POST to `/agents/<name>/uamp`,
 * `/uamp/stream`, `/chat/completions` or `/v1/chat/completions` is refused
 * from the request line alone before any route runs. The endpoint cases in
 * this file are about what those routes DO once a caller is let in, so they
 * present a credential rather than assert a world where a billable model
 * endpoint answers anyone who can reach the port.
 *
 * The floor checks presence, not validity (`hasCredential`), and the fixture
 * agent carries no AuthSkill, so a dummy bearer is the honest instrument. The
 * value is the one the Python suite uses (python/tests/server/conftest.py,
 * AUTHED_HEADERS), so the two suites agree on what "authenticated enough"
 * looks like. The floor itself is asserted anonymously in the describe at the
 * bottom of this file and exhaustively in tests/unit/server/billable-routes.test.ts.
 */
const AUTHED_HEADERS = {
  'Content-Type': 'application/json',
  Authorization: 'Bearer test-service-token',
};

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

async function makeRequest(app: ReturnType<WebAgentsServer['getApp']>, path: string, init?: RequestInit) {
  const req = new Request(`http://localhost${path}`, init);
  return app.fetch(req);
}

describe('WebAgentsServer', () => {
  let server: WebAgentsServer;
  let agent: BaseAgent;

  beforeEach(async () => {
    server = new WebAgentsServer({
      port: 0,
      identity: {
        publicUrl: 'https://agents.example.com',
      },
    });

    agent = new BaseAgent({
      name: 'echo',
      description: 'An echo agent',
      skills: [new EchoLLM()],
    });

    await server.addAgent('echo', agent);
  });

  describe('global endpoints', () => {
    it('GET /health returns agent list', async () => {
      const res = await makeRequest(server.getApp(), '/health');
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe('ok');
      expect(body.agents).toHaveLength(1);
      expect(body.agents[0].name).toBe('echo');
    });

    it('GET /agents lists registered agents', async () => {
      const res = await makeRequest(server.getApp(), '/agents');
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.agents).toHaveLength(1);
      expect(body.agents[0].name).toBe('echo');
      expect(body.agents[0].mountPath).toBe('/agents/echo');
    });
  });

  describe('per-agent transport endpoints', () => {
    it('GET /agents/echo/ returns agent health', async () => {
      const res = await makeRequest(server.getApp(), '/agents/echo/');
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.agent).toBe('echo');
    });

    it('GET /agents/echo/info returns agent info', async () => {
      const res = await makeRequest(server.getApp(), '/agents/echo/info');
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.name).toBe('echo');
      expect(body.capabilities).toBeDefined();
    });

    it('POST /agents/echo/uamp processes UAMP events', async () => {
      const events = [
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('Test'),
        createResponseCreateEvent(),
      ];
      const res = await makeRequest(server.getApp(), '/agents/echo/uamp', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify(events),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(Array.isArray(body)).toBe(true);
      expect(body.some((e: { type: string }) => e.type === 'response.done')).toBe(true);
    });

    it('POST /agents/echo/uamp/stream returns SSE', async () => {
      const events = [
        createSessionCreateEvent({ modalities: ['text'] }),
        createInputTextEvent('Streaming'),
        createResponseCreateEvent(),
      ];
      const res = await makeRequest(server.getApp(), '/agents/echo/uamp/stream', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify(events),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get('Content-Type')).toBe('text/event-stream');

      const text = await res.text();
      expect(text).toContain('data:');
    });

    it('POST /agents/echo/chat/completions returns completion', async () => {
      const res = await makeRequest(server.getApp(), '/agents/echo/chat/completions', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'Hello' }],
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices).toBeDefined();
      expect(body.choices[0].message.role).toBe('assistant');
    });

    it('POST /agents/echo/v1/chat/completions also works', async () => {
      const res = await makeRequest(server.getApp(), '/agents/echo/v1/chat/completions', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'Hello' }],
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices).toBeDefined();
    });
  });

  describe('A2A agent card', () => {
    it('GET /.well-known/agent.json returns agent card', async () => {
      const { A2ATransportSkill } = await import('../../src/skills/transport/a2a/skill.js');
      const a2aServer = new WebAgentsServer({ port: 0, logging: false });
      const a2aAgent = new BaseAgent({
        name: 'echo',
        description: 'An echo agent',
        skills: [new EchoLLM(), new A2ATransportSkill()],
      });
      await a2aServer.addAgent('echo', a2aAgent);

      const res = await makeRequest(a2aServer.getApp(), '/agents/echo/.well-known/agent.json');
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.name).toBe('echo');
      expect(body.description).toBe('An echo agent');
      expect(body.capabilities.streaming).toBe(true);
    });
  });

  describe('AOAuth endpoints', () => {
    it('GET /.well-known/jwks.json returns JWKS', async () => {
      const res = await makeRequest(server.getApp(), '/agents/echo/.well-known/jwks.json');
      expect(res.status).toBe(200);
      expect(res.headers.get('Cache-Control')).toContain('public');

      const body = await res.json();
      expect(body.keys).toBeDefined();
      expect(body.keys).toHaveLength(1);
      expect(body.keys[0].kty).toBe('OKP');
      expect(body.keys[0].crv).toBe('Ed25519');
      expect(body.keys[0].kid).toBe('echo');
      expect(body.keys[0].alg).toBe('EdDSA');
    });

    it('GET /.well-known/openid-configuration returns discovery doc', async () => {
      const res = await makeRequest(server.getApp(), '/agents/echo/.well-known/openid-configuration');
      expect(res.status).toBe(200);

      const body = await res.json();
      expect(body.issuer).toBe('https://agents.example.com/agents/echo');
      expect(body.jwks_uri).toBe('https://agents.example.com/agents/echo/.well-known/jwks.json');
      expect(body.grant_types_supported).toContain('client_credentials');
    });

    it('identity can mint verifiable tokens', async () => {
      const identity = server.getIdentity('echo');
      expect(identity).toBeDefined();

      const token = await identity!.mintToken('https://target.com', 'read write');
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3);
    });
  });

  describe('AOAuth disabled', () => {
    it('returns 404 for JWKS when identity not configured', async () => {
      const noIdServer = new WebAgentsServer({ port: 0 });
      const a = new BaseAgent({
        name: 'plain',
        description: 'No AOAuth',
        skills: [new EchoLLM()],
      });
      await noIdServer.addAgent('plain', a);

      const res = await makeRequest(noIdServer.getApp(), '/agents/plain/.well-known/jwks.json');
      expect(res.status).toBe(404);
    });
  });

  describe('unknown agent', () => {
    it('returns 404 for non-existent agent', async () => {
      const res = await makeRequest(server.getApp(), '/agents/unknown/health');
      expect(res.status).toBe(404);
    });
  });

  describe('agent management', () => {
    it('removeAgent removes identity too', async () => {
      expect(server.getIdentity('echo')).toBeDefined();
      server.removeAgent('echo');
      expect(server.getIdentity('echo')).toBeUndefined();
      expect(server.getAgent('echo')).toBeUndefined();
    });
  });

  describe('transport skill routing', () => {
    it('routeToAgent dispatches to httpRegistry before hardcoded routes', async () => {
      const { CompletionsTransportSkill } = await import('../../src/skills/transport/completions/skill.js');

      const transportServer = new WebAgentsServer({ port: 0, logging: false });
      const transportAgent = new BaseAgent({
        name: 'transport-agent',
        skills: [new EchoLLM(), new CompletionsTransportSkill()],
      });
      await transportServer.addAgent('transport', transportAgent);

      // The CompletionsTransportSkill registers /v1/chat/completions
      const res = await makeRequest(transportServer.getApp(), '/agents/transport/v1/chat/completions', {
        method: 'POST',
        headers: AUTHED_HEADERS,
        body: JSON.stringify({
          model: 'test',
          messages: [{ role: 'user', content: 'test' }],
        }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.choices).toBeDefined();
    });

    it('agents with transport skills expose all registered endpoints', async () => {
      const { A2ATransportSkill } = await import('../../src/skills/transport/a2a/skill.js');
      const { CompletionsTransportSkill } = await import('../../src/skills/transport/completions/skill.js');

      const transportServer = new WebAgentsServer({ port: 0, logging: false });
      const transportAgent = new BaseAgent({
        name: 'full-transport',
        skills: [new EchoLLM(), new CompletionsTransportSkill(), new A2ATransportSkill()],
      });
      await transportServer.addAgent('full', transportAgent);

      // A2A agent card
      const cardRes = await makeRequest(transportServer.getApp(), '/agents/full/.well-known/agent.json');
      expect(cardRes.status).toBe(200);
      const card = await cardRes.json();
      expect(card.name).toBe('full-transport');

      // /v1/models via CompletionsTransportSkill
      const modelsRes = await makeRequest(transportServer.getApp(), '/agents/full/v1/models');
      expect(modelsRes.status).toBe(200);
    });
  });

  // Every endpoint case above now presents a credential, which on its own is
  // indistinguishable from the floor having been removed. This block is the
  // witness: the same routes, anonymous, through the real Hono app, must be
  // refused, and the public GETs must not be. billable-routes.test.ts walks the
  // route tables exhaustively; this is the one whole-server check in the
  // integration suite for this class.
  describe('the credential floor on WebAgentsServer', () => {
    const billable = [
      '/agents/echo/uamp',
      '/agents/echo/uamp/stream',
      '/agents/echo/chat/completions',
      '/agents/echo/v1/chat/completions',
    ];
    const uampBody = JSON.stringify([
      createSessionCreateEvent({ modalities: ['text'] }),
      createInputTextEvent('anonymous'),
      createResponseCreateEvent(),
    ]);
    const completionsBody = JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }] });
    const bodyFor = (path: string) => (path.includes('/uamp') ? uampBody : completionsBody);

    it('refuses an anonymous POST to every billable route with 401', async () => {
      for (const path of billable) {
        const res = await makeRequest(server.getApp(), path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: bodyFor(path),
        });
        expect(res.status, `POST ${path}`).toBe(401);
        const body = await res.json();
        expect(body.error.code, `POST ${path}`).toBe('unauthorized');
      }
    });

    it('lets the same routes through once a credential is present', async () => {
      // Guard the guard: a typo'd path would 404 and never prove the 401
      // above was the floor talking.
      for (const path of billable) {
        const res = await makeRequest(server.getApp(), path, {
          method: 'POST',
          headers: AUTHED_HEADERS,
          body: bodyFor(path),
        });
        expect(res.status, `POST ${path}`).toBe(200);
      }
    });

    it('is not blanket: the public GETs stay anonymous', async () => {
      for (const path of ['/health', '/agents', '/agents/echo/', '/agents/echo/info']) {
        const res = await makeRequest(server.getApp(), path);
        expect(res.status, `GET ${path}`).toBe(200);
      }
    });
  });
});
