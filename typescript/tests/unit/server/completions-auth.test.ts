/**
 * `POST {basePath}/chat/completions` — authentication and sender attribution.
 *
 * Two things are pinned here, both through a REAL request against the fetch
 * handler rather than a hand-built context:
 *
 *  1. the endpoint is not anonymous. It runs the agent's model on the owner's
 *     credit, so a request with no credential is refused before the model is
 *     reached, and a credential the AuthSkill refuses is a 401 (not a 500, and
 *     not a 200 with the refusal buried in an SSE body).
 *
 *  2. the request body's `metadata.sender` reaches `context.metadata`. That is
 *     the ONLY thing that attributes a platform-routed turn to a person: the
 *     service token names the router, so without the sender every
 *     `scope: 'owner'` tool is unreachable on every routed call.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SignJWT, exportJWK, generateKeyPair, createLocalJWKSet } from 'jose';
import { BaseAgent } from '../../../src/core/agent.js';
import { Skill } from '../../../src/core/skill.js';
import { handoff } from '../../../src/core/decorators.js';
import { AuthSkill } from '../../../src/skills/auth/skill.js';
import { JWKSManager } from '../../../src/crypto/jwks.js';
import { AuthScope } from '../../../src/core/types.js';
import type { AuthInfo, Context } from '../../../src/core/types.js';
import type { ClientEvent, ServerEvent } from '../../../src/uamp/events.js';
import {
  createResponseDeltaEvent,
  createResponseDoneEvent,
  generateEventId,
} from '../../../src/uamp/events.js';
import { createFetchHandler } from '../../../src/server/handler.js';

const PLATFORM = 'https://robutler.ai';
const OWNER_ID = 'owner-user-1';

let privateKey: CryptoKey;

/** Capture what the run's context looked like by the time the LLM was reached. */
class ProbeLLM extends Skill {
  seenAuth: AuthInfo | undefined;
  seenMetadata: Record<string, unknown> | undefined;

  @handoff({ name: 'probe-llm' })
  async *processUAMP(_events: ClientEvent[], context: Context): AsyncGenerator<ServerEvent> {
    this.seenAuth = context.auth;
    this.seenMetadata = context.metadata as Record<string, unknown>;
    const responseId = generateEventId();
    yield { type: 'response.created', event_id: generateEventId(), response_id: responseId } as ServerEvent;
    yield createResponseDeltaEvent(responseId, { type: 'text', text: 'ok' });
    yield createResponseDoneEvent(responseId, [{ type: 'text', text: 'ok' }]);
  }
}

async function signServiceToken(): Promise<string> {
  return new SignJWT({ scopes: ['agents:*'] })
    .setProtectedHeader({ alg: 'RS256', kid: 'test-sig-key' })
    .setIssuer(PLATFORM)
    .setSubject('service:robutler-router')
    .setIssuedAt()
    .setExpirationTime('1h')
    .sign(privateKey);
}

function completionsRequest(
  headers: Record<string, string>,
  body: Record<string, unknown>,
): Request {
  return new Request('https://agent.example.com/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
}

describe('chat/completions auth', () => {
  let probe: ProbeLLM;
  let handler: (request: Request) => Promise<Response>;

  beforeEach(async () => {
    const kp = await generateKeyPair('RS256');
    privateKey = kp.privateKey;
    const pub = await exportJWK(kp.publicKey);
    const publicJwk = { ...pub, kid: 'test-sig-key', use: 'sig', alg: 'RS256' };

    const jwksManager = new JWKSManager({
      platformApiUrl: PLATFORM,
      platformIssuer: PLATFORM,
    });
    const localJwks = createLocalJWKSet({ keys: [publicJwk] as never });
    (jwksManager as unknown as { jwksCache: Map<string, unknown> }).jwksCache.set(
      `${PLATFORM}/.well-known/jwks.json`,
      localJwks,
    );

    // No api-key validation service in a unit test: every /api/auth/validate-key
    // call fails, so auth falls through to the service-token path.
    vi.stubGlobal('fetch', async () => new Response('nope', { status: 401 }));

    probe = new ProbeLLM();
    const agent = new BaseAgent({
      name: 'probe',
      skills: [probe, new AuthSkill({ jwksManager, requireAuth: true, ownerUserId: OWNER_ID })],
    });
    handler = createFetchHandler(agent);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('refuses a completions request that carries no credential at all', async () => {
    const res = await handler(
      completionsRequest({}, { messages: [{ role: 'user', content: 'Hi' }] }),
    );
    expect(res.status).toBe(401);
    expect(probe.seenAuth).toBeUndefined();
  });

  it('refuses a bearer that does not verify (401, not 500)', async () => {
    const res = await handler(
      completionsRequest(
        { authorization: 'Bearer not-a-real-token' },
        { messages: [{ role: 'user', content: 'Hi' }] },
      ),
    );
    expect(res.status).toBe(401);
    expect(probe.seenAuth?.authenticated).not.toBe(true);
  });

  it('refuses an unauthenticated STREAMING request before the body starts', async () => {
    const res = await handler(
      completionsRequest({}, { messages: [{ role: 'user', content: 'Hi' }], stream: true }),
    );
    expect(res.status).toBe(401);
    expect(res.headers.get('content-type')).toContain('application/json');
  });

  it('refuses an anonymous request with a malformed body as 401, not a parser 500', async () => {
    // Cross-SDK parity, and a real property in its own right: the credential
    // floor runs BEFORE `await request.json()`. While it ran after, an
    // anonymous caller could make the server parse an arbitrary body, and this
    // exact request came back 500 `completions_error` carrying the JSON
    // parser's message while Python answered 401 — same refusal, two different
    // observables for anyone probing the two SDKs.
    // The Python half asserts the same in
    // python/tests/server/test_completions_floor.py.
    const res = await handler(
      new Request('https://agent.example.com/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{ this is not json',
      }),
    );
    expect(res.status).toBe(401);
    expect(probe.seenAuth).toBeUndefined();
  });

  // The named twins of the Python `test_static_uamp_completions_refuses_an_
  // anonymous_caller`. `POST /uamp` and `/uamp/stream` call `agent.processUAMP`
  // — the same model, the same owner's credit as `/chat/completions` — and were
  // anonymous 200s until the floor moved above route dispatch. The enumerating
  // test in billable-routes.test.ts discovers these too; they are spelled out
  // here because a door that was actually found open deserves a case with its
  // name on it.
  it('refuses an anonymous POST /uamp (same model, different wire format)', async () => {
    const res = await handler(
      new Request('https://agent.example.com/uamp', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify([]),
      }),
    );
    expect(res.status).toBe(401);
    expect(probe.seenAuth).toBeUndefined();
  });

  it('refuses an anonymous POST /uamp/stream', async () => {
    const res = await handler(
      new Request('https://agent.example.com/uamp/stream', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify([]),
      }),
    );
    expect(res.status).toBe(401);
    expect(res.headers.get('content-type')).toContain('application/json');
    expect(probe.seenAuth).toBeUndefined();
  });

  it('attributes a platform service token to the sender in the request body', async () => {
    const token = await signServiceToken();
    const res = await handler(
      completionsRequest(
        { authorization: `Bearer ${token}` },
        {
          messages: [{ role: 'user', content: 'Hi' }],
          metadata: {
            chat_id: 'chat-1',
            platform: 'robutler',
            sender: { id: 'user-42', username: 'alice', account_type: 'user' },
          },
        },
      ),
    );

    expect(res.status).toBe(200);
    expect(probe.seenAuth?.authenticated).toBe(true);
    expect(probe.seenAuth?.user_id).toBe('user-42');
    expect(probe.seenAuth?.scope).toBe(AuthScope.USER);
    // The sender survived onto the context — this is the wiring that was dead.
    expect((probe.seenMetadata?.sender as { id?: string } | undefined)?.id).toBe('user-42');
    expect(probe.seenMetadata?.chatId).toBe('chat-1');
  });

  it('elevates to OWNER when the relayed sender is the agent owner', async () => {
    const token = await signServiceToken();
    const res = await handler(
      completionsRequest(
        { authorization: `Bearer ${token}` },
        {
          messages: [{ role: 'user', content: 'Hi' }],
          metadata: { sender: { id: OWNER_ID, username: 'owner', account_type: 'user' } },
        },
      ),
    );

    expect(res.status).toBe(200);
    expect(probe.seenAuth?.scope).toBe(AuthScope.OWNER);
    expect(probe.seenAuth?.user_id).toBe(OWNER_ID);
  });

  it('a request body cannot overwrite the credential header with its own metadata', async () => {
    const token = await signServiceToken();
    const res = await handler(
      completionsRequest(
        { authorization: `Bearer ${token}` },
        {
          messages: [{ role: 'user', content: 'Hi' }],
          metadata: {
            authorization: 'Bearer forged',
            sender: { id: OWNER_ID },
          },
        },
      ),
    );

    expect(res.status).toBe(200);
    expect(probe.seenMetadata?.authorization).toBe(`Bearer ${token}`);
  });
});
