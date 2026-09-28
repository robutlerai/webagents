/**
 * The TrustFlow lookup client (plan item 2.5, 2026-09-26): which route, which
 * query, which credential, what is held and for how long, and the sentence
 * for every failure, as the shared contract pins them
 * (`python/tests/fixtures/trust/trust_tool_definition.json`; Python
 * tests/trustflow/test_trust_lookup_w2trust.py).
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  NO_TRUST_CREDENTIAL,
  TRUST_CACHE_TTL_MS,
  TRUST_LOOKUP_PATH,
  TRUST_RECORD_PATH,
  TrustLookup,
  TrustLookupError,
  platformCredentialFor,
  trustFailureMessage,
} from '../../../src/trustflow/trust-lookup';
import { AgentIdentity } from '../../../src/crypto/identity';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/trust/trust_tool_definition.json'), 'utf8'));
const PORTAL = 'https://portal.test';
const SCOUT = 'https://bot.acme.com/agents/scout';

function response(status: number, body: unknown): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('the contract', () => {
  it('names the routes, the query keys and the cache TTL the fixture pins', () => {
    expect(TRUST_LOOKUP_PATH).toBe(FIXTURE.routes.lookup);
    expect(TRUST_RECORD_PATH).toBe(FIXTURE.routes.record);
    expect(TRUST_CACHE_TTL_MS).toBe(FIXTURE.cache_ttl_ms);
    expect(NO_TRUST_CREDENTIAL).toBe(FIXTURE.no_credential);
  });

  it('says the fixture sentences for every failure', () => {
    expect(trustFailureMessage('unreachable', '@x')).toBe(FIXTURE.messages.unreachable.replace('{agent}', '@x'));
    expect(trustFailureMessage('not_found', '@x')).toBe(FIXTURE.messages.not_found.replace('{agent}', '@x'));
    expect(trustFailureMessage('refused', '@x', 503)).toBe(FIXTURE.messages.refused.replace('{agent}', '@x').replace('{status}', '503'));
    expect(trustFailureMessage('unreadable', '@x')).toBe(FIXTURE.messages.unreadable.replace('{agent}', '@x'));
    expect(trustFailureMessage('no_credential', '@x')).toBe(FIXTURE.no_credential);
  });
});

describe('lookup', () => {
  it('asks the lookup route for the agent and topic, as a bearer, and returns the answer unchanged', async () => {
    const fetch = vi.fn(async () => response(200, FIXTURE.lookup_response));
    const client = new TrustLookup({ platformUrl: PORTAL, credential: { kind: 'bearer', key: 'k1' }, fetch });
    const result = await client.lookup(SCOUT, 'billing');
    expect(result).toEqual(FIXTURE.lookup_response);
    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${PORTAL}${TRUST_LOOKUP_PATH}?agent=${encodeURIComponent(SCOUT)}&topic=billing`);
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer k1');
    expect(init.method).toBe('GET');
  });

  it('leaves the topic out of the query when none is asked', async () => {
    const fetch = vi.fn(async () => response(200, FIXTURE.lookup_response));
    const client = new TrustLookup({ platformUrl: PORTAL, credential: { kind: 'bearer', key: 'k' }, fetch });
    await client.lookup('@scout');
    expect((fetch.mock.calls[0] as unknown as [string])[0]).toBe(`${PORTAL}${TRUST_LOOKUP_PATH}?agent=%40scout`);
  });

  it('holds an answer for the TTL, per agent and topic', async () => {
    const clock = { t: 1_000_000 };
    const fetch = vi.fn(async () => response(200, FIXTURE.lookup_response));
    const client = new TrustLookup({ platformUrl: PORTAL, credential: { kind: 'bearer', key: 'k' }, fetch, now: () => clock.t });
    await client.lookup(SCOUT, 'billing');
    await client.lookup(SCOUT, 'billing');
    expect(fetch).toHaveBeenCalledTimes(1);
    await client.lookup(SCOUT);
    expect(fetch).toHaveBeenCalledTimes(2);
    clock.t += TRUST_CACHE_TTL_MS + 1;
    await client.lookup(SCOUT, 'billing');
    expect(fetch).toHaveBeenCalledTimes(3);
    client.clearCache();
    await client.lookup(SCOUT, 'billing');
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it('signs the request with an identity that can sign, and sends no bearer beside it', async () => {
    const identity = new AgentIdentity({ agentId: 'scout', issuer: SCOUT });
    await identity.initialize();
    const seen: Request[] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      seen.push(input as Request);
      return response(200, FIXTURE.lookup_response);
    }) as typeof fetch;
    const client = new TrustLookup({ platformUrl: PORTAL, credential: platformCredentialFor({ identity }) });
    await client.lookup('@other');
    expect(seen).toHaveLength(1);
    expect(seen[0].headers.get('signature-input')).toBeTruthy();
    expect(seen[0].headers.get('authorization')).toBeNull();
    expect(seen[0].url).toBe(`${PORTAL}${TRUST_LOOKUP_PATH}?agent=%40other`);
  });

  it('is refused up front with the fixture sentence when there is no credential; nothing is dialled', async () => {
    const fetch = vi.fn();
    const client = new TrustLookup({ platformUrl: PORTAL, fetch });
    const error = await client.lookup(SCOUT).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TrustLookupError);
    expect((error as TrustLookupError).code).toBe('no_credential');
    expect((error as Error).message).toBe(FIXTURE.no_credential);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('names each failure, and holds none of them', async () => {
    const cases: Array<[() => Promise<Response>, string, string]> = [
      [async () => response(404, { error: 'Agent not found' }), 'not_found', FIXTURE.messages.not_found.replace('{agent}', SCOUT)],
      [async () => response(503, {}), 'refused', FIXTURE.messages.refused.replace('{agent}', SCOUT).replace('{status}', '503')],
      [async () => { throw new TypeError('fetch failed'); }, 'unreachable', FIXTURE.messages.unreachable.replace('{agent}', SCOUT)],
      [async () => response(200, 'not json'), 'unreadable', FIXTURE.messages.unreadable.replace('{agent}', SCOUT)],
      [async () => response(200, { subject: {} }), 'unreadable', FIXTURE.messages.unreadable.replace('{agent}', SCOUT)],
    ];
    for (const [answer, code, message] of cases) {
      const fetch = vi.fn(answer);
      const client = new TrustLookup({ platformUrl: PORTAL, credential: { kind: 'bearer', key: 'k' }, fetch });
      const error = (await client.lookup(SCOUT).catch((e: unknown) => e)) as TrustLookupError;
      expect(error.code, code).toBe(code);
      expect(error.message).toBe(message);
      await client.lookup(SCOUT).catch(() => undefined);
      expect(fetch, `${code} is not held`).toHaveBeenCalledTimes(2);
    }
  });
});

describe('record', () => {
  it('asks the record route and holds the answer', async () => {
    const answer = { record: 'a.b.c', payload: { exp: 1 }, jwks_url: `${PORTAL}/.well-known/jwks.json` };
    const fetch = vi.fn(async () => response(200, answer));
    const client = new TrustLookup({ platformUrl: PORTAL, credential: { kind: 'bearer', key: 'k' }, fetch });
    expect(await client.record(SCOUT)).toEqual(answer);
    expect(await client.record(SCOUT)).toEqual(answer);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect((fetch.mock.calls[0] as unknown as [string])[0]).toBe(`${PORTAL}${TRUST_RECORD_PATH}?agent=${encodeURIComponent(SCOUT)}`);
  });

  it('an answer without a record is unreadable', async () => {
    const fetch = vi.fn(async () => response(200, { payload: {} }));
    const client = new TrustLookup({ platformUrl: PORTAL, credential: { kind: 'bearer', key: 'k' }, fetch });
    await expect(client.record(SCOUT)).rejects.toMatchObject({ code: 'unreadable' });
  });
});

describe('the credential rule', () => {
  it('an identity that can sign signs; a key alone is a bearer; neither is the refusal', async () => {
    const identity = new AgentIdentity({ agentId: 'scout', issuer: SCOUT });
    await identity.initialize();
    expect(platformCredentialFor({ identity, apiKey: 'k' })).toEqual({ kind: 'signature', identity });
    expect(platformCredentialFor({ apiKey: ' k ' })).toEqual({ kind: 'bearer', key: 'k' });
    expect(platformCredentialFor({})).toEqual({ kind: 'none', reason: NO_TRUST_CREDENTIAL });
    expect(platformCredentialFor(undefined)).toEqual({ kind: 'none', reason: NO_TRUST_CREDENTIAL });
  });

  it('an identity the platform cannot fetch keys from falls back to the key, or names the reason', async () => {
    const loopback = new AgentIdentity({ agentId: 'scout', issuer: 'http://127.0.0.1:3000/agents/scout' });
    await loopback.initialize();
    expect(platformCredentialFor({ identity: loopback, apiKey: 'k' })).toEqual({ kind: 'bearer', key: 'k' });
    const none = platformCredentialFor({ identity: loopback });
    expect(none.kind).toBe('none');
    expect((none as { reason: string }).reason).toContain('loopback');
  });
});
