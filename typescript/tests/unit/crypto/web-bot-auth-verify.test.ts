/**
 * The inbound Web Bot Auth verifier (ADR-0045 section 3).
 *
 * `python/tests/fixtures/web_bot_auth/verify-cases.json` holds signed requests
 * and the outcome both SDK verifiers must reach; the Python suite runs the same
 * file (tests/crypto/test_web_bot_auth_verify.py). Then: this SDK's own signer
 * round-trips through the verifier, the vectors the portal verifies build the
 * same base here, and the key-set fetcher's refusals.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { readFileSync } from 'node:fs';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  KeySetFetcher,
  MemoryNonceStore,
  parseKeySet,
  verifyWebBotAuth,
  type Discovery,
  type KeySetOutcome,
} from '../../../src/crypto/web-bot-auth-verify';
import { parseDictionary, parseItem, StructuredFieldParseError } from '../../../src/crypto/structured-fields';
import { signMessage } from '../../../src/crypto/http-signature';
import { AgentIdentity } from '../../../src/crypto/identity';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/web_bot_auth');
const CASES = JSON.parse(readFileSync(path.join(FIXTURES, 'verify-cases.json'), 'utf8')) as {
  authority: string;
  scheme: 'https';
  signerKey: Record<string, unknown>;
  cases: Array<{
    name: string;
    request: { method: string; target: string; headers: Record<string, string>; body: string };
    now: number;
    keys?: Array<Record<string, unknown>>;
    repeat?: number;
    expect: { ok: boolean; principal?: string; thumbprints?: string[]; identifier?: string; code?: string; description?: string };
  }>;
};

class StubKeySets {
  constructor(private readonly keys: Array<Record<string, unknown>>) {}
  async get(discovery: Discovery): Promise<KeySetOutcome> {
    const parsed = await parseKeySet({ keys: this.keys }, { wellKnownDirectory: discovery.type === 'directory' });
    return parsed.ok ? { ok: true, keys: parsed.keys, ttlS: 300 } : { ok: false, code: 'key_set_invalid', reason: parsed.reason };
  }
}

describe('the shared cases', () => {
  for (const c of CASES.cases) {
    it(c.name, async () => {
      const nonces = new MemoryNonceStore();
      let outcome: Awaited<ReturnType<typeof verifyWebBotAuth>> | undefined;
      for (let i = 0; i < (c.repeat ?? 1); i += 1) {
        outcome = await verifyWebBotAuth(
          {
            method: c.request.method,
            target: c.request.target,
            headers: c.request.headers,
            body: Uint8Array.from(atob(c.request.body), (ch) => ch.charCodeAt(0)),
          },
          {
            authorities: [CASES.authority],
            scheme: CASES.scheme,
            keySets: new StubKeySets(c.keys ?? [CASES.signerKey]),
            nonces,
            now: () => c.now,
          },
        );
      }
      if (c.expect.ok) {
        expect(outcome!.ok ? undefined : outcome!.refusal).toBeUndefined();
        if (outcome!.ok) {
          expect(outcome!.agent.principal).toBe(c.expect.principal);
          expect(outcome!.agent.thumbprints).toEqual(c.expect.thumbprints);
          expect(outcome!.agent.identifier).toBe(c.expect.identifier);
        }
      } else {
        expect(outcome!.ok).toBe(false);
        if (!outcome!.ok) {
          expect(outcome!.refusal.code).toBe(c.expect.code);
          expect(outcome!.refusal.description).toBe(c.expect.description);
        }
      }
    });
  }
});

describe('round trips', () => {
  it("this SDK's signer verifies", async () => {
    const identity = new AgentIdentity({ agentId: 'scout', issuer: 'https://caller.example/agents/scout' });
    await identity.initialize();
    const body = new TextEncoder().encode('{}');
    const signed = await signMessage(identity, { method: 'POST', url: 'https://agent.example/agents/mini/chat/completions?stream=1', body });
    const jwks = identity.getJwks() as unknown as { keys: Array<Record<string, unknown>> };
    const outcome = await verifyWebBotAuth(
      { method: 'POST', target: '/agents/mini/chat/completions?stream=1', headers: { host: 'agent.example', ...signed.headers }, body },
      { authorities: ['agent.example'], scheme: 'https', keySets: new StubKeySets(jwks.keys), nonces: new MemoryNonceStore() },
    );
    expect(outcome.ok ? 'ok' : outcome.refusal).toBe('ok');
    if (outcome.ok) expect(outcome.agent.principal).toBe('https://caller.example/agents/scout');
  });

  it('the vectors the portal verifies verify here', async () => {
    const vectors = JSON.parse(readFileSync(path.join(FIXTURES, 'vectors.json'), 'utf8'));
    const request = vectors.request;
    for (const vector of vectors.vectors) {
      const headers: Record<string, string> = { host: request.authority };
      for (const [k, v] of Object.entries(vector.headers as Record<string, string>)) headers[k.toLowerCase()] = v;
      const key = { thumbprint: vectors.key.thumbprint, x: vectors.key.publicJwk.x };
      const outcome = await verifyWebBotAuth(
        {
          method: request.method,
          target: request.path + (request.query !== '?' ? request.query : ''),
          headers,
          body: typeof request.body === 'string' ? new TextEncoder().encode(request.body) : undefined,
        },
        {
          authorities: [request.authority],
          scheme: 'https',
          keySets: { get: async () => ({ ok: true, keys: [key], ttlS: 300 }) },
          nonces: new MemoryNonceStore(),
          now: () => vectors.params.created,
        },
      );
      expect(outcome.ok ? 'ok' : `${vector.id}: ${outcome.refusal.code}`).toBe('ok');
    }
  });
});

describe('structured fields', () => {
  it('a dictionary with an inner list', () => {
    const [[label, member]] = parseDictionary('sig1=("@method" "signature-agent";key="sig1");created=1;tag="web-bot-auth"');
    expect(label).toBe('sig1');
    expect('items' in member && member.items.map((i) => i.value)).toEqual(['@method', 'signature-agent']);
  });

  for (const text of [
    'a=1, a=2',
    'a=1;x;x',
    'a=1.5',
    'a=@1700000000',
    'a=%"x"',
    'a=:YWI:',
    'a=:YWJjZA:',
    'a=1,',
    'A=1',
    'a="\u0001"',
    'a=(1 2',
    `a=${'1'.repeat(16)}`,
    Array.from({ length: 17 }, (_, i) => `k${i}=1`).join(', '),
    `a=${'x'.repeat(9000)}`,
  ]) {
    it(`refuses ${JSON.stringify(text.slice(0, 40))}`, () => {
      expect(() => parseDictionary(text)).toThrow(StructuredFieldParseError);
    });
  }

  it('a legacy string', () => {
    expect(parseItem('"https://caller.example"').value).toBe('https://caller.example');
  });
});

// -- the key-set fetcher, against a local server ----------------------------------------------

let server: http.Server;
let port = 0;
let thumbprint = '';

beforeAll(async () => {
  const identity = new AgentIdentity({ agentId: 'scout', issuer: 'https://caller.example/agents/scout' });
  await identity.initialize();
  const keySet = identity.getJwks();
  thumbprint = keySet.keys[0].kid;
  server = http.createServer((req, res) => {
    const send = (status: number, body: string | Buffer, type: string, extra: Record<string, string> = {}): void => {
      const bytes = typeof body === 'string' ? Buffer.from(body) : body;
      res.writeHead(status, { 'Content-Type': type, 'Content-Length': String(bytes.length), ...extra });
      res.end(bytes);
    };
    switch (req.url) {
      case '/agents/scout/.well-known/jwks.json':
        return send(200, JSON.stringify(keySet), 'application/json', { 'Cache-Control': 'max-age=10' });
      case '/redirect/.well-known/jwks.json':
        return send(302, '', 'text/plain', { Location: '/agents/scout/.well-known/jwks.json' });
      case '/big/.well-known/jwks.json':
        return send(200, Buffer.alloc(70 * 1024, 'x'), 'application/json');
      case '/notjson/.well-known/jwks.json':
        return send(200, '<html>', 'text/html');
      case '/.well-known/http-message-signatures-directory':
        return send(200, JSON.stringify(keySet), 'application/json');
      default:
        return send(404, 'no', 'text/plain');
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  port = (server.address() as AddressInfo).port;
});

afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function jwks(prefix: string): Discovery {
  const url = `http://127.0.0.1:${port}${prefix}/.well-known/jwks.json`;
  return { type: 'jwks_uri', principal: url.split('/.well-known')[0], identifier: url, fetchUrl: url };
}

describe('the key-set fetcher', () => {
  it('fetches and parses, the TTL clamped up to the floor', async () => {
    const outcome = await new KeySetFetcher({ allowPrivate: true }).get(jwks('/agents/scout'));
    expect(outcome.ok && outcome.keys[0].thumbprint).toBe(thumbprint);
    expect(outcome.ok && outcome.ttlS).toBe(300);
  });

  it('private addresses need the switch', async () => {
    expect(await new KeySetFetcher().get(jwks('/agents/scout'))).toEqual({
      ok: false,
      code: 'key_set_unreachable',
      reason: '127.0.0.1 is not a public address, so it is not called.',
    });
  });

  it('no redirects, a size cap, JSON only', async () => {
    const fetcher = new KeySetFetcher({ allowPrivate: true });
    expect(await fetcher.get(jwks('/redirect'))).toEqual({
      ok: false,
      code: 'key_set_unreachable',
      reason: 'it answered with a redirect, which is not followed',
    });
    expect(await fetcher.get(jwks('/big'))).toEqual({ ok: false, code: 'key_set_invalid', reason: 'it is larger than 64 KiB' });
    expect(await fetcher.get(jwks('/notjson'))).toEqual({ ok: false, code: 'key_set_invalid', reason: 'it is not JSON' });
  });

  it('a directory needs its media type', async () => {
    const origin = `http://127.0.0.1:${port}`;
    const url = `${origin}/.well-known/http-message-signatures-directory`;
    const outcome = await new KeySetFetcher({ allowPrivate: true }).get({
      type: 'directory',
      principal: origin,
      identifier: url,
      fetchUrl: url,
      mediaType: 'application/http-message-signatures-directory+json',
    });
    expect(outcome).toEqual({
      ok: false,
      code: 'key_set_invalid',
      reason: 'a directory must be served as application/http-message-signatures-directory+json',
    });
  });

  it('the RFC test key poisons the set', async () => {
    const vectors = JSON.parse(readFileSync(path.join(FIXTURES, 'vectors.json'), 'utf8'));
    expect(await parseKeySet({ keys: [vectors.key.publicJwk] }, { wellKnownDirectory: false })).toEqual({
      ok: false,
      reason: 'it carries a known test key (RFC 9421 Appendix B.1)',
    });
  });
});
