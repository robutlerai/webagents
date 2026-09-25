/**
 * `examples/intent-discovery.ts` is EXECUTED here against a stub platform
 * (loopback only, no model call, no platform key), and the docs' snippet is
 * asserted to be generated from that exact file, so the documented way to
 * publish and search intents cannot rot (2026-09-23).
 *
 * What the stub checks is the wire: the publish and the search both arrive
 * SIGNED (RFC 9421, Web Bot Auth), verifiable under the key set the example's
 * own server publishes, and with no bearer anywhere. That is the whole point
 * of the page these snippets land on: intent discovery with a signing
 * identity and no platform key.
 *
 * The example imports from 'webagents' (the documented specifier); vi.mock
 * aliases it onto src/ so the module loads identically under the SDK's own
 * vitest config and the monorepo's.
 */

import { describe, it, expect, vi, afterAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createServer } from 'node:http';
import { createPublicKey, verify as cryptoVerify } from 'node:crypto';
import { tempDirs } from '../../../helpers/cli';

// Every folder a test here makes is removed after the file (tests/helpers/cli.ts).
const tempDir = tempDirs();

vi.mock('webagents', async () => await import('../../../../src/index'));

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.resolve(HERE, '../../../../examples');
const DOCS = path.resolve(HERE, '../../../../../docs');

// The example binds a real socket; port 0 keeps that to an ephemeral port.
// The key must not land in the developer's home directory either, and the
// card's `url` is the CONFIGURED public address, never the request host.
process.env.PORT = '0';
process.env.WEBAGENTS_KEYS_DIR = tempDir('webagents-discovery-example-keys-');
process.env.OPENAI_API_KEY ??= 'test-key-not-used';
process.env.WEBAGENTS_PUBLIC_URL = 'https://agent.example.com';
delete process.env.WEBAGENTS_API_KEY;
delete process.env.WEBAGENTS_AGENT_TOKEN;

const AGENT_URL = 'https://agent.example.com/agents/translator';
const KEY_SET = `${AGENT_URL}/.well-known/jwks.json`;

interface Seen {
  path: string;
  method: string;
  host: string;
  headers: Record<string, string | undefined>;
  body: string;
}

const closers: Array<() => Promise<void>> = [];
afterAll(async () => {
  for (const close of closers) await close().catch(() => {});
});

/** Rebuild the base as the platform does and verify `sig1` under the served key set. */
function verifiesWith(seen: Seen, keys: Array<{ kid: string; x: string }>): boolean {
  const input = seen.headers['signature-input']!;
  const keyid = /keyid="([^"]+)"/.exec(input)![1];
  const key = keys.find((k) => k.kid === keyid);
  if (!key) return false;
  const [, pathname, search = ''] = /^([^?]*)(\?.*)?$/.exec(seen.path)!;
  const lines = [
    `"@method": ${seen.method}`,
    `"@authority": ${seen.host}`,
    `"@path": ${pathname}`,
    `"@query": ${search || '?'}`,
    ...(seen.headers['content-digest'] ? [`"content-digest": ${seen.headers['content-digest']}`] : []),
    `"signature-agent";key="sig1": "${KEY_SET}";type=jwks_uri`,
    `"@signature-params": ${input.slice('sig1='.length)}`,
  ].join('\n');
  const signature = /^sig1=:([A-Za-z0-9+/=]+):$/.exec(seen.headers.signature!)![1];
  const publicKey = createPublicKey({ key: { kty: 'OKP', crv: 'Ed25519', x: key.x }, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(lines, 'ascii'), publicKey, Buffer.from(signature, 'base64'));
}

describe('intent-discovery example', () => {
  it('publishes and searches with signed requests the platform can verify, and no platform key', async () => {
    const seen: Seen[] = [];
    const stub = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (c: Buffer) => chunks.push(c));
      req.on('end', () => {
        const h = (name: string) => {
          const v = req.headers[name];
          return Array.isArray(v) ? v.join(', ') : v;
        };
        seen.push({
          path: req.url ?? '',
          method: req.method ?? '',
          host: h('host') ?? '',
          headers: {
            authorization: h('authorization'),
            'signature-agent': h('signature-agent'),
            'signature-input': h('signature-input'),
            signature: h('signature'),
            'content-digest': h('content-digest'),
          },
          body: Buffer.concat(chunks).toString('utf8'),
        });
        const route = (req.url ?? '').split('?')[0];
        const answer = (status: number, body: unknown) => {
          res.writeHead(status, { 'content-type': 'application/json' });
          res.end(JSON.stringify(body));
        };
        if (route === '/api/auth/cli/token') {
          return answer(200, { access_token: 'platform-token', user_id: 'user-42', username: 'com.example.agent.translator' });
        }
        if (route === '/api/intents/create') return answer(201, { results: [{ id: 'i-1' }, { id: 'i-2' }], count: 2 });
        if (route === '/api/intents/search') {
          return answer(200, {
            results: [
              {
                id: 'i-9',
                intent: 'translate legal documents into German',
                agentId: 'agent-9',
                description: 'Certified legal translation',
                rank: 0,
                url: 'https://legal.example.com/agents/jurist',
                protocol: 'completions',
                similarity: 0.87,
              },
            ],
          });
        }
        answer(404, { error: 'not found' });
      });
    });
    await new Promise<void>((resolve) => stub.listen(0, '127.0.0.1', () => resolve()));
    const stubPort = (stub.address() as { port: number }).port;
    process.env.ROBUTLER_API_URL = `http://127.0.0.1:${stubPort}`;
    closers.push(() => new Promise<void>((resolve) => stub.close(() => resolve())));

    const example = await import('../../../../examples/intent-discovery.ts');
    closers.push(() => example.server.close());

    // `serve()` handed the agent the identity it publishes, and the skill
    // reads it from there: no `identity` and no `apiKey` in the example.
    expect(example.agent.identity).toBe(example.server.identity);
    expect(example.server.identity.issuer).toBe(AGENT_URL);
    expect(example.discovery.credential()).toEqual({ kind: 'signature', identity: example.server.identity });

    // The key set the platform would fetch to verify what follows.
    const jwks = await fetch(`http://127.0.0.1:${example.server.port}/agents/translator/.well-known/jwks.json`);
    expect(jwks.status).toBe(200);
    const { keys } = (await jwks.json()) as { keys: Array<{ kid: string; x: string }> };

    // Registration happened first (the first signed request), then the publish.
    expect(example.registration.ok, example.registration.error).toBe(true);
    expect(example.published).toEqual({ ok: true, status: 201 });

    const publish = seen.find((r) => r.path === '/api/intents/create')!;
    expect(publish).toBeDefined();
    expect(publish.method).toBe('POST');
    expect(publish.headers.authorization).toBeUndefined();
    expect(publish.headers['signature-agent']).toBe(`sig1="${KEY_SET}";type=jwks_uri`);
    expect(JSON.parse(publish.body)).toMatchObject({
      intents: ['translate documents between English and German', 'proofread German business correspondence'],
      description: 'Translates and proofreads English and German text.',
    });
    expect(verifiesWith(publish, keys)).toBe(true);
    expect(seen.findIndex((r) => r.path === '/api/auth/cli/token')).toBeLessThan(seen.indexOf(publish));

    // The other side: the same identity, the same verification, and the
    // results as the platform shaped them.
    const found = await example.findAgentFor('translate a contract into German');
    const search = seen.find((r) => r.path === '/api/intents/search')!;
    expect(search).toBeDefined();
    expect(search.headers.authorization).toBeUndefined();
    expect(JSON.parse(search.body)).toEqual({ query: 'translate a contract into German', limit: 10 });
    expect(verifiesWith(search, keys)).toBe(true);
    expect(found.intents).toEqual([
      expect.objectContaining({ intent: 'translate legal documents into German', agentId: 'agent-9', similarity: 0.87 }),
    ]);

    // Nothing this example sent carried a bearer, except presence: the
    // heartbeat carries the bearer registration minted, handed over by
    // `registerWithPlatform` (2026-09-24). The platform refuses a heartbeat
    // with no credential header to a handful per address (a signed one would
    // make it fetch a key set), so presence cannot be signed-only yet. The
    // key the page says you do not need is never used.
    for (const request of seen) {
      if (request.path === '/api/agents/heartbeat') {
        expect(request.headers.authorization).toBe('Bearer platform-token');
        continue;
      }
      expect(request.headers.authorization, request.path).toBeUndefined();
    }
  }, 20000);

  it('intent-discovery.md carries the example file verbatim', () => {
    const text = readFileSync(path.join(EXAMPLES, 'intent-discovery.ts'), 'utf8');
    const m = text.match(/^\s*\/\*\*[\s\S]*?\*\/\s*\n/);
    const code = (m ? text.slice(m[0].length) : text).trim();
    const doc = readFileSync(path.join(DOCS, 'guides/intent-discovery.md'), 'utf8');
    expect(doc).toContain('<!-- BEGIN GENERATED: typescript/examples/intent-discovery.ts,python/examples/intent_discovery.py -->');
    expect(doc).toContain(code);
  });
});
