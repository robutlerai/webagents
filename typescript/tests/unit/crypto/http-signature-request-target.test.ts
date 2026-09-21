/**
 * Cross-language vectors for the REQUEST TARGET (2026-09-19):
 * `python/tests/fixtures/web_bot_auth/vectors-request-target.json`.
 *
 * The first vector file signs one ordinary URL, so it never noticed that the
 * two SDKs read `@path` and `@query` differently. This signer takes them from
 * the WHATWG `URL` parser (`pathname`, `search`), which is also how the
 * platform rebuilds them (`new URL(request.url)`, lib/auth/agent-auth.ts step
 * 10); the Python signer took them from `urlsplit` or `httpx`'s `raw_path`
 * untouched, so a dot-segment path signed `/a/../b` there and `/b` here, and
 * the platform verified `/b`. This file pins the cases the first one lacks:
 * dot segments (plain and `%2e`-spelled), a body with a query string, a query
 * that needs percent-encoding, an empty query and a backslash path.
 *
 * Both SDKs regenerate the WHOLE document from its inputs and compare it with
 * the file byte for byte; neither overwrites it (the Python twin is
 * `python/tests/crypto/test_request_target_vectors.py`). The top-level
 * `source` sentence is prose and is read back from the file. The three older
 * vector files are untouched: the portal reads them.
 */

import { describe, it, expect } from 'vitest';
import { createPublicKey, verify as cryptoVerify } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importJWK, type KeyLike } from 'jose';
import { AgentIdentity } from '../../../src/crypto/identity';
import { signMessage, signRequest, type SignatureAgentForm } from '../../../src/crypto/http-signature';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VECTORS_PATH = path.resolve(HERE, '../../../../python/tests/fixtures/web_bot_auth/vectors-request-target.json');

const B14 = {
  name: 'rfc9421-B.1.4',
  publicJwk: { kty: 'OKP', crv: 'Ed25519', x: 'JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs' },
  privateJwk: {
    kty: 'OKP',
    crv: 'Ed25519',
    x: 'JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs',
    d: 'n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU',
  },
  thumbprint: 'poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U',
};
const AGENT_URL = 'https://agent.example/agents/mini';
const CREATED = 1758067200;
const EXPIRES = 1758067260;
const FIXED_NONCE = Buffer.from(Array.from({ length: 64 }, (_, i) => i)).toString('base64');

/** `[id, method, url, body]`, in file order. ASCII only: `JSON.stringify` and `json.dumps` differ on anything else. */
const CASES: ReadonlyArray<readonly [string, string, string, string]> = [
  ['dot-segments', 'GET', 'https://robutler.ai/api/./agents/x/../mini/feed', ''],
  ['encoded-dot-segments', 'GET', 'https://robutler.ai/api/agents/%2e%2E/agents/mini/%2e/feed/..', ''],
  ['body-and-query', 'POST', 'https://robutler.ai/api/mpp/credits?amount=500&note=a%20b', '{"amount":500}'],
  ['query-encoding', 'GET', 'https://Robutler.AI:443/api/search?q=a b\'c"d<e>&k=%7e', ''],
  ['dot-segments-body-and-query', 'POST', 'https://robutler.ai/api/agents/../mpp/./credits?x=1', '{}'],
  ['empty-query', 'GET', 'https://robutler.ai/api/feed?', ''],
  ['backslash-path', 'GET', 'https://robutler.ai\\api\\feed\\..\\me', ''],
];

/** What the WHATWG parser makes of each case: `[path, query, url]`. */
const EXPECTED_TARGETS: Record<string, readonly [string, string, string]> = {
  'dot-segments': ['/api/agents/mini/feed', '?', 'https://robutler.ai/api/agents/mini/feed'],
  'encoded-dot-segments': ['/api/agents/mini/', '?', 'https://robutler.ai/api/agents/mini/'],
  'body-and-query': ['/api/mpp/credits', '?amount=500&note=a%20b', 'https://robutler.ai/api/mpp/credits?amount=500&note=a%20b'],
  'query-encoding': ['/api/search', '?q=a%20b%27c%22d%3Ce%3E&k=%7e', 'https://robutler.ai/api/search?q=a%20b%27c%22d%3Ce%3E&k=%7e'],
  'dot-segments-body-and-query': ['/api/mpp/credits', '?x=1', 'https://robutler.ai/api/mpp/credits?x=1'],
  'empty-query': ['/api/feed', '?', 'https://robutler.ai/api/feed?'],
  'backslash-path': ['/api/me', '?', 'https://robutler.ai/api/me'],
};

interface VectorCase {
  id: string;
  request: { method: string; url: string; body: string };
  target: { authority: string; path: string; query: string; url: string };
  headers: { 'content-digest'?: string; 'signature-agent': string; 'signature-input': string; signature: string };
  signatureBaseLines: string[];
}

interface VectorFile {
  source: string;
  key: typeof B14;
  agentUrl: string;
  form: SignatureAgentForm;
  params: { label: string; created: number; expires: number; nonce: string; alg: string; tag: string };
  cases: VectorCase[];
}

async function identity(): Promise<AgentIdentity> {
  const id = new AgentIdentity({
    agentId: 'mini',
    issuer: AGENT_URL,
    privateKey: (await importJWK(B14.privateJwk, 'EdDSA')) as KeyLike,
    publicKey: (await importJWK(B14.publicJwk, 'EdDSA')) as KeyLike,
  });
  await id.initialize();
  return id;
}

function verifies(base: string, signatureHeader: string, label: string): boolean {
  const m = new RegExp(`(?:^|, )${label}=:([A-Za-z0-9+/=]+):`).exec(signatureHeader);
  if (!m) return false;
  const key = createPublicKey({ key: B14.publicJwk as never, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(base, 'ascii'), key, new Uint8Array(Buffer.from(m[1], 'base64')));
}

async function buildVectors(source: string): Promise<VectorFile> {
  const id = await identity();
  const cases: VectorCase[] = [];
  for (const [caseId, method, url, body] of CASES) {
    const signed = await signMessage(
      id,
      { method, url, body: new TextEncoder().encode(body) },
      { form: 'dictionary-typed', label: 'sig1', created: CREATED, lifetimeSeconds: EXPIRES - CREATED, nonce: FIXED_NONCE },
    );
    const parsed = new URL(url);
    const digest = signed.headers['content-digest'];
    cases.push({
      id: caseId,
      request: { method, url, body },
      target: {
        authority: parsed.host,
        path: parsed.pathname,
        query: parsed.search === '' ? '?' : parsed.search,
        // What `fetch` puts on the wire: the serialised URL without its fragment.
        url: parsed.href.split('#')[0],
      },
      headers: {
        ...(digest ? { 'content-digest': digest } : {}),
        'signature-agent': signed.headers['signature-agent'],
        'signature-input': signed.headers['signature-input'],
        signature: signed.headers.signature,
      },
      signatureBaseLines: signed.labels[0].base.split('\n'),
    });
  }
  return {
    source,
    key: B14,
    agentUrl: AGENT_URL,
    form: 'dictionary-typed',
    params: { label: 'sig1', created: CREATED, expires: EXPIRES, nonce: FIXED_NONCE, alg: 'ed25519', tag: 'web-bot-auth' },
    cases,
  };
}

const serialize = (file: VectorFile) => `${JSON.stringify(file, null, 2)}\n`;

describe('request-target cross-language vectors', () => {
  it('signs the WHATWG target for every case, and the signature verifies over it', async () => {
    const built = await buildVectors('unused');
    expect(built.cases.map((c) => c.id)).toEqual(CASES.map((c) => c[0]));
    for (const c of built.cases) {
      const [pathname, query, url] = EXPECTED_TARGETS[c.id];
      expect(c.target, c.id).toEqual({ authority: 'robutler.ai', path: pathname, query, url });
      expect(c.signatureBaseLines[1], c.id).toBe('"@authority": robutler.ai');
      expect(c.signatureBaseLines[2], c.id).toBe(`"@path": ${pathname}`);
      expect(c.signatureBaseLines[3], c.id).toBe(`"@query": ${query}`);
      expect('content-digest' in c.headers, c.id).toBe(c.request.body.length > 0);
      expect(verifies(c.signatureBaseLines.join('\n'), c.headers.signature, 'sig1'), c.id).toBe(true);
    }
  });

  it('the URL sent is the URL signed: a Request carries the normalised spelling', async () => {
    const id = await identity();
    for (const [caseId, method, url, body] of CASES) {
      const signed = await signRequest(id, new Request(url, { method, body: body || undefined }), { created: CREATED, nonce: FIXED_NONCE });
      expect(signed.url, caseId).toBe(EXPECTED_TARGETS[caseId][2]);
    }
  });

  it('matches the committed file byte for byte, or writes it when no SDK has yet', async () => {
    if (!existsSync(VECTORS_PATH)) {
      mkdirSync(path.dirname(VECTORS_PATH), { recursive: true });
      writeFileSync(
        VECTORS_PATH,
        serialize(
          await buildVectors(
            'Request-target cross-language vectors, written 2026-09-19 by the TypeScript SDK signer ' +
              '(webagents/typescript/tests/unit/crypto/http-signature-request-target.test.ts). Inputs: the RFC 9421 ' +
              'Appendix B.1.4 Ed25519 test key (denylisted by the platform profile), created 1758067200, expires ' +
              '1758067260, nonce = bytes 0x00..0x3f in standard base64, agent URL https://agent.example/agents/mini, ' +
              'the dictionary-typed Signature-Agent form. Each case gives a method, a URL as a caller might spell it ' +
              'and a body; `target` is what the WHATWG URL parser makes of the URL, which is what both SDKs sign and ' +
              'send and what the platform rebuilds. Ed25519 is deterministic so the signature bytes are pinned too.',
          ),
        ),
      );
      console.log(`[http-signature-request-target.test] wrote ${VECTORS_PATH}`);
    }
    const committed = readFileSync(VECTORS_PATH, 'utf8');
    const source = (JSON.parse(committed) as VectorFile).source;
    expect(typeof source === 'string' && source.length > 0).toBe(true);
    const produced = serialize(await buildVectors(source));
    // Byte for byte: inputs, targets, headers, base lines and the layout itself.
    expect(produced).toBe(committed);

    // Independently of this signer: every committed signature verifies under the committed key over the committed base.
    for (const c of (JSON.parse(committed) as VectorFile).cases) {
      expect(verifies(c.signatureBaseLines.join('\n'), c.headers.signature, 'sig1'), c.id).toBe(true);
    }
  });
});
