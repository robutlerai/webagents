/**
 * The signed UAMP upgrade, pinned across both SDKs (2026-09-18).
 *
 * WHY. Nothing signed a UAMP upgrade until the fix pass of 2026-09-18, so
 * the platform's socket door (`signedUpgradeEligible`,
 * lib/payments/machine-door-socket.ts) never saw a signed agent and never
 * sent an SDK client the in-band `mpp` entry. `MppBuyer.upgradeHeaders`
 * (TS) and `MppBuyer.upgrade_headers` (Python) now sign a GET on the
 * http(s) form of the socket URL, the request the door rebuilds from the
 * upgrade's Host header and path. The mapping (`wss:` to `https:`, `ws:`
 * to `http:`, host, port, path and query unchanged) and the covered hint
 * are new, so they get their own vector file rather than a change to
 * `vectors.json` or `vectors-covered-headers.json`, which stay
 * byte-identical for their consumers.
 *
 * THE VECTOR FILE. `webagents/python/tests/fixtures/web_bot_auth/
 * vectors-upgrade.json`. Same rule as the other two: whichever SDK lands
 * first writes it (this suite did, on 2026-09-18) and the other compares
 * byte for byte. Each case records the socket URL, the realm allowlist, the
 * hint the buyer covers, the http(s) request that is signed, and per
 * Signature-Agent form every header `upgradeHeaders` returns (lowercased
 * names) plus the signature base lines.
 */

import { describe, it, expect } from 'vitest';
import { createPublicKey, verify as cryptoVerify } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importJWK, type KeyLike } from 'jose';
import { AgentIdentity } from '../../../../src/crypto/identity';
import { SIGNATURE_AGENT_FORMS, signMessage, type SignatureAgentForm } from '../../../../src/crypto/http-signature';
import { MppBuyer, PAYMENT_METHODS_HINT_HEADER } from '../../../../src/skills/payments/mpp-buyer';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VECTORS_PATH = path.resolve(HERE, '../../../../../python/tests/fixtures/web_bot_auth/vectors-upgrade.json');

// RFC 9421 Appendix B.1.4, test-key-ed25519: denylisted by the platform, so its private half is safe beside a fixture.
const B14 = {
  name: 'rfc9421-B.1.4',
  publicJwk: { kty: 'OKP', crv: 'Ed25519', x: 'JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs' } as const,
  privateJwk: {
    kty: 'OKP',
    crv: 'Ed25519',
    x: 'JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs',
    d: 'n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU',
  } as const,
  thumbprint: 'poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U',
};

const AGENT_URL = 'https://agent.example/agents/mini';
const CREATED = 1758067200;
const EXPIRES = 1758067260;
const FIXED_NONCE = Buffer.from(Array.from({ length: 64 }, (_, i) => i)).toString('base64');

interface UpgradeCase {
  name: string;
  socketUrl: string;
  realms: string[];
  /** The one configured source, which decides the hint. */
  source: 'card' | 'stablecoin';
  hints: Record<string, string>;
  signedRequest: { method: 'GET'; url: string; authority: string; path: string; query: string };
  vectors: Array<{ form: SignatureAgentForm; label: string; headers: Record<string, string>; signatureBaseLines: string[] }>;
}

interface UpgradeVectorFile {
  source: string;
  key: typeof B14;
  agentUrl: string;
  params: { label: string; created: number; expires: number; nonce: string; alg: string; tag: string };
  cases: UpgradeCase[];
}

async function identity(): Promise<AgentIdentity> {
  const id = new AgentIdentity({
    agentId: 'mini',
    issuer: AGENT_URL,
    privateKey: (await importJWK(B14.privateJwk as never, 'EdDSA')) as KeyLike,
    publicKey: (await importJWK(B14.publicJwk as never, 'EdDSA')) as KeyLike,
  });
  await id.initialize();
  return id;
}

function lower(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]));
}

function verifies(base: string, signature: Uint8Array, jwk: { x: string }): boolean {
  const key = createPublicKey({ key: { kty: 'OKP', crv: 'Ed25519', x: jwk.x }, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(base, 'utf8'), key, Buffer.from(signature));
}

async function produce(c: Omit<UpgradeCase, 'vectors'>): Promise<UpgradeCase['vectors']> {
  const id = await identity();
  const out: UpgradeCase['vectors'] = [];
  for (const form of SIGNATURE_AGENT_FORMS) {
    const sign = { form, created: CREATED, nonce: FIXED_NONCE, allowHttp: true };
    const buyer = new MppBuyer({
      identity: id,
      sources: c.source === 'card' ? { card: { getSpt: async () => 'spt_x' } } : { stablecoin: { signTempoTransfer: async () => '0x7600' } },
      policy: { maxPerPurchaseCents: 1, acceptTerms: '2026-07-31', realms: c.realms },
      sign,
    });
    const headers = lower(await buyer.upgradeHeaders(c.socketUrl));
    const reference = await signMessage(
      id,
      { method: 'GET', url: c.signedRequest.url, headers: new Headers(c.hints) },
      { ...sign, coveredHeaders: Object.keys(c.hints) },
    );
    out.push({ form, label: 'sig1', headers, signatureBaseLines: reference.labels[0].base.split('\n') });
  }
  return out;
}

const CASES: Array<Omit<UpgradeCase, 'vectors'>> = [
  {
    name: 'wss to the platform, card-only buyer',
    socketUrl: 'wss://robutler.ai/agents/acme/uamp',
    realms: ['robutler.ai'],
    source: 'card',
    hints: { 'robutler-payment-methods': 'stripe' },
    signedRequest: { method: 'GET', url: 'https://robutler.ai/agents/acme/uamp', authority: 'robutler.ai', path: '/agents/acme/uamp', query: '?' },
  },
  {
    name: 'ws on a local port with a query, stablecoin-only buyer',
    socketUrl: 'ws://localhost:3000/agents/acme/uamp?token=abc',
    realms: ['localhost:3000'],
    source: 'stablecoin',
    hints: { 'robutler-payment-methods': 'tempo' },
    signedRequest: { method: 'GET', url: 'http://localhost:3000/agents/acme/uamp?token=abc', authority: 'localhost:3000', path: '/agents/acme/uamp', query: '?token=abc' },
  },
];

describe('cross-language vectors for the signed UAMP upgrade', () => {
  it('matches the committed vectors byte for byte, or writes them when no SDK has yet', async () => {
    if (!existsSync(VECTORS_PATH)) {
      const cases: UpgradeCase[] = [];
      for (const c of CASES) cases.push({ ...c, vectors: await produce(c) });
      const file: UpgradeVectorFile = {
        source:
          'Signed UAMP upgrade cross-language vectors, written 2026-09-18 by the TypeScript SDK ' +
          '(webagents/typescript/tests/unit/skills/payments/mpp-buyer-upgrade-vectors.test.ts). MppBuyer.upgradeHeaders ' +
          '(TS) and MppBuyer.upgrade_headers (Python) sign a GET on the http(s) form of the socket URL (wss: as https:, ' +
          'ws: as http:, host, port, path and query unchanged), which is the request the platform socket door rebuilds ' +
          'from the upgrade Host header and path, with the buyer method hint covered. Inputs: the RFC 9421 Appendix ' +
          'B.1.4 Ed25519 test key (denylisted by the platform profile), agent URL https://agent.example/agents/mini, ' +
          'created 1758067200, expires 1758067260, nonce = bytes 0x00..0x3f in standard base64. Per case and per ' +
          'Signature-Agent form: every header the buyer returns (names lowercased) and the signature base lines.',
        key: B14,
        agentUrl: AGENT_URL,
        params: { label: 'sig1', created: CREATED, expires: EXPIRES, nonce: FIXED_NONCE, alg: 'ed25519', tag: 'web-bot-auth' },
        cases,
      };
      mkdirSync(path.dirname(VECTORS_PATH), { recursive: true });
      writeFileSync(VECTORS_PATH, `${JSON.stringify(file, null, 2)}\n`);
      console.log(`[mpp-buyer-upgrade-vectors.test] wrote the cross-language vectors to ${VECTORS_PATH}`);
    }

    const committed = JSON.parse(readFileSync(VECTORS_PATH, 'utf8')) as UpgradeVectorFile;
    expect(committed.key.thumbprint).toBe(B14.thumbprint);
    expect(committed.cases.map((c) => c.name)).toEqual(CASES.map((c) => c.name));
    for (const theirs of committed.cases) {
      const mine = await produce(theirs);
      for (const vector of theirs.vectors) {
        const ours = mine.find((v) => v.form === vector.form)!;
        expect(ours.headers, `${theirs.name} / ${vector.form}`).toEqual(vector.headers);
        expect(ours.signatureBaseLines, `${theirs.name} / ${vector.form}: base`).toEqual(vector.signatureBaseLines);
        expect(vector.headers[PAYMENT_METHODS_HINT_HEADER.toLowerCase()]).toBe(theirs.hints['robutler-payment-methods']);
        expect(vector.signatureBaseLines[0]).toBe('"@method": GET');
        expect(vector.signatureBaseLines[1]).toBe(`"@authority": ${theirs.signedRequest.authority}`);
        expect(vector.signatureBaseLines[2]).toBe(`"@path": ${theirs.signedRequest.path}`);
        expect(vector.signatureBaseLines[3]).toBe(`"@query": ${theirs.signedRequest.query}`);
        const m = new RegExp(`^${vector.label}=:([A-Za-z0-9+/=]+):$`).exec(vector.headers.signature);
        expect(m, `${theirs.name} / ${vector.form}: one signature member`).not.toBeNull();
        expect(verifies(vector.signatureBaseLines.join('\n'), new Uint8Array(Buffer.from(m![1], 'base64')), committed.key.publicJwk)).toBe(true);
      }
    }
  });
});
