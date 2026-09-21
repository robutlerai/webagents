/**
 * Covered headers on the Web Bot Auth signer (machine-purchase design
 * section 6.4; pass P9a, 2026-09-18). The paid retry of a purchase carries
 * `Payment-Authorization` and `Robutler-Terms-Accepted`, and the platform
 * admits them only when both are among the signature's covered
 * components. This suite pins how `SignRequestOptions.coveredHeaders`
 * places and values them, what it refuses, and the cross-language vectors
 * for them.
 *
 * THE VECTOR FILE. `webagents/python/tests/fixtures/web_bot_auth/
 * vectors-covered-headers.json` is a SECOND file beside `vectors.json`,
 * which stays byte-identical: the existing three consumers pin that file
 * as a whole document, so extending it would have failed every one of
 * them until each was rewritten. Same rule as the first file: whichever
 * SDK lands first writes it (this suite did, on 2026-09-18); the other
 * compares byte for byte and fails on any difference, and the portal
 * verifies every vector through its pure signature layer
 * (`tests/unit/auth/web-bot-auth-covered-headers-vectors.test.ts`). The
 * inputs are the first file's key, agent URL and parameters, a `POST` to
 * the purchase URL with body `{}`, and two covered headers whose values
 * are themselves pinned: the credential is built from the fixed challenge
 * the file records, so the credential encoder is pinned across languages
 * as well as the signer.
 */

import { describe, it, expect } from 'vitest';
import { createHash, createPublicKey, verify as cryptoVerify } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importJWK, type KeyLike } from 'jose';
import { AgentIdentity } from '../../../src/crypto/identity';
import {
  COVERED_HEADERS_RESERVED,
  normalizeCoveredHeaders,
  SIGNATURE_AGENT_FORMS,
  signMessage,
  signRequest,
  StructuredFieldError,
  type SignatureAgentForm,
} from '../../../src/crypto/http-signature';
import {
  base64urlEncode,
  encodeMppCredential,
  jcsCanonicalize,
  type MppChallengeFields,
} from '../../../src/skills/payments/mpp-buyer';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VECTORS_PATH = path.resolve(HERE, '../../../../python/tests/fixtures/web_bot_auth/vectors-covered-headers.json');

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
const PURCHASE_URL = 'https://robutler.ai/api/mpp/credits';
const CREATED = 1758067200;
const EXPIRES = 1758067260;
const FIXED_NONCE = Buffer.from(Array.from({ length: 64 }, (_, i) => i)).toString('base64');
const TERMS_VERSION = '2026-07-31';

const body = (text: string) => new TextEncoder().encode(text);

async function identityFor(issuer = AGENT_URL) {
  const identity = new AgentIdentity({
    agentId: 'mini',
    issuer,
    privateKey: (await importJWK(B14.privateJwk as never, 'EdDSA')) as KeyLike,
    publicKey: (await importJWK(B14.publicJwk as never, 'EdDSA')) as KeyLike,
  });
  await identity.initialize();
  return identity;
}

/** The fixed challenge the vector's credential echoes: every value deterministic, shaped as the platform mints it. */
function fixedChallenge(): { fields: MppChallengeFields; payload: Record<string, unknown> } {
  const request = { amount: '500', currency: 'usd', methodDetails: { networkId: 'profile_vector', paymentMethodTypes: ['card'] } };
  const opaque = {
    packId: 'mpp_5',
    userId: '00000000-0000-4000-8000-000000000001',
    principal: AGENT_URL,
    thumbprint: B14.thumbprint,
    kind: 'pack',
    terms: TERMS_VERSION,
    origin: createHash('sha256').update('POST /api/mpp/credits', 'utf8').digest('hex'),
  };
  return {
    fields: {
      id: createHash('sha256').update('mpp covered-headers vector', 'utf8').digest('base64url'),
      realm: 'robutler.ai',
      method: 'stripe',
      intent: 'charge',
      request: base64urlEncode(jcsCanonicalize(request)),
      expires: '2026-09-18T12:05:00.000Z',
      opaque: base64urlEncode(jcsCanonicalize(opaque)),
      header: 'Payment-Authorization',
    },
    payload: { spt: 'spt_vector_0000000000000001' },
  };
}

function signatureMembers(header: string): Map<string, Uint8Array> {
  const out = new Map<string, Uint8Array>();
  for (const part of header.split(', ')) {
    const m = /^([a-z0-9_*.-]+)=:([A-Za-z0-9+/=]+):$/.exec(part);
    if (!m) throw new Error(`not a byte-sequence member: ${part}`);
    out.set(m[1], new Uint8Array(Buffer.from(m[2], 'base64')));
  }
  return out;
}

function verifies(base: string, signature: Uint8Array, publicJwk: object): boolean {
  const key = createPublicKey({ key: publicJwk as never, format: 'jwk' });
  return cryptoVerify(null, Buffer.from(base, 'ascii'), key, signature);
}

/** The component identifiers of a `Signature-Input` member, in wire order. */
function coveredOf(signatureInput: string, label = 'sig1'): string[] {
  const member = signatureInput.split(', ').find((m) => m.startsWith(`${label}=(`));
  if (!member) throw new Error(`no ${label} member`);
  const inner = member.slice(member.indexOf('(') + 1, member.indexOf(')'));
  return inner.split(' ');
}

describe('normalizeCoveredHeaders', () => {
  it('trims, lowercases and deduplicates in first-seen order', () => {
    expect(normalizeCoveredHeaders([' Payment-Authorization ', 'robutler-terms-accepted', 'PAYMENT-AUTHORIZATION'])).toEqual([
      'payment-authorization',
      'robutler-terms-accepted',
    ]);
    expect(normalizeCoveredHeaders(undefined)).toEqual([]);
  });

  it('refuses the reserved names, derived components, non-field names and non-strings', () => {
    for (const reserved of COVERED_HEADERS_RESERVED) {
      expect(() => normalizeCoveredHeaders([reserved])).toThrow(/covered by the signer itself/);
    }
    expect(() => normalizeCoveredHeaders(['Content-Digest'])).toThrow(/whenever the message has a body/);
    expect(() => normalizeCoveredHeaders(['@method'])).toThrow(/derived component/);
    expect(() => normalizeCoveredHeaders(['bad name'])).toThrow(/not a header field name/);
    expect(() => normalizeCoveredHeaders([''])).toThrow(/not a header field name/);
    expect(() => normalizeCoveredHeaders([42 as unknown as string])).toThrow(/header field names/);
  });
});

describe('signMessage with coveredHeaders (design section 6.4)', () => {
  const message = {
    method: 'POST',
    url: PURCHASE_URL,
    body: body('{}'),
    headers: { 'Payment-Authorization': '  Payment abc  ', 'Robutler-Terms-Accepted': TERMS_VERSION, 'X-Other': 'ignored' },
  };
  const options = { created: CREATED, nonce: FIXED_NONCE, coveredHeaders: ['Payment-Authorization', 'Robutler-Terms-Accepted'] };

  it('places the covered headers after content-digest and before the signature-agent member, lowercased and bare', async () => {
    const identity = await identityFor();
    const signed = await signMessage(identity, message, options);
    expect(coveredOf(signed.headers['signature-input'])).toEqual([
      '"@method"',
      '"@authority"',
      '"@path"',
      '"@query"',
      '"content-digest"',
      '"payment-authorization"',
      '"robutler-terms-accepted"',
      '"signature-agent";key="sig1"',
    ]);
  });

  it('values the covered lines with the field value trimmed, and the signature verifies over that base', async () => {
    const identity = await identityFor();
    const signed = await signMessage(identity, message, options);
    const lines = signed.labels[0].base.split('\n');
    expect(lines[5]).toBe('"payment-authorization": Payment abc');
    expect(lines[6]).toBe(`"robutler-terms-accepted": ${TERMS_VERSION}`);
    expect(lines).toHaveLength(9);
    const sig = signatureMembers(signed.headers.signature).get('sig1')!;
    expect(verifies(signed.labels[0].base, sig, B14.publicJwk)).toBe(true);
  });

  it('without a body the covered headers follow @query directly', async () => {
    const identity = await identityFor();
    const signed = await signMessage(identity, { ...message, body: undefined }, options);
    expect(coveredOf(signed.headers['signature-input'])).toEqual([
      '"@method"',
      '"@authority"',
      '"@path"',
      '"@query"',
      '"payment-authorization"',
      '"robutler-terms-accepted"',
      '"signature-agent";key="sig1"',
    ]);
    expect(signed.headers['content-digest']).toBeUndefined();
  });

  it('under the legacy-string form the covered headers still precede the bare signature-agent component', async () => {
    const identity = await identityFor();
    const signed = await signMessage(identity, message, { ...options, form: 'legacy-string' });
    const covered = coveredOf(signed.headers['signature-input']);
    expect(covered.slice(-3)).toEqual(['"payment-authorization"', '"robutler-terms-accepted"', '"signature-agent"']);
  });

  it('reads the header value case-insensitively from a record and from a Headers', async () => {
    const identity = await identityFor();
    const fromRecord = await signMessage(identity, { ...message, headers: { 'PAYMENT-authorization': 'Payment abc', 'robutler-terms-accepted': TERMS_VERSION } }, options);
    const fromHeaders = await signMessage(identity, { ...message, headers: new Headers({ 'Payment-Authorization': 'Payment abc', 'Robutler-Terms-Accepted': TERMS_VERSION }) }, options);
    expect(fromRecord.labels[0].base).toBe(fromHeaders.labels[0].base);
    expect(fromRecord.headers.signature).toBe(fromHeaders.headers.signature);
  });

  it('a Headers joins a repeated field with ", " and that joined value is what is covered (RFC 9421 section 2.1 step 4)', async () => {
    const identity = await identityFor();
    const headers = new Headers();
    headers.append('Robutler-Terms-Accepted', TERMS_VERSION);
    headers.append('Robutler-Terms-Accepted', '2026-01-01');
    headers.set('Payment-Authorization', 'Payment abc');
    const signed = await signMessage(identity, { ...message, headers }, options);
    expect(signed.labels[0].base.split('\n')[6]).toBe(`"robutler-terms-accepted": ${TERMS_VERSION}, 2026-01-01`);
  });

  it('refuses a covered header the message does not carry, and a message with no headers at all', async () => {
    const identity = await identityFor();
    await expect(signMessage(identity, { ...message, headers: { 'Payment-Authorization': 'Payment abc' } }, options)).rejects.toThrow(
      /cannot cover header robutler-terms-accepted: the message does not carry it/,
    );
    await expect(signMessage(identity, { ...message, headers: undefined }, options)).rejects.toThrow(/cannot cover header payment-authorization/);
  });

  it('refuses a covered value that is not printable ASCII (a CR or LF could forge a base line)', async () => {
    const identity = await identityFor();
    await expect(
      signMessage(identity, { ...message, headers: { 'Payment-Authorization': 'Payment abc', 'Robutler-Terms-Accepted': 'vé' } }, options),
    ).rejects.toThrow(StructuredFieldError);
  });

  it('with no coveredHeaders the signature is byte-identical to the pre-P9a signer (the first vector file still holds)', async () => {
    const identity = await identityFor();
    const before = await signMessage(identity, { method: 'POST', url: PURCHASE_URL, body: body('{}') }, { created: CREATED, nonce: FIXED_NONCE });
    const after = await signMessage(identity, { ...message }, { created: CREATED, nonce: FIXED_NONCE });
    expect(after.headers).toEqual(before.headers);
    expect(after.labels[0].base).toBe(before.labels[0].base);
  });
});

describe('signRequest with coveredHeaders', () => {
  it('reads the covered values off the Request, keeps them on the signed Request, and covers them', async () => {
    const identity = await identityFor();
    const request = new Request(PURCHASE_URL, {
      method: 'POST',
      headers: { 'Payment-Authorization': 'Payment abc', 'Robutler-Terms-Accepted': TERMS_VERSION, 'content-type': 'application/json' },
      body: '{}',
    });
    const signed = await signRequest(identity, request, { coveredHeaders: ['payment-authorization', 'Robutler-Terms-Accepted'] });
    expect(signed.headers.get('payment-authorization')).toBe('Payment abc');
    expect(signed.headers.get('robutler-terms-accepted')).toBe(TERMS_VERSION);
    expect(coveredOf(signed.headers.get('signature-input')!)).toContain('"payment-authorization"');
    expect(coveredOf(signed.headers.get('signature-input')!)).toContain('"robutler-terms-accepted"');
    expect(await signed.text()).toBe('{}');
  });

  it('refuses before sending when the Request lacks a named header', async () => {
    const identity = await identityFor();
    const request = new Request(PURCHASE_URL, { method: 'POST', body: '{}' });
    await expect(signRequest(identity, request, { coveredHeaders: ['payment-authorization'] })).rejects.toThrow(/does not carry it/);
  });
});

// ---------------------------------------------------------------------------
// Cross-language vectors with covered headers (design section 6.4)
// ---------------------------------------------------------------------------

interface CoveredVectorFile {
  source: string;
  key: { name: string; publicJwk: object; privateJwk: object; thumbprint: string };
  agentUrl: string;
  request: {
    method: string;
    url: string;
    authority: string;
    path: string;
    query: string;
    body: string;
    headers: { 'payment-authorization': string; 'robutler-terms-accepted': string };
  };
  coveredHeaders: string[];
  credential: { challenge: MppChallengeFields; payload: Record<string, unknown>; header: string };
  params: { label: string; created: number; expires: number; nonce: string; alg: string; tag: string };
  vectors: Array<{
    id: string;
    form: SignatureAgentForm;
    label: string;
    headers: { 'content-digest': string; 'signature-agent': string; 'signature-input': string; signature: string };
    signatureBaseLines: string[];
  }>;
}

async function produceCoveredVectors(file: Omit<CoveredVectorFile, 'vectors' | 'source'>): Promise<CoveredVectorFile['vectors']> {
  const identity = await identityFor(file.agentUrl);
  const out: CoveredVectorFile['vectors'] = [];
  for (const form of SIGNATURE_AGENT_FORMS) {
    const signed = await signMessage(
      identity,
      { method: file.request.method, url: file.request.url, body: body(file.request.body), headers: file.request.headers },
      {
        form,
        label: file.params.label,
        created: file.params.created,
        lifetimeSeconds: file.params.expires - file.params.created,
        nonce: file.params.nonce,
        coveredHeaders: file.coveredHeaders,
      },
    );
    out.push({
      id: form,
      form,
      label: file.params.label,
      headers: {
        'content-digest': signed.headers['content-digest']!,
        'signature-agent': signed.headers['signature-agent'],
        'signature-input': signed.headers['signature-input'],
        signature: signed.headers.signature,
      },
      signatureBaseLines: signed.labels[0].base.split('\n'),
    });
  }
  return out;
}

describe('cross-language vectors with covered headers (design section 6.4)', () => {
  const challenge = fixedChallenge();
  const credentialHeader = encodeMppCredential(challenge.fields, challenge.payload);
  const inputs: Omit<CoveredVectorFile, 'vectors' | 'source'> = {
    key: B14,
    agentUrl: AGENT_URL,
    request: {
      method: 'POST',
      url: PURCHASE_URL,
      authority: 'robutler.ai',
      path: '/api/mpp/credits',
      query: '',
      body: '{}',
      headers: { 'payment-authorization': credentialHeader, 'robutler-terms-accepted': TERMS_VERSION },
    },
    coveredHeaders: ['payment-authorization', 'robutler-terms-accepted'],
    credential: { challenge: challenge.fields, payload: challenge.payload, header: credentialHeader },
    params: { label: 'sig1', created: CREATED, expires: EXPIRES, nonce: FIXED_NONCE, alg: 'ed25519', tag: 'web-bot-auth' },
  };

  it('matches the committed vectors byte for byte, or writes them when no SDK has yet', async () => {
    if (!existsSync(VECTORS_PATH)) {
      const vectors = await produceCoveredVectors(inputs);
      const file: CoveredVectorFile = {
        source:
          'Machine-purchase design section 6.4 cross-language vectors with covered headers, written 2026-09-18 by the ' +
          'TypeScript SDK signer (webagents/typescript/tests/unit/crypto/http-signature-covered-headers.test.ts). Inputs: ' +
          'the RFC 9421 Appendix B.1.4 Ed25519 test key (denylisted by the platform profile), POST /api/mpp/credits at ' +
          'robutler.ai with body {}, created 1758067200, expires 1758067260, nonce = bytes 0x00..0x3f in standard base64, ' +
          'agent URL https://agent.example/agents/mini, and two covered headers: payment-authorization, whose value is ' +
          'the MPP credential built from the fixed challenge and payload under `credential` (Payment + base64url of the ' +
          'RFC 8785 JCS of {challenge, payload}), and robutler-terms-accepted, the Terms version. The covered headers ' +
          'follow content-digest and precede the signature-agent member, lowercased and bare, valued by the field value ' +
          'trimmed. One vector per Signature-Agent form. A consumer rebuilds the credential header from `credential`, ' +
          'signs the inputs and compares every header and every base line; Ed25519 is deterministic so the signature ' +
          'bytes are pinned too.',
        ...inputs,
        vectors,
      };
      mkdirSync(path.dirname(VECTORS_PATH), { recursive: true });
      writeFileSync(VECTORS_PATH, `${JSON.stringify(file, null, 2)}\n`);
      console.log(`[http-signature-covered-headers.test] wrote the cross-language vectors to ${VECTORS_PATH}`);
    }

    const committed = JSON.parse(readFileSync(VECTORS_PATH, 'utf8')) as CoveredVectorFile;
    for (const field of ['key', 'agentUrl', 'request', 'coveredHeaders', 'credential', 'params', 'vectors'] as const) {
      expect(committed[field], `vectors-covered-headers.json has no ${field}`).toBeDefined();
    }
    expect(committed.key.thumbprint).toBe(B14.thumbprint);
    expect(committed.coveredHeaders).toEqual(['payment-authorization', 'robutler-terms-accepted']);
    expect(committed.vectors.map((v) => v.form).sort()).toEqual([...SIGNATURE_AGENT_FORMS].sort());

    // The credential encoder is pinned too: the header the file's request carries is what this encoder makes of the file's challenge and payload.
    expect(encodeMppCredential(committed.credential.challenge, committed.credential.payload)).toBe(committed.credential.header);
    expect(committed.request.headers['payment-authorization']).toBe(committed.credential.header);

    const mine = await produceCoveredVectors(committed);
    for (const theirs of committed.vectors) {
      const ours = mine.find((v) => v.form === theirs.form)!;
      for (const header of ['content-digest', 'signature-agent', 'signature-input', 'signature'] as const) {
        expect(ours.headers[header], `${theirs.form}: ${header}`).toBe(theirs.headers[header]);
      }
      expect(ours.signatureBaseLines, `${theirs.form}: signature base`).toEqual(theirs.signatureBaseLines);
      expect(theirs.signatureBaseLines).toContain(`"payment-authorization": ${committed.credential.header}`);
      expect(theirs.signatureBaseLines).toContain(`"robutler-terms-accepted": ${committed.request.headers['robutler-terms-accepted']}`);
      const sig = signatureMembers(theirs.headers.signature).get(theirs.label)!;
      expect(verifies(theirs.signatureBaseLines.join('\n'), sig, committed.key.publicJwk), `${theirs.form}: verify`).toBe(true);
    }
  });

  it('the first vector file is untouched by this suite (it stays byte-identical for its three consumers)', () => {
    const first = path.resolve(HERE, '../../../../python/tests/fixtures/web_bot_auth/vectors.json');
    const committed = JSON.parse(readFileSync(first, 'utf8')) as { request: Record<string, unknown>; vectors: Array<{ signatureBaseLines: string[] }> };
    expect(committed.request.headers).toBeUndefined();
    for (const v of committed.vectors) {
      expect(v.signatureBaseLines.some((line) => line.startsWith('"payment-authorization"'))).toBe(false);
    }
  });
});
