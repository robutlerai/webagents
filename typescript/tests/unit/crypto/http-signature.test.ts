/**
 * The Web Bot Auth request signer, pinned to the W2 design's wire contract
 * (ADR 0038 step 5, 2026-09-17): sections 2.1 to 2.7 for the headers, the
 * covered set, the parameters, the three `Signature-Agent` forms and the
 * two-label rotation signing, and section 10.3 for the cross-language
 * vectors both SDKs and the portal verifier must agree on.
 *
 * Every signature here is made with the RFC 9421 Appendix B.1.4 Ed25519
 * test key, whose thumbprint is `poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U`.
 * That key is denylisted by the platform's profile (P section 6.8), which
 * is exactly why it is safe to commit its private half beside a fixture:
 * nothing it signs is ever admitted. Ed25519 is deterministic (RFC 8032),
 * so with a fixed key, `created`, `expires` and nonce every byte of the
 * three headers is reproducible, and the vectors pin them.
 *
 * THE VECTOR FILE (design section 10.3). `webagents/python/tests/fixtures/
 * web_bot_auth/vectors.json` is one file with three consumers: the Python
 * signer, this signer and the portal verifier. Whichever SDK lands first
 * writes it; the other compares byte for byte and fails on any difference.
 * So: if the file exists, this suite reads the INPUTS from it (key, request,
 * `created`, `expires`, nonce, agent URL) and asserts that this signer maps
 * them to the file's outputs; if it does not, this suite writes it from
 * this implementation and says so on stdout.
 */

import { describe, it, expect, beforeAll, vi } from 'vitest';
import { createPublicKey, verify as cryptoVerify, generateKeyPairSync } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exportJWK, generateKeyPair, importJWK, type KeyLike } from 'jose';
import { AgentIdentity } from '../../../src/crypto/identity';
import {
  assertSignableAgentUrl,
  buildSignatureBase,
  contentDigest,
  randomNonce,
  serializeBareItem,
  serializeDictionary,
  serializeInnerList,
  serializeItem,
  serializeParameters,
  SfToken,
  SIGNATURE_AGENT_FORMS,
  signatureAgentValue,
  signedFetch,
  signMessage,
  signRequest,
  StructuredFieldError,
  type SignatureAgentForm,
  type SignedMessage,
} from '../../../src/crypto/http-signature';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VECTORS_PATH = path.resolve(HERE, '../../../../python/tests/fixtures/web_bot_auth/vectors.json');

// RFC 9421 Appendix B.1.4, test-key-ed25519.
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
const KEY_SET_URL = `${AGENT_URL}/.well-known/jwks.json`;
const PLATFORM = 'https://robutler.ai';
const TOKEN_URL = `${PLATFORM}/api/auth/cli/token`;
// Design section 2.7's example window.
const CREATED = 1758067200;
const EXPIRES = 1758067260;
// The fixed nonce of the vectors: bytes 0x00 to 0x3f, standard base64 (88 characters).
const FIXED_NONCE = Buffer.from(Array.from({ length: 64 }, (_, i) => i)).toString('base64');
// SHA-256 over the two bytes `{}` (design section 2.7).
const EMPTY_OBJECT_DIGEST = 'sha-256=:RBNvo1WzZ4oRRq0W9+hknpT7T8If536DEMBg9hyq/4o=:';

const body = (text: string) => new TextEncoder().encode(text);

async function identityFor(
  privateJwk: object,
  publicJwk: object,
  issuer = AGENT_URL,
  previousKeys: Array<{ privateKey: KeyLike; publicKey: KeyLike }> = [],
) {
  const identity = new AgentIdentity({
    agentId: 'mini',
    issuer,
    privateKey: (await importJWK(privateJwk as never, 'EdDSA')) as KeyLike,
    publicKey: (await importJWK(publicJwk as never, 'EdDSA')) as KeyLike,
    previousKeys,
  });
  await identity.initialize();
  return identity;
}

/** The `label=:base64:` members of a `Signature` header. */
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

/** `Signature-Input` member parameters in wire order, as `[name, raw]` pairs. */
function inputParams(header: string, label: string): Array<[string, string]> {
  const member = header.split(', ').find((m) => m.startsWith(`${label}=(`));
  if (!member) throw new Error(`no ${label} member in ${header}`);
  const afterList = member.slice(member.indexOf(')') + 1);
  return afterList
    .split(';')
    .filter(Boolean)
    .map((p) => {
      const eq = p.indexOf('=');
      return [p.slice(0, eq), p.slice(eq + 1)] as [string, string];
    });
}

describe('RFC 9651 serialising subset', () => {
  it('serialises Strings quoted and escaped, printable ASCII only', () => {
    expect(serializeBareItem('a"b\\c')).toBe('"a\\"b\\\\c"');
    expect(() => serializeBareItem('café')).toThrow(StructuredFieldError);
    expect(() => serializeBareItem('a\nb')).toThrow(StructuredFieldError);
  });

  it('serialises Tokens bare and refuses a value that is not a token', () => {
    expect(serializeBareItem(new SfToken('jwks_uri'))).toBe('jwks_uri');
    expect(serializeBareItem(new SfToken('*x/y:z'))).toBe('*x/y:z');
    expect(() => serializeBareItem(new SfToken('1abc'))).toThrow(StructuredFieldError);
    expect(() => serializeBareItem(new SfToken('a b'))).toThrow(StructuredFieldError);
  });

  it('serialises Integers in decimal and refuses decimals and out-of-range values', () => {
    expect(serializeBareItem(1758067200)).toBe('1758067200');
    expect(serializeBareItem(-5)).toBe('-5');
    expect(() => serializeBareItem(1.5)).toThrow(StructuredFieldError);
    expect(() => serializeBareItem(1e15)).toThrow(StructuredFieldError);
  });

  it('serialises Byte Sequences as padded standard base64 between colons, never base64url', () => {
    expect(serializeBareItem(new Uint8Array([0xfb, 0xff]))).toBe(':+/8=:');
    expect(serializeBareItem(new Uint8Array(0))).toBe('::');
  });

  it('serialises Booleans, and a true parameter as its bare key', () => {
    expect(serializeBareItem(true)).toBe('?1');
    expect(serializeBareItem(false)).toBe('?0');
    expect(serializeParameters([['req', true], ['sf', false]])).toBe(';req;sf=?0');
  });

  it('serialises Items, Inner Lists and Dictionaries the way the verifier re-serialises them', () => {
    expect(serializeItem({ value: KEY_SET_URL, params: [['type', new SfToken('jwks_uri')]] })).toBe(
      `"${KEY_SET_URL}";type=jwks_uri`,
    );
    expect(
      serializeInnerList({
        items: [{ value: '@method' }, { value: 'signature-agent', params: [['key', 'sig1']] }],
        params: [['created', 1], ['tag', 'web-bot-auth']],
      }),
    ).toBe('("@method" "signature-agent";key="sig1");created=1;tag="web-bot-auth"');
    expect(
      serializeDictionary([
        ['sig1', { value: new Uint8Array([1, 2, 3]) }],
        ['sig2', { items: [{ value: '@path' }], params: [] }],
        ['flag', { value: true, params: [['p', 2]] }],
      ]),
    ).toBe('sig1=:AQID:, sig2=("@path"), flag;p=2');
    expect(() => serializeDictionary([['Bad', { value: 1 }]])).toThrow(StructuredFieldError);
  });
});

describe('the pieces', () => {
  it('Content-Digest is sha-256 as a Byte Sequence, and {} digests to the design 2.7 value', async () => {
    expect(await contentDigest(body('{}'))).toBe(EMPTY_OBJECT_DIGEST);
    expect(await contentDigest(body('{"a":1}'))).toMatch(/^sha-256=:[A-Za-z0-9+/]{43}=:$/);
  });

  it('the nonce is 64 random bytes in standard base64', () => {
    const a = randomNonce();
    const b = randomNonce();
    expect(a).toMatch(/^[A-Za-z0-9+/]{86}==$/);
    expect(Buffer.from(a, 'base64')).toHaveLength(64);
    expect(a).not.toBe(b);
  });

  it('Signature-Agent names the key set for the dictionary forms and the origin for the legacy string', () => {
    expect(signatureAgentValue(AGENT_URL, 'dictionary-typed')).toBe(KEY_SET_URL);
    expect(signatureAgentValue(`${AGENT_URL}/`, 'dictionary-untyped')).toBe(KEY_SET_URL);
    expect(signatureAgentValue(AGENT_URL, 'legacy-string')).toBe('https://agent.example');
    expect(signatureAgentValue('https://agent.example', 'dictionary-typed')).toBe(
      'https://agent.example/.well-known/jwks.json',
    );
  });

  it('the signature base is LF separated with no trailing newline and refuses a CR or LF in a value', () => {
    expect(buildSignatureBase([['"@method"', 'POST'], ['"@path"', '/x']], '("@method" "@path");created=1')).toBe(
      '"@method": POST\n"@path": /x\n"@signature-params": ("@method" "@path");created=1',
    );
    expect(() => buildSignatureBase([['"@path"', '/x\n"@method": GET']], '()')).toThrow(StructuredFieldError);
  });

  it('refuses to sign for a loopback or plaintext agent URL, naming the variable to set', () => {
    expect(() => assertSignableAgentUrl('http://localhost:8000')).toThrow(/WEBAGENTS_PUBLIC_URL/);
    expect(() => assertSignableAgentUrl('http://127.0.0.1:8000')).toThrow(/loopback/);
    expect(() => assertSignableAgentUrl('https://[::1]/agents/x')).toThrow(/loopback/);
    expect(() => assertSignableAgentUrl('http://agent.example/agents/x')).toThrow(/not https/);
    expect(() => assertSignableAgentUrl('not a url')).toThrow(/not a URL/);
    expect(assertSignableAgentUrl('https://agent.example/agents/x').hostname).toBe('agent.example');
    // The platform's local-overlay switch, honoured by name: plaintext, never loopback.
    expect(assertSignableAgentUrl('http://agent.internal/agents/x', { allowHttp: true }).protocol).toBe('http:');
    expect(() => assertSignableAgentUrl('http://localhost/agents/x', { allowHttp: true })).toThrow(/loopback/);
    vi.stubEnv('ROBUTLER_AGENT_URL_ALLOW_PRIVATE', '1');
    try {
      expect(assertSignableAgentUrl('http://agent.internal/agents/x').protocol).toBe('http:');
    } finally {
      vi.unstubAllEnvs();
    }
  });
});

describe('signMessage (design sections 2.1 to 2.7)', () => {
  let identity: AgentIdentity;

  beforeAll(async () => {
    identity = await identityFor(B14.privateJwk, B14.publicJwk);
  });

  const registration = () => ({ method: 'POST', url: TOKEN_URL, body: body('{}') });

  it('reproduces the design 2.7 example byte for byte', async () => {
    const signed = await signMessage(identity, registration(), { created: CREATED, nonce: FIXED_NONCE });

    expect(signed.headers['content-digest']).toBe(EMPTY_OBJECT_DIGEST);
    expect(signed.headers['signature-agent']).toBe(`sig1="${KEY_SET_URL}";type=jwks_uri`);
    expect(signed.headers['signature-input']).toBe(
      'sig1=("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="sig1")' +
        `;created=${CREATED};expires=${EXPIRES};keyid="${B14.thumbprint}";alg="ed25519";nonce="${FIXED_NONCE}";tag="web-bot-auth"`,
    );
    expect(signed.headers.signature).toMatch(/^sig1=:[A-Za-z0-9+/]{86}==:$/);
    expect(signed.created).toBe(CREATED);
    expect(signed.expires).toBe(EXPIRES);

    expect(signed.labels).toHaveLength(1);
    expect(signed.labels[0].base).toBe(
      [
        '"@method": POST',
        '"@authority": robutler.ai',
        '"@path": /api/auth/cli/token',
        '"@query": ?',
        `"content-digest": ${EMPTY_OBJECT_DIGEST}`,
        `"signature-agent";key="sig1": "${KEY_SET_URL}";type=jwks_uri`,
        '"@signature-params": ("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="sig1")' +
          `;created=${CREATED};expires=${EXPIRES};keyid="${B14.thumbprint}";alg="ed25519";nonce="${FIXED_NONCE}";tag="web-bot-auth"`,
      ].join('\n'),
    );
    const sig = signatureMembers(signed.headers.signature).get('sig1')!;
    expect(sig).toHaveLength(64);
    expect(verifies(signed.labels[0].base, sig, B14.publicJwk)).toBe(true);
    // The signature is over the base and nothing else: one flipped byte of the base fails.
    expect(verifies(signed.labels[0].base.replace('POST', 'GET'), sig, B14.publicJwk)).toBe(false);
  });

  it('is deterministic for a fixed key, window and nonce (Ed25519 has no signer randomness)', async () => {
    const a = await signMessage(identity, registration(), { created: CREATED, nonce: FIXED_NONCE });
    const b = await signMessage(identity, registration(), { created: CREATED, nonce: FIXED_NONCE });
    expect(a.headers).toEqual(b.headers);
  });

  it('carries every parameter of design 2.3, in order, with the design defaults', async () => {
    const before = Math.floor(Date.now() / 1000);
    const signed = await signMessage(identity, registration());
    const params = inputParams(signed.headers['signature-input'], 'sig1');
    expect(params.map(([k]) => k)).toEqual(['created', 'expires', 'keyid', 'alg', 'nonce', 'tag']);
    const created = Number(params[0][1]);
    expect(created).toBeGreaterThanOrEqual(before);
    expect(created).toBeLessThanOrEqual(before + 5);
    expect(Number(params[1][1])).toBe(created + 60);
    expect(params[2][1]).toBe(`"${B14.thumbprint}"`);
    expect(params[3][1]).toBe('"ed25519"');
    expect(params[4][1]).toMatch(/^"[A-Za-z0-9+/]{86}=="$/);
    expect(params[5][1]).toBe('"web-bot-auth"');
    // A fresh nonce every time.
    const again = await signMessage(identity, registration());
    expect(inputParams(again.headers['signature-input'], 'sig1')[4][1]).not.toBe(params[4][1]);
  });

  it('covers exactly the design 2.2 components, in order', async () => {
    const withBody = await signMessage(identity, registration(), { created: CREATED, nonce: FIXED_NONCE });
    expect(withBody.headers['signature-input'].split(')')[0]).toBe(
      'sig1=("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="sig1"',
    );
    const withoutBody = await signMessage(identity, { method: 'get', url: `${PLATFORM}/api/agents/me` });
    expect(withoutBody.headers['signature-input'].split(')')[0]).toBe(
      'sig1=("@method" "@authority" "@path" "@query" "signature-agent";key="sig1"',
    );
    // `@method` uppercased, `@query` is `?` alone without a query.
    expect(withoutBody.labels[0].base.split('\n').slice(0, 4)).toEqual([
      '"@method": GET',
      '"@authority": robutler.ai',
      '"@path": /api/agents/me',
      '"@query": ?',
    ]);
  });

  it('Content-Digest is present exactly when there is a non-empty body', async () => {
    const get = await signMessage(identity, { method: 'GET', url: TOKEN_URL });
    expect(get.headers['content-digest']).toBeUndefined();
    expect(get.headers['signature-input']).not.toContain('content-digest');

    const emptyBody = await signMessage(identity, { method: 'POST', url: TOKEN_URL, body: new Uint8Array(0) });
    expect(emptyBody.headers['content-digest']).toBeUndefined();
    expect(emptyBody.headers['signature-input']).not.toContain('content-digest');

    const withBody = await signMessage(identity, { method: 'POST', url: TOKEN_URL, body: body('{"a":1}') });
    expect(withBody.headers['content-digest']).toBe(await contentDigest(body('{"a":1}')));
    expect(withBody.headers['signature-input']).toContain('"content-digest"');
    expect(withBody.labels[0].base).toContain(`"content-digest": ${withBody.headers['content-digest']}`);
  });

  it('normalises @authority, keeps @path and @query as the URL parser spells them', async () => {
    const signed = await signMessage(identity, {
      method: 'POST',
      url: 'https://ROBUTLER.AI:443/api/x/y?b=2&a=%201',
      body: body('{}'),
    });
    expect(signed.labels[0].base.split('\n').slice(0, 4)).toEqual([
      '"@method": POST',
      '"@authority": robutler.ai',
      '"@path": /api/x/y',
      '"@query": ?b=2&a=%201',
    ]);
    const port = await signMessage(identity, { method: 'GET', url: 'https://host.example:8443/' });
    expect(port.labels[0].base.split('\n').slice(1, 3)).toEqual(['"@authority": host.example:8443', '"@path": /']);
  });

  it('sends the three Signature-Agent forms and covers the member accordingly', async () => {
    const typed = await signMessage(identity, registration(), { form: 'dictionary-typed' });
    expect(typed.headers['signature-agent']).toBe(`sig1="${KEY_SET_URL}";type=jwks_uri`);
    expect(typed.labels[0].base).toContain(`\n"signature-agent";key="sig1": "${KEY_SET_URL}";type=jwks_uri\n`);

    const untyped = await signMessage(identity, registration(), { form: 'dictionary-untyped' });
    expect(untyped.headers['signature-agent']).toBe(`sig1="${KEY_SET_URL}"`);
    expect(untyped.headers['signature-input']).toContain('"signature-agent";key="sig1"');
    expect(untyped.labels[0].base).toContain(`\n"signature-agent";key="sig1": "${KEY_SET_URL}"\n`);

    const legacy = await signMessage(identity, registration(), { form: 'legacy-string' });
    expect(legacy.headers['signature-agent']).toBe('"https://agent.example"');
    // The bare field is covered, with no `key` parameter, and its value is the whole field.
    expect(legacy.headers['signature-input']).toContain('"content-digest" "signature-agent")');
    expect(legacy.labels[0].base).toContain('\n"signature-agent": "https://agent.example"\n');

    for (const form of SIGNATURE_AGENT_FORMS) {
      const signed = await signMessage(identity, registration(), { form });
      const sig = signatureMembers(signed.headers.signature).get('sig1')!;
      expect(verifies(signed.labels[0].base, sig, B14.publicJwk), form).toBe(true);
    }
    await expect(signMessage(identity, registration(), { form: 'cimd' as SignatureAgentForm })).rejects.toThrow(/form/);
  });

  it('signs once per held key while rotating: sig1 with the current key, sig2 with the previous one', async () => {
    const previous = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
    const previousPublicJwk = await exportJWK(previous.publicKey);
    const rotating = await identityFor(B14.privateJwk, B14.publicJwk, AGENT_URL, [previous]);
    const [current, old] = rotating.getHeldKeys();
    expect(current.kid).toBe(B14.thumbprint);
    expect(old.kid).not.toBe(B14.thumbprint);

    const signed = await signMessage(rotating, registration(), { created: CREATED });
    expect(signed.labels.map((l) => l.label)).toEqual(['sig1', 'sig2']);
    expect(signed.labels.map((l) => l.kid)).toEqual([current.kid, old.kid]);
    expect(signed.labels[0].nonce).not.toBe(signed.labels[1].nonce);

    // One member per label in every field, each member keyed by its own label.
    expect(signed.headers['signature-agent']).toBe(
      `sig1="${KEY_SET_URL}";type=jwks_uri, sig2="${KEY_SET_URL}";type=jwks_uri`,
    );
    const inputs = signed.headers['signature-input'].split(', sig2=');
    expect(inputs).toHaveLength(2);
    expect(inputs[0]).toContain('"signature-agent";key="sig1")');
    expect(inputs[0]).toContain(`keyid="${current.kid}"`);
    expect(`sig2=${inputs[1]}`).toContain('"signature-agent";key="sig2")');
    expect(inputs[1]).toContain(`keyid="${old.kid}"`);
    expect(signed.labels[1].base).toContain(`"signature-agent";key="sig2": "${KEY_SET_URL}";type=jwks_uri`);

    const sigs = signatureMembers(signed.headers.signature);
    expect([...sigs.keys()]).toEqual(['sig1', 'sig2']);
    expect(verifies(signed.labels[0].base, sigs.get('sig1')!, B14.publicJwk)).toBe(true);
    expect(verifies(signed.labels[1].base, sigs.get('sig2')!, previousPublicJwk)).toBe(true);
    // And not the other way round: the label selects the key.
    expect(verifies(signed.labels[1].base, sigs.get('sig2')!, B14.publicJwk)).toBe(false);

    // ONE NONCE PER KEY (2026-09-19). A plain string is one nonce; it used to
    // be written into BOTH labels, and the platform, which spends each nonce
    // once, answered `signature_replayed` to sig2: the rotation request itself
    // was the one refused. Refused before any key signs; the function form
    // is the only fixed-nonce form for two keys, and it must not repeat.
    await expect(signMessage(rotating, registration(), { created: CREATED, nonce: FIXED_NONCE })).rejects.toThrow(
      /every label needs its own nonce/,
    );
    await expect(signMessage(rotating, registration(), { created: CREATED, nonce: () => FIXED_NONCE })).rejects.toThrow(
      /same value for two labels/,
    );
    const fixed = await signMessage(rotating, registration(), { created: CREATED, nonce: (i) => `${FIXED_NONCE.slice(0, -4)}${i}A==` });
    expect(fixed.labels.map((l) => l.nonce)).toEqual([`${FIXED_NONCE.slice(0, -4)}0A==`, `${FIXED_NONCE.slice(0, -4)}1A==`]);
    expect(fixed.headers['signature-input']).toContain(`nonce="${fixed.labels[0].nonce}"`);
    expect(fixed.headers['signature-input']).toContain(`nonce="${fixed.labels[1].nonce}"`);
    // One held key: the plain string is still the vectors' test seam.
    expect((await signMessage(identity, registration(), { created: CREATED, nonce: FIXED_NONCE })).labels[0].nonce).toBe(FIXED_NONCE);

    // A custom first label keeps its stem for the second.
    const labelled = await signMessage(rotating, registration(), { label: 'reg1' });
    expect(labelled.labels.map((l) => l.label)).toEqual(['reg1', 'reg2']);
    // The legacy string is one field covered bare by both labels.
    const legacy = await signMessage(rotating, registration(), { form: 'legacy-string' });
    expect(legacy.headers['signature-agent']).toBe('"https://agent.example"');
    expect(legacy.labels.every((l) => l.base.includes('\n"signature-agent": "https://agent.example"\n'))).toBe(true);
  });

  it('refuses what the verifier would refuse, before signing', async () => {
    await expect(signMessage(identity, registration(), { lifetimeSeconds: 3601 })).rejects.toThrow(/3600/);
    await expect(signMessage(identity, registration(), { lifetimeSeconds: 0 })).rejects.toThrow(/lifetime/);
    await expect(signMessage(identity, registration(), { label: 'Sig1' })).rejects.toThrow(StructuredFieldError);
    const loopback = await identityFor(B14.privateJwk, B14.publicJwk, 'http://localhost:8000/agents/mini');
    await expect(signMessage(loopback, registration())).rejects.toThrow(/WEBAGENTS_PUBLIC_URL/);
    const plaintext = await identityFor(B14.privateJwk, B14.publicJwk, 'http://agent.example/agents/mini');
    await expect(signMessage(plaintext, registration())).rejects.toThrow(/not https/);
    const allowed = await signMessage(plaintext, registration(), { allowHttp: true });
    expect(allowed.headers['signature-agent']).toBe('sig1="http://agent.example/agents/mini/.well-known/jwks.json";type=jwks_uri');
    const uninitialised = new AgentIdentity({ agentId: 'x', issuer: AGENT_URL });
    await expect(signMessage(uninitialised, registration())).rejects.toThrow(/not initialized/);
  });

  it('a signature of the wrong length from a held key is refused rather than sent', async () => {
    const short = {
      issuer: AGENT_URL,
      getHeldKeys: () => [{ kid: B14.thumbprint, sign: async () => new Uint8Array(63) }],
    };
    await expect(signMessage(short, registration())).rejects.toThrow(/63-byte/);
  });
});

describe('signRequest and signedFetch', () => {
  let identity: AgentIdentity;

  beforeAll(async () => {
    identity = await identityFor(B14.privateJwk, B14.publicJwk);
  });

  it('returns a new Request carrying the four headers, the original headers, method, URL and body', async () => {
    const original = new Request(TOKEN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Robutler-Owner-Key': 'rk_test' },
      body: '{}',
    });
    const signed = await signRequest(identity, original, { created: CREATED, nonce: FIXED_NONCE });
    expect(signed).not.toBe(original);
    expect(signed.method).toBe('POST');
    expect(signed.url).toBe(TOKEN_URL);
    expect(signed.headers.get('content-type')).toBe('application/json');
    expect(signed.headers.get('x-robutler-owner-key')).toBe('rk_test');
    expect(signed.headers.get('content-digest')).toBe(EMPTY_OBJECT_DIGEST);
    expect(signed.headers.get('signature-agent')).toBe(`sig1="${KEY_SET_URL}";type=jwks_uri`);
    expect(signed.headers.get('signature-input')).toContain(`keyid="${B14.thumbprint}"`);
    expect(signed.headers.get('signature')).toMatch(/^sig1=:[A-Za-z0-9+/=]+:$/);
    expect(await signed.text()).toBe('{}');
    // The signed request carries no bearer: the signature IS the credential.
    expect(signed.headers.get('authorization')).toBeNull();
  });

  it('signs a GET with no body and no Content-Digest', async () => {
    const signed = await signRequest(identity, new Request(`${PLATFORM}/api/agents/me`));
    expect(signed.body).toBeNull();
    expect(signed.headers.get('content-digest')).toBeNull();
    expect(signed.headers.get('signature-input')).not.toContain('content-digest');
  });

  it('the headers a Request carries verify against the base the signer reports', async () => {
    const message = { method: 'POST', url: TOKEN_URL, body: body('{"x":1}') };
    const reported: SignedMessage = await signMessage(identity, message, { created: CREATED, nonce: FIXED_NONCE });
    const signed = await signRequest(identity, new Request(TOKEN_URL, { method: 'POST', body: '{"x":1}' }), {
      created: CREATED,
      nonce: FIXED_NONCE,
    });
    expect(signed.headers.get('signature')).toBe(reported.headers.signature);
    const sig = signatureMembers(signed.headers.get('signature')!).get('sig1')!;
    expect(verifies(reported.labels[0].base, sig, B14.publicJwk)).toBe(true);
  });

  it('signedFetch hands fetch the signed request', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}', { status: 200 }));
    try {
      const res = await signedFetch(identity, TOKEN_URL, { method: 'POST', body: '{}' }, { form: 'dictionary-untyped' });
      expect(res.status).toBe(200);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const sent = fetchSpy.mock.calls[0][0] as Request;
      expect(sent).toBeInstanceOf(Request);
      expect(sent.url).toBe(TOKEN_URL);
      expect(sent.headers.get('signature-agent')).toBe(`sig1="${KEY_SET_URL}"`);
      expect(sent.headers.get('content-digest')).toBe(EMPTY_OBJECT_DIGEST);
      expect(await sent.text()).toBe('{}');
    } finally {
      fetchSpy.mockRestore();
    }
  });
});

// ---------------------------------------------------------------------------
// Cross-language vectors (design section 10.3)
// ---------------------------------------------------------------------------

interface VectorFile {
  source: string;
  key: { name: string; publicJwk: object; privateJwk: object; thumbprint: string };
  agentUrl: string;
  request: { method: string; url: string; authority: string; path: string; query: string; body: string };
  params: { label: string; created: number; expires: number; nonce: string; alg: string; tag: string };
  vectors: Array<{
    id: string;
    form: SignatureAgentForm;
    label: string;
    headers: { 'content-digest': string; 'signature-agent': string; 'signature-input': string; signature: string };
    signatureBaseLines: string[];
  }>;
}

async function produceVectors(file: Omit<VectorFile, 'vectors' | 'source'>): Promise<VectorFile['vectors']> {
  const identity = await identityFor(file.key.privateJwk, file.key.publicJwk, file.agentUrl);
  const out: VectorFile['vectors'] = [];
  for (const form of SIGNATURE_AGENT_FORMS) {
    const signed = await signMessage(
      identity,
      { method: file.request.method, url: file.request.url, body: body(file.request.body) },
      {
        form,
        label: file.params.label,
        created: file.params.created,
        lifetimeSeconds: file.params.expires - file.params.created,
        nonce: file.params.nonce,
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

describe('cross-language vectors (design section 10.3)', () => {
  const inputs: Omit<VectorFile, 'vectors' | 'source'> = {
    key: B14,
    agentUrl: AGENT_URL,
    request: { method: 'POST', url: TOKEN_URL, authority: 'robutler.ai', path: '/api/auth/cli/token', query: '', body: '{}' },
    params: { label: 'sig1', created: CREATED, expires: EXPIRES, nonce: FIXED_NONCE, alg: 'ed25519', tag: 'web-bot-auth' },
  };

  it('matches the committed vectors byte for byte, or writes them when no SDK has yet', async () => {
    if (!existsSync(VECTORS_PATH)) {
      const vectors = await produceVectors(inputs);
      const file: VectorFile = {
        source:
          'W2 design section 10.3 cross-language vectors, written 2026-09-17 by the TypeScript SDK signer ' +
          '(webagents/typescript/tests/unit/crypto/http-signature.test.ts). Inputs: the RFC 9421 Appendix B.1.4 ' +
          'Ed25519 test key (denylisted by the platform profile, P section 6.8), POST /api/auth/cli/token at ' +
          'robutler.ai with body {}, created 1758067200, expires 1758067260, nonce = bytes 0x00..0x3f in standard ' +
          'base64, agent URL https://agent.example/agents/mini, one vector per Signature-Agent form. A consumer ' +
          'signs the inputs and compares every header and every base line; Ed25519 is deterministic so the ' +
          'signature bytes are pinned too.',
        ...inputs,
        vectors,
      };
      mkdirSync(path.dirname(VECTORS_PATH), { recursive: true });
      writeFileSync(VECTORS_PATH, `${JSON.stringify(file, null, 2)}\n`);
      console.log(`[http-signature.test] wrote the cross-language vectors to ${VECTORS_PATH}`);
    }

    const committed = JSON.parse(readFileSync(VECTORS_PATH, 'utf8')) as VectorFile;
    // The inputs come from the FILE, whichever SDK wrote it; this signer must map them to its outputs.
    for (const field of ['key', 'agentUrl', 'request', 'params', 'vectors'] as const) {
      expect(committed[field], `vectors.json has no ${field}`).toBeDefined();
    }
    expect(committed.key.thumbprint).toBe(B14.thumbprint);
    expect(committed.vectors.map((v) => v.form).sort()).toEqual([...SIGNATURE_AGENT_FORMS].sort());

    const mine = await produceVectors(committed);
    for (const theirs of committed.vectors) {
      const ours = mine.find((v) => v.form === theirs.form)!;
      for (const header of ['content-digest', 'signature-agent', 'signature-input', 'signature'] as const) {
        expect(ours.headers[header], `${theirs.form}: ${header}`).toBe(theirs.headers[header]);
      }
      expect(ours.signatureBaseLines, `${theirs.form}: signature base`).toEqual(theirs.signatureBaseLines);
      // Independently: the committed signature verifies under the committed key over the committed base.
      const sig = signatureMembers(theirs.headers.signature).get(theirs.label)!;
      expect(verifies(theirs.signatureBaseLines.join('\n'), sig, committed.key.publicJwk), `${theirs.form}: verify`).toBe(true);
    }
  });

  it('the vector inputs are the ones design 10.3 names', () => {
    expect(inputs.request).toMatchObject({ method: 'POST', authority: 'robutler.ai', path: '/api/auth/cli/token', body: '{}' });
    expect(inputs.params.expires - inputs.params.created).toBe(60);
    expect(Buffer.from(inputs.params.nonce, 'base64')).toHaveLength(64);
    expect(generateKeyPairSync('ed25519').publicKey.export({ format: 'jwk' })).toMatchObject({ kty: 'OKP', crv: 'Ed25519' });
  });
});

// 2026-09-18 (W2 review): the platform verifies at most AOAUTH_MAX_LABELS = 2
// `web-bot-auth` labels per request and answers `signature_malformed` above
// that. `AgentIdentity` refuses a third key at initialize(); a caller's own
// SigningIdentity is refused here, before a request the platform would reject.
import { MAX_HELD_KEYS } from '../../../src/crypto/identity';

describe('the held-key cap (design section 2.5, platform AOAUTH_MAX_LABELS)', () => {
  async function heldKey() {
    const pair = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
    const identity = new AgentIdentity({ agentId: 'k', issuer: AGENT_URL, ...pair });
    await identity.initialize();
    return identity.getHeldKeys()[0];
  }

  it('is two: the current key and one previous key', () => {
    expect(MAX_HELD_KEYS).toBe(2);
  });

  it('signs with two held keys and refuses three', async () => {
    const keys = [await heldKey(), await heldKey(), await heldKey()];
    const message = { method: 'POST', url: TOKEN_URL, body: body('{}') };
    const two = await signMessage({ issuer: AGENT_URL, getHeldKeys: () => keys.slice(0, 2) }, message);
    expect(two.labels.map((l) => l.label)).toEqual(['sig1', 'sig2']);
    await expect(signMessage({ issuer: AGENT_URL, getHeldKeys: () => keys }, message)).rejects.toThrow(
      /at most 2 signatures/,
    );
  });
});
