/**
 * AgentIdentity: the Ed25519 key set an agent publishes and signs with
 * (ADR 0038 step 5, W2 design sections 2.3, 2.5, 3.1, 7.2 and 9.1,
 * 2026-09-17).
 *
 * What these pin, and why each matters to the platform's verifier:
 *   * `kid` IS the RFC 7638 thumbprint, computed here and checked against an
 *     independent computation, because the verifier selects the key by
 *     `keyid` alone and a chosen `kid` would let two agents publish the same
 *     id for different keys;
 *   * the published entry carries the three thumbprint members, `kid` and
 *     `use`, and NO `alg` (the verifier never reads one; WebCrypto refuses
 *     `alg: "ed25519"` if it is ever passed through);
 *   * the claim token is the one JWT left, and it verifies BY `kid` against
 *     the key set, which is how the claim route selects the key;
 *   * rotation publishes every held key, current first, and `getHeldKeys()`
 *     hands the signer one entry per key;
 *   * Ed25519 only: any other key type is refused at `initialize()`.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { createHash, generateKeyPairSync } from 'node:crypto';
import { createLocalJWKSet, decodeProtectedHeader, generateKeyPair, importJWK, jwtVerify, type KeyLike } from 'jose';
import { AgentIdentity, MAX_HELD_KEYS, canonicalAgentUrl } from '../../../src/crypto/identity';

const ISSUER = 'https://example.com/agents/test-agent';
const PLATFORM = 'https://platform.example';

/** RFC 7638: SHA-256 over the lexically ordered required members, base64url unpadded. */
function thumbprintOf(x: string): string {
  return createHash('sha256').update(JSON.stringify({ crv: 'Ed25519', kty: 'OKP', x })).digest('base64url');
}

describe('AgentIdentity', () => {
  let identity: AgentIdentity;

  beforeEach(async () => {
    identity = new AgentIdentity({ agentId: 'test-agent', issuer: ISSUER });
    await identity.initialize();
  });

  describe('initialization', () => {
    it('generates a key pair on initialize', () => {
      expect(identity.publicKey).toBeDefined();
      expect(identity.privateKey).toBeDefined();
    });

    it('sets agentId and issuer, and strips one trailing slash', () => {
      expect(identity.agentId).toBe('test-agent');
      expect(identity.issuer).toBe(ISSUER);
      const id = new AgentIdentity({ agentId: 'a', issuer: `${ISSUER}/` });
      expect(id.issuer).toBe(ISSUER);
    });

    it('stores the issuer in the spelling the platform derives: host lowercased, default port dropped (2026-09-18)', () => {
      // The platform reads the principal off Signature-Agent through a WHATWG
      // parse and compares the card to it by string equality; a verbatim
      // issuer made a verifying signature `card_not_self_naming`.
      const id = new AgentIdentity({ agentId: 'a', issuer: 'https://Example.COM:443/agents/test-agent/' });
      expect(id.issuer).toBe(ISSUER);
      expect(id.keySetUrl).toBe(`${ISSUER}/.well-known/jwks.json`);
      expect(new AgentIdentity({ agentId: 'a', issuer: 'https://example.com:8443/x' }).issuer).toBe(
        'https://example.com:8443/x',
      );
      expect(new AgentIdentity({ agentId: 'a', issuer: 'http://Host.Example:80/x' }).issuer).toBe('http://host.example/x');
      expect(canonicalAgentUrl('/agents/mini/')).toBe('/agents/mini');
      expect(canonicalAgentUrl('https://Example.COM:443/')).toBe('https://example.com');
      expect(canonicalAgentUrl(ISSUER)).toBe(ISSUER);
    });

    it('derives the key set and card URLs from the issuer (design section 3.1)', () => {
      expect(identity.keySetUrl).toBe(`${ISSUER}/.well-known/jwks.json`);
      expect(identity.cardUrl).toBe(`${ISSUER}/.well-known/agent.json`);
    });

    it('refuses a key that is not Ed25519, at initialize, loudly', async () => {
      const rsa = generateKeyPairSync('rsa', { modulusLength: 2048 });
      const wrong = new AgentIdentity({
        agentId: 'rsa',
        issuer: ISSUER,
        privateKey: rsa.privateKey as unknown as KeyLike,
        publicKey: rsa.publicKey as unknown as KeyLike,
      });
      await expect(wrong.initialize()).rejects.toThrow(/only Ed25519/);
    });

    it('no longer takes a chosen kid or an agent_path', () => {
      // Both were bearer-era knobs: the platform composed `iss + agent_path +
      // sub` and selected the key by a `kid` the agent chose. The principal
      // is the issuer now and the kid is the thumbprint, so a config carrying
      // either is a type error, and at runtime it is simply ignored.
      const legacy = new AgentIdentity({ agentId: 'a', issuer: ISSUER, kid: 'chosen', agentPath: '/agents' } as never);
      expect('agentPath' in legacy).toBe(false);
      expect(() => legacy.kid).toThrow('not initialized');
    });
  });

  describe('kid and the key set', () => {
    it('kid equals the RFC 7638 thumbprint of the public key', () => {
      const [key] = identity.getJwks().keys;
      expect(identity.kid).toBe(thumbprintOf(key.x));
      expect(identity.kid).toMatch(/^[A-Za-z0-9_-]{43}$/);
      expect(key.kid).toBe(identity.kid);
    });

    it('publishes exactly kty, crv, x, kid and use, with no alg', () => {
      const jwks = identity.getJwks();
      expect(jwks.keys).toHaveLength(1);
      expect(Object.keys(jwks.keys[0]).sort()).toEqual(['crv', 'kid', 'kty', 'use', 'x']);
      expect(jwks.keys[0]).toMatchObject({ kty: 'OKP', crv: 'Ed25519', use: 'sig' });
      expect('alg' in jwks.keys[0]).toBe(false);
      expect('d' in jwks.keys[0]).toBe(false);
    });

    it('the RFC 9421 B.1.4 test key gets the thumbprint the draft vectors carry', async () => {
      const x = 'JrQLj5P_89iXES9-vFgrIy29clF9CC_oPPsw3c5D0bs';
      const d = 'n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU';
      const fixed = new AgentIdentity({
        agentId: 'b14',
        issuer: ISSUER,
        privateKey: (await importJWK({ kty: 'OKP', crv: 'Ed25519', x, d }, 'EdDSA')) as KeyLike,
        publicKey: (await importJWK({ kty: 'OKP', crv: 'Ed25519', x }, 'EdDSA')) as KeyLike,
      });
      await fixed.initialize();
      expect(fixed.kid).toBe('poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U');
    });

    it('throws if not initialized', () => {
      const uninitialized = new AgentIdentity({ agentId: 'x', issuer: ISSUER });
      expect(() => uninitialized.getJwks()).toThrow('not initialized');
      expect(() => uninitialized.kid).toThrow('not initialized');
      expect(() => uninitialized.getHeldKeys()).toThrow('not initialized');
    });

    it('still exports the SPKI PEM for the platform-side bridge', () => {
      expect(identity.getPublicKeySpki()).toMatch(/^-----BEGIN PUBLIC KEY-----/);
    });
  });

  describe('rotation (design section 2.5)', () => {
    it('publishes every held key, current first, and hands the signer one entry per key', async () => {
      const previous = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
      const rotating = new AgentIdentity({
        agentId: 'rot',
        issuer: ISSUER,
        previousKeys: [previous],
      });
      await rotating.initialize();

      const keys = rotating.getJwks().keys;
      expect(keys).toHaveLength(2);
      expect(keys[0].kid).toBe(rotating.kid);
      expect(new Set(keys.map((k) => k.kid)).size).toBe(2);
      for (const key of keys) expect(key.kid).toBe(thumbprintOf(key.x));

      const held = rotating.getHeldKeys();
      expect(held.map((h) => h.kid)).toEqual(keys.map((k) => k.kid));
      // Each entry signs with ITS key: the signature verifies under that entry's public key only.
      const data = new TextEncoder().encode('base');
      const sigs = await Promise.all(held.map((h) => h.sign(data)));
      for (let i = 0; i < held.length; i += 1) {
        expect(sigs[i]).toBeInstanceOf(Uint8Array);
        expect(sigs[i]).toHaveLength(64);
        const pub = await importJWK({ kty: 'OKP', crv: 'Ed25519', x: keys[i].x }, 'EdDSA');
        const ok = await crypto.subtle.verify('Ed25519', await toCryptoKey(pub as KeyLike, keys[i].x), sigs[i], data);
        expect(ok, `entry ${i}`).toBe(true);
        const other = await toCryptoKey(pub as KeyLike, keys[(i + 1) % held.length].x);
        expect(await crypto.subtle.verify('Ed25519', other, sigs[i], data), `entry ${i} under another key`).toBe(false);
      }
    });

    it('refuses a second previous key, because the platform verifies at most two labels (2026-09-18)', async () => {
      // AOAUTH_MAX_LABELS = 2 on the platform: a request with three
      // `web-bot-auth` labels is `signature_malformed`, so a third held key
      // would break every signed request rather than ease a rotation.
      expect(MAX_HELD_KEYS).toBe(2);
      const previous = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
      const older = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
      const rotating = new AgentIdentity({ agentId: 'rot', issuer: ISSUER, previousKeys: [previous, older] });
      await expect(rotating.initialize()).rejects.toThrow(/at most 2 signatures/);
      // The same key listed twice is one key, not a third: still admitted.
      const twice = new AgentIdentity({ agentId: 'rot', issuer: ISSUER, previousKeys: [previous, previous] });
      await twice.initialize();
      expect(twice.getHeldKeys()).toHaveLength(2);
    });

    it('lists a key once even when the previous key is the current one', async () => {
      const pair = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
      const dup = new AgentIdentity({ agentId: 'dup', issuer: ISSUER, ...pair, previousKeys: [pair] });
      await dup.initialize();
      expect(dup.getJwks().keys).toHaveLength(1);
      expect(dup.getHeldKeys()).toHaveLength(1);
    });

    it('refuses a previous key that is not Ed25519', async () => {
      const rsa = generateKeyPairSync('rsa', { modulusLength: 2048 });
      const mixed = new AgentIdentity({
        agentId: 'mixed',
        issuer: ISSUER,
        previousKeys: [{ privateKey: rsa.privateKey as unknown as KeyLike, publicKey: rsa.publicKey as unknown as KeyLike }],
      });
      await expect(mixed.initialize()).rejects.toThrow(/only Ed25519/);
    });
  });

  describe('sign()', () => {
    it('signs with the current key and answers standard base64 of 64 bytes', async () => {
      const data = new TextEncoder().encode('hello');
      const b64 = await identity.sign(data);
      const bytes = Buffer.from(b64, 'base64');
      expect(bytes).toHaveLength(64);
      expect(b64).toMatch(/^[A-Za-z0-9+/]{86}==$/);
      const [key] = identity.getJwks().keys;
      const pub = await toCryptoKey(null, key.x);
      expect(await crypto.subtle.verify('Ed25519', pub, bytes, data)).toBe(true);
    });

    it('throws if not initialized', async () => {
      const uninitialized = new AgentIdentity({ agentId: 'x', issuer: ISSUER });
      await expect(uninitialized.sign(new Uint8Array(1))).rejects.toThrow('not initialized');
    });
  });

  describe('getOpenIdConfiguration', () => {
    it('names the issuer and the key set URL', () => {
      const config = identity.getOpenIdConfiguration();
      expect(config.issuer).toBe(ISSUER);
      expect(config.jwks_uri).toBe(`${ISSUER}/.well-known/jwks.json`);
      expect(config.grant_types_supported).toContain('client_credentials');
      expect(config.id_token_signing_alg_values_supported as string[]).toContain('EdDSA');
    });
  });

  describe('mintClaimToken (design section 7.2): the one JWT left', () => {
    it('mintToken is gone', () => {
      expect('mintToken' in identity).toBe(false);
      expect(typeof (identity as unknown as { mintToken?: unknown }).mintToken).toBe('undefined');
    });

    it('verifies by kid against the published key set, EdDSA only', async () => {
      const token = await identity.mintClaimToken(PLATFORM);
      expect(token.split('.')).toHaveLength(3);

      const header = decodeProtectedHeader(token);
      expect(header.alg).toBe('EdDSA');
      expect(header.kid).toBe(identity.kid);

      // `createLocalJWKSet` selects the key by the header's `kid`, as the claim route does.
      const { payload, protectedHeader } = await jwtVerify(token, createLocalJWKSet(identity.getJwks()), {
        issuer: ISSUER,
        audience: `${PLATFORM}/claim`,
        algorithms: ['EdDSA'],
      });
      expect(protectedHeader.kid).toBe(identity.kid);
      expect(payload.sub).toBe('test-agent');
      expect(payload.iss).toBe(ISSUER);
      expect(payload.aud).toBe(`${PLATFORM}/claim`);
      expect(payload.scope).toBe('agent:claim');
      expect(typeof payload.jti).toBe('string');
      expect(payload.exp! - payload.iat!).toBe(600);
      expect(payload.nbf).toBe(payload.iat);
      // Bearer-era claims are gone.
      expect('client_id' in payload).toBe(false);
      expect('token_type' in payload).toBe(false);
      expect('agent_path' in payload).toBe(false);
    });

    it('a token without the published kid does not verify by kid', async () => {
      const token = await identity.mintClaimToken(PLATFORM);
      const other = new AgentIdentity({ agentId: 'other', issuer: ISSUER });
      await other.initialize();
      await expect(jwtVerify(token, createLocalJWKSet(other.getJwks()), { algorithms: ['EdDSA'] })).rejects.toThrow();
    });

    it('addresses the platform claim endpoint whatever the trailing slash, and honours the TTL', async () => {
      const token = await identity.mintClaimToken(`${PLATFORM}///`, 60);
      const { payload } = await jwtVerify(token, createLocalJWKSet(identity.getJwks()));
      expect(payload.aud).toBe(`${PLATFORM}/claim`);
      expect(payload.exp! - payload.iat!).toBe(60);
    });

    it('mints a unique jti each time', async () => {
      const a = await jwtVerify(await identity.mintClaimToken(PLATFORM), createLocalJWKSet(identity.getJwks()));
      const b = await jwtVerify(await identity.mintClaimToken(PLATFORM), createLocalJWKSet(identity.getJwks()));
      expect(a.payload.jti).not.toBe(b.payload.jti);
    });

    it('throws if not initialized', async () => {
      const uninitialized = new AgentIdentity({ agentId: 'x', issuer: ISSUER });
      await expect(uninitialized.mintClaimToken(PLATFORM)).rejects.toThrow('not initialized');
    });
  });
});

/** A WebCrypto verifying key for a public `x`; `_` keeps the jose import path exercised beside it. */
async function toCryptoKey(_: KeyLike | null, x: string): Promise<CryptoKey> {
  return crypto.subtle.importKey('jwk', { kty: 'OKP', crv: 'Ed25519', x }, { name: 'Ed25519' }, false, ['verify']);
}
