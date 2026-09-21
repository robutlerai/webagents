/**
 * Agent identity: the Ed25519 key set an agent publishes and signs with.
 *
 * Rewritten 2026-09-17 for ADR 0038 step 5 (W2 design sections 2.3, 3.1
 * and 9.1). Before that day the identity minted a bearer JWT that the platform
 * verified against a PEM read off the agent card; it is now a Web Bot Auth
 * signer (RFC 9421, draft-ietf-webbotauth-httpsig-protocol-00) and the only
 * JWT left is the claim token. Three facts about it decide everything else:
 *
 *  * `issuer` IS the agent URL, the principal the platform registers
 *    (`agent_registrations.agent_url`, design section 3.1). The key set is
 *    served at `${issuer}/.well-known/jwks.json` and that is the value every
 *    signature carries in `Signature-Agent`, so `issuer` must be the URL the
 *    agent is actually mounted at (`serve()` composes `publicUrl` + `basePath`),
 *    never the bare origin of a prefixed agent.
 *  * `kid` is the RFC 7638 thumbprint of the public key (design section 2.3),
 *    computed in `initialize()`, and nothing else: an operator-chosen key id
 *    would let two agents publish the same `kid` for different keys, and the
 *    platform selects the key by `keyid` alone. There is no `alg` on the
 *    published entry either (the verifier never reads one).
 *  * Ed25519 only (ADR operator decision 3). A key of any other type is
 *    refused at `initialize()`, loudly, rather than published as a set the
 *    platform will call `key_set_invalid`.
 *
 * ROTATION (design section 2.5). While an agent moves to a new key it holds
 * the outgoing one beside it (`previousKeys`): the key set lists every held
 * key, current first, and `getHeldKeys()` hands the signer one entry per key
 * so a request is signed once per key (`sig1`, `sig2`). That is what lets the
 * platform's continuity rule (admit a new key only when a key it already holds
 * co-signed) be switched on with no SDK change.
 *
 * SIGNING RUNS ON WEBCRYPTO, from a key imported out of the JWK. jose's Node
 * build returns `KeyObject`s from `generateKeyPair` and `importJWK`, and
 * `crypto.subtle.sign` refuses those ("not of type CryptoKey"); the previous
 * `sign()` called exactly that and had no caller, so nothing noticed. A
 * `CryptoKey` (browser build, or a caller's own) is used as it is.
 */

import {
  calculateJwkThumbprint,
  exportJWK,
  exportSPKI,
  generateKeyPair,
  SignJWT,
  type JWK,
  type KeyLike,
} from 'jose';

/** One key pair the identity holds beside its current key while rotating. */
export interface HeldKeyPair {
  privateKey: KeyLike;
  publicKey: KeyLike;
}

export interface AgentIdentityConfig {
  /** Agent ID (`sub` of the claim token). */
  agentId: string;
  /**
   * The agent URL: the principal. `https://host/agents/mini` for an agent
   * served under `/agents/mini`. The key set lives at
   * `${issuer}/.well-known/jwks.json` and the card at
   * `${issuer}/.well-known/agent.json`. Stored in the canonical spelling
   * (`canonicalAgentUrl`), whatever spelling is passed.
   */
  issuer: string;
  /** Pre-generated Ed25519 private key. If omitted, a new pair is generated. */
  privateKey?: KeyLike;
  /** Pre-generated Ed25519 public key. */
  publicKey?: KeyLike;
  /**
   * The ONE key still held from before a rotation. It is published in the
   * key set and co-signs every request (design section 2.5). Drop it once
   * the platform has admitted the new one. A list, for the shape's sake, but
   * `initialize()` refuses more than one distinct previous key: the platform
   * verifies at most `MAX_HELD_KEYS` labels per request and refuses the whole
   * request above that, so a second previous key would break every signed
   * request rather than ease an overlapping rotation.
   */
  previousKeys?: HeldKeyPair[];
}

/**
 * The most keys an identity holds, and so the most signatures a request
 * carries: the current key and one previous key. Equal to the platform's
 * `AOAUTH_MAX_LABELS` (`lib/auth/web-bot-auth/profile.ts`), which answers
 * `signature_malformed` to a request with more `web-bot-auth` labels. Found
 * 2026-09-18 in the W2 review: `previousKeys` was unbounded and the signer
 * signed once per held key, so two outgoing keys during overlapping rotations
 * broke registration and every later platform call.
 */
export const MAX_HELD_KEYS = 2;

/**
 * The one spelling of an agent URL: `origin + pathname` as the WHATWG parser
 * reads it (host lowercased, a default port dropped), trailing slashes
 * stripped. A value that is not an absolute http(s) URL (a relative
 * principal such as `/agents/mini`) comes back with only the trailing
 * slashes stripped.
 *
 * WHY (2026-09-18, W2 review): the platform derives the principal from
 * `Signature-Agent` through exactly that parse and compares the card's `url`,
 * `client_id` and `jwks_uri` to it by string equality. The signer already
 * spelled its value from `url.origin`, but the identity kept `issuer`
 * verbatim and the card derived from the identity, so a `publicUrl` spelled
 * `https://Agents.Example.com:443` signed a request that verified and was
 * refused `card_not_self_naming`. The identity and the card now both read
 * this one function, so they cannot disagree.
 */
export function canonicalAgentUrl(value: string): string {
  const trimmed = value.replace(/\/+$/, '');
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return trimmed;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return trimmed;
  return `${url.origin}${url.pathname}`.replace(/\/+$/, '');
}

/**
 * What the request signer needs from one held key: its thumbprint and a raw
 * Ed25519 signature. Raw bytes, not base64: the signer serialises them as an
 * RFC 9651 Byte Sequence itself, and a string here would only be decoded
 * again on the next line.
 */
export interface HeldKey {
  /** RFC 7638 thumbprint, base64url, 43 characters: the `keyid` parameter. */
  readonly kid: string;
  /** Ed25519 over `data`: the 64 signature bytes (RFC 8032 section 5.1.6). */
  sign(data: Uint8Array): Promise<Uint8Array>;
}

interface HeldKeyState {
  kid: string;
  x: string;
  privateKey: KeyLike;
  signingKey: CryptoKey;
}

/** The public JWK of a held key as the key set publishes it: the three thumbprint members, `kid`, `use`, no `alg`. */
export interface PublishedJwk {
  kty: 'OKP';
  crv: 'Ed25519';
  x: string;
  kid: string;
  use: 'sig';
}

function isCryptoKey(key: unknown): key is CryptoKey {
  return typeof CryptoKey !== 'undefined' && key instanceof CryptoKey;
}

async function publicMembersOf(publicKey: KeyLike): Promise<{ kty: 'OKP'; crv: 'Ed25519'; x: string }> {
  const jwk: JWK = await exportJWK(publicKey);
  if (jwk.kty !== 'OKP' || jwk.crv !== 'Ed25519' || typeof jwk.x !== 'string') {
    throw new Error(
      `AgentIdentity: only Ed25519 keys are supported (got kty=${jwk.kty ?? '?'} crv=${jwk.crv ?? '?'})`,
    );
  }
  return { kty: 'OKP', crv: 'Ed25519', x: jwk.x };
}

/**
 * A WebCrypto key `crypto.subtle.sign('Ed25519', ...)` accepts. A `CryptoKey`
 * is taken as it is; a jose `KeyObject` (the Node build) is exported to a
 * private JWK and imported non-extractable.
 */
async function toSigningKey(privateKey: KeyLike): Promise<CryptoKey> {
  if (isCryptoKey(privateKey)) return privateKey;
  const jwk = await exportJWK(privateKey);
  if (jwk.kty !== 'OKP' || jwk.crv !== 'Ed25519' || typeof jwk.d !== 'string') {
    throw new Error('AgentIdentity: the private key is not an Ed25519 key');
  }
  return crypto.subtle.importKey('jwk', jwk as JsonWebKey, { name: 'Ed25519' }, false, ['sign']);
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

async function signWith(key: CryptoKey, data: Uint8Array): Promise<Uint8Array> {
  const sig = await crypto.subtle.sign('Ed25519', key, data as unknown as ArrayBuffer);
  return new Uint8Array(sig);
}

export class AgentIdentity {
  readonly agentId: string;
  readonly issuer: string;

  private _privateKey: KeyLike | null = null;
  private _publicKey: KeyLike | null = null;
  private _previous: HeldKeyPair[];
  private _held: HeldKeyState[] = [];
  private _jwksJson: { keys: PublishedJwk[] } | null = null;
  private _spkiPem: string | null = null;

  constructor(config: AgentIdentityConfig) {
    this.agentId = config.agentId;
    this.issuer = canonicalAgentUrl(config.issuer);
    this._previous = config.previousKeys ?? [];
    if (config.privateKey) this._privateKey = config.privateKey;
    if (config.publicKey) this._publicKey = config.publicKey;
  }

  async initialize(): Promise<void> {
    if (!this._privateKey || !this._publicKey) {
      const { privateKey, publicKey } = await generateKeyPair('EdDSA', { crv: 'Ed25519' });
      this._privateKey = privateKey;
      this._publicKey = publicKey;
    }

    const held: HeldKeyState[] = [];
    for (const pair of [{ privateKey: this._privateKey, publicKey: this._publicKey }, ...this._previous]) {
      const members = await publicMembersOf(pair.publicKey);
      const kid = await calculateJwkThumbprint(members, 'sha256');
      // The same key listed twice (a "previous" key that is the current one)
      // would make the signer sign twice with one key and the set carry a
      // duplicate `kid`; one entry per thumbprint.
      if (held.some((h) => h.kid === kid)) continue;
      held.push({ kid, x: members.x, privateKey: pair.privateKey, signingKey: await toSigningKey(pair.privateKey) });
    }
    if (held.length > MAX_HELD_KEYS) {
      throw new Error(
        `AgentIdentity: ${held.length} distinct keys held, but the platform verifies at most ` +
          `${MAX_HELD_KEYS} signatures per request; hold the current key and one previous key ` +
          'while rotating, and retire the older one first',
      );
    }
    this._held = held;
    this._jwksJson = {
      keys: held.map((h) => ({ kty: 'OKP', crv: 'Ed25519', x: h.x, kid: h.kid, use: 'sig' })),
    };
    try {
      this._spkiPem = await exportSPKI(this._publicKey as Parameters<typeof exportSPKI>[0]);
    } catch {
      this._spkiPem = null;
    }
  }

  /** The RFC 7638 thumbprint of the current key: the `keyid` on the wire and the `kid` in the key set. */
  get kid(): string {
    const current = this._held[0];
    if (!current) throw new Error('AgentIdentity not initialized');
    return current.kid;
  }

  /** JWK Set for `/.well-known/jwks.json`: every held key, current first. */
  getJwks(): { keys: PublishedJwk[] } {
    if (!this._jwksJson) throw new Error('AgentIdentity not initialized');
    return this._jwksJson;
  }

  /** Where the key set is published: the identifier every signature names (design section 2.4). */
  get keySetUrl(): string {
    return `${this.issuer}/.well-known/jwks.json`;
  }

  /** Where the registration-time card is served (design section 3.3). */
  get cardUrl(): string {
    return `${this.issuer}/.well-known/agent.json`;
  }

  /**
   * The current public key as an SPKI PEM.
   *
   * NOTHING NEEDS THIS ANY MORE (2026-09-19). The card stopped publishing the
   * PEM, and the one reason this method survived that, the platform's pre-W2
   * row bridge (a registration made before key sets adopted a key whose SPKI
   * equalled the PEM pinned on its row), is gone: the operator ruled that no
   * back-compat is owed, so the platform reads no PEM on any path and a
   * registration holding no key row is decided by the key admission mode like
   * any other. Its only caller in this repository is
   * `tests/unit/crypto/identity.test.ts`, which asserts the PEM shape; no SDK
   * code path calls it. It stays because it is a public method of an exported
   * class and an agent may hold the PEM form for its own tooling. Do not
   * reach for it as an identifier: the key's identity is its RFC 7638
   * thumbprint, which is what `keyid` carries and what the platform stores.
   */
  getPublicKeySpki(): string {
    if (!this._spkiPem) throw new Error('AgentIdentity not initialized (or key not exportable)');
    return this._spkiPem;
  }

  /** OpenID configuration for /.well-known/openid-configuration */
  getOpenIdConfiguration(): Record<string, unknown> {
    return {
      issuer: this.issuer,
      jwks_uri: this.keySetUrl,
      response_types_supported: ['token'],
      subject_types_supported: ['public'],
      id_token_signing_alg_values_supported: ['EdDSA', 'RS256'],
      scopes_supported: ['read', 'write', 'admin', 'namespace:*', 'tools:*'],
      token_endpoint_auth_methods_supported: ['client_secret_basic', 'client_secret_post'],
      grant_types_supported: ['client_credentials'],
    };
  }

  /**
   * The claim token: the ONE JWT this identity still mints (design section
   * 7.2). It is not an HTTP request from the agent, so it cannot be a signed
   * request; it is what a human pastes into `POST /api/agents/{id}/claim` to
   * prove the agent consents to being owned. `EdDSA`, `kid` = thumbprint so
   * the platform selects the key from the registration's key set, `aud` =
   * `${platform}/claim`, `scope` = `agent:claim`, a `jti` the platform spends
   * once, ten minutes by default.
   */
  async mintClaimToken(platformUrl: string, ttlSeconds = 600): Promise<string> {
    if (!this._privateKey || this._held.length === 0) throw new Error('AgentIdentity not initialized');
    const base = platformUrl.replace(/\/+$/, '');
    const now = Math.floor(Date.now() / 1000);
    return new SignJWT({ scope: 'agent:claim' })
      .setProtectedHeader({ alg: 'EdDSA', kid: this.kid })
      .setIssuedAt(now)
      .setNotBefore(now)
      .setExpirationTime(now + ttlSeconds)
      .setSubject(this.agentId)
      .setIssuer(this.issuer)
      .setAudience(`${base}/claim`)
      .setJti(crypto.randomUUID())
      .sign(this._privateKey);
  }

  /**
   * Ed25519 over `data` with the CURRENT key, the 64 signature bytes as
   * standard base64 (what a `Signature` Byte Sequence carries between colons).
   */
  async sign(data: Uint8Array): Promise<string> {
    const current = this._held[0];
    if (!current) throw new Error('AgentIdentity not initialized');
    return bytesToBase64(await signWith(current.signingKey, data));
  }

  /**
   * Every key the identity holds, current first: the request signer produces
   * one signature label per entry (design section 2.5).
   */
  getHeldKeys(): ReadonlyArray<HeldKey> {
    if (this._held.length === 0) throw new Error('AgentIdentity not initialized');
    return this._held.map((h) => ({ kid: h.kid, sign: (data: Uint8Array) => signWith(h.signingKey, data) }));
  }

  get publicKey(): KeyLike | null {
    return this._publicKey;
  }

  get privateKey(): KeyLike | null {
    return this._privateKey;
  }
}
