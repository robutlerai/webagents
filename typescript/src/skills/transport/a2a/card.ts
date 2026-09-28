/**
 * The A2A v1.0 agent card and its signature (A2A v1.0 sections 8 and 8.4).
 *
 * TWO CARDS, ON PURPOSE (plan item 1.3, 2026-09-26). The registration card at
 * `/.well-known/agent.json` (`server/card.ts`) is what the platform reads once
 * per principal and it refuses a card that does not name itself
 * (`lib/auth/agent-auth.ts`), so it stays byte-compatible and is not touched.
 * The v1.0 card served BESIDE it at `/.well-known/agent-card.json` is for
 * peers: OpenClaw and Hermes both look there first (Hermes falls back to
 * `agent.json`, and reads it as a v1.0 card, which is why the legacy card must
 * keep serving something harmless rather than something v1.0-shaped and
 * wrong). The JSON-RPC interface is listed first, because Hermes takes the
 * first `JSONRPC` entry without checking its version and OpenClaw posts to a
 * configured URL without reading the card at all.
 *
 * THE SIGNATURE IS DETACHED JWS OVER THE JCS FORM OF THE PARSED CARD. Section
 * 8.4: drop `signatures`, drop every field left at its default, keep required
 * fields even when empty, keep `optional` (explicit-presence) fields even at
 * their default, canonicalise with RFC 8785, sign
 * `base64url(protected) + "." + base64url(payload)` and transmit the header
 * and the signature only. The tables below say which fields are required and
 * which have explicit presence, straight from `a2a.proto`.
 *
 * WHERE a2a-python DIVERGES (src/a2a/utils/signing.py, released in 1.1.4 on
 * 2026-09-08), so a reviewer comparing bytes is not surprised:
 *   1. its `_clean_empty` drops every `""`, `[]` and `{}`, REQUIRED ones
 *      included: for a card with an empty `description` or no `skills` the two
 *      payloads differ, and only then. This implementation keeps them, as the
 *      spec says, and the shared fixture pins that
 *      (`python/tests/fixtures/a2a/vectors.json`, `signing_payload_cases`);
 *   2. before 1.1.4 it signed `json.dumps(sort_keys=True)` with ASCII escapes,
 *      which is not JCS for non-ASCII text or for numbers such as `1.0`;
 *   3. it canonicalises the parsed proto, and so must we: a server adds v0.3
 *      compatibility fields after signing, and a field sent at its default
 *      (`"required": false`) must be dropped before canonicalising, which the
 *      tables here do without a proto parser.
 *
 * Signing runs on WebCrypto. EdDSA with the agent's Ed25519 identity is the
 * production path (`kid` = the RFC 7638 thumbprint the key set publishes,
 * `jku` = the agent's own `jwks.json`); ES256 verifies for peers that sign
 * that way (a2a-python's sample does); HS256 exists only for the spec's
 * byte-assembly vector and needs a caller-supplied secret.
 */

import { canonicalize } from './jcs';
import { isBlockedIpLiteral } from '../../../crypto/jwks';
import type { HeldKey } from '../../../crypto/identity';

export const AGENT_CARD_WELL_KNOWN_SUFFIX = '/.well-known/agent-card.json';
export const LEGACY_CARD_WELL_KNOWN_SUFFIX = '/.well-known/agent.json';
export const KEY_SET_WELL_KNOWN_SUFFIX = '/.well-known/jwks.json';
/** Where the JSON-RPC and HTTP+JSON bindings are served, under the principal. */
export const A2A_RPC_SUBPATH = '/a2a';

export interface AgentInterface {
  url: string;
  protocolBinding: 'JSONRPC' | 'GRPC' | 'HTTP+JSON' | string;
  protocolVersion: string;
  tenant?: string;
}

export interface AgentSkill {
  id: string;
  name: string;
  description: string;
  tags: string[];
  examples?: string[];
  inputModes?: string[];
  outputModes?: string[];
  securityRequirements?: unknown[];
}

export interface AgentProvider {
  organization: string;
  url: string;
}

export interface AgentCardSignature {
  protected: string;
  signature: string;
  header?: Record<string, unknown>;
}

export interface AgentCardV1 {
  name: string;
  description: string;
  supportedInterfaces: AgentInterface[];
  provider?: AgentProvider;
  version: string;
  documentationUrl?: string;
  iconUrl?: string;
  capabilities: {
    streaming?: boolean;
    pushNotifications?: boolean;
    extendedAgentCard?: boolean;
    extensions?: Array<{ uri: string; description?: string; required?: boolean; params?: Record<string, unknown> }>;
  };
  securitySchemes?: Record<string, unknown>;
  securityRequirements?: Array<{ schemes: Record<string, { list: string[] }> }>;
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: AgentSkill[];
  signatures?: AgentCardSignature[];
}

export interface BuildCardOptions {
  /** The agent URL: the interfaces are `${principal}${rpcPath}`. */
  principal: string;
  rpcPath?: string;
  version?: string;
  provider?: AgentProvider | null;
  documentationUrl?: string | null;
  iconUrl?: string | null;
  streaming?: boolean;
  /** What a default-group caller may use; never owner-only tools. */
  skills: AgentSkill[];
  inputModes?: string[];
  outputModes?: string[];
}

export const BEARER_SCHEME_DESCRIPTION =
  'A Robutler platform token, an api key this agent accepts, or a peer token configured out of band.';
export const HTTPSIG_SCHEME_DESCRIPTION =
  "Web Bot Auth: RFC 9421 HTTP Message Signatures with Signature-Agent naming the caller's key set.";

/** The v1.0 card for `agent` served at `principal`. Pure: no environment, no request. */
export function buildA2AAgentCard(
  agent: { name: string; description?: string },
  options: BuildCardOptions,
): AgentCardV1 {
  const base = options.principal.replace(/\/+$/, '');
  const rpcUrl = `${base}${options.rpcPath ?? A2A_RPC_SUBPATH}`;
  const card: AgentCardV1 = {
    name: agent.name,
    description: agent.description ?? '',
    supportedInterfaces: [
      { url: rpcUrl, protocolBinding: 'JSONRPC', protocolVersion: '1.0' },
      { url: rpcUrl, protocolBinding: 'HTTP+JSON', protocolVersion: '1.0' },
    ],
    ...(options.provider ? { provider: options.provider } : {}),
    version: options.version ?? '1.0.0',
    ...(options.documentationUrl ? { documentationUrl: options.documentationUrl } : {}),
    ...(options.iconUrl ? { iconUrl: options.iconUrl } : {}),
    capabilities: { streaming: options.streaming ?? true, pushNotifications: false },
    securitySchemes: {
      bearer: { httpAuthSecurityScheme: { scheme: 'Bearer', bearerFormat: 'JWT', description: BEARER_SCHEME_DESCRIPTION } },
      httpsig: { httpAuthSecurityScheme: { scheme: 'HTTPSig', description: HTTPSIG_SCHEME_DESCRIPTION } },
    },
    securityRequirements: [{ schemes: { bearer: { list: [] } } }, { schemes: { httpsig: { list: [] } } }],
    defaultInputModes: options.inputModes ?? ['text/plain', 'application/json'],
    defaultOutputModes: options.outputModes ?? ['text/plain'],
    skills: options.skills,
  };
  return card;
}

/** Card skills from tool definitions: one skill per tool, tagged `tool`. */
export function skillsFromTools(
  tools: ReadonlyArray<{ type?: string; function?: { name: string; description?: string } }>,
): AgentSkill[] {
  const skills: AgentSkill[] = [];
  for (const tool of tools) {
    const fn = tool.function;
    if (!fn || (tool.type && tool.type !== 'function')) continue;
    skills.push({ id: fn.name, name: fn.name, description: fn.description ?? '', tags: ['tool'] });
  }
  return skills;
}

// ---------------------------------------------------------------------------
// The signing payload (section 8.4)
// ---------------------------------------------------------------------------

/**
 * Required fields per message type, keyed by the path of the type: `''` is
 * the card, `x[]` the items of the repeated field `x`. Always kept, even
 * empty. From `a2a.proto` and section 8 of the spec.
 */
const REQUIRED_FIELDS: Record<string, readonly string[]> = {
  '': ['name', 'description', 'supportedInterfaces', 'version', 'capabilities', 'defaultInputModes', 'defaultOutputModes', 'skills'],
  'supportedInterfaces[]': ['url', 'protocolBinding', 'protocolVersion'],
  'skills[]': ['id', 'name', 'description', 'tags'],
  'capabilities.extensions[]': ['uri'],
  provider: ['url', 'organization'],
};

/**
 * Explicit-presence (`optional`) fields per type: kept when present, even at
 * their default, because a sender that wrote `"streaming": false` set it.
 */
const EXPLICIT_PRESENCE_FIELDS: Record<string, readonly string[]> = {
  capabilities: ['streaming', 'pushNotifications', 'extendedAgentCard'],
  'supportedInterfaces[]': ['tenant'],
};

function isDefault(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  if (value === false || value === 0 || value === '') return true;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value as object).length === 0;
  return false;
}

function cleanValue(value: unknown, typePath: string): unknown {
  if (Array.isArray(value)) return value.map((item) => cleanValue(item, `${typePath}[]`));
  if (value && typeof value === 'object') return cleanObject(value as Record<string, unknown>, typePath);
  return value;
}

function cleanObject(value: Record<string, unknown>, typePath: string): Record<string, unknown> {
  const required = new Set(REQUIRED_FIELDS[typePath] ?? []);
  const explicit = new Set(EXPLICIT_PRESENCE_FIELDS[typePath] ?? []);
  const out: Record<string, unknown> = {};
  for (const [key, raw] of Object.entries(value)) {
    if (typePath === '' && key === 'signatures') continue;
    if (raw === null || raw === undefined) continue;
    const childPath = typePath ? `${typePath}.${key}` : key;
    const cleaned = cleanValue(raw, childPath);
    if (isDefault(cleaned) && !required.has(key) && !explicit.has(key)) continue;
    out[key] = cleaned;
  }
  return out;
}

/** The card as it is signed: no `signatures`, defaults removed per section 8.4. */
export function cardSigningPayload(card: Record<string, unknown>): Record<string, unknown> {
  return cleanObject(card, '');
}

/** The canonical (JCS) bytes a card signature covers. */
export function cardSigningBytes(card: Record<string, unknown>): Uint8Array {
  return new TextEncoder().encode(canonicalize(cardSigningPayload(card)));
}

// ---------------------------------------------------------------------------
// base64url
// ---------------------------------------------------------------------------

export function base64url(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function base64urlDecode(text: string): Uint8Array {
  const padded = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

export type CardSignatureAlg = 'EdDSA' | 'ES256' | 'HS256';

export interface CardSigner {
  alg: CardSignatureAlg;
  kid: string;
  /** The key set URL peers fetch `kid` from; the agent's own `jwks.json`. */
  jku?: string;
  sign(data: Uint8Array): Promise<Uint8Array>;
}

/** The protected header, canonical so both SDKs emit identical bytes. */
export function protectedHeader(signer: Pick<CardSigner, 'alg' | 'kid' | 'jku'>): Record<string, string> {
  return { alg: signer.alg, kid: signer.kid, typ: 'JOSE', ...(signer.jku ? { jku: signer.jku } : {}) };
}

/** The bytes a card signature is computed over: `protected "." base64url(JCS payload)`. */
export function signingInput(protectedB64: string, card: Record<string, unknown>): Uint8Array {
  return new TextEncoder().encode(`${protectedB64}.${base64url(cardSigningBytes(card))}`);
}

/** `card` with one more detached signature. The card itself is not changed. */
export async function signAgentCard<T extends Record<string, unknown>>(card: T, signer: CardSigner): Promise<T & { signatures: AgentCardSignature[] }> {
  const header = protectedHeader(signer);
  const protectedB64 = base64url(new TextEncoder().encode(canonicalize(header)));
  const signature = base64url(await signer.sign(signingInput(protectedB64, card)));
  const existing = Array.isArray(card.signatures) ? (card.signatures as AgentCardSignature[]) : [];
  return { ...card, signatures: [...existing, { protected: protectedB64, signature }] };
}

/** An EdDSA signer for the agent's own identity: the current key, `kid` its thumbprint, `jku` its key set. */
export function identityCardSigner(identity: { issuer: string; getHeldKeys(): ReadonlyArray<HeldKey> }): CardSigner {
  const current = identity.getHeldKeys()[0];
  if (!current) throw new Error('A2A card signer: the identity holds no key');
  return {
    alg: 'EdDSA',
    kid: current.kid,
    jku: `${identity.issuer}${KEY_SET_WELL_KNOWN_SUFFIX}`,
    sign: (data) => current.sign(data),
  };
}

/** An HS256 signer, for the spec's byte-assembly vector and tests only. */
export function hmacCardSigner(kid: string, secret: Uint8Array): CardSigner {
  return {
    alg: 'HS256',
    kid,
    async sign(data) {
      const key = await crypto.subtle.importKey('raw', secret as unknown as ArrayBuffer, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
      return new Uint8Array(await crypto.subtle.sign('HMAC', key, data as unknown as ArrayBuffer));
    },
  };
}

// ---------------------------------------------------------------------------
// Verifying
// ---------------------------------------------------------------------------

export interface VerifyCardOptions {
  /** Keys to try by `kid` before any fetch (a peer's key set already held). */
  keys?: JsonWebKey[];
  /** Resolve the key a signature names; return null for none. Replaces the `jku` fetch. */
  resolveKey?: (header: { kid?: string; jku?: string; alg?: string }) => Promise<JsonWebKey | null>;
  /** The HS256 secret, for the test vector only. */
  secret?: Uint8Array;
  /** Algorithms accepted. Default: EdDSA and ES256 (HS256 only with `secret`). */
  allowedAlgs?: CardSignatureAlg[];
  /** Where the card was fetched from: a `jku` must share its origin (or an interface's). */
  cardUrl?: string;
  /** Allow a plaintext `jku` (tests against loopback). */
  allowHttp?: boolean;
  fetch?: typeof fetch;
}

export interface VerifyCardResult {
  ok: boolean;
  /** The signature that verified. */
  kid?: string;
  alg?: string;
  /** Why none did. */
  reason?: string;
  /** How many signatures the card carried. */
  checked: number;
}

/**
 * Whether at least one of the card's signatures verifies (section 8.4: "at
 * least one signature valid"). The key comes from `keys`, `resolveKey`, or the
 * signature's `jku`, and a `jku` is fetched only from the origin the card was
 * fetched from or one of its own interfaces, over https unless `allowHttp`,
 * never from a blocked address: a card must not be able to send a verifier
 * to an arbitrary host.
 */
export async function verifyAgentCard(card: Record<string, unknown>, options: VerifyCardOptions = {}): Promise<VerifyCardResult> {
  const signatures = Array.isArray(card.signatures) ? (card.signatures as AgentCardSignature[]) : [];
  if (signatures.length === 0) return { ok: false, reason: 'card carries no signatures', checked: 0 };
  const allowed = new Set<string>(options.allowedAlgs ?? (options.secret ? ['EdDSA', 'ES256', 'HS256'] : ['EdDSA', 'ES256']));
  let lastReason = 'no signature verified';
  for (const entry of signatures) {
    let header: { alg?: string; kid?: string; jku?: string };
    try {
      header = JSON.parse(new TextDecoder().decode(base64urlDecode(entry.protected))) as typeof header;
    } catch {
      lastReason = 'protected header is not base64url JSON';
      continue;
    }
    if (!header.alg || !allowed.has(header.alg)) {
      lastReason = `alg ${header.alg ?? '(none)'} is not accepted`;
      continue;
    }
    if (!header.kid) {
      lastReason = 'protected header has no kid';
      continue;
    }
    try {
      const data = signingInput(entry.protected, card);
      const signature = base64urlDecode(entry.signature);
      const ok = await verifyOne(header as { alg: CardSignatureAlg; kid: string; jku?: string }, data, signature, card, options);
      if (ok === true) return { ok: true, kid: header.kid, alg: header.alg, checked: signatures.length };
      lastReason = ok;
    } catch (error) {
      lastReason = (error as Error).message;
    }
  }
  return { ok: false, reason: lastReason, checked: signatures.length };
}

async function verifyOne(
  header: { alg: CardSignatureAlg; kid: string; jku?: string },
  data: Uint8Array,
  signature: Uint8Array,
  card: Record<string, unknown>,
  options: VerifyCardOptions,
): Promise<true | string> {
  if (header.alg === 'HS256') {
    if (!options.secret) return 'HS256 needs a secret';
    const key = await crypto.subtle.importKey('raw', options.secret as unknown as ArrayBuffer, { name: 'HMAC', hash: 'SHA-256' }, false, ['verify']);
    const valid = await crypto.subtle.verify('HMAC', key, signature as unknown as ArrayBuffer, data as unknown as ArrayBuffer);
    return valid ? true : 'HS256 signature does not verify';
  }
  const jwk = await resolveJwk(header, card, options);
  if (!jwk) return `no key for kid ${header.kid}`;
  if (header.alg === 'EdDSA') {
    if (jwk.kty !== 'OKP' || jwk.crv !== 'Ed25519') return `kid ${header.kid} is not an Ed25519 key`;
    const key = await crypto.subtle.importKey('jwk', { kty: 'OKP', crv: 'Ed25519', x: jwk.x }, { name: 'Ed25519' }, false, ['verify']);
    const valid = await crypto.subtle.verify('Ed25519', key, signature as unknown as ArrayBuffer, data as unknown as ArrayBuffer);
    return valid ? true : 'EdDSA signature does not verify';
  }
  if (jwk.kty !== 'EC' || jwk.crv !== 'P-256') return `kid ${header.kid} is not a P-256 key`;
  const key = await crypto.subtle.importKey('jwk', { kty: 'EC', crv: 'P-256', x: jwk.x, y: jwk.y }, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
  const valid = await crypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, key, signature as unknown as ArrayBuffer, data as unknown as ArrayBuffer);
  return valid ? true : 'ES256 signature does not verify';
}

async function resolveJwk(
  header: { kid: string; jku?: string; alg?: string },
  card: Record<string, unknown>,
  options: VerifyCardOptions,
): Promise<JsonWebKey | null> {
  const held = (options.keys ?? []).find((k) => (k as JsonWebKey & { kid?: string }).kid === header.kid);
  if (held) return held;
  if (options.resolveKey) return options.resolveKey(header);
  if (!header.jku) return null;
  if (!jkuAllowed(header.jku, card, options)) throw new Error(`jku ${header.jku} is not an origin this card is served from`);
  const doFetch = options.fetch ?? fetch;
  const response = await doFetch(header.jku, { headers: { accept: 'application/json' } });
  if (!response.ok) throw new Error(`jku ${header.jku} answered ${response.status}`);
  const body = (await response.json()) as { keys?: JsonWebKey[] };
  return (body.keys ?? []).find((k) => (k as JsonWebKey & { kid?: string }).kid === header.kid) ?? null;
}

/** A `jku` is fetched only from an origin the card itself is tied to. */
export function jkuAllowed(jku: string, card: Record<string, unknown>, options: Pick<VerifyCardOptions, 'cardUrl' | 'allowHttp'>): boolean {
  let url: URL;
  try {
    url = new URL(jku);
  } catch {
    return false;
  }
  if (url.protocol !== 'https:' && !(options.allowHttp && url.protocol === 'http:')) return false;
  if (!options.allowHttp && isBlockedIpLiteral(url.hostname)) return false;
  const origins = new Set<string>();
  if (options.cardUrl) {
    try {
      origins.add(new URL(options.cardUrl).origin);
    } catch {
      // not absolute: nothing to tie to
    }
  }
  for (const iface of (card.supportedInterfaces as Array<{ url?: string }> | undefined) ?? []) {
    try {
      if (iface.url) origins.add(new URL(iface.url).origin);
    } catch {
      // relative interface URL: nothing to tie to
    }
  }
  return origins.has(url.origin);
}
