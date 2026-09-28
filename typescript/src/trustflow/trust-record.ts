/**
 * The exportable TrustFlow record: verifying one, and carrying one in an A2A
 * card (webagents gap-closure plan item 2.7, 2026-09-26). The Python twin is
 * `python/webagents/trustflow/trust_record.py`; both verify the vectors in
 * `python/tests/fixtures/trust/trustflow_record.json`, which the portal's
 * signer produces byte for byte.
 *
 * A RECORD is a compact JWS the platform signs with its keyring (RS256, the
 * `kid` published at `<issuer>/.well-known/jwks.json`): `iss`, `sub` (the
 * agent URL), `aud` (`urn:robutler:trustflow-record`), `iat`, `exp`, and the
 * record itself (`agent`, `score` in [0, 1], `tier`, `topics`, `computed_at`,
 * `methodology`). It travels with the agent, so anyone holding the platform
 * key set can verify it with no call to the platform: that is the point of
 * it, and why the checks here are strict.
 *
 * WHAT VERIFYING CHECKS, in order: the three-part shape; `alg` RS256 and
 * `typ` `trustflow+jwt` (a platform token is not a record); the key by `kid`
 * from the held keys or the ISSUER's key set; the signature; `iss` equal to
 * the issuer the CALLER expects; `aud`; `exp` and `iat`; the claim shape;
 * and, when asked, that the record is about the subject the caller has in
 * hand (URL, id or username). The expected issuer comes from configuration
 * (`issuer`, else ROBUTLER_PLATFORM_ISSUER, else the platform URL the skills
 * resolve), NEVER from the record's own `iss`, and the key set URL is derived
 * from that issuer under `keySetUrlFromIssuer`'s filter: a record must not be
 * able to send its verifier to a host of its choosing (S-135, `crypto/jwks.ts`).
 *
 * THE CARD EXTENSION. An agent references its record in its A2A v1.0 card
 * (`skills/transport/a2a/card.ts`) as `capabilities.extensions[]` with the
 * Robutler URI below and `params.record` the JWS. The card's own signature
 * covers the extension, so a peer reading the card gets both proofs.
 */

import { keySetUrlFromIssuer } from '../crypto/jwks';
import { envVar, resolveSkillPlatformUrl, trimmedUrl } from '../skills/platform-url';

export const TRUSTFLOW_RECORD_TYP = 'trustflow+jwt';
export const TRUSTFLOW_RECORD_AUDIENCE = 'urn:robutler:trustflow-record';
export const TRUSTFLOW_RECORD_EXTENSION_URI = 'https://robutler.ai/a2a/extensions/trustflow-record/v1';
export const TRUSTFLOW_RECORD_EXTENSION_DESCRIPTION =
  'A TrustFlow record signed by Robutler: params.record is a compact JWS verifiable against the issuer key set.';
/** How long a fetched key set is held (the platform serves it with max-age 3600). */
export const TRUST_KEY_SET_TTL_MS = 3_600_000;
/** How far ahead of the clock an `iat` may be. */
export const TRUST_RECORD_LEEWAY_SECONDS = 60;

export interface TrustRecord {
  iss: string;
  sub: string;
  aud: string;
  iat: number;
  exp: number;
  agent: { id: string; username: string; url: string | null };
  score: number;
  tier: string;
  topics: Array<{ id: string; label: string; score: number }>;
  computed_at: string;
  methodology: string;
}

export type TrustRecordRefusal =
  | 'malformed'
  | 'alg'
  | 'typ'
  | 'key_set'
  | 'no_key'
  | 'signature'
  | 'issuer'
  | 'audience'
  | 'expired'
  | 'not_yet_valid'
  | 'shape'
  | 'subject';

export interface TrustRecordSubject {
  url?: string;
  id?: string;
  username?: string;
}

export interface VerifyTrustRecordOptions {
  /** The issuer expected; see the file comment for the default. */
  issuer?: string;
  /** The platform's keys, held; no fetch when the `kid` is among them. */
  keys?: JsonWebKey[];
  /** Where to fetch keys from instead of `<issuer>/.well-known/jwks.json`. */
  jwksUrl?: string;
  fetch?: typeof fetch;
  /** The agent the record must be about. */
  subject?: TrustRecordSubject;
  /** Seconds since the epoch; the clock when unset. */
  now?: number;
}

export type VerifyTrustRecordResult =
  | { ok: true; record: TrustRecord; kid: string }
  | { ok: false; code: TrustRecordRefusal; reason: string };

// ---------------------------------------------------------------------------
// base64url and JSON
// ---------------------------------------------------------------------------

function base64urlDecode(text: string): Uint8Array {
  const padded = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function decodeJson(segment: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(new TextDecoder().decode(base64urlDecode(segment))) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/** The header and claims of a record, unverified: for display, never for a decision. */
export function decodeTrustRecord(jws: string): { header: Record<string, unknown>; payload: Record<string, unknown> } | null {
  const parts = typeof jws === 'string' ? jws.split('.') : [];
  if (parts.length !== 3 || parts.some((p) => !p)) return null;
  const header = decodeJson(parts[0]);
  const payload = decodeJson(parts[1]);
  return header && payload ? { header, payload } : null;
}

/** The one spelling of an agent URL: lower-case origin plus path, no trailing slash. */
export function canonicalRecordUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  try {
    const url = new URL(trimmed);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return trimmed;
    return `${url.origin}${url.pathname}`.replace(/\/+$/, '');
  } catch {
    return trimmed;
  }
}

// ---------------------------------------------------------------------------
// The key set
// ---------------------------------------------------------------------------

interface HeldKeySet {
  keys: JsonWebKey[];
  expires: number;
}

const keySets = new Map<string, HeldKeySet>();

/** Tests: forget every fetched key set. */
export function _resetTrustKeySets(): void {
  keySets.clear();
}

async function fetchKeySet(url: string, doFetch: typeof fetch, force: boolean): Promise<JsonWebKey[]> {
  const hit = keySets.get(url);
  if (hit && !force && hit.expires > Date.now()) return hit.keys;
  const response = await doFetch(url, { headers: { accept: 'application/json' } });
  if (!response.ok) throw new Error(`key set ${url} answered ${response.status}`);
  const body = (await response.json()) as { keys?: JsonWebKey[] };
  const keys = Array.isArray(body.keys) ? body.keys : [];
  keySets.set(url, { keys, expires: Date.now() + TRUST_KEY_SET_TTL_MS });
  return keys;
}

function findKey(keys: JsonWebKey[] | undefined, kid: string): JsonWebKey | undefined {
  return (keys ?? []).find((k) => (k as JsonWebKey & { kid?: string }).kid === kid);
}

/** The issuer to expect when the caller names none (file comment). */
export async function defaultTrustIssuer(): Promise<string> {
  return trimmedUrl(envVar('ROBUTLER_PLATFORM_ISSUER')) ?? (await resolveSkillPlatformUrl());
}

// ---------------------------------------------------------------------------
// Verifying
// ---------------------------------------------------------------------------

function isVector(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value);
}

/** The claims as a `TrustRecord`, or null when the shape is not a record's. */
export function trustRecordShape(payload: Record<string, unknown>): TrustRecord | null {
  const agent = payload.agent as Record<string, unknown> | undefined;
  if (
    typeof payload.iss !== 'string' ||
    typeof payload.sub !== 'string' ||
    typeof payload.aud !== 'string' ||
    !isVector(payload.iat) ||
    !isVector(payload.exp) ||
    !agent || typeof agent !== 'object' ||
    typeof agent.id !== 'string' || !agent.id ||
    typeof agent.username !== 'string' ||
    !(agent.url === null || typeof agent.url === 'string') ||
    !isVector(payload.score) || (payload.score as number) < 0 || (payload.score as number) > 1 ||
    typeof payload.tier !== 'string' ||
    !Array.isArray(payload.topics) ||
    typeof payload.computed_at !== 'string' ||
    typeof payload.methodology !== 'string' || !payload.methodology
  ) {
    return null;
  }
  for (const topic of payload.topics as unknown[]) {
    const t = topic as Record<string, unknown> | null;
    if (!t || typeof t !== 'object' || typeof t.id !== 'string' || typeof t.label !== 'string' || !isVector(t.score)) return null;
  }
  return payload as unknown as TrustRecord;
}

/** Whether `record` is about `subject`: every field the caller named must match. */
export function recordMatchesSubject(record: TrustRecord, subject: TrustRecordSubject): boolean {
  let named = 0;
  if (subject.url !== undefined) {
    named += 1;
    const wanted = canonicalRecordUrl(subject.url);
    const recorded = record.agent.url ? canonicalRecordUrl(record.agent.url) : null;
    if (wanted !== recorded && wanted !== canonicalRecordUrl(record.sub)) return false;
  }
  if (subject.id !== undefined) {
    named += 1;
    if (subject.id !== record.agent.id) return false;
  }
  if (subject.username !== undefined) {
    named += 1;
    if (subject.username.replace(/^@/, '').toLowerCase() !== record.agent.username.toLowerCase()) return false;
  }
  return named > 0;
}

async function verifyRs256(jwk: JsonWebKey, data: Uint8Array, signature: Uint8Array): Promise<boolean> {
  if (jwk.kty !== 'RSA' || typeof jwk.n !== 'string' || typeof jwk.e !== 'string') return false;
  const key = await crypto.subtle.importKey(
    'jwk',
    { kty: 'RSA', n: jwk.n, e: jwk.e, alg: 'RS256', ext: true },
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['verify'],
  );
  return crypto.subtle.verify('RSASSA-PKCS1-v1_5', key, signature as unknown as ArrayBuffer, data as unknown as ArrayBuffer);
}

/** Verify `jws` as a TrustFlow record (file comment: what is checked, in order). Never throws. */
export async function verifyTrustRecord(jws: string, options: VerifyTrustRecordOptions = {}): Promise<VerifyTrustRecordResult> {
  const refuse = (code: TrustRecordRefusal, reason: string): VerifyTrustRecordResult => ({ ok: false, code, reason });
  const parts = typeof jws === 'string' ? jws.split('.') : [];
  if (parts.length !== 3 || parts.some((p) => !p)) return refuse('malformed', 'not a compact JWS');
  const header = decodeJson(parts[0]);
  if (!header) return refuse('malformed', 'protected header is not base64url JSON');
  if (header.alg !== 'RS256') return refuse('alg', `alg ${String(header.alg ?? '(none)')} is not RS256`);
  if (header.typ !== TRUSTFLOW_RECORD_TYP) return refuse('typ', `typ ${String(header.typ ?? '(none)')} is not ${TRUSTFLOW_RECORD_TYP}`);
  const kid = typeof header.kid === 'string' && header.kid ? header.kid : null;
  if (!kid) return refuse('no_key', 'protected header has no kid');
  const payload = decodeJson(parts[1]);
  if (!payload) return refuse('malformed', 'payload is not base64url JSON');

  const issuer = trimmedUrl(options.issuer) ?? (await defaultTrustIssuer());

  let jwk = findKey(options.keys, kid);
  if (!jwk) {
    const url = options.jwksUrl ?? keySetUrlFromIssuer(issuer);
    if (!url) return refuse('key_set', `no key set can be fetched for issuer ${issuer}`);
    const doFetch = options.fetch ?? fetch;
    try {
      jwk = findKey(await fetchKeySet(url, doFetch, false), kid) ?? findKey(await fetchKeySet(url, doFetch, true), kid);
    } catch (err) {
      return refuse('key_set', (err as Error).message);
    }
  }
  if (!jwk) return refuse('no_key', `no key for kid ${kid}`);

  let valid = false;
  try {
    valid = await verifyRs256(jwk, new TextEncoder().encode(`${parts[0]}.${parts[1]}`), base64urlDecode(parts[2]));
  } catch {
    valid = false;
  }
  if (!valid) return refuse('signature', 'signature does not verify');

  if (typeof payload.iss !== 'string' || trimmedUrl(payload.iss) !== issuer) {
    return refuse('issuer', `issuer ${String(payload.iss ?? '(none)')} is not ${issuer}`);
  }
  if (payload.aud !== TRUSTFLOW_RECORD_AUDIENCE) return refuse('audience', `aud ${String(payload.aud ?? '(none)')} is not a trust record`);
  const now = options.now ?? Math.floor(Date.now() / 1000);
  if (!isVector(payload.exp) || (payload.exp as number) <= now) return refuse('expired', 'the record has expired');
  if (!isVector(payload.iat) || (payload.iat as number) > now + TRUST_RECORD_LEEWAY_SECONDS) return refuse('not_yet_valid', 'the record is dated in the future');
  const record = trustRecordShape(payload);
  if (!record) return refuse('shape', 'the claims are not a trust record');
  if (options.subject && !recordMatchesSubject(record, options.subject)) return refuse('subject', 'the record is about another agent');
  return { ok: true, record, kid };
}

// ---------------------------------------------------------------------------
// The card extension
// ---------------------------------------------------------------------------

export interface TrustRecordExtension {
  uri: string;
  description: string;
  params: { record: string };
}

/** The extension entry for `record`. */
export function trustRecordExtension(record: string): TrustRecordExtension {
  return { uri: TRUSTFLOW_RECORD_EXTENSION_URI, description: TRUSTFLOW_RECORD_EXTENSION_DESCRIPTION, params: { record } };
}

/** `card` with its record set (replacing an earlier one); the card itself is not changed. */
export function withTrustRecordExtension<T extends Record<string, unknown>>(card: T, record: string): T {
  const capabilities = { ...((card.capabilities as Record<string, unknown> | undefined) ?? {}) };
  const existing = Array.isArray(capabilities.extensions) ? (capabilities.extensions as Array<{ uri?: string }>) : [];
  capabilities.extensions = [...existing.filter((e) => e?.uri !== TRUSTFLOW_RECORD_EXTENSION_URI), trustRecordExtension(record)];
  return { ...card, capabilities };
}

/** The record a card carries, or null. Unverified: hand it to `verifyTrustRecord`. */
export function trustRecordFromCard(card: Record<string, unknown>): string | null {
  const capabilities = card.capabilities as { extensions?: Array<{ uri?: string; params?: { record?: unknown } }> } | undefined;
  const entry = capabilities?.extensions?.find((e) => e?.uri === TRUSTFLOW_RECORD_EXTENSION_URI);
  const record = entry?.params?.record;
  return typeof record === 'string' && record ? record : null;
}
