/**
 * Verifying a Web Bot Auth signature on a request an agent RECEIVES
 * (ADR-0045 section 3, 2026-09-25).
 *
 * Both SDKs could sign what they send and neither could check what they were
 * sent: the only verifier was the portal's (`lib/auth/agent-auth.ts`,
 * `lib/auth/http-signatures/`, `lib/auth/web-bot-auth/`). This is that verifier
 * ported, step for step and refusal code for refusal code, minus what is the
 * platform's own business (agent registrations, owner keys, Redis). The Python
 * twin is `python/webagents/crypto/web_bot_auth_verify.py`; both run the same
 * cases.
 *
 * THE STEPS, in the portal's order:
 *   1. `Signature-Input` and `Signature` parse, label for label.
 *   2. Only `tag="web-bot-auth"` labels count; none, or more than 2, refuses.
 *   3. Each label's parameters: integer `created` and `expires`, a window of
 *      at most 3600 s, 60 s of clock skew, `keyid` an RFC 7638 thumbprint,
 *      `alg` absent or `ed25519`, a 1 to 256 character `nonce`, 64 signature
 *      bytes.
 *   4. The request's `Host` is THIS agent's own authority (its public URL);
 *      a signature made for another host is not ours to accept.
 *   5. Coverage: `@method @authority @path @query`, `content-digest` when there
 *      is a body, and exactly one covered `Signature-Agent` member.
 *   6. That member names where the keys are: a `jwks_uri` ending in
 *      `/.well-known/jwks.json` (what both SDKs send; the principal is the URL
 *      before it) or a `directory` origin. Every label must name the same
 *      principal.
 *   7. The key set is fetched through the address guard (`../net`): https
 *      only, public addresses only, no redirects, 64 KiB, 5 s, cached.
 *   8. The key whose thumbprint is `keyid` verifies Ed25519 over the base.
 *   9. `Content-Digest`, when covered, matches the body.
 *  10. The nonce is spent: a second use within the window is a replay.
 *
 * `cimd` (a card naming the key set) is refused with a reason, not guessed at.
 */

import { calculateJwkThumbprint, type JWK } from 'jose';

import {
  SfToken,
  serializeInnerList,
  serializeItem,
  type SfBareItem,
  type SfInnerList,
  type SfItem,
  type SfParameters,
} from './http-signature';
import { isInnerList, paramOf, parseDictionary, parseItem, StructuredFieldParseError, type SfMember } from './structured-fields';
import { parseAllowList, type AllowEntry } from '../net/addresses';
import { GuardError, exchange, headerOf, resolveAllowed } from '../net/guarded-request';

export const WEB_BOT_AUTH_TAG = 'web-bot-auth';
export const CLOCK_TOLERANCE_S = 60;
export const MAX_SIGNATURE_LIFETIME_S = 3600;
export const NONCE_MAX_CHARS = 256;
export const MAX_LABELS = 2;
export const KEY_SET_MAX_KEYS = 16;
export const KEY_SET_MAX_BYTES = 64 * 1024;
export const KEY_SET_TIMEOUT_MS = 5000;
export const KEY_SET_MIN_TTL_S = 300;
export const KEY_SET_MAX_TTL_S = 86400;
export const KEY_SET_FAILURE_TTL_S = 60;
export const SIGNED_BODY_MAX_BYTES = 4 * 1024 * 1024;
export const KEY_SET_WELL_KNOWN_SUFFIX = '/.well-known/jwks.json';
export const CARD_WELL_KNOWN_SUFFIX = '/.well-known/agent.json';
export const DIRECTORY_WELL_KNOWN_PATH = '/.well-known/http-message-signatures-directory';
export const DIRECTORY_MEDIA_TYPE = 'application/http-message-signatures-directory+json';
const REQUIRED_COMPONENTS = ['@method', '@authority', '@path', '@query'];
const REFUSED_DERIVED = new Set(['@query-param', '@status', '@request-target']);
const REFUSED_PARAMS = new Set(['sf', 'bs', 'tr', 'name', 'req']);
const THUMBPRINT_RE = /^[A-Za-z0-9_-]{43}$/;
const X_RE = /^[A-Za-z0-9_-]{43}$/;
const JWK_ALG_ACCEPTED = new Set(['ed25519', 'EdDSA', 'Ed25519']);
/** RFC 9421 Appendix B.1 test keys, by thumbprint: a key set carrying one is a copy of an example. */
export const TEST_KEY_THUMBPRINTS: ReadonlySet<string> = new Set([
  'poqkLGiymh_W0uP6PZFw-dvez3QJT5SolqXBCW38r0U',
  'oD0HwocPBSfpNy5W3bpJeyFGY_IQ_YpqxSjQ3Yd-CLA',
]);

export type RefusalCode =
  | 'signature_malformed'
  | 'signature_params_invalid'
  | 'signature_expired'
  | 'signature_coverage_insufficient'
  | 'signature_authority_mismatch'
  | 'signature_agent_invalid'
  | 'signature_body_too_large'
  | 'content_digest_mismatch'
  | 'key_set_unreachable'
  | 'key_set_invalid'
  | 'signature_key_unknown'
  | 'signature_invalid'
  | 'signature_replayed';

export interface Refusal {
  code: RefusalCode;
  /** One sentence about the caller's own request; never a secret. */
  description: string;
}

/** A caller whose signature verified. */
export interface VerifiedAgent {
  /** The agent URL the key set was published under, no trailing slash: the `agent:` principal. */
  principal: string;
  /** Thumbprints of the keys that verified, one per label: the `key:` principals. */
  thumbprints: string[];
  /** Where the keys were read. */
  identifier: string;
}

export type VerifyOutcome = { ok: true; agent: VerifiedAgent } | { ok: false; refusal: Refusal };

export interface InboundRequest {
  method: string;
  /** The request line's path and query as received, e.g. `/agents/mini/chat/completions`. */
  target: string;
  headers: Pick<Headers, 'get'> | Readonly<Record<string, string | string[] | undefined>>;
  /** The body bytes; absent or empty for none. */
  body?: Uint8Array;
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/** Remembers spent nonces until their signature expires. Refuses when full rather than forgetting early. */
export class MemoryNonceStore {
  private readonly spent = new Map<string, number>();

  constructor(private readonly max = 100_000) {}

  /** True when `nonce` from `principal` was unspent and is now spent until `untilS`. */
  spend(principal: string, nonce: string, untilS: number, nowS: number): boolean {
    const key = `${principal}\n${nonce}`;
    const known = this.spent.get(key);
    if (known !== undefined && known >= nowS) return false;
    if (this.spent.size >= this.max) {
      for (const [k, until] of this.spent) if (until < nowS) this.spent.delete(k);
      if (this.spent.size >= this.max) return false;
    }
    this.spent.set(key, untilS);
    return true;
  }
}

// ---------------------------------------------------------------------------
// Key sets
// ---------------------------------------------------------------------------

export interface DiscoveredKey {
  thumbprint: string;
  x: string;
}

export type KeySetOutcome = { ok: true; keys: DiscoveredKey[]; ttlS: number } | { ok: false; code: 'key_set_unreachable' | 'key_set_invalid'; reason: string };

export interface Discovery {
  type: 'jwks_uri' | 'directory';
  principal: string;
  identifier: string;
  fetchUrl: string;
  mediaType?: string;
}

async function thumbprintOf(entry: Record<string, unknown>): Promise<string | null> {
  const s = (k: string): string | undefined => (typeof entry[k] === 'string' ? (entry[k] as string) : undefined);
  let members: JWK | null = null;
  if (entry.kty === 'OKP' && s('crv') && s('x')) members = { kty: 'OKP', crv: s('crv'), x: s('x') };
  else if (entry.kty === 'RSA' && s('n') && s('e')) members = { kty: 'RSA', n: s('n'), e: s('e') };
  else if (entry.kty === 'EC' && s('crv') && s('x') && s('y')) members = { kty: 'EC', crv: s('crv'), x: s('x'), y: s('y') };
  if (!members) return null;
  try {
    return await calculateJwkThumbprint(members, 'sha256');
  } catch {
    return null;
  }
}

/** A JWK Set body to its usable Ed25519 keys (the portal's `parseKeySet` rules). */
export async function parseKeySet(body: unknown, opts: { wellKnownDirectory: boolean }): Promise<{ ok: true; keys: DiscoveredKey[] } | { ok: false; reason: string }> {
  if (!body || typeof body !== 'object' || !Array.isArray((body as { keys?: unknown }).keys)) {
    return { ok: false, reason: 'the document has no keys array' };
  }
  const entries = (body as { keys: unknown[] }).keys;
  if (entries.length > KEY_SET_MAX_KEYS) return { ok: false, reason: `it carries more than ${KEY_SET_MAX_KEYS} entries` };
  const keys: DiscoveredKey[] = [];
  const seen = new Set<string>();
  for (const raw of entries) {
    if (!raw || typeof raw !== 'object') continue;
    const entry = raw as Record<string, unknown>;
    const thumbprint = await thumbprintOf(entry);
    if (thumbprint && TEST_KEY_THUMBPRINTS.has(thumbprint)) {
      return { ok: false, reason: 'it carries a known test key (RFC 9421 Appendix B.1)' };
    }
    if (!thumbprint) continue;
    if (entry.kty !== 'OKP' || entry.crv !== 'Ed25519' || typeof entry.x !== 'string' || !X_RE.test(entry.x)) continue;
    if (entry.use !== undefined && entry.use !== 'sig') continue;
    if (entry.key_ops !== undefined && !(Array.isArray(entry.key_ops) && entry.key_ops.includes('verify'))) continue;
    if (entry.alg !== undefined && !(typeof entry.alg === 'string' && JWK_ALG_ACCEPTED.has(entry.alg))) continue;
    if (opts.wellKnownDirectory && entry.kid !== undefined && entry.kid !== thumbprint) {
      return { ok: false, reason: 'a kid at the well-known directory must equal the key thumbprint' };
    }
    if (seen.has(thumbprint)) continue;
    seen.add(thumbprint);
    keys.push({ thumbprint, x: entry.x });
  }
  if (keys.length === 0) return { ok: false, reason: 'it carries no usable Ed25519 key' };
  return { ok: true, keys };
}

function cacheTtl(cacheControl: string | undefined): number {
  const match = /(?:^|,)\s*max-age\s*=\s*(\d+)/i.exec(cacheControl ?? '');
  const asked = match ? Number(match[1]) : KEY_SET_MIN_TTL_S;
  return Math.min(KEY_SET_MAX_TTL_S, Math.max(KEY_SET_MIN_TTL_S, asked));
}

/** Where a verifier gets key sets: fetched through the address guard and cached. */
export class KeySetFetcher {
  private readonly cache = new Map<string, { until: number; outcome: KeySetOutcome }>();
  private inFlight = 0;
  private readonly perHost = new Map<string, number>();

  constructor(
    private readonly opts: { allowPrivate?: boolean; maxEntries?: number; now?: () => number } = {},
  ) {}

  private now(): number {
    return this.opts.now ? this.opts.now() : Math.floor(Date.now() / 1000);
  }

  async get(discovery: Discovery): Promise<KeySetOutcome> {
    const now = this.now();
    const hit = this.cache.get(discovery.fetchUrl);
    if (hit && hit.until > now) return hit.outcome;
    const outcome = await this.fetch(discovery);
    const ttl = outcome.ok ? outcome.ttlS : KEY_SET_FAILURE_TTL_S;
    const max = this.opts.maxEntries ?? 1000;
    if (this.cache.size >= max) {
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
    this.cache.set(discovery.fetchUrl, { until: now + ttl, outcome });
    return outcome;
  }

  private async fetch(discovery: Discovery): Promise<KeySetOutcome> {
    const url = new URL(discovery.fetchUrl);
    const host = url.hostname.startsWith('[') ? url.hostname.slice(1, -1) : url.hostname;
    if (this.inFlight >= 32 || (this.perHost.get(host) ?? 0) >= 2) {
      return { ok: false, code: 'key_set_unreachable', reason: 'too many key set fetches are in flight' };
    }
    this.inFlight += 1;
    this.perHost.set(host, (this.perHost.get(host) ?? 0) + 1);
    try {
      const allow: AllowEntry[] = this.opts.allowPrivate ? parseAllowList(['0.0.0.0/0', '::/0']) : [];
      const port = url.port ? Number(url.port) : url.protocol === 'https:' ? 443 : 80;
      const address = await resolveAllowed(host, port, allow);
      const target = url.href.slice(`${url.protocol}//${url.host}`.length) || '/';
      const answer = await exchange({
        method: 'GET',
        scheme: url.protocol === 'https:' ? 'https' : 'http',
        host,
        port,
        target,
        headers: [
          ['Host', url.host],
          ['Accept', discovery.mediaType ? `${discovery.mediaType}, application/json` : 'application/json, application/jwk-set+json'],
          ['Accept-Encoding', 'identity'],
          ['User-Agent', 'WebAgents (+https://robutler.ai)'],
        ],
        body: Buffer.alloc(0),
        address,
        deadline: Date.now() + KEY_SET_TIMEOUT_MS,
        maxBytes: KEY_SET_MAX_BYTES + 1,
      });
      if (answer.status >= 300 && answer.status < 400) {
        return { ok: false, code: 'key_set_unreachable', reason: 'it answered with a redirect, which is not followed' };
      }
      if (answer.status !== 200) return { ok: false, code: 'key_set_unreachable', reason: `it answered ${answer.status}` };
      if (answer.truncated || answer.body.length > KEY_SET_MAX_BYTES) {
        return { ok: false, code: 'key_set_invalid', reason: 'it is larger than 64 KiB' };
      }
      const mediaType = (headerOf(answer, 'content-type') ?? '').split(';')[0].trim().toLowerCase();
      if (discovery.mediaType && mediaType !== discovery.mediaType) {
        return { ok: false, code: 'key_set_invalid', reason: `a directory must be served as ${discovery.mediaType}` };
      }
      let body: unknown;
      try {
        body = JSON.parse(answer.body.toString('utf8'));
      } catch {
        return { ok: false, code: 'key_set_invalid', reason: 'it is not JSON' };
      }
      const parsed = await parseKeySet(body, { wellKnownDirectory: discovery.type === 'directory' });
      if (!parsed.ok) return { ok: false, code: 'key_set_invalid', reason: parsed.reason };
      return { ok: true, keys: parsed.keys, ttlS: cacheTtl(headerOf(answer, 'cache-control')) };
    } catch (err) {
      if (err instanceof GuardError) {
        return { ok: false, code: 'key_set_unreachable', reason: err.code === 'timeout' ? 'no answer within 5 s' : err.message };
      }
      return { ok: false, code: 'key_set_unreachable', reason: 'the fetch failed' };
    } finally {
      this.inFlight -= 1;
      const left = (this.perHost.get(host) ?? 1) - 1;
      if (left <= 0) this.perHost.delete(host);
      else this.perHost.set(host, left);
    }
  }
}

// ---------------------------------------------------------------------------
// Authority, Signature-Agent, coverage
// ---------------------------------------------------------------------------

const AUTHORITY_RE =
  /^(?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)*\.?|\[[0-9a-f:.]+\])(?::[0-9]{1,5})?$/;

/** `host[:port]` lower-cased with the default ports stripped, or null when it is not an authority. */
export function normalizeAuthority(host: string | null | undefined): string | null {
  if (typeof host !== 'string') return null;
  const lower = host.trim().toLowerCase();
  if (lower.length === 0 || lower.length > 255 || !AUTHORITY_RE.test(lower)) return null;
  return lower.replace(/:(?:443|80)$/, '');
}

interface AgentMember {
  value: string;
  type: string | null;
  invalid: string | null;
}

const DOT_SEGMENT_RE = /^(?:\.|%2e){1,2}$/i;

function checkValue(value: string, allowHttp: boolean): { ok: true; url: URL } | { ok: false; reason: string } {
  if (value.length === 0 || value.length > 2048) return { ok: false, reason: 'the value is empty or too long' };
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return { ok: false, reason: 'the value is not an absolute URL' };
  }
  if (url.protocol !== 'https:' && !(allowHttp && url.protocol === 'http:')) return { ok: false, reason: 'the value must use https' };
  if (url.username || url.password) return { ok: false, reason: 'the value carries userinfo' };
  if (value.includes('#')) return { ok: false, reason: 'the value carries a fragment' };
  if (value.includes('?')) return { ok: false, reason: 'the value carries a query' };
  if (!url.hostname) return { ok: false, reason: 'the value has no host' };
  const pathStart = value.indexOf('/', value.indexOf('//') + 2);
  const rawPath = pathStart === -1 ? '' : value.slice(pathStart);
  const segments = rawPath.split('/').slice(1);
  if (segments.some((s) => DOT_SEGMENT_RE.test(s))) return { ok: false, reason: 'the value carries a dot segment' };
  if (segments.slice(0, -1).some((s) => s.length === 0)) return { ok: false, reason: 'the value carries an empty path segment' };
  if ((rawPath === '' ? '/' : rawPath) !== url.pathname) return { ok: false, reason: 'the value path is not in canonical form' };
  return { ok: true, url };
}

/** The design section 3.1 table: `jwks_uri` and `directory`. */
export function resolveDiscovery(member: AgentMember, opts: { allowHttp: boolean; legacy: boolean }): { ok: true; discovery: Discovery } | { ok: false; reason: string } {
  if (member.invalid) return { ok: false, reason: member.invalid };
  const checked = checkValue(member.value, opts.allowHttp);
  if (!checked.ok) return checked;
  const { url } = checked;
  const isOrigin = member.value === url.origin || member.value === `${url.origin}/`;
  const directory = (): { ok: true; discovery: Discovery } | { ok: false; reason: string } =>
    isOrigin
      ? {
          ok: true,
          discovery: {
            type: 'directory',
            principal: url.origin,
            identifier: `${url.origin}${DIRECTORY_WELL_KNOWN_PATH}`,
            fetchUrl: `${url.origin}${DIRECTORY_WELL_KNOWN_PATH}`,
            mediaType: DIRECTORY_MEDIA_TYPE,
          },
        }
      : { ok: false, reason: 'a directory value must be an origin (scheme and host only, no path)' };
  const jwks = (): { ok: true; discovery: Discovery } | { ok: false; reason: string } => {
    if (!url.pathname.endsWith(KEY_SET_WELL_KNOWN_SUFFIX)) return { ok: false, reason: `a key set URL must end in ${KEY_SET_WELL_KNOWN_SUFFIX}` };
    const principal = `${url.origin}${url.pathname.slice(0, -KEY_SET_WELL_KNOWN_SUFFIX.length).replace(/\/+$/, '')}`;
    const identifier = `${url.origin}${url.pathname}`;
    return { ok: true, discovery: { type: 'jwks_uri', principal, identifier, fetchUrl: identifier } };
  };
  if (opts.legacy) return directory();
  if (member.type === 'jwks_uri') return jwks();
  if (member.type === 'directory') return directory();
  if (member.type === 'cimd') return { ok: false, reason: 'card-based key discovery (type=cimd) is not supported by this verifier' };
  if (member.type === null) {
    if (isOrigin) return directory();
    if (url.pathname.endsWith(KEY_SET_WELL_KNOWN_SUFFIX)) return jwks();
    if (url.pathname.endsWith(CARD_WELL_KNOWN_SUFFIX)) {
      return { ok: false, reason: 'card-based key discovery (type=cimd) is not supported by this verifier' };
    }
    return { ok: false, reason: `an untyped value must be an origin or a key set URL ending in ${KEY_SET_WELL_KNOWN_SUFFIX}` };
  }
  return { ok: false, reason: `unsupported type ${member.type}` };
}

function readMember(value: SfBareItem, params: SfParameters | undefined): AgentMember {
  let type: string | null = null;
  let invalid: string | null = null;
  if (typeof value !== 'string') invalid = 'the member value is not a string';
  for (const [name, param] of params ?? []) {
    if (name === 'type') {
      if (!(param instanceof SfToken)) invalid ??= 'the type parameter is not a token';
      else type = param.value;
      continue;
    }
    invalid ??= `unknown member parameter ${name}`;
  }
  return { value: typeof value === 'string' ? value : '', type, invalid };
}

interface ParsedAgentField {
  legacy: boolean;
  legacyValue: string | null;
  members: Map<string, AgentMember>;
}

function parseSignatureAgent(raw: string): ParsedAgentField | null {
  const trimmed = raw.trim();
  try {
    if (trimmed.startsWith('"')) {
      const item = parseItem(trimmed);
      if (typeof item.value !== 'string' || (item.params ?? []).length > 0) return null;
      return { legacy: true, legacyValue: item.value, members: new Map() };
    }
    const dictionary = parseDictionary(trimmed);
    if (dictionary.length === 0) return null;
    const members = new Map<string, AgentMember>();
    for (const [key, member] of dictionary) {
      members.set(
        key,
        isInnerList(member)
          ? { value: '', type: null, invalid: 'the member is an inner list' }
          : readMember(member.value, member.params),
      );
    }
    return { legacy: false, legacyValue: null, members };
  } catch (err) {
    if (err instanceof StructuredFieldParseError) return null;
    throw err;
  }
}

// ---------------------------------------------------------------------------
// The signature base
// ---------------------------------------------------------------------------

interface Context {
  method: string;
  authority: string;
  scheme: 'https' | 'http';
  path: string;
  query: string;
  header(name: string): string | null;
}

class BaseError extends Error {}

function headerReader(headers: InboundRequest['headers']): (name: string) => string | null {
  if (typeof (headers as Pick<Headers, 'get'>).get === 'function') {
    return (name) => (headers as Pick<Headers, 'get'>).get(name);
  }
  return (name) => {
    for (const [key, value] of Object.entries(headers as Record<string, string | string[] | undefined>)) {
      if (key.toLowerCase() !== name) continue;
      if (value === undefined) return null;
      return Array.isArray(value) ? value.join(', ') : value;
    }
    return null;
  };
}

function requireAscii(value: string): string {
  if (!/^[\x20-\x7e]*$/.test(value)) throw new BaseError('a component value is not printable ASCII');
  return value;
}

function componentValue(ctx: Context, item: SfItem): string {
  if (typeof item.value !== 'string') throw new BaseError('a component identifier is not a string');
  const name = item.value;
  let key: string | null = null;
  for (const [param, value] of item.params ?? []) {
    if (param === 'key') {
      if (typeof value !== 'string') throw new BaseError('a key parameter is not a string');
      if (name.startsWith('@')) throw new BaseError('a key parameter on a derived component');
      key = value;
      continue;
    }
    throw new BaseError(REFUSED_PARAMS.has(param) ? `the ${param} parameter is refused` : `unknown parameter ${param}`);
  }
  if (name.startsWith('@')) {
    switch (name) {
      case '@method':
        return requireAscii(ctx.method);
      case '@authority':
        return requireAscii(ctx.authority);
      case '@scheme':
        return ctx.scheme;
      case '@path':
        return requireAscii(ctx.path === '' ? '/' : ctx.path);
      case '@query':
        return requireAscii(ctx.query === '' ? '?' : ctx.query);
      case '@target-uri':
        return requireAscii(`${ctx.scheme}://${ctx.authority}${ctx.path === '' ? '/' : ctx.path}${ctx.query}`);
      default:
        throw new BaseError(REFUSED_DERIVED.has(name) ? `${name} is refused` : `unknown derived component ${name}`);
    }
  }
  if (name !== name.toLowerCase()) throw new BaseError('a field component name is not lower-case');
  const raw = ctx.header(name);
  if (raw === null) throw new BaseError(`the ${name} field is absent`);
  if (key === null) return requireAscii(raw.trim());
  let dictionary: Array<[string, SfMember]>;
  try {
    dictionary = parseDictionary(raw);
  } catch {
    throw new BaseError(`the ${name} field is not a dictionary`);
  }
  const member = dictionary.find(([k]) => k === key)?.[1];
  if (!member) throw new BaseError(`the ${name} field has no member ${key}`);
  return requireAscii(isInnerList(member) ? serializeInnerList(member) : serializeItem(member));
}

function signatureBase(ctx: Context, input: SfInnerList): string {
  const seen = new Set<string>();
  const lines: string[] = [];
  for (const item of input.items) {
    const identifier = serializeItem(item);
    if (seen.has(identifier)) throw new BaseError('a component is covered twice');
    seen.add(identifier);
    lines.push(`${identifier}: ${componentValue(ctx, item)}`);
  }
  lines.push(`"@signature-params": ${requireAscii(serializeInnerList(input))}`);
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

function refuse(code: RefusalCode, description: string): VerifyOutcome {
  return { ok: false, refusal: { code, description } };
}

const MALFORMED =
  'Signature-Input and Signature must be RFC 9651 dictionaries whose members match label for label, ' +
  'each Signature member a 64-byte Ed25519 signature.';
const REQUIRED_SET_SENTENCE = '("@method" "@authority" "@path" "@query" "content-digest" "signature-agent";key="<label>")';

interface Label {
  label: string;
  input: SfInnerList;
  signature: Uint8Array;
}

async function verifyEd25519(x: string, base: string, signature: Uint8Array): Promise<boolean> {
  if (signature.byteLength !== 64) return false;
  try {
    const key = await crypto.subtle.importKey('jwk', { kty: 'OKP', crv: 'Ed25519', x }, { name: 'Ed25519' }, false, ['verify']);
    return await crypto.subtle.verify({ name: 'Ed25519' }, key, signature as unknown as ArrayBuffer, new TextEncoder().encode(base));
  } catch {
    return false;
  }
}

async function digestMatches(headerValue: string, body: Uint8Array): Promise<boolean> {
  let entries: Array<[string, SfMember]>;
  try {
    entries = parseDictionary(headerValue);
  } catch {
    return false;
  }
  if (entries.length === 0) return false;
  let matched = false;
  for (const [alg, member] of entries) {
    if (isInnerList(member) || !(member.value instanceof Uint8Array)) return false;
    const algorithm = alg === 'sha-256' ? 'SHA-256' : alg === 'sha-512' ? 'SHA-512' : null;
    if (!algorithm) continue;
    const expected = new Uint8Array(await crypto.subtle.digest(algorithm, body as unknown as ArrayBuffer));
    const given = member.value;
    if (given.length !== expected.length) return false;
    let diff = 0;
    for (let i = 0; i < expected.length; i += 1) diff |= expected[i] ^ given[i];
    if (diff !== 0) return false;
    matched = true;
  }
  return matched;
}

export interface VerifierOptions {
  /** This agent's own authorities, from its public URL (`host[:port]`, default port omitted). */
  authorities: readonly string[];
  /** The scheme of this agent's public URL, for `@scheme` and `@target-uri`. */
  scheme: 'https' | 'http';
  keySets: Pick<KeySetFetcher, 'get'>;
  nonces: Pick<MemoryNonceStore, 'spend'>;
  /** Plain http `Signature-Agent` values, for local development (`ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1`). */
  allowHttp?: boolean;
  /** Seconds since the epoch; injected by tests. */
  now?: () => number;
}

/** Verify the Web Bot Auth signature on an inbound request. Never throws for anything the caller sent. */
export async function verifyWebBotAuth(request: InboundRequest, options: VerifierOptions): Promise<VerifyOutcome> {
  const header = headerReader(request.headers);
  const nowS = options.now ? options.now() : Math.floor(Date.now() / 1000);

  // 1. The two fields, label for label.
  let inputs: Array<[string, SfMember]>;
  let signatures: Array<[string, SfMember]>;
  try {
    inputs = parseDictionary(header('signature-input') ?? '');
    signatures = parseDictionary(header('signature') ?? '');
  } catch (err) {
    if (err instanceof StructuredFieldParseError) return refuse('signature_malformed', MALFORMED);
    throw err;
  }
  if (inputs.length === 0) return refuse('signature_malformed', MALFORMED);
  const signatureMap = new Map(signatures);
  for (const [label] of signatures) if (!inputs.some(([l]) => l === label)) return refuse('signature_malformed', MALFORMED);
  const parsed: Label[] = [];
  for (const [label, member] of inputs) {
    const sig = signatureMap.get(label);
    if (!isInnerList(member) || !sig || isInnerList(sig) || !(sig.value instanceof Uint8Array)) {
      return refuse('signature_malformed', MALFORMED);
    }
    if (member.items.some((item) => typeof item.value !== 'string')) return refuse('signature_malformed', MALFORMED);
    parsed.push({ label, input: member, signature: sig.value });
  }

  // 2. Our tag, at most two.
  const labels = parsed.filter((p) => paramOf(p.input.params, 'tag') === WEB_BOT_AUTH_TAG);
  if (labels.length === 0) {
    return refuse(
      'signature_malformed',
      `No signature carries tag="${WEB_BOT_AUTH_TAG}". Each Signature-Input member this agent verifies must set that tag.`,
    );
  }
  if (labels.length > MAX_LABELS) {
    return refuse(
      'signature_malformed',
      `At most ${MAX_LABELS} signatures tagged "${WEB_BOT_AUTH_TAG}" are verified per request; send one per key you hold while rotating and no more.`,
    );
  }

  // 3. Parameters.
  for (const { input, signature } of labels) {
    const p = input.params;
    const created = paramOf(p, 'created');
    const expires = paramOf(p, 'expires');
    const keyid = paramOf(p, 'keyid');
    const alg = paramOf(p, 'alg');
    const nonce = paramOf(p, 'nonce');
    if (typeof created !== 'number' || !Number.isInteger(created)) {
      return refuse('signature_params_invalid', 'The created parameter is required and must be an integer number of seconds since the epoch.');
    }
    if (typeof expires !== 'number' || !Number.isInteger(expires)) {
      return refuse('signature_params_invalid', 'The expires parameter is required and must be an integer number of seconds since the epoch.');
    }
    if (typeof keyid !== 'string' || !THUMBPRINT_RE.test(keyid)) {
      return refuse(
        'signature_params_invalid',
        'The keyid parameter must be the RFC 7638 SHA-256 thumbprint of the signing key: base64url, no padding, 43 characters.',
      );
    }
    if (alg !== undefined && alg !== 'ed25519') {
      return refuse('signature_params_invalid', 'The alg parameter, when present, must be "ed25519"; this agent verifies Ed25519 signatures only.');
    }
    if (typeof nonce !== 'string' || nonce.length === 0 || nonce.length > NONCE_MAX_CHARS) {
      return refuse(
        'signature_params_invalid',
        `The nonce parameter is required: a string of 1 to ${NONCE_MAX_CHARS} characters, random and used once (64 random bytes in base64 is the convention).`,
      );
    }
    if (expires <= created) return refuse('signature_params_invalid', 'The expires parameter must be greater than created.');
    if (expires - created > MAX_SIGNATURE_LIFETIME_S) {
      return refuse(
        'signature_params_invalid',
        `The signature window (expires minus created) may be at most ${MAX_SIGNATURE_LIFETIME_S} seconds; sign one request at a time with a short window.`,
      );
    }
    if (signature.byteLength !== 64) return refuse('signature_malformed', MALFORMED);
    if (created > nowS + CLOCK_TOLERANCE_S) {
      return refuse('signature_expired', `The created parameter is more than ${CLOCK_TOLERANCE_S} seconds in the future; check the signer's clock.`);
    }
    if (expires < nowS - CLOCK_TOLERANCE_S) {
      return refuse('signature_expired', `The signature expired more than ${CLOCK_TOLERANCE_S} seconds ago; sign a fresh request.`);
    }
  }

  // 4. Authority: this agent's own.
  const accepted = options.authorities.map(normalizeAuthority).filter((a): a is string => a !== null);
  const authority = normalizeAuthority(header('host'));
  if (!authority || !accepted.includes(authority)) {
    return refuse(
      'signature_authority_mismatch',
      `This agent verifies signatures made for its own address only; sign the request for ${accepted[0] ?? 'its public host'} and send it there.`,
    );
  }

  // 5. Body, coverage, and the Signature-Agent member each label covers.
  const body = request.body ?? new Uint8Array(0);
  if (body.byteLength > SIGNED_BODY_MAX_BYTES) {
    return refuse('signature_body_too_large', `A signed request body may be at most ${SIGNED_BODY_MAX_BYTES / (1024 * 1024)} MiB.`);
  }
  const hasBody = body.byteLength > 0;
  const agentRaw = header('signature-agent');
  const agentField = agentRaw === null ? null : parseSignatureAgent(agentRaw);
  if (agentRaw === null) return refuse('signature_agent_invalid', 'The Signature-Agent header is absent.');
  if (!agentField) return refuse('signature_malformed', 'Signature-Agent must be an RFC 9651 dictionary of strings, or one string.');

  let discovery: Discovery | null = null;
  const covered: Array<Label & { keyid: string; nonce: string; expires: number; digestCovered: boolean }> = [];
  for (const sig of labels) {
    const names = sig.input.items.map((item) => item.value as string);
    for (const required of REQUIRED_COMPONENTS) {
      if (!names.includes(required)) {
        return refuse(
          'signature_coverage_insufficient',
          `The signature must cover ${required}. Cover at least ${REQUIRED_SET_SENTENCE}, omitting content-digest only when the request has no body.`,
        );
      }
    }
    const digestCovered = names.includes('content-digest');
    if (hasBody && !digestCovered) {
      return refuse(
        'signature_coverage_insufficient',
        'The request has a body, so it must send Content-Digest (sha-256 over the body bytes) and cover "content-digest" in the signature.',
      );
    }
    const agentItems = sig.input.items.filter((item) => item.value === 'signature-agent');
    if (agentItems.length !== 1) {
      return refuse(
        'signature_coverage_insufficient',
        'The signature must cover the Signature-Agent member for its label exactly once, as "signature-agent";key="<label>".',
      );
    }
    const key = paramOf(agentItems[0].params, 'key');
    let member: AgentMember | undefined;
    if (agentField.legacy) {
      if (key !== undefined) {
        return refuse(
          'signature_coverage_insufficient',
          'Signature-Agent is a bare string, so the covered component must be bare "signature-agent" with no key parameter.',
        );
      }
      member = { value: agentField.legacyValue ?? '', type: null, invalid: null };
    } else {
      if (typeof key !== 'string' || key.length === 0) {
        return refuse(
          'signature_coverage_insufficient',
          'Signature-Agent is a dictionary, so the signature must cover one of its members as "signature-agent";key="<label>".',
        );
      }
      member = agentField.members.get(key);
      if (!member) return refuse('signature_agent_invalid', `The Signature-Agent dictionary has no member keyed ${key}.`);
    }
    const resolved = resolveDiscovery(member, { allowHttp: options.allowHttp === true, legacy: agentField.legacy });
    if (!resolved.ok) return refuse('signature_agent_invalid', `The Signature-Agent member this signature covers is not usable: ${resolved.reason}.`);
    if (discovery && resolved.discovery.principal !== discovery.principal) {
      return refuse('signature_agent_invalid', 'Every signature on one request must name the same agent URL.');
    }
    discovery = resolved.discovery;
    covered.push({
      ...sig,
      keyid: paramOf(sig.input.params, 'keyid') as string,
      nonce: paramOf(sig.input.params, 'nonce') as string,
      expires: paramOf(sig.input.params, 'expires') as number,
      digestCovered,
    });
  }
  if (!discovery) return refuse('signature_malformed', MALFORMED);

  // The base's context.
  let path: string;
  let query: string;
  try {
    const url = new URL(request.target, `${options.scheme}://${authority}`);
    path = url.pathname;
    query = url.search;
  } catch {
    return refuse('signature_malformed', MALFORMED);
  }
  const ctx: Context = { method: request.method, authority, scheme: options.scheme, path, query, header };

  // 7. The keys.
  const keySet = await options.keySets.get(discovery);
  if (!keySet.ok) {
    return refuse(
      keySet.code,
      keySet.code === 'key_set_unreachable'
        ? `The key set Signature-Agent names could not be fetched: ${keySet.reason}.`
        : `The key set Signature-Agent names is not usable: ${keySet.reason}.`,
    );
  }

  // 8. Each label verifies.
  const thumbprints: string[] = [];
  for (const sig of covered) {
    const key = keySet.keys.find((k) => k.thumbprint === sig.keyid);
    if (!key) {
      return refuse('signature_key_unknown', 'No key in the published key set has the thumbprint this signature names as keyid.');
    }
    let base: string;
    try {
      base = signatureBase(ctx, sig.input);
    } catch (err) {
      if (err instanceof BaseError) return refuse('signature_malformed', MALFORMED);
      throw err;
    }
    if (!(await verifyEd25519(key.x, base, sig.signature))) {
      return refuse('signature_invalid', 'The signature does not verify under the published key.');
    }
    thumbprints.push(key.thumbprint);
  }

  // 9. The body the signature vouches for.
  if (covered.some((c) => c.digestCovered)) {
    const digestHeader = header('content-digest');
    if (digestHeader === null || !(await digestMatches(digestHeader, body))) {
      return refuse('content_digest_mismatch', 'Content-Digest does not match the request body.');
    }
  }

  // 10. Spend each nonce.
  for (const sig of covered) {
    if (!options.nonces.spend(discovery.principal, sig.nonce, sig.expires + CLOCK_TOLERANCE_S, nowS)) {
      return refuse('signature_replayed', "This signature's nonce was already used; sign each request afresh.");
    }
  }

  return { ok: true, agent: { principal: discovery.principal, thumbprints, identifier: discovery.identifier } };
}
