/**
 * The TrustFlow lookup: an agent's score and topic summary, from the platform
 * (webagents gap-closure plan item 2.5, 2026-09-26). The Python twin is
 * `python/webagents/trustflow/trust_lookup.py`; the contract both read is
 * `python/tests/fixtures/trust/trust_tool_definition.json`.
 *
 * TRUSTFLOW IS A PLATFORM SERVICE. The score is computed by Robutler from
 * interactions on the platform (`lib/reputation/batch.ts` there), so an SDK
 * agent cannot compute it and does not try: it asks `GET /api/trust/lookup`,
 * AUTHENTICATED AS ITSELF, and holds the answer for a minute. The platform
 * decides who is asking from the credential and what they may see; this
 * class only carries the credential and the subject.
 *
 * THE CREDENTIAL is the discovery skill's rule (`skills/discovery/skill.ts`,
 * "HOW A CALL IS AUTHENTICATED"), because it is the same platform and the
 * same routes' gate: an identity that can sign signs the request (Web Bot
 * Auth) and no bearer rides beside it; otherwise a platform key is the
 * bearer; otherwise the call is refused up front with the sentence naming the
 * fix, never sent to get a 401.
 *
 * FAILURE IS AN ERROR, NEVER A ZERO. A lookup that cannot be made (no
 * credential, unreachable, refused, unreadable) throws `TrustLookupError`, so
 * a caller gating on trust (`skills/access/skill.ts`) fails CLOSED and a tool
 * says why; a zero would read as "this agent is not trusted", which the
 * platform never said.
 */

import { assertSignableAgentUrl, signedFetch, type SigningIdentity } from '../crypto/http-signature';
import { configuredPlatformUrl, resolveSkillPlatformUrl } from '../skills/platform-url';

export const TRUST_LOOKUP_PATH = '/api/trust/lookup';
export const TRUST_RECORD_PATH = '/api/trust/record';
/** How long an answer is held; the platform marks it `max-age=60` too. */
export const TRUST_CACHE_TTL_MS = 60_000;
/** How long a signed record is held: a record is valid for a week. */
export const TRUST_RECORD_CACHE_TTL_MS = 3_600_000;
export const TRUST_LOOKUP_TIMEOUT_MS = 8_000;

/** The sentence for an agent with neither credential, the same in both SDKs (fixture `no_credential`). */
export const NO_TRUST_CREDENTIAL =
  'No credential for the platform: this agent has no signing identity and no platform key, ' +
  'so it cannot ask Robutler for a TrustFlow score. ' +
  'Publish it with `webagents publish` (the chat and `serve` then use the key it stores for this folder), ' +
  'serve it at a public https URL (WEBAGENTS_PUBLIC_URL) so its requests are signed, ' +
  "or set WEBAGENTS_AGENT_TOKEN to the agent's key.";

/** How a platform call is authenticated (file comment). */
export type PlatformCredential =
  | { kind: 'signature'; identity: SigningIdentity }
  | { kind: 'bearer'; key: string }
  | { kind: 'none'; reason: string };

/** Duck-typed `SigningIdentity`, so nothing here imports the concrete class. */
function signingIdentityOf(value: unknown): SigningIdentity | undefined {
  const candidate = value as { issuer?: unknown; getHeldKeys?: unknown } | undefined;
  if (candidate && typeof candidate.issuer === 'string' && typeof candidate.getHeldKeys === 'function') {
    return candidate as SigningIdentity;
  }
  return undefined;
}

/**
 * The credential for `holder` (an agent, or anything carrying `identity` and
 * `apiKey`): its identity when that can sign for a URL the platform can fetch
 * keys from, else its key as a bearer, else the refusal (file comment).
 */
export function platformCredentialFor(holder: { identity?: unknown; apiKey?: string | null } | undefined): PlatformCredential {
  const key = holder?.apiKey?.trim() || undefined;
  const identity = signingIdentityOf(holder?.identity);
  if (identity) {
    try {
      assertSignableAgentUrl(identity.issuer);
      return { kind: 'signature', identity };
    } catch (err) {
      if (!key) return { kind: 'none', reason: (err as Error).message };
    }
  }
  if (key) return { kind: 'bearer', key };
  return { kind: 'none', reason: NO_TRUST_CREDENTIAL };
}

export interface TrustTopicScore {
  id: string;
  label: string;
  score: number;
}

/** What `GET /api/trust/lookup` answers. */
export interface TrustLookupResult {
  subject: { id: string; username: string; url: string | null };
  /** TrustFlow, 0 to 1. */
  score: number;
  tier: string;
  trust_level: string;
  /** The topic asked for; `score` null when the platform could not score it. */
  topic?: { query: string; score: number | null };
  topics: TrustTopicScore[];
  computed_at: string;
  methodology: string;
}

/** What `GET /api/trust/record` answers: the compact JWS and its claims. */
export interface TrustRecordResult {
  record: string;
  payload: Record<string, unknown>;
  jwks_url: string;
}

export type TrustLookupFailure = 'no_credential' | 'unreachable' | 'refused' | 'not_found' | 'unreadable';

/** The sentences a tool says for each failure, the same in both SDKs (fixture `messages`). */
export function trustFailureMessage(code: TrustLookupFailure, agent: string, status?: number, reason?: string): string {
  switch (code) {
    case 'no_credential':
      return reason ?? NO_TRUST_CREDENTIAL;
    case 'unreachable':
      return `The platform could not be reached, so there is no TrustFlow score for ${agent}.`;
    case 'not_found':
      return `The platform knows no agent ${agent} that you may look up.`;
    case 'refused':
      return `The platform refused the lookup for ${agent} (${status ?? 'no status'}).`;
    case 'unreadable':
      return `The platform's answer for ${agent} could not be read.`;
  }
}

export class TrustLookupError extends Error {
  constructor(
    readonly code: TrustLookupFailure,
    readonly agent: string,
    readonly status?: number,
    reason?: string,
  ) {
    super(trustFailureMessage(code, agent, status, reason));
    this.name = 'TrustLookupError';
  }
}

export interface TrustLookupOptions {
  /** The platform; unset means the skills' resolution (`platform-url.ts`). */
  platformUrl?: string;
  /** The credential, or how to get it per call (an identity is attached after construction). */
  credential?: PlatformCredential | (() => PlatformCredential | Promise<PlatformCredential>);
  fetch?: typeof fetch;
  ttlMs?: number;
  recordTtlMs?: number;
  timeoutMs?: number;
  /** The clock in ms, for tests. */
  now?: () => number;
}

interface Held<T> {
  value: T;
  expires: number;
}

/** The lookup client: `lookup` and `record`, each held for its TTL per subject and topic. */
export class TrustLookup {
  private readonly options: TrustLookupOptions;
  private readonly held = new Map<string, Held<TrustLookupResult>>();
  private readonly heldRecords = new Map<string, Held<TrustRecordResult>>();
  private platform?: string;

  constructor(options: TrustLookupOptions = {}) {
    this.options = options;
    this.platform = configuredPlatformUrl(options.platformUrl);
  }

  /** The platform's base URL: configured, else the environment, else the CLI's, else https://robutler.ai. */
  async platformUrl(): Promise<string> {
    if (this.platform) return this.platform;
    this.platform = await resolveSkillPlatformUrl();
    return this.platform;
  }

  /** The credential the next call carries. */
  async credential(): Promise<PlatformCredential> {
    const c = this.options.credential;
    if (typeof c === 'function') return c();
    return c ?? { kind: 'none', reason: NO_TRUST_CREDENTIAL };
  }

  clearCache(): void {
    this.held.clear();
    this.heldRecords.clear();
  }

  private now(): number {
    return this.options.now ? this.options.now() : Date.now();
  }

  /**
   * The agent's trust summary, on `topic` when given. `agent` is its URL,
   * `@username` or platform id, as the platform reads it. Throws
   * `TrustLookupError` when there is no answer (file comment).
   */
  async lookup(agent: string, topic?: string): Promise<TrustLookupResult> {
    const subject = agent.trim();
    const key = `${subject}\u0000${topic?.trim() ?? ''}`;
    const hit = this.held.get(key);
    if (hit && hit.expires > this.now()) return hit.value;
    const query = new URLSearchParams({ agent: subject });
    if (topic?.trim()) query.set('topic', topic.trim());
    const body = await this.getJson(`${await this.platformUrl()}${TRUST_LOOKUP_PATH}?${query}`, subject);
    if (typeof body.score !== 'number' || !body.subject || typeof body.subject !== 'object') {
      throw new TrustLookupError('unreadable', subject);
    }
    const result = body as unknown as TrustLookupResult;
    this.held.set(key, { value: result, expires: this.now() + (this.options.ttlMs ?? TRUST_CACHE_TTL_MS) });
    return result;
  }

  /** The agent's signed record (`trust-record.ts` verifies it). Held for an hour. */
  async record(agent: string): Promise<TrustRecordResult> {
    const subject = agent.trim();
    const hit = this.heldRecords.get(subject);
    if (hit && hit.expires > this.now()) return hit.value;
    const query = new URLSearchParams({ agent: subject });
    const body = await this.getJson(`${await this.platformUrl()}${TRUST_RECORD_PATH}?${query}`, subject);
    if (typeof body.record !== 'string' || !body.payload || typeof body.payload !== 'object') {
      throw new TrustLookupError('unreadable', subject);
    }
    const result = body as unknown as TrustRecordResult;
    this.heldRecords.set(subject, { value: result, expires: this.now() + (this.options.recordTtlMs ?? TRUST_RECORD_CACHE_TTL_MS) });
    return result;
  }

  private async getJson(url: string, subject: string): Promise<Record<string, unknown>> {
    const credential = await this.credential();
    if (credential.kind === 'none') throw new TrustLookupError('no_credential', subject, undefined, credential.reason);
    const init: RequestInit = { method: 'GET', headers: { accept: 'application/json' }, signal: AbortSignal.timeout(this.options.timeoutMs ?? TRUST_LOOKUP_TIMEOUT_MS) };
    let response: Response;
    try {
      if (credential.kind === 'signature') {
        response = await signedFetch(credential.identity, url, init);
      } else {
        const doFetch = this.options.fetch ?? fetch;
        response = await doFetch(url, { ...init, headers: { ...(init.headers as Record<string, string>), Authorization: `Bearer ${credential.key}` } });
      }
    } catch {
      throw new TrustLookupError('unreachable', subject);
    }
    if (response.status === 404) throw new TrustLookupError('not_found', subject, 404);
    if (!response.ok) throw new TrustLookupError('refused', subject, response.status);
    try {
      const data = (await response.json()) as unknown;
      if (data && typeof data === 'object' && !Array.isArray(data)) return data as Record<string, unknown>;
    } catch {
      // unreadable, below
    }
    throw new TrustLookupError('unreadable', subject);
  }
}
