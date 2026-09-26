/**
 * Portal Discovery Skill
 *
 * Unified search across the Robutler platform:
 * - Intent-based agent search (semantic)
 * - Agent listing and info
 * - Content search (posts, channels, users, tags)
 * - Intent publishing (`publishIntents()`, and on initialise with `autoPublish`)
 *
 * HOW A CALL IS AUTHENTICATED (2026-09-23). Every platform route this skill
 * calls (`/api/intents/search`, `/api/intents/create`, `/api/discovery/*`)
 * authenticates through the platform's `authenticateAgentRequest`, which
 * takes an RFC 9421 signed request (Web Bot Auth; the platform's profile of
 * it is called AOAuth) BEFORE it looks for a bearer. Until today this skill
 * only ever sent `Authorization: Bearer <WEBAGENTS_API_KEY>`, so an agent
 * that already held a signing identity, the one `serve()` persists and
 * publishes at `{agentUrl}/.well-known/jwks.json`, was told to go and obtain
 * a platform key for a call the platform would have accepted signed.
 *
 * The rule now, in this order:
 *
 *  1. AN IDENTITY SIGNS. `config.identity`, else the identity `serve()` and
 *     `WebAgentsServer.addAgent()` hand the agent (`agent.identity`, read
 *     through `setAgent`). The request goes out through `signedFetch`, the
 *     same signer registration uses, and NO bearer rides beside it: the
 *     platform decides identity from the signature whenever one is present
 *     and ignores a bearer next to it, so sending both would only hand the
 *     key to a route that does not read it.
 *  2. A KEY IS A BEARER. `config.apiKey`, else `WEBAGENTS_API_KEY`. Used when
 *     there is no identity, or when the identity cannot sign: a loopback or
 *     plaintext agent URL, which the platform refuses before it resolves
 *     anything (`assertSignableAgentUrl` is the rule the signer applies, run
 *     here first so the fallback can happen).
 *  3. NEITHER IS A REFUSAL, decided up front with the fix named, in place of
 *     one 401 per request logged as "FAILED".
 *
 * The key stays optional. It was never the credential the platform needed
 * from an agent that can sign, and a platform-hosted agent (no key set of its
 * own) still presents one exactly as before.
 *
 * THE PLATFORM URL is `portalUrl`, else `ROBUTLER_API_URL`, else
 * `ROBUTLER_INTERNAL_API_URL`, else the CLI's `platform.url` (the portal
 * `webagents login` signed in to), else https://robutler.ai. A signature
 * covers the platform's host, so the URL registration signs for is the URL
 * this skill must sign for, or every signed call is
 * `signature_authority_mismatch`. The last two steps are 2026-09-25: the
 * default was `https://portal.webagents.ai`, a host that does not serve the
 * platform, so an agent file naming `discovery` searched nothing.
 *
 * ONE TOOL IN BOTH SDKS (2026-09-25). The Python skill offered
 * `discovery_tool` on `POST /api/discovery`, which drops the `channel`, `tag`
 * and `sort` filters, and an owner-only `publish_intents_tool`, so an agent
 * file naming `discovery` gave its model different tools under each CLI. Both
 * now offer this `search`, with the definition in
 * `python/tests/fixtures/discovery_tool/definition.json`, the same routes and
 * the same result shapes (`python/webagents/agents/skills/robutler/discovery/skill.py`).
 * Agents carry their URL, and posts are cut to an excerpt, as the platform's
 * own search tool returns them (`lib/agents/portal-discovery-skill.ts`), where
 * this returned whole post bodies. Progress lines go to the agent trace, which
 * the CLI keeps off the terminal, and never carry the query.
 */

import { Skill } from '../../core/skill';
import { resolveAgentCredential } from '../../server/agent-credential';
import { tool } from '../../core/decorators';
import { agentTrace } from '../../core/trace';
import type { Context } from '../../core/types';
import { assertSignableAgentUrl, signedFetch, type SigningIdentity } from '../../crypto/http-signature';
import {
  DEFAULT_PLATFORM_URL,
  configuredPlatformUrl,
  envVar,
  resolveSkillPlatformUrl,
} from '../platform-url';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DiscoveryConfig {
  /**
   * Platform base URL. Defaults to ROBUTLER_API_URL, then
   * ROBUTLER_INTERNAL_API_URL, then the CLI's `platform.url`, then
   * https://robutler.ai (file comment).
   */
  portalUrl?: string;
  /**
   * A platform API key, presented as a bearer. OPTIONAL: an agent with a
   * signing identity needs none (file comment). Defaults to WEBAGENTS_API_KEY.
   */
  apiKey?: string;
  /**
   * The identity to sign platform calls with. Usually left unset: `serve()`
   * and `WebAgentsServer.addAgent()` hand the agent its identity and the
   * skill reads it from there. Set it to sign with a different key, or when
   * this SDK does not serve the agent.
   */
  identity?: SigningIdentity;
  /** Request timeout in ms */
  timeout?: number;
  /** Agent ID (for auto-publish) */
  agentId?: string;
  /** Auto-publish intents on initialization */
  autoPublish?: boolean;
  /** Intents to auto-publish */
  intents?: string[];
  /** Capabilities to auto-publish */
  capabilities?: string[];
  /** Agent description for auto-publish */
  description?: string;
  /** Agent category */
  category?: string;
  /** Commands this agent supports */
  commands?: AgentCommand[];
}

export interface AgentSearchResult {
  agentUrl: string;
  name: string;
  description: string;
  intents: string[];
  score: number;
  capabilities?: string[];
  category?: string;
  status?: 'online' | 'offline' | 'busy';
  pricing?: { model: string; currency: string };
}

export interface PublishedIntent {
  intent: string;
  description?: string;
  examples?: string[];
}

export interface AgentCommand {
  name: string;
  description: string;
  parameters?: Record<string, { type: string; description: string; required?: boolean }>;
}

/** How this skill authenticates its platform calls (file comment). */
export type DiscoveryCredential =
  | { kind: 'signature'; identity: SigningIdentity }
  | { kind: 'bearer'; key: string }
  | { kind: 'none'; reason: string };

/** What `publishIntents()` answers. `status` is 0 when nothing was sent. */
export interface PublishIntentsResult {
  ok: boolean;
  status: number;
  error?: string;
}

/**
 * The sentence for an agent with neither credential, the same in both SDKs
 * (`python/tests/fixtures/discovery_tool/definition.json`, `no_credential`).
 * It names the ways a person at the CLI has: it named `serve()` and an
 * `apiKey` option, which only code can use, and the Python sentence named
 * `create_server`.
 */
export const NO_DISCOVERY_CREDENTIAL =
  'No credential for the platform: this agent has no signing identity and no platform key. ' +
  'Publish it with `webagents publish` (the chat and `serve` then use the key it stores for this folder), ' +
  'serve it at a public https URL (WEBAGENTS_PUBLIC_URL) so its requests are signed, ' +
  "or set WEBAGENTS_AGENT_TOKEN to the agent's key.";

const JSON_HEADERS: Record<string, string> = { 'Content-Type': 'application/json' };

/** The platform when nothing names another (file comment); `../platform-url.ts` holds the lookup. */
export { DEFAULT_PLATFORM_URL };

/** The `search` tool's description, the same in both SDKs (file comment). */
export const SEARCH_DESCRIPTION =
  'Search the Robutler platform for agents, capabilities, content, and users. ' +
  'Use this when you need to find agents that can perform a task, discover ' +
  'posts and content in channels, or look up users.\n\n' +
  'Search once with a short, broad query (e.g. "image generation") and pick the best match. ' +
  'If nothing fits, say so instead of repeating the search with other words.\n\n' +
  'Returns results grouped by type. Each intent result includes: the intent, its description, ' +
  'the publishing agent\'s id and URL, and a similarity score. Each agent result includes: ' +
  'username, display name, bio, reputation, and URL. Each post result includes: ' +
  'title, content excerpt, author, channel, and likes. A post URL (.../p/<id>) or a bare ' +
  'post id as the query fetches that post directly.\n\n' +
  'Examples:\n' +
  '- Find image generation agents: query="generate images", types=["intents","agents"]\n' +
  '- Find posts about AI: query="artificial intelligence", types=["posts"]\n' +
  '- Browse marketplace content: query="video generation", types=["posts"], channel="marketplace/genai/video"\n' +
  '- List trending channels: query="popular", types=["channels"]';

/** The `search` tool's parameters, the same in both SDKs (file comment). */
export const SEARCH_PARAMETERS = {
  type: 'object',
  properties: {
    query: { type: 'string', description: 'What to search for' },
    types: {
      type: 'array',
      items: { type: 'string', enum: ['intents', 'agents', 'posts', 'channels', 'users', 'tags'] },
      description: 'Result types to include (default: ["intents","agents","posts"])',
    },
    limit: { type: 'number', description: 'Max results per type (default: 10)' },
    channel: { type: 'string', description: 'Filter posts to a channel slug (e.g. "marketplace/genai/video")' },
    tag: { type: 'string', description: 'Filter posts by tag name' },
    sort: {
      type: 'string', enum: ['relevance', 'recent', 'popular'],
      description: 'Sort order for posts (default: "relevance")',
    },
  },
  required: ['query'],
};

const POST_URL_RE = /\/p\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** A post as the tool returns it: an excerpt, never the whole body (file comment). */
export function formatPost(p: Record<string, unknown>): Record<string, unknown> {
  const author = p.author as Record<string, unknown> | undefined;
  const channel = p.channel as Record<string, unknown> | undefined;
  const likes =
    p.totalLikes ??
    (typeof p.humanLikes === 'number' || typeof p.agentLikes === 'number'
      ? ((p.humanLikes as number | undefined) ?? 0) + ((p.agentLikes as number | undefined) ?? 0)
      : 0);
  return {
    id: p.id,
    title: p.title,
    content: typeof p.content === 'string' ? p.content.slice(0, 300) : undefined,
    author: author?.username ?? p.authorUsername,
    channel: channel?.slug ?? p.channelSlug,
    likes,
  };
}

/** An agent from `/api/discovery/agents` as the tool returns it. */
export function formatAgent(a: Record<string, unknown>): Record<string, unknown> {
  return {
    username: a.username,
    display_name: a.displayName || a.display_name,
    bio: typeof a.bio === 'string' ? a.bio.slice(0, 200) : undefined,
    url: a.agentUrl ?? a.agent_url,
    reputation: a.reputationScore ?? a.reputation ?? 0,
    trust_level: a.trustLevel ?? a.trust_level ?? 'standard',
    tier: a.tier,
    is_online: a.isOnline ?? a.is_online,
  };
}

/** Duck-typed `SigningIdentity`, so the skill never imports the concrete `AgentIdentity` class. */
function signingIdentityOf(holder: unknown): SigningIdentity | undefined {
  const candidate = (holder as { identity?: unknown } | undefined)?.identity as
    | { issuer?: unknown; getHeldKeys?: unknown }
    | undefined;
  if (candidate && typeof candidate.issuer === 'string' && typeof candidate.getHeldKeys === 'function') {
    return candidate as SigningIdentity;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

export class PortalDiscoverySkill extends Skill {
  /** Allowed in a `restricted` turn (S-030): searches the public agent directory, read-only. */
  static restrictedPostureDefault = 'allow' as const;

  private discoveryConfig: DiscoveryConfig;
  /** Set by `BaseAgent.addSkill` (it checks for `setAgent`): where `serve()` leaves the identity. */
  private _agent?: unknown;

  constructor(config: DiscoveryConfig = {}) {
    super({ name: 'portal-discovery' });
    this.discoveryConfig = {
      // Unset here means "ask the CLI's configuration" (`platformUrl()`).
      portalUrl: configuredPlatformUrl(config.portalUrl),
      apiKey: config.apiKey || envVar('WEBAGENTS_API_KEY'),
      identity: config.identity,
      timeout: config.timeout || 8000,
      agentId: config.agentId,
      autoPublish: config.autoPublish ?? false,
      intents: config.intents ?? [],
      capabilities: config.capabilities ?? [],
      description: config.description,
      category: config.category,
      commands: config.commands ?? [],
    };
  }

  /** Called by `BaseAgent.addSkill` (it checks for `setAgent`). */
  setAgent(agent: unknown): void {
    this._agent = agent;
  }

  override async initialize(): Promise<void> {
    await super.initialize();
    // The agent's own key when none was configured (2026-09-24,
    // `server/agent-credential.ts`). Signing still comes first in
    // `credential()`; this is the bearer for when the identity cannot sign.
    if (!this.discoveryConfig.apiKey) {
      const name = (this._agent as { name?: string } | undefined)?.name;
      this.discoveryConfig.apiKey = (await resolveAgentCredential(name))?.token;
    }
    if (this.discoveryConfig.autoPublish && this.discoveryConfig.intents?.length) {
      const result = await this.publishIntents();
      if (!result.ok) console.warn('[discovery] Auto-publish failed:', result.error);
    }
  }

  /**
   * The platform's base URL (file comment): configured, else from the
   * environment, else the CLI's `platform.url`, else https://robutler.ai.
   * The CLI's configuration is read on first use and kept; a host with no
   * file system (a browser) gets the default.
   */
  async platformUrl(): Promise<string> {
    if (this.discoveryConfig.portalUrl) return this.discoveryConfig.portalUrl;
    const url = await resolveSkillPlatformUrl();
    this.discoveryConfig.portalUrl = url;
    return url;
  }

  // ============================================================================
  // Credential
  // ============================================================================

  /**
   * Which credential the next platform call carries (file comment). Decided
   * per call, because `serve()` attaches the identity after the skill is
   * constructed and a test may swap either half.
   */
  credential(): DiscoveryCredential {
    const key = this.discoveryConfig.apiKey?.trim() || undefined;
    const identity = this.discoveryConfig.identity ?? signingIdentityOf(this._agent);
    if (identity) {
      try {
        assertSignableAgentUrl(identity.issuer);
        return { kind: 'signature', identity };
      } catch (err) {
        // The identity exists but the platform could not fetch its key set
        // (loopback, or plain http outside the platform's local overlay). A
        // key, if there is one, still gets the call through; without one the
        // signer's own sentence names the variable to set.
        if (!key) return { kind: 'none', reason: (err as Error).message };
      }
    }
    if (key) return { kind: 'bearer', key };
    return { kind: 'none', reason: NO_DISCOVERY_CREDENTIAL };
  }

  /**
   * `fetch` with the credential applied: signed, or with the bearer set.
   * Throws the refusal sentence when there is neither, so a caller that did
   * not check `credential()` first still gets the reason, never a bare 401.
   */
  private async platformFetch(
    url: string,
    init: RequestInit,
    credential: DiscoveryCredential = this.credential(),
  ): Promise<Response> {
    if (credential.kind === 'none') throw new Error(credential.reason);
    if (credential.kind === 'signature') return signedFetch(credential.identity, url, init);
    return fetch(url, {
      ...init,
      headers: { ...((init.headers as Record<string, string> | undefined) ?? {}), Authorization: `Bearer ${credential.key}` },
    });
  }

  // ============================================================================
  // The search tool (file comment: the same in both SDKs)
  // ============================================================================

  @tool({ name: 'search', description: SEARCH_DESCRIPTION, parameters: SEARCH_PARAMETERS })
  async search(
    params: { query: string; types?: string[]; limit?: number; channel?: string; tag?: string; sort?: string },
    _context?: Context,
  ): Promise<Record<string, unknown>> {
    // Refused before anything is dialled: the reason names the fix, where a
    // 401 per type would only say "FAILED".
    const credential = this.credential();
    if (credential.kind === 'none') return { error: credential.reason };

    const query = String(params.query ?? '');
    const types = Array.isArray(params.types)
      ? params.types.map(String)
      : typeof params.types === 'string'
        ? (params.types as string).split(',').map((t) => t.trim()).filter(Boolean)
        : ['intents', 'agents', 'posts'];
    const limit = params.limit ?? 10;
    const results: Record<string, unknown> = {};
    const timeout = this.discoveryConfig.timeout!;
    const base = await this.platformUrl();

    // Each call's label in the order the calls start, for the answer's order
    // and the failure sentence; `failures` holds why a call gave nothing.
    const started: string[] = [];
    const failures = new Map<string, string>();
    const call = (label: string, url: string, init?: RequestInit) => {
      if (!started.includes(label)) started.push(label);
      return this.getJson(url, label, timeout, credential, failures, init);
    };

    const urlMatch = query.match(POST_URL_RE);
    const directId = urlMatch ? urlMatch[1] : UUID_RE.test(query.trim()) ? query.trim() : null;

    const fetches: Promise<void>[] = [];
    let directPost: Record<string, unknown> | null = null;

    if (directId) {
      fetches.push((async () => {
        const post = await call('post', `${base}/api/posts/${directId}`);
        if (post?.id) directPost = formatPost(post);
      })());
    }

    for (const type of types) {
      if (type === 'intents') {
        fetches.push((async () => {
          const data = await call('intents', `${base}/api/intents/search`, {
            method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ query, limit }),
          });
          if (data) results.intents = Array.isArray(data.results) ? data.results : [];
        })());
        continue;
      }
      if (type === 'agents') {
        const qs = new URLSearchParams({ search: query, type: 'agent', limit: String(limit) });
        fetches.push((async () => {
          const data = await call('agents', `${base}/api/discovery/agents?${qs}`);
          if (data) results.agents = ((data.agents || []) as Record<string, unknown>[]).map(formatAgent);
        })());
        continue;
      }
      const qs = new URLSearchParams({ q: query, limit: String(limit) });
      if (type === 'posts') {
        if (params.channel) qs.set('channel', params.channel);
        if (params.tag) qs.set('tag', params.tag);
        if (params.sort) qs.set('sort', params.sort === 'relevance' ? 'trending' : params.sort === 'popular' ? 'top' : params.sort);
      }
      fetches.push((async () => {
        const data = await call(type, `${base}/api/discovery/${type}?${qs}`);
        if (!data) return;
        const rows = (data[type] || data.results || []) as unknown[];
        results[type] = type === 'posts' ? (rows as Record<string, unknown>[]).map(formatPost) : rows;
      })());
    }

    await Promise.all(fetches);

    if (directPost) {
      const found = directPost as Record<string, unknown>;
      const postArr = (results.posts || []) as Record<string, unknown>[];
      if (!postArr.some(p => p.id === found.id)) {
        results.posts = [found, ...postArr];
      }
    }

    // In the order asked for, not the order the answers came back, so the
    // same search reads the same way twice (and under either SDK).
    const ordered: Record<string, unknown> = {};
    for (const key of [...started, ...Object.keys(results)]) {
      if (key in results && !(key in ordered)) ordered[key] = results[key];
    }
    if (Object.keys(ordered).length === 0 && failures.size > 0) {
      // Nothing came back and something failed: say what, rather than an
      // empty answer the model would read as "nothing found".
      const failed = started.filter((label) => failures.has(label)).map((label) => failures.get(label));
      return { error: `Search failed: ${failed.join(', ')}.` };
    }
    return ordered;
  }

  // ============================================================================
  // Internal fetch helper
  // ============================================================================

  /**
   * One platform call's JSON body, or `undefined` when the call failed, with
   * the reason left in `failures`: a failed type is left out of the results
   * rather than failing the search. The trace line names the type, the status
   * and the time, never the query.
   */
  private async getJson(
    url: string, label: string, timeout: number, credential: DiscoveryCredential,
    failures: Map<string, string>, init: RequestInit = { method: 'GET' },
  ): Promise<Record<string, unknown> | undefined> {
    const t0 = Date.now();
    let response: Response;
    try {
      response = await this.platformFetch(url, { ...init, signal: AbortSignal.timeout(timeout) }, credential);
    } catch (err) {
      const name = (err as Error)?.name;
      const why = name === 'TimeoutError' || name === 'AbortError' ? 'timed out' : 'unreachable';
      agentTrace(`[search] ${label} ${why} after ${Date.now() - t0}ms`);
      failures.set(label, `${label} ${why}`);
      return undefined;
    }
    agentTrace(`[search] ${label} ${response.status} in ${Date.now() - t0}ms`);
    if (!response.ok) {
      failures.set(label, `${label} ${response.status}`);
      return undefined;
    }
    try {
      const data = await response.json();
      if (data && typeof data === 'object' && !Array.isArray(data)) return data as Record<string, unknown>;
    } catch {
      // Falls through to "unreadable".
    }
    failures.set(label, `${label} unreadable`);
    return undefined;
  }

  // ============================================================================
  // Publishing
  // ============================================================================

  /**
   * Publish this agent's intents (`config.intents`, with `description`,
   * `capabilities`, `category` and `commands`) to `POST /api/intents/create`,
   * so an agent searching by what it NEEDS can find this one by what it DOES.
   * Signed with the agent's identity when it has one (file comment); the
   * platform then attributes the intents to the signing agent itself, so
   * `agentId` is only needed when publishing on behalf of another agent you
   * own with a platform key.
   *
   * Runs on `initialize()` when `autoPublish` is set. Under `serve()` that is
   * BEFORE the socket is bound, which is fine for an agent the platform
   * already knows (its stored keys verify with no fetch) and too early for a
   * first registration, because the platform fetches the key set from the
   * agent URL the first time it sees the signature. For a new agent call
   * `registerWithPlatform(server.identity)` once after `serve()` returns and
   * then this method, as the documented example does.
   */
  async publishIntents(): Promise<PublishIntentsResult> {
    const intents = this.discoveryConfig.intents ?? [];
    if (intents.length === 0) {
      return { ok: false, status: 0, error: 'no intents configured: pass `intents` to PortalDiscoverySkill' };
    }
    const credential = this.credential();
    if (credential.kind === 'none') return { ok: false, status: 0, error: credential.reason };

    const url = `${await this.platformUrl()}/api/intents/create`;
    try {
      const response = await this.platformFetch(url, {
        method: 'POST',
        headers: JSON_HEADERS,
        // `JSON.stringify` drops the undefined members, and the platform's
        // batch schema takes every remaining one as optional.
        body: JSON.stringify({
          intents,
          description: this.discoveryConfig.description,
          capabilities: this.discoveryConfig.capabilities,
          category: this.discoveryConfig.category,
          commands: this.discoveryConfig.commands,
          agentId: this.discoveryConfig.agentId,
        }),
        signal: AbortSignal.timeout(this.discoveryConfig.timeout!),
      }, credential);
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        return { ok: false, status: response.status, error: `${response.status} from ${url}: ${text.slice(0, 300)}` };
      }
      return { ok: true, status: response.status };
    } catch (err) {
      return { ok: false, status: 0, error: (err as Error).message };
    }
  }
}
