/**
 * The access skill: who is calling, and which group they are in (ADR-0045).
 *
 * An agent file with an `access:` block gets this skill. On every turn, before
 * the model runs (`on_connection`, after an auth skill and before payments), it:
 *
 *   1. takes the caller's TIER as already established by something this process
 *      trusts: the local chat's owner (`RunOptions.auth`), or an auth skill that
 *      verified a platform credential. Nothing in the request can set it;
 *   2. collects the caller's PRINCIPALS from verified credentials only:
 *        - `user:<id>` from a platform credential an auth skill verified (an API
 *          key or an owner assertion), or from the `sender` the platform signs
 *          into a service token whose audience is this agent's own URL
 *          (S-240); the request body's sender names no one;
 *        - `agent:<URL>` and `key:<thumbprint>` from a Web Bot Auth signature,
 *          verified here (`crypto/web-bot-auth-verify.ts`). A signature that is
 *          present and does not verify refuses the request (401); it is never
 *          read as "anonymous". A bearer the agent has no way to verify names
 *          no one, as it always has;
 *   3. decides with the block (`access/policy.ts`): refused (403), or let in
 *      with its groups, which become `group:<name>` scopes on the context.
 *
 * The request reaches it as `_inboundRequest` in the run's session data, which
 * only the server sets (`server/handler.ts`); the request body's `metadata`
 * cannot. It also adds each group's instructions file as a prompt scoped to that
 * group, and one short prompt telling the model who this turn is from.
 *
 * The Python twin is `python/webagents/agents/skills/local/access/skill.py`.
 */

import { Skill } from '../../core/skill';
import { hook } from '../../core/decorators';
import type { AuthInfo, Context, HookData, HookResult, Prompt } from '../../core/types';
import {
  decide,
  trustKey,
  trustRequirements,
  verifiedAgentOf,
  type AccessPolicy,
  type TrustEvidence,
} from '../../access/policy';
import { channelIdentityOf } from '../../access/caller';
import { GROUP_PREFIX } from '../../core/scopes';
import { TrustLookup, platformCredentialFor } from '../../trustflow/trust-lookup';
import { envVar } from '../platform-url';
import {
  KeySetFetcher,
  MemoryNonceStore,
  normalizeAuthority,
  verifyWebBotAuth,
  type InboundRequest,
} from '../../crypto/web-bot-auth-verify';

export const NO_ADDRESS =
  'This agent has no public address, so it cannot check a signature made for one. ' +
  'Serve it with WEBAGENTS_PUBLIC_URL set to the address callers sign for.';
export const FORBIDDEN = 'This agent does not accept requests from this caller.';

/** The session-data key the server puts the inbound request under. */
export const INBOUND_REQUEST_KEY = '_inboundRequest';

/** A request the access block refuses. Named so the server's auth handling answers it. */
export class AccessRefusedError extends Error {
  constructor(
    readonly statusCode: 401 | 403,
    readonly code: string,
    message: string,
  ) {
    super(message);
    // `server/handler.ts` answers errors by these two names.
    this.name = statusCode === 401 ? 'AuthenticationError' : 'AuthorizationError';
  }
}

export function tierOf(auth: Partial<AuthInfo> | undefined): 'owner' | 'admin' | null {
  if (!auth || auth.authenticated === false) return null;
  const scopes = new Set<string>([...(auth.scopes ?? []), ...(auth.scope ? [String(auth.scope)] : [])]);
  if (scopes.has('admin')) return 'admin';
  if (scopes.has('owner')) return 'owner';
  return null;
}

export function userPrincipals(auth: Partial<AuthInfo> | undefined): string[] {
  if (!auth || !auth.authenticated || auth.provider === 'local') return [];
  let userId: unknown;
  let username: unknown;
  if (auth.provider === 'service_token') {
    // A service token names the router. The person is the platform's signed
    // `sender` claim, and only on a token whose audience is this agent's own
    // URL: any other could have been minted for another agent and replayed
    // here (S-240). The body's sender names no one.
    const sender = auth.audienceVerified
      ? (auth.claims as { sender?: { id?: unknown; username?: unknown } } | undefined)?.sender
      : undefined;
    userId = sender?.id;
    username = sender?.username;
  } else {
    userId = auth.user_id;
    username = (auth as { username?: unknown }).username;
  }
  const out: string[] = [];
  // A sender the platform relayed from a connected channel (plan item 2.2)
  // is FIRST: `channel:<type>:<sender id>` is the stable identity of the
  // person (the synthetic `user:` id behind it is a platform artefact), so
  // it is what an access block names and what the memory skill keys the
  // caller's namespace on (`namespaceOf` takes the first verified principal).
  const channel = channelIdentityOf((auth as { channel?: unknown }).channel);
  if (channel) out.push(`channel:${channel.type}:${channel.sender_id}`);
  if (typeof userId === 'string' && userId) out.push(`user:${userId}`);
  if (typeof username === 'string' && username) out.push(`user:@${username}`);
  return out;
}

function ownAddress(publicUrl: string | undefined): { authority: string; scheme: 'https' | 'http' } | null {
  const raw = publicUrl ?? (typeof process !== 'undefined' ? process.env.WEBAGENTS_PUBLIC_URL : undefined);
  if (!raw) return null;
  let url: URL;
  try {
    url = new URL(raw.trim());
  } catch {
    return null;
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return null;
  const authority = normalizeAuthority(url.host);
  return authority ? { authority, scheme: url.protocol === 'https:' ? 'https' : 'http' } : null;
}

/** One sentence for the model about this turn's caller. */
export function whoIsCalling(auth: Partial<AuthInfo> | undefined): string {
  const tier = tierOf(auth);
  if (tier === 'owner') return "This turn is from this agent's owner.";
  if (tier === 'admin') return 'This turn is from an admin of this agent.';
  const principals = ((auth as { principals?: string[] } | undefined)?.principals ?? []) as string[];
  const groups = (auth?.scopes ?? []).filter((s) => s.startsWith(GROUP_PREFIX)).map((s) => s.slice(GROUP_PREFIX.length));
  const channel = principals.find((p) => p.startsWith('channel:'))?.slice('channel:'.length);
  const agent = principals.find((p) => p.startsWith('agent:'))?.slice('agent:'.length);
  const user = principals.find((p) => p.startsWith('user:'))?.slice('user:'.length);
  const sep = channel?.indexOf(':') ?? -1;
  const who = channel && sep > 0
    ? `the sender ${channel.slice(sep + 1)} on ${channel.slice(0, sep)}, relayed by Robutler`
    : agent
      ? `the agent ${agent}, which proved it with a Web Bot Auth signature`
      : user
        ? `the Robutler user ${user}`
        : 'a caller who did not identify itself';
  return `This turn is from ${who}. Groups: ${groups.join(', ') || 'none'}.`;
}

/**
 * Where a trust-gated group's scores come from (plan item 2.5): the platform,
 * through `trustflow/trust-lookup.ts`, asked as THIS agent. A test stubs it.
 */
export interface TrustScoreSource {
  lookup(agent: string, topic?: string): Promise<{ score: number; topic?: { query: string; score: number | null } }>;
}

export interface AccessSkillConfig {
  policy: AccessPolicy;
  /** Group name to the text of its instructions file. */
  instructionTexts?: Map<string, string>;
  keySets?: Pick<KeySetFetcher, 'get'>;
  nonces?: Pick<MemoryNonceStore, 'spend'>;
  /** The address callers sign for; `WEBAGENTS_PUBLIC_URL` when unset. */
  publicUrl?: string;
  /** Trust scores for `trust:` groups; built from the agent's own credential when unset. */
  trust?: TrustScoreSource;
}

export class AccessSkill extends Skill {
  /**
   * Establishes who is calling (`BaseAgent.identifyCaller`): a scoped
   * `@http` or `@websocket` endpoint runs this skill's `on_connection` hook.
   */
  static readonly identifiesCaller = true;

  /** It adds no tools, so a restricted turn loses nothing by letting it run. */
  static restrictedPostureDefault = 'allow' as const;

  readonly policy: AccessPolicy;
  keySets: Pick<KeySetFetcher, 'get'>;
  nonces: Pick<MemoryNonceStore, 'spend'>;
  publicUrl?: string;
  /** Trust scores for `trust:` groups (`TrustScoreSource`); the platform, as this agent, when unset. */
  trust?: TrustScoreSource;
  private readonly groupPrompts: Prompt[];
  /** Set by `BaseAgent.addSkill` (it checks for `setAgent`): where `serve()` leaves the identity the lookup signs with. */
  private _agent?: { identity?: unknown; apiKey?: unknown };

  constructor(config: AccessSkillConfig) {
    super({ name: 'access' });
    this.policy = config.policy;
    this.trust = config.trust;
    const allowPrivate = typeof process !== 'undefined' && process.env.ROBUTLER_AGENT_URL_ALLOW_PRIVATE === '1';
    this.keySets = config.keySets ?? new KeySetFetcher({ allowPrivate });
    this.nonces = config.nonces ?? new MemoryNonceStore();
    this.publicUrl = config.publicUrl;
    this.groupPrompts = [
      {
        name: 'accessCaller',
        priority: 4,
        scope: 'all',
        handler: (context: Context) => `## Caller\n${whoIsCalling(context.auth)}`,
      },
      ...[...(config.instructionTexts ?? new Map()).entries()].map(([group, text]) => ({
        name: `accessInstructions_${group}`,
        priority: 5,
        scope: `${GROUP_PREFIX}${group}`,
        handler: () => text,
      })),
    ];
  }

  /** The caller sentence and each group's instructions (see the file comment). */
  get prompts(): Prompt[] {
    return this.groupPrompts;
  }

  /** Called by `BaseAgent.addSkill` (it checks for `setAgent`). */
  setAgent(agent: unknown): void {
    this._agent = agent as { identity?: unknown; apiKey?: unknown };
  }

  /** The configured source, else the platform asked as this agent (its identity signs; else its key is the bearer). */
  private trustSource(): TrustScoreSource {
    if (!this.trust) {
      this.trust = new TrustLookup({
        credential: () =>
          platformCredentialFor({
            identity: this._agent?.identity,
            apiKey:
              (typeof this._agent?.apiKey === 'string' ? this._agent.apiKey : undefined) ??
              envVar('WEBAGENTS_AGENT_TOKEN') ??
              envVar('WEBAGENTS_API_KEY'),
          }),
      });
    }
    return this.trust;
  }

  /**
   * What the platform says about the verified calling agent, for every trust
   * requirement the block makes: one lookup per topic key. A lookup that
   * fails leaves its key out, and a caller with no `agent:` principal gets no
   * evidence at all; `decide` then fails CLOSED for those groups
   * (`access/policy.ts`). Null when there is nothing to look up.
   */
  async trustEvidence(principals: readonly string[]): Promise<TrustEvidence | null> {
    const requirements = trustRequirements(this.policy);
    const agent = verifiedAgentOf(principals);
    if (requirements.length === 0 || !agent) return null;
    const source = this.trustSource();
    const scores: Record<string, number> = {};
    await Promise.all(
      requirements.map(async (requirement) => {
        try {
          const result = await source.lookup(agent, requirement.topic ?? undefined);
          const score = requirement.topic ? result.topic?.score : result.score;
          if (typeof score === 'number' && Number.isFinite(score)) scores[trustKey(requirement.topic)] = score;
        } catch {
          // Fail closed: no score for this key, so the group is not joined.
        }
      }),
    );
    return { agent, scores };
  }

  @hook({ lifecycle: 'on_connection', priority: 1 })
  async admit(_data: HookData, context: Context): Promise<HookResult | void> {
    const auth = context.auth as Partial<AuthInfo> | undefined;
    const tier = tierOf(auth);
    const principals = userPrincipals(auth);

    const inbound = context.get<InboundRequest>(INBOUND_REQUEST_KEY);
    const signed = inbound ? headerValue(inbound.headers, 'signature-input') !== null : false;
    if (inbound && signed) {
      const own = ownAddress(this.publicUrl);
      if (!own) throw new AccessRefusedError(401, 'signature_authority_mismatch', NO_ADDRESS);
      const outcome = await verifyWebBotAuth(inbound, {
        authorities: [own.authority],
        scheme: own.scheme,
        keySets: this.keySets,
        nonces: this.nonces,
        allowHttp: typeof process !== 'undefined' && process.env.ROBUTLER_AGENT_URL_ALLOW_PRIVATE === '1',
      });
      if (!outcome.ok) throw new AccessRefusedError(401, outcome.refusal.code, outcome.refusal.description);
      principals.push(`agent:${outcome.agent.principal}`, ...outcome.agent.thumbprints.map((t) => `key:${t}`));
    }

    const decision = decide(this.policy, principals, tier, await this.trustEvidence(principals));
    if (!decision.allow) throw new AccessRefusedError(403, 'forbidden', FORBIDDEN);

    // The groups this block decided, and none a caller brought along.
    const kept = (auth?.scopes ?? []).filter((s) => !s.startsWith(GROUP_PREFIX));
    const next: AuthInfo & { principals?: string[] } = {
      ...(auth ?? {}),
      authenticated: Boolean(tier || principals.length || auth?.authenticated),
      scopes: [...kept, ...decision.groups.map((g) => `${GROUP_PREFIX}${g}`)],
      principals,
    } as AuthInfo & { principals?: string[] };
    try {
      (context as { auth: AuthInfo }).auth = next;
    } catch {
      // read-only in some contexts
    }
    const setter = (context as unknown as { setAuth?: (a: AuthInfo) => void }).setAuth;
    if (typeof setter === 'function') setter.call(context, next);
  }
}

function headerValue(headers: InboundRequest['headers'], name: string): string | null {
  if (typeof (headers as Pick<Headers, 'get'>).get === 'function') return (headers as Pick<Headers, 'get'>).get(name);
  for (const [key, value] of Object.entries(headers as Record<string, string | string[] | undefined>)) {
    if (key.toLowerCase() === name && value !== undefined) return Array.isArray(value) ? value.join(', ') : value;
  }
  return null;
}
