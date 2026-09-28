/**
 * The trust skill: the `trust` tool, and this agent's own signed TrustFlow
 * record (webagents gap-closure plan items 2.5 and 2.7, 2026-09-26). An agent
 * file names it as `- trust` (`skills/resolve.ts`); the Python twin is
 * `python/webagents/agents/skills/robutler/trust/trust_skill.py`, and the tool
 * both offer is pinned by `python/tests/fixtures/trust/trust_tool_definition.json`.
 *
 * TRUSTFLOW IS A PLATFORM SERVICE (the tool's description says so to the
 * model): the score comes from `GET /api/trust/lookup`, asked as this agent,
 * through `trustflow/trust-lookup.ts`, and held for a minute. The credential
 * is the discovery skill's rule: the agent's identity signs when it can,
 * else the agent's platform key (`server/agent-credential.ts`) is the bearer,
 * else the tool answers with the sentence naming the fix.
 *
 * Text in an answer that other people wrote (a username, a topic label) is
 * the platform's own data about a platform record, not free text an agent
 * chose, so it is not fenced the way search results are.
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';
import type { SigningIdentity } from '../../crypto/http-signature';
import { resolveAgentCredential } from '../../server/agent-credential';
import { configuredPlatformUrl, envVar } from '../platform-url';
import {
  TrustLookup,
  TrustLookupError,
  platformCredentialFor,
  type PlatformCredential,
  type TrustLookupResult,
  type TrustRecordResult,
} from '../../trustflow/trust-lookup';
import { verifyTrustRecord, type VerifyTrustRecordOptions, type VerifyTrustRecordResult } from '../../trustflow/trust-record';

/** The tool's description, the same in both SDKs (fixture `definition`). */
export const TRUST_DESCRIPTION =
  "Look up an agent's TrustFlow score on the Robutler platform. TrustFlow is a platform service: " +
  'Robutler computes the score from verified interactions on the platform, so it cannot be computed ' +
  'locally and this tool asks the platform. Use it before delegating to an agent you do not know, or ' +
  'to check whether an agent is trusted on a subject.\n\n' +
  "`agent` is the agent's URL, @username or platform id. `topic` (optional) scores the agent on that " +
  'subject, for example "billing".\n\n' +
  "Returns: subject (id, username, url), score (0 to 1, the agent's overall TrustFlow), topic (the " +
  "subject asked for and the agent's score on it, null when the platform could not score it), tier, " +
  "trust_level (none, silver, gold, platinum, flagged or suspended), topics (the subjects the agent's " +
  'earned reputation is strongest in, with a score each), computed_at and methodology. Answers are held ' +
  'for a minute.';

/** The tool's parameters, the same in both SDKs (fixture `definition`). */
export const TRUST_PARAMETERS = {
  type: 'object',
  properties: {
    agent: { type: 'string', description: 'The agent: its URL, @username or platform id' },
    topic: { type: 'string', description: 'A subject to score the agent on, for example "billing"' },
  },
  required: ['agent'],
};

export interface TrustSkillConfig {
  /** Platform base URL; unset means the skills' resolution (`platform-url.ts`). */
  portalUrl?: string;
  /** A platform key, presented as a bearer when the identity cannot sign. Defaults to the agent's own credential. */
  apiKey?: string;
  /** The identity to sign with; usually the one `serve()` attaches to the agent. */
  identity?: SigningIdentity;
  /** Per platform call, ms. */
  timeout?: number;
  /** How long an answer is held, ms. */
  ttl?: number;
}

export class TrustSkill extends Skill {
  /** Read-only, about public records: allowed in a restricted turn (S-030). */
  static restrictedPostureDefault = 'allow' as const;

  private readonly trustConfig: TrustSkillConfig;
  private _agent?: { name?: string; identity?: unknown };
  private client?: TrustLookup;

  constructor(config: TrustSkillConfig = {}) {
    super({ name: 'trust' });
    this.trustConfig = {
      ...config,
      portalUrl: configuredPlatformUrl(config.portalUrl),
      apiKey: config.apiKey || envVar('WEBAGENTS_API_KEY'),
    };
  }

  /** Called by `BaseAgent.addSkill` (it checks for `setAgent`): where `serve()` leaves the identity. */
  setAgent(agent: unknown): void {
    this._agent = agent as { name?: string; identity?: unknown };
  }

  override async initialize(): Promise<void> {
    await super.initialize();
    if (!this.trustConfig.apiKey) {
      this.trustConfig.apiKey = (await resolveAgentCredential(this._agent?.name))?.token;
    }
  }

  /** Which credential the next platform call carries, decided per call (an identity is attached after construction). */
  credential(): PlatformCredential {
    return platformCredentialFor({ identity: this.trustConfig.identity ?? this._agent?.identity, apiKey: this.trustConfig.apiKey });
  }

  /** The lookup client, built once. */
  get lookup(): TrustLookup {
    if (!this.client) {
      this.client = new TrustLookup({
        platformUrl: this.trustConfig.portalUrl,
        credential: () => this.credential(),
        timeoutMs: this.trustConfig.timeout,
        ttlMs: this.trustConfig.ttl,
      });
    }
    return this.client;
  }

  @tool({ name: 'trust', description: TRUST_DESCRIPTION, parameters: TRUST_PARAMETERS })
  async trust(params: { agent: string; topic?: string }, _context?: Context): Promise<Record<string, unknown>> {
    const agent = String(params?.agent ?? '').trim();
    if (!agent) return { error: 'agent is required: the agent URL, @username or platform id.' };
    const topic = typeof params?.topic === 'string' && params.topic.trim() ? params.topic.trim() : undefined;
    try {
      const result: TrustLookupResult = await this.lookup.lookup(agent, topic);
      return result as unknown as Record<string, unknown>;
    } catch (err) {
      if (err instanceof TrustLookupError) return { error: err.message };
      return { error: `The lookup failed: ${(err as Error).message}` };
    }
  }

  /** This agent's own signed record (its identity's URL), or another agent's when `agent` is given. */
  async record(agent?: string): Promise<TrustRecordResult> {
    const subject = agent ?? (this.trustConfig.identity ?? (this._agent?.identity as SigningIdentity | undefined))?.issuer;
    if (!subject) throw new TrustLookupError('no_credential', '(this agent)', undefined, 'This agent has no identity, so it has no record of its own to fetch; name the agent.');
    return this.lookup.record(subject);
  }

  /** Verify a record from any agent against the platform's key set (`trustflow/trust-record.ts`). */
  async verify(record: string, options: VerifyTrustRecordOptions = {}): Promise<VerifyTrustRecordResult> {
    return verifyTrustRecord(record, options);
  }
}
