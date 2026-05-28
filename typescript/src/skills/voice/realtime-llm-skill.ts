/**
 * RealtimeLLMSkill (Plan v3-03).
 *
 * Layers chat-completion-style ergonomics over the underlying
 * `RealtimeTransportSkill` for voice agents that want to:
 *   - Maintain a conversation context server-side (so reconnects /
 *     stream interruptions don't lose state).
 *   - Forward tool-call invocations to other skills on the same agent
 *     (e.g. compose with Plan 2's `BrowserControlSkill` for a
 *     voice-controlled browser agent — see
 *     `docs/public/guides/agents/voice-with-tools.md`).
 *   - Stream responses back through the realtime transport without
 *     skill authors having to know the underlying UAMP event shape.
 *
 * Provider-agnostic: the skill takes a `provider` + `model` + `voiceId`
 * triple at construction and delegates the wire-protocol to the
 * transport. Provider-specific HTTP wire is intentionally stubbed for
 * v3-03 (TODO markers identify the integration points); the public
 * surface area is locked so consumer agents can compose against the
 * stable shape today.
 *
 * Composability example (informational, not executed here):
 *
 * ```ts
 * import { RealtimeLLMSkill } from 'webagents/skills/voice';
 * import { BrowserControlSkill } from 'webagents/skills/browser';
 *
 * const agent = new Agent({
 *   skills: [
 *     new RealtimeLLMSkill({
 *       provider: 'openai',
 *       model: 'gpt-4o-realtime-preview',
 *       voiceId: 'alloy',
 *       systemPrompt: 'You can drive a browser via voice.',
 *     }),
 *     new BrowserControlSkill({ ... }),
 *   ],
 * });
 * ```
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context } from '../../core/types';
import type { VoiceProvider } from './types';

/**
 * Plan v3-03 #10 — payment hook surface. Same dependency-injection
 * pattern as `BrowserControlSkill.hooks`: the agent runtime binds
 * portal-side `lockPaymentToken` / `settlePaymentToken` at activation
 * time so the SDK stays standalone (no portal imports).
 *
 * When omitted, the skill runs in "standalone" mode (tests + local
 * dev): `onActivate` / `onDeactivate` succeed but no lock/settle wire
 * is hit. Production agents always supply hooks via the runtime.
 */
export interface VoicePaymentHooks {
  lockPayment: (args: {
    tokenOrJti: string;
    agentId: string;
    amount: bigint;
  }) => Promise<{ lockId: string; lockedAmount: bigint } | { error: string; available: bigint }>;
  settlePayment: (args: {
    lockId: string;
    amount: bigint;
    agentId: string;
    description?: string;
  }) => Promise<{ success: boolean; charged: bigint; remaining: bigint; error?: string }>;
}

/**
 * Runtime-bound context for billing — supplied by the agent runtime,
 * not the skill author. Mirrors `BrowserControlSkill.runtimeContext`.
 */
export interface VoiceBillingRuntimeContext {
  userId: string;
  agentId: string;
  paymentTokenOrJti: string;
  /**
   * Per-minute cost in nano-dollars (1e9 = $1). Sum of input + output
   * rates resolved by the runtime from env (`VOICE_RATE_INPUT_PER_MIN`
   * + `VOICE_RATE_OUTPUT_PER_MIN`, defaults $0.06/min + $0.24/min).
   */
  costPerMinNanoCents: bigint;
  /**
   * Provider-side session TTL in seconds — drives the initial lock
   * amount (`durationMin × costPerMin`). Defaults to 30min when zero.
   */
  providerSessionTtlSeconds: number;
}

export interface RealtimeLLMConfig {
  name?: string;
  enabled?: boolean;
  provider: VoiceProvider;
  /** Provider-side model name, e.g. `gpt-4o-realtime-preview`. */
  model: string;
  /** Provider-side voice id, e.g. `alloy` for OpenAI. */
  voiceId: string;
  /** Agent persona / instructions; injected as `session.update`. */
  systemPrompt: string;
  /**
   * Optional override for the underlying transport skill name. The
   * skill resolves the matching `RealtimeTransportSkill` instance from
   * the agent's skill registry via this name. Defaults to the canonical
   * `realtime-transport` name.
   */
  transportSkillName?: string;
  /**
   * Plan v3-03 #10 — billing hooks + runtime context. When both are
   * supplied, `onActivate` upfront-locks `costPerMin × ttlMin` and
   * `onDeactivate` settles for the actual elapsed minutes. Omit for
   * standalone test mode.
   */
  paymentHooks?: VoicePaymentHooks;
  billingContext?: VoiceBillingRuntimeContext;
}

/**
 * Lightweight upstream-spec shape returned by the transport — kept
 * here as a local interface so the LLM skill doesn't have to import
 * runtime types from the transport package (which would create a
 * publishing-graph cycle when these skills get split apart).
 */
export interface RealtimeUpstreamSpec {
  provider: VoiceProvider;
  /**
   * Provider-side session id. Plan 3's voice dispatcher handler
   * propagates this to the widget; the corresponding ephemeral
   * provider token is NEVER propagated (see ADR-v3-12).
   */
  providerSessionId: string;
  /** Optional websocket endpoint, kept server-side only. */
  websocketUrl?: string;
}

interface ConversationTurn {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  toolCallId?: string;
  ts: number;
}

export class RealtimeLLMSkill extends Skill {
  private readonly provider: VoiceProvider;
  private readonly model: string;
  private readonly voiceId: string;
  private readonly systemPrompt: string;
  /**
   * Name of the underlying transport skill instance to compose with.
   * Resolved lazily at runtime (see TODO in `getUpstreamSpec`) — kept
   * as a configured value so the dispatcher / skill registry can
   * locate the right transport when multiple are mounted on the same
   * agent (rare but supported).
   */
  protected readonly transportSkillName: string;

  /**
   * Per-session conversation context. Keyed by the underlying
   * transport's session id so we can serve interleaved tool calls and
   * streamed audio responses without leaking state across sessions.
   */
  private readonly conversations = new Map<string, ConversationTurn[]>();

  // Plan v3-03 #10 billing state — set during onActivate, settled in onDeactivate.
  private readonly paymentHooks?: VoicePaymentHooks;
  private readonly billingContext?: VoiceBillingRuntimeContext;
  private paymentLockId: string | null = null;
  private activatedAt: number | null = null;
  private lastSessionId: string | null = null;

  constructor(config: RealtimeLLMConfig) {
    super({
      ...config,
      name: config.name || 'realtime-llm',
    });
    this.provider = config.provider;
    this.model = config.model;
    this.voiceId = config.voiceId;
    this.systemPrompt = config.systemPrompt;
    this.transportSkillName = config.transportSkillName || 'realtime-transport';
    this.paymentHooks = config.paymentHooks;
    this.billingContext = config.billingContext;
  }

  /**
   * Plan v3-03 #10 — upfront-lock the worst-case session cost. Called
   * by the dispatcher right after `getUpstreamSpec` resolves but BEFORE
   * the provider session goes hot. Failure to lock rethrows so the
   * caller short-circuits cleanly without burning a provider session.
   *
   * Mirrors `BrowserControlSkill.openInternal`'s lockPayment step.
   */
  async onActivate(): Promise<void> {
    if (!this.paymentHooks || !this.billingContext) {
      // Standalone mode (tests / local dev) — no lock wire.
      this.activatedAt = Date.now();
      return;
    }
    const ttlSec = this.billingContext.providerSessionTtlSeconds > 0
      ? this.billingContext.providerSessionTtlSeconds
      : 1800; // 30 min default
    const durationMin = ttlSec / 60;
    // Worst-case lock amount: ceil(durationMin) × costPerMin so we never
    // under-reserve due to truncation.
    const lockAmount = BigInt(Math.ceil(durationMin)) * this.billingContext.costPerMinNanoCents;
    const lock = await this.paymentHooks.lockPayment({
      tokenOrJti: this.billingContext.paymentTokenOrJti,
      agentId: this.billingContext.agentId,
      amount: lockAmount,
    });
    if ('error' in lock) {
      throw new Error(`voice payment lock failed: ${lock.error}`);
    }
    this.paymentLockId = lock.lockId;
    this.activatedAt = Date.now();
  }

  /**
   * Plan v3-03 #10 — settle the lock based on actual elapsed minutes.
   * Wrapped in try/finally by the caller; we additionally guard our
   * own state cleanup so a settle error never strands state.
   */
  async onDeactivate(): Promise<void> {
    const lockId = this.paymentLockId;
    const startedAt = this.activatedAt;
    this.paymentLockId = null;
    this.activatedAt = null;
    if (!this.paymentHooks || !this.billingContext || !lockId || startedAt == null) {
      return;
    }
    const actualDurationMin = Math.max(0, (Date.now() - startedAt) / 60_000);
    // Charge at least 1 cent equivalent if any time elapsed — provider
    // sessions still cost the upstream provider even for short bursts.
    const chargeNano = BigInt(Math.ceil(actualDurationMin * Number(this.billingContext.costPerMinNanoCents)));
    try {
      await this.paymentHooks.settlePayment({
        lockId,
        amount: chargeNano,
        agentId: this.billingContext.agentId,
        description: `voice session (${actualDurationMin.toFixed(2)}min, provider=${this.provider})`,
      });
    } catch (err) {
      // Don't rethrow — caller is already in teardown; log and move on.
      console.error('[realtime-llm] settlePayment failed:', err);
    }
  }

  /**
   * Resolve the upstream provider session spec for the dispatcher.
   *
   * Plan 3's `voiceDispatchHandler` (server-side, in the portal) calls
   * this for Mode 2 agents — it gets back ONLY the non-secret
   * `{ provider, providerSessionId }` pair, never an ephemeral provider
   * token.
   *
   * The session id is minted here and used as the key under which the
   * dispatcher registers the configured `RealtimeTransportSkill` in the
   * relay registry. The actual provider WebSocket (Gemini Live) is
   * opened lazily when the widget connects to the relay — so resolving a
   * spec the user never connects to costs nothing upstream.
   */
  async getUpstreamSpec(opts: {
    runtimeContext?: Record<string, unknown>;
  }): Promise<RealtimeUpstreamSpec> {
    void opts;
    const providerSessionId =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `${this.provider}_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    this.lastSessionId = providerSessionId;
    return {
      provider: this.provider,
      providerSessionId,
    };
  }

  /** Most recently minted provider session id (read by the dispatcher). */
  getLastSessionId(): string | null {
    return this.lastSessionId;
  }

  /**
   * Append a turn to the agent's conversation context. Voice agent
   * implementations call this from a `hook({ lifecycle: 'message' })`
   * to keep the context in sync with the realtime stream.
   */
  appendTurn(sessionId: string, turn: ConversationTurn): void {
    const ctx = this.conversations.get(sessionId);
    if (ctx) {
      ctx.push(turn);
    } else {
      this.conversations.set(sessionId, [
        { role: 'system', content: this.systemPrompt, ts: Date.now() },
        turn,
      ]);
    }
  }

  getConversation(sessionId: string): ReadonlyArray<ConversationTurn> {
    return this.conversations.get(sessionId) ?? [];
  }

  @tool({
    name: 'realtime_llm_describe',
    description:
      'Describe the configured voice LLM (provider, model, voice). Read-only — useful for agent self-introspection.',
    parameters: { type: 'object', properties: {} },
  })
  async realtimeLLMDescribe(
    _params: Record<string, unknown>,
    _context: Context,
  ): Promise<{ provider: VoiceProvider; model: string; voiceId: string }> {
    return {
      provider: this.provider,
      model: this.model,
      voiceId: this.voiceId,
    };
  }

  override async cleanup(): Promise<void> {
    // Safety net: if `onDeactivate` was not called (e.g. transport crash
    // before the dispatcher's teardown ran), settle the lock here so a
    // crashed session doesn't strand a payment reservation. No-op when
    // `paymentLockId` is already null (deactivate ran cleanly).
    if (this.paymentLockId !== null) {
      try {
        await this.onDeactivate();
      } catch (err) {
        console.error('[realtime-llm] cleanup-time deactivate failed:', err);
      }
    }
    this.conversations.clear();
  }
}
