/**
 * PaymentSkill - Full verify→lock→settle payment lifecycle
 *
 * Ports the Python PaymentSkill to TypeScript for the WebAgents framework.
 * All cost computation is done server-side: agents forward raw usage records
 * (model, prompt_tokens, completion_tokens, cached tokens) to /settle,
 * and the platform computes the dollar cost from MODEL_PRICING.
 */

import { Skill } from '../../core/skill';
import { resolveAgentCredential } from '../../server/agent-credential';
import { hook, getPricingForTool } from '../../core/decorators';
import type { HookData, HookResult, Context, PricingConfig } from '../../core/types';
import type { PaymentVerifyResult, PaymentSettleResult } from './types';
import { readSettleResult } from './settle-result';
import { OTEL_RUN_CONTEXT_KEY, errorType, type AgentRun, type PaymentSettleInfo } from '../../observability/otel';
import { PaymentRequiredError } from './x402';
import {
  IDEMPOTENCY_KEY_BODY_FIELD,
  freshSettleIdempotencyKey,
  idempotencyHeaders,
  settleIdempotencyKey,
} from './idempotency';
import { Paywall, chainSellerFromEnv, type ChainSellerConfig, type PaywallConfig } from './paywall';
import { platformCreditsClient } from './x402-credits';
import { DEFAULT_PLATFORM_URL, resolveSkillPlatformUrl } from '../platform-url';
import { facilitatorFromConfig, type FacilitatorClient, type FacilitatorConfig } from './x402-facilitator';
import { MppSeller, type MppSellerConfig } from './mpp-seller';

// ============================================================================
// Helpers
// ============================================================================

function getEnv(name: string): string | undefined {
  if (typeof process !== 'undefined' && process.env) {
    return process.env[name];
  }
  return undefined;
}

/**
 * The usage settle as an OpenTelemetry span under the run's (plan item 2.4,
 * 2026-09-26): the amount in credits, the lock, the outcome. The run handle
 * travels on the run context (`observability/otel.ts`); with none, or with
 * the switch off, nothing is recorded.
 */
function recordSettleSpan(context: Context, info: PaymentSettleInfo & { lockId: string }): void {
  const run = context.get<AgentRun>(OTEL_RUN_CONTEXT_KEY);
  if (run?.active) run.paymentSettle(info);
}

/** What a settle charged, in credits: the platform's credit amount, else its nanocredit string. */
function settledCredits(result: PaymentSettleResult): number {
  if (typeof result.chargedDollars === 'number') return result.chargedDollars;
  const charged = result.charged;
  if (typeof charged === 'number') return charged / 1e9;
  if (typeof charged === 'string' && charged.trim()) {
    const nano = Number(charged);
    return Number.isFinite(nano) ? nano / 1e9 : 0;
  }
  return 0;
}

// ============================================================================
// Config & Supporting Types
// ============================================================================

export interface PaymentSkillConfig {
  enableBilling?: boolean;
  /** Platform URL for payment APIs. Defaults to env ROBUTLER_PLATFORM_URL / ROBUTLER_API_URL */
  platformUrl?: string;
  /** @deprecated Use platformUrl */
  platformApiUrl?: string;
  apiKey?: string;
  minimumBalance?: number;
  perMessageLock?: number;
  defaultToolLock?: number;
  /** Structured per-token pricing override: { inputPer1k, outputPer1k, cacheReadPer1k? } */
  creditsPerToken?: { inputPer1k: string; outputPer1k: string; cacheReadPer1k?: string } | null;
  /** Fixed agent fee charged per request during finalization */
  agentFee?: number;
  agentName?: string;
  agentId?: string;
  /**
   * Selling over x402 on priced `@http` endpoints (2026-09-26, `./paywall.ts`).
   * Absent means: take credits through the platform this skill already
   * talks to, and no chain scheme unless the environment names one
   * (`X402_PAY_TO`). See {@link X402SellerConfig}.
   */
  x402?: X402SellerConfig;
}

/**
 * How this agent sells over x402 (and, alongside, MPP) on its priced
 * endpoints. The credits scheme is on by default: Robutler is its seller of
 * record and the agent is credited through the settle it already makes.
 * A chain scheme names WHO RECEIVES the payment, and that is the one thing
 * the credits rules care about: on a self-hosted agent it is the developer's
 * own address, in the developer's own software; the platform's dispatcher
 * never reads this config and keeps its own chain path off until counsel
 * clears it (lib/payments/x402-custom-http.ts on the portal).
 */
export interface X402SellerConfig {
  /** Take Robutler credits (default true). */
  credits?: boolean;
  /** The HMAC secret behind the credits nonces; one per fleet, random per process when unset. */
  nonceSecret?: string;
  /** Local verification of payment tokens against the platform's key set, before the verify API. */
  jwks?: import('../../crypto/jwks').JWKSManager;
  /** A chain seller; the environment (`X402_PAY_TO`, `X402_NETWORK`, `X402_ASSET`, `X402_ASSET_DECIMALS`) is the fallback. */
  chain?: Partial<Omit<ChainSellerConfig, 'facilitator'>> & { facilitator?: FacilitatorClient | FacilitatorConfig };
  /** What the 402 says about the service. */
  resource?: PaywallConfig['resource'];
  /** MPP alongside x402 on the same endpoints (`./mpp-seller.ts`). */
  mpp?: MppSellerConfig;
}

export interface UsageRecord {
  type: 'llm' | 'tool';
  model?: string;
  promptTokens?: number;
  completionTokens?: number;
  cachedReadTokens?: number;
  toolName?: string;
  pricing?: { credits: number; reason?: string; metadata?: Record<string, unknown> };
}

/**
 * A usage record as the settle route reads it (2026-09-25): snake_case, the
 * shape of `usageRecordSchema` in the portal's `lib/payments/usage.ts`, which
 * the Python skill already sends. This sent the camelCase fields as they are,
 * the schema dropped them as unknown keys, and every LLM record priced at 0.
 */
export function wireUsageRecord(record: UsageRecord): Record<string, unknown> {
  const wire: Record<string, unknown> = { type: record.type };
  if (record.model) wire.model = record.model;
  if (record.promptTokens !== undefined) wire.prompt_tokens = record.promptTokens;
  if (record.completionTokens !== undefined) wire.completion_tokens = record.completionTokens;
  if (record.cachedReadTokens !== undefined) wire.cached_read_tokens = record.cachedReadTokens;
  if (record.toolName) wire.tool_name = record.toolName;
  if (record.pricing) {
    wire.pricing = {
      credits: record.pricing.credits,
      ...(record.pricing.reason ? { reason: record.pricing.reason } : {}),
    };
  }
  return wire;
}

export class PaymentContext {
  paymentToken?: string;
  userId?: string;
  agentId?: string;
  lockId?: string;
  lockedAmountDollars: number = 0;
  paymentSuccessful: boolean = false;
  /** A settle charged less than it asked (2026-09-18, `readSettleResult`): the run is NOT charged in full. */
  settlePartial: boolean = false;
  /** What the partial settles left uncharged, in dollars. */
  unbilledDollars: number = 0;
  usageRecords: UsageRecord[] = [];
}

// ============================================================================
// PaymentSkill
// ============================================================================

export class PaymentSkill extends Skill {
  private enableBilling: boolean;
  private platformApiUrl: string;
  /** Whether `platformApiUrl` was named (config or a variable) rather than defaulted (B12). */
  private platformNamed: boolean;
  private apiKey: string | undefined;
  private minimumBalance: number;
  private perMessageLock: number;
  private defaultToolLock: number;
  private agentFee: number;
  private agentId: string | undefined;
  private readonly x402Config: X402SellerConfig;
  private _paywall: Paywall | undefined;

  constructor(config: PaymentSkillConfig = {}) {
    super({ name: 'PaymentSkill' });

    this.enableBilling = config.enableBilling ?? true;
    // NEVER LOCALHOST BY DEFAULT (B12, 2026-09-28): the last resort was
    // `http://localhost:3000`, which a priced endpoint's 402 published as the
    // platform to pay through. Nothing named: the default platform now, and
    // the CLI's `platform.url` once `initialize()` has read it
    // (`resolveSkillPlatformUrl`), as `PaymentX402Skill` does.
    const named = (
      config.platformUrl
      || config.platformApiUrl
      || getEnv('ROBUTLER_PLATFORM_URL')
      || getEnv('ROBUTLER_INTERNAL_API_URL')
      || getEnv('ROBUTLER_API_URL')
      || ''
    ).replace(/\/$/, '');
    this.platformNamed = Boolean(named);
    this.platformApiUrl = named || DEFAULT_PLATFORM_URL;
    // `ROBUTLER_API_KEY` is the older name for the agent's key here, and still
    // read. When neither is set, `initialize()` finds the key `publish` stored.
    this.apiKey = config.apiKey || getEnv('ROBUTLER_API_KEY');
    this.minimumBalance = config.minimumBalance ?? parseFloat(getEnv('MINIMUM_BALANCE') || '0.01');
    this.perMessageLock = config.perMessageLock ?? parseFloat(getEnv('PER_MESSAGE_LOCK') || '0.005');
    this.defaultToolLock = config.defaultToolLock ?? parseFloat(getEnv('DEFAULT_TOOL_LOCK') || '0.20');
    this.agentFee = config.agentFee ?? 0;
    this.agentId = config.agentId ?? config.agentName;
    this.x402Config = config.x402 ?? {};
  }

  /**
   * The paywall the servers ask for a priced `@http` endpoint (2026-09-26,
   * `./paywall.ts`): built once, on first use, so `initialize()` has had its
   * chance to find the agent's key. Undefined only when this skill has
   * nothing to sell with: credits switched off and no chain seller named.
   */
  get paywall(): Paywall | undefined {
    if (this._paywall) return this._paywall;
    const config: PaywallConfig = { resource: this.x402Config.resource };
    if (this.x402Config.credits !== false) {
      config.credits = {
        client: platformCreditsClient({ platformUrl: this.platformApiUrl, apiKey: this.apiKey, jwks: this.x402Config.jwks }),
        nonceSecret: this.x402Config.nonceSecret ?? getEnv('X402_NONCE_SECRET'),
        platformUrl: this.platformApiUrl,
      };
    }
    const chain = this.chainSellerConfig();
    if (chain) config.chain = chain;
    if (!config.credits && !config.chain) return undefined;
    if (this.x402Config.mpp) config.mpp = new MppSeller(this.x402Config.mpp);
    this._paywall = new Paywall(config);
    return this._paywall;
  }

  /**
   * The chain seller: the environment's (`chainSellerFromEnv`, `X402_PAY_TO`
   * switches it on), with every field the config names laid over it. A
   * configured address alone switches it on too.
   */
  private chainSellerConfig(): ChainSellerConfig | undefined {
    const c = this.x402Config.chain ?? {};
    const facilitator: FacilitatorClient =
      c.facilitator && 'verify' in c.facilitator ? c.facilitator : facilitatorFromConfig(c.facilitator as FacilitatorConfig | undefined);
    const env: Record<string, string | undefined> = typeof process !== 'undefined' && process.env ? process.env : {};
    // The configured address, laid over the environment's under the same
    // variable name, so one builder writes the entry for both.
    const { payTo: configured, ...rest } = c;
    const fromEnv = chainSellerFromEnv(configured ? { ...env, X402_PAY_TO: configured } : env, { facilitator, schemes: rest.schemes });
    if (!fromEnv) return undefined;
    return {
      ...fromEnv,
      ...(rest.network ? { network: rest.network } : {}),
      ...(rest.asset ? { asset: rest.asset } : {}),
      ...(rest.decimals !== undefined ? { decimals: rest.decimals } : {}),
      ...(rest.unitsPerCredit !== undefined ? { unitsPerCredit: rest.unitsPerCredit } : {}),
      ...(rest.maxTimeoutSeconds !== undefined ? { maxTimeoutSeconds: rest.maxTimeoutSeconds } : {}),
      ...(rest.extra ? { extra: rest.extra } : {}),
    };
  }

  /** The agent this skill serves, for finding its key (`BaseAgent.addSkill` calls this). */
  private agentName?: string;

  setAgent(agent: unknown): void {
    this.agentName = (agent as { name?: string })?.name;
  }

  /**
   * The agent's own key when none was configured (2026-09-24): the key
   * `webagents publish` stored for the agent this directory is linked to
   * (`server/agent-credential.ts`). It had to be exported by hand, and under a
   * name (`ROBUTLER_API_KEY`) that registration uses for the OWNER's key.
   */
  override async initialize(): Promise<void> {
    await super.initialize();
    if (!this.apiKey) this.apiKey = (await resolveAgentCredential(this.agentName))?.token;
    if (!this.platformNamed) this.platformApiUrl = await resolveSkillPlatformUrl();
  }

  // ==========================================================================
  // Lifecycle Hooks
  // ==========================================================================

  /**
   * Verify payment token → lock budget on connection open.
   */
  @hook({ lifecycle: 'on_connection', priority: 10 })
  async setupPaymentContext(_data: HookData, context: Context): Promise<HookResult | void> {
    const paymentCtx = new PaymentContext();
    paymentCtx.agentId = this.agentId;

    if (!this.enableBilling) {
      const token = this._extractPaymentToken(context);
      if (token) {
        paymentCtx.paymentToken = token;
        context.set('_payment_context', paymentCtx);
        context.payment = { valid: false, token };
      }
      return;
    }

    const token = this._extractPaymentToken(context);

    const callerUserId = context.auth?.user_id;
    const assertedAgentId = context.auth?.agent_id;

    paymentCtx.paymentToken = token;
    paymentCtx.userId = callerUserId;
    paymentCtx.agentId = assertedAgentId ?? this.agentId;

    if (token) {
      // 1. Verify token balance
      const verification = await this._verifyToken(token);
      if (!verification.valid) {
        throw new PaymentRequiredError(
          `Payment token invalid: ${verification.invalidReason ?? 'validation failed'}`,
          { maxAmountRequired: this.minimumBalance },
        );
      }

      const balance = verification.balance ?? 0;
      const minUsable = 0.001;
      if (balance < minUsable) {
        throw new PaymentRequiredError(
          `Insufficient token balance: $${balance.toFixed(4)} (need at least $${minUsable})`,
          { maxAmountRequired: minUsable },
        );
      }

      // 2. Lock budget
      const lockAmount = Math.min(balance, this.perMessageLock);
      try {
        const lock = await this._lockBudget(token, lockAmount);
        paymentCtx.lockId = lock.lockId;
        paymentCtx.lockedAmountDollars = lock.lockedAmountDollars;
      } catch (err: unknown) {
        const status = (err as { status?: number }).status;
        if (status === 400) {
          // Refused, not failed: the token cannot back this lock right now
          // (other locks hold its balance, or it has too many). This tried a
          // zero-amount lock "for tracking", which the route never accepts,
          // and the request then ran with nothing charged (S-259).
          throw new PaymentRequiredError(
            `Insufficient token balance: it cannot cover this request's $${lockAmount.toFixed(4)} lock right now`,
            { maxAmountRequired: lockAmount },
          );
        }
        throw err;
      }

      // 3. Publish to context
      context.payment = {
        valid: true,
        token,
        balance,
        currency: 'USD',
        lockId: paymentCtx.lockId,
        lockedAmount: paymentCtx.lockedAmountDollars,
      };

    } else if (this.minimumBalance > 0) {
      // Billing enabled but no token — send 402 with x402 accepts
      throw new PaymentRequiredError(
        'Payment required. Provide a valid payment token.',
        {
          maxAmountRequired: this.minimumBalance,
          accepts: [{
            scheme: 'token',
            network: 'robutler',
            amount: String(this.minimumBalance),
            asset: 'robutler:credits',
            maxTimeoutSeconds: 300,
            extra: { tokenType: 'jwt' },
          }],
        },
      );
    }

    context.set('_payment_context', paymentCtx);
  }

  /**
   * Lock funds for LLM call based on adapter capabilities.
   * PaymentSkill reads _llm_capabilities to size the lock.
   */
  @hook({ lifecycle: 'before_llm_call', priority: 10 })
  async lockForLLMCall(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.enableBilling) return;

    const paymentCtx = context.get<PaymentContext>('_payment_context');
    if (!paymentCtx?.paymentToken) return;

    const capabilities = context.get<{
      model: string;
      provider: string;
      maxOutputTokens: number;
      pricing: { inputPer1k: number; outputPer1k: number };
    }>('_llm_capabilities');

    if (!capabilities) return;

    const estimatedCost = Math.max(
      this.perMessageLock,
      (capabilities.maxOutputTokens / 1000) * capabilities.pricing.outputPer1k,
    );

    if (estimatedCost <= 0) return;

    if (!paymentCtx.lockId) {
      try {
        const lock = await this._lockBudget(paymentCtx.paymentToken, estimatedCost);
        paymentCtx.lockId = lock.lockId;
        paymentCtx.lockedAmountDollars = lock.lockedAmountDollars;
      } catch {
        // Non-fatal: finalization will handle
      }
    } else {
      try {
        await this._extendLock(paymentCtx.lockId, estimatedCost);
        paymentCtx.lockedAmountDollars += estimatedCost;
      } catch {
        // Non-fatal
      }
    }
  }

  /**
   * Settle LLM usage after the call completes.
   * Reads _llm_usage from context. Skips LLM billing if is_byok.
   */
  @hook({ lifecycle: 'after_llm_call', priority: 10 })
  async settleForLLMCall(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.enableBilling) return;

    const paymentCtx = context.get<PaymentContext>('_payment_context');
    if (!paymentCtx?.lockId) return;

    const usage = context.get<{
      model: string;
      provider: string;
      input_tokens: number;
      output_tokens: number;
      is_byok: boolean;
    }>('_llm_usage');

    if (!usage) return;

    // BYOK: user pays provider directly, no LLM billing
    if (usage.is_byok) return;

    paymentCtx.usageRecords.push({
      type: 'llm',
      model: usage.model,
      promptTokens: usage.input_tokens,
      completionTokens: usage.output_tokens,
    });
  }

  /**
   * Lock funds for the incoming message/request.
   * Ensures the per-message lock is established before any tool execution.
   */
  @hook({ lifecycle: 'on_message', priority: 15 })
  async lockFundsForMessage(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.enableBilling) return;

    const paymentCtx = context.get<PaymentContext>('_payment_context');
    if (!paymentCtx?.paymentToken) return;

    // If we already have a lock from on_connection, nothing to do
    if (paymentCtx.lockId) return;

    // Create a per-message lock from the token
    if (!this.perMessageLock || this.perMessageLock <= 0) return;

    try {
      const lock = await this._lockBudget(paymentCtx.paymentToken, this.perMessageLock);
      paymentCtx.lockId = lock.lockId;
      paymentCtx.lockedAmountDollars = lock.lockedAmountDollars;
    } catch {
      // Non-fatal: tool-level locks will handle authorization
    }
  }

  /**
   * Extend the payment lock before executing a priced tool.
   */
  @hook({ lifecycle: 'before_toolcall', priority: 20 })
  async preauthToolLock(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.enableBilling) return;

    // Read from context (set by agent loop) or from HookData
    const toolName = context.get<string>('tool_name') ?? _data.tool_name;
    if (!toolName) return;

    const pricingConfig = this._findPricingForTool(toolName, context);
    const toolParams = context.get<Record<string, unknown>>('tool_params') ?? _data.tool_params as Record<string, unknown> | undefined;
    let lockAmount: number;
    if (typeof pricingConfig?.lock === 'function' && toolParams) {
      lockAmount = pricingConfig.lock(toolParams);
    } else {
      lockAmount = (typeof pricingConfig?.lock === 'number' ? pricingConfig.lock : undefined)
        ?? pricingConfig?.creditsPerCall
        ?? this.defaultToolLock;
    }

    if (lockAmount <= 0) return;

    const paymentCtx = context.get<PaymentContext>('_payment_context');

    if (!paymentCtx?.lockId) {
      // Try a fresh lock if we have a token but no lock yet
      if (paymentCtx?.paymentToken) {
        try {
          const lock = await this._lockBudget(paymentCtx.paymentToken, lockAmount);
          paymentCtx.lockId = lock.lockId;
          paymentCtx.lockedAmountDollars = lock.lockedAmountDollars;
          return;
        } catch {
          // Fall through to block
        }
      }

      context.set('tool_result',
        `Tool '${toolName}' blocked: no payment lock available. ` +
        `Required $${lockAmount.toFixed(4)}. Provide a valid payment token with sufficient balance.`);
      context.set('tool_skipped', true);
      return;
    }

    const result = await this._extendLock(paymentCtx.lockId, lockAmount);
    if (!result.success) {
      context.set('tool_result',
        `Tool '${toolName}' blocked: spending limit exceeded. ` +
        `Required $${lockAmount.toFixed(4)}, insufficient token balance.`);
      context.set('tool_skipped', true);
      return;
    }

    paymentCtx.lockedAmountDollars += lockAmount;
  }

  /**
   * Record a tool's fee (its @pricing metadata) as usage, charged at finalize.
   */
  @hook({ lifecycle: 'after_toolcall', priority: 20 })
  async handleToolCompletion(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.enableBilling) return;

    const paymentCtx = context.get<PaymentContext>('_payment_context');
    if (!paymentCtx?.lockId) return;

    const toolName = context.get<string>('tool_name') ?? _data.tool_name;
    const toolResult = context.get<unknown>('tool_result') ?? _data.tool_result;
    const isError = typeof toolResult === 'string' && toolResult.toLowerCase().includes('error');

    if (isError) return; // Don't charge for failed tools

    const pricingConfig = toolName ? this._findPricingForTool(toolName, context) : undefined;
    let toolFee: number | undefined;
    if (pricingConfig?.settle && toolResult != null) {
      const toolParams = context.get<Record<string, unknown>>('tool_params') ?? _data.tool_params as Record<string, unknown> | undefined;
      toolFee = pricingConfig.settle(toolResult, toolParams ?? {});
    } else {
      toolFee = pricingConfig?.creditsPerCall;
    }

    if (toolFee && toolFee > 0) {
      // Charged with the run's usage in the one settle at finalize, as the
      // Python skill does (2026-09-25). This settled each fee on its own with
      // `chargeType: 'tool_fee'`, which the settle route does not accept, so
      // no tool fee was ever charged; and the record it then kept would have
      // charged the fee a second time at finalize once that settle went
      // through. A usage settle credits the agent as `agent_fee` does.
      paymentCtx.usageRecords.push({
        type: 'tool',
        toolName,
        pricing: {
          credits: toolFee,
          reason: pricingConfig?.reason ?? `Tool '${toolName}' execution`,
        },
      });
    }
  }

  /**
   * Settle remaining agent_fee charges and release unused locks.
   */
  @hook({ lifecycle: 'finalize_connection', priority: 95 })
  async finalizePayment(_data: HookData, context: Context): Promise<HookResult | void> {
    if (!this.enableBilling) return;

    const paymentCtx = context.get<PaymentContext>('_payment_context');
    if (!paymentCtx) return;

    try {
      const lockId = paymentCtx.lockId;

      // Settle agent_fee (fixed per-request charge) if configured
      if (lockId && this.agentFee > 0) {
        try {
          this._notePartial(
            paymentCtx,
            await this._settlePayment(lockId, {
              amount: this.agentFee,
              chargeType: 'agent_fee',
              description: 'Per-request agent fee',
            }),
          );
        } catch {
          // Best-effort agent fee settlement
        }
      }

      const usageRecords = context.get<UsageRecord[]>('usage') ?? paymentCtx.usageRecords;
      const llmRecords = usageRecords.filter(r => r.type === 'llm');
      const toolRecords = usageRecords.filter(r => r.type === 'tool');
      const hasUsage = llmRecords.length > 0 || toolRecords.length > 0;

      if (!hasUsage) {
        if (lockId) {
          try {
            await this._settlePayment(lockId, { amount: 0, release: true });
          } catch {
            // Best-effort release
          }
        }
        return;
      }

      if (!lockId) return;

      // Platform billing: forward all usage, server computes cost from MODEL_PRICING
      const settleStartedAt = Date.now();
      let settled: PaymentSettleResult;
      try {
        settled = await this._settlePayment(lockId, {
          usage: [...llmRecords, ...toolRecords],
          description: 'LLM + tool usage',
        });
      } catch (error) {
        recordSettleSpan(context, { lockId, credits: 0, startedAt: settleStartedAt, error: errorType(error) });
        throw error;
      }
      this._notePartial(paymentCtx, settled);
      recordSettleSpan(context, { lockId, credits: settledCredits(settled), startedAt: settleStartedAt, ...(settled.success ? {} : { error: settled.error || 'settle_failed' }) });

      // Release remaining locked balance
      try {
        await this._settlePayment(lockId, { amount: 0, release: true });
      } catch {
        // Best-effort release
      }

      paymentCtx.paymentSuccessful = true;
      // A partial settle committed, so the run is settled, but it is never
      // reported as charged in full (2026-09-18).
      context.payment = {
        ...context.payment,
        settled: true,
        ...(paymentCtx.settlePartial ? { partial: true, unbilledDollars: paymentCtx.unbilledDollars } : {}),
      };
    } catch {
      // Payment finalization is best-effort; don't crash the connection
    }
  }

  // ==========================================================================
  // Internal Methods
  // ==========================================================================
  //
  // The private-scheme x402 code that lived here (`createX402Requirements`,
  // `verifyX402Payment`: `scheme: 'token'` on `network: 'robutler'`, a
  // `payTo:` naming the agent, settled by token on every call) is retired
  // (2026-09-26). A priced `@http` endpoint is served through `this.paywall`
  // by the servers themselves, in the standard x402 shape, with the credits
  // scheme settled once after the handler answered (`./paywall.ts`).

  private _extractPaymentToken(context: Context): string | undefined {
    // 1. Transport-agnostic: set by transport layer
    const explicit = context.get<string>('payment_token');
    if (explicit?.trim()) return explicit.trim();

    // 2. Metadata headers (backward compat for HTTP)
    const meta = context.metadata ?? {};
    const fromHeader = (
      (meta['x-payment-token'] as string | undefined)
      ?? (meta['X-Payment-Token'] as string | undefined)
      ?? (meta['x-payment'] as string | undefined)
      ?? (meta['X-PAYMENT'] as string | undefined)
    );
    if (fromHeader?.trim()) return fromHeader.trim();

    return undefined;
  }

  private async _verifyToken(token: string): Promise<PaymentVerifyResult> {
    const res = await fetch(`${this.platformApiUrl}/api/payments/verify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify({ token }),
    });
    return (await res.json()) as PaymentVerifyResult;
  }

  private async _lockBudget(
    token: string,
    amount: number,
  ): Promise<{ lockId: string; lockedAmountDollars: number }> {
    const res = await fetch(`${this.platformApiUrl}/api/payments/lock`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify({ token, amount }),
    });
    if (!res.ok) {
      const err = new Error(`Lock failed: HTTP ${res.status}`) as Error & { status: number };
      err.status = res.status;
      throw err;
    }
    const data = await res.json() as Record<string, unknown>;
    return {
      lockId: data.lockId as string,
      lockedAmountDollars: (data.lockedAmountDollars as number) ?? amount,
    };
  }

  private async _extendLock(
    lockId: string,
    amount: number,
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const res = await fetch(`${this.platformApiUrl}/api/payments/lock/${encodeURIComponent(lockId)}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        // The route reads `additionalAmount` (`app/api/payments/lock/[id]/route.ts`);
        // this sent `amount`, so every extension was refused with a 400
        // (2026-09-25). The Python client's `extend_lock` sends the same body.
        body: JSON.stringify({ additionalAmount: amount }),
      });
      if (!res.ok) {
        return { success: false, error: `HTTP ${res.status}` };
      }
      return { success: true };
    } catch (err: unknown) {
      return { success: false, error: String(err) };
    }
  }

  /**
   * The Idempotency-Key for a settle this skill issues (`./idempotency.ts`):
   * stable for the settles the lifecycle names, so a repeat is answered the
   * first result and never charged twice; fresh for one nothing names.
   *
   *   release-only          settle:<lock>:release
   *   usage array           settle:<lock>:usage
   *   amount + chargeType   settle:<lock>:<chargeType>            (the agent fee)
   *   … made for a tool call settle:<lock>:<chargeType>:<callId>   (a per-call settle)
   *   amount alone          settle:client:<uuid>                  (a new settle per call)
   *
   * `callId` (wave-0 review finding 5): a settle issued for one tool call
   * carries that call's id in its purpose, so two calls on one lock never
   * share a key and the second is never answered the first's numbers.
   */
  private _settleIdempotencyKey(
    lockId: string,
    options: { amount?: number; usage?: UsageRecord[]; chargeType?: string; release?: boolean; callId?: string },
  ): string {
    if (options.release && (options.amount ?? 0) === 0) return settleIdempotencyKey(lockId, 'release');
    if (options.usage !== undefined) return settleIdempotencyKey(lockId, 'usage', options.callId);
    if (options.chargeType) return settleIdempotencyKey(lockId, options.chargeType, options.callId);
    return freshSettleIdempotencyKey('client');
  }

  private async _settlePayment(
    lockId: string,
    options: {
      amount?: number;
      usage?: UsageRecord[];
      description?: string;
      chargeType?: string;
      release?: boolean;
      /** Overrides the derived key; a caller that retries its own settle passes the key it used. */
      idempotencyKey?: string;
      /** The tool call this settle charges for, when it is one call's: part of the derived key. */
      callId?: string;
    } = {},
  ): Promise<PaymentSettleResult> {
    const idempotencyKey = options.idempotencyKey ?? this._settleIdempotencyKey(lockId, options);
    const body: Record<string, unknown> = {
      lockId,
      description: options.description,
      chargeType: options.chargeType,
      release: options.release,
      // Header AND body: the platform reads either, and a proxy that drops
      // unknown headers must not turn a safe retry into a second charge.
      [IDEMPOTENCY_KEY_BODY_FIELD]: idempotencyKey,
    };
    if (options.amount !== undefined) body.amount = options.amount;
    if (options.usage !== undefined) body.usage = options.usage.map(wireUsageRecord);

    const res = await fetch(`${this.platformApiUrl}/api/payments/settle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...idempotencyHeaders(idempotencyKey),
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify(body),
    });
    return readSettleResult(await res.json(), `settle ${options.chargeType ?? (options.release ? 'release' : 'usage')}`);
  }

  /** Record a partial settle on the run's payment context (2026-09-18, `readSettleResult`). */
  private _notePartial(paymentCtx: PaymentContext, result: PaymentSettleResult): void {
    if (!result.partial) return;
    paymentCtx.settlePartial = true;
    paymentCtx.unbilledDollars += result.unbilledDollars ?? 0;
  }

  private _findPricingForTool(toolName: string, context: Context): PricingConfig | undefined {
    const skills = context.get<Array<{ constructor: Function }>>('_skills');
    if (skills) {
      const decoratorPricing = getPricingForTool(skills, toolName);
      if (decoratorPricing) return decoratorPricing;
      for (const skill of skills) {
        for (const tool of (skill as any).tools ?? []) {
          if (tool.name === toolName && tool.pricing) return tool.pricing;
        }
      }
    }
    return undefined;
  }
}
