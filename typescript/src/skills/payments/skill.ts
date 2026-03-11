/**
 * PaymentSkill - Full verify→lock→settle payment lifecycle
 *
 * Ports the Python PaymentSkill to TypeScript for the WebAgents framework.
 * All cost computation is done server-side: agents forward raw usage records
 * (model, prompt_tokens, completion_tokens, cached tokens) to /settle,
 * and the platform computes the dollar cost from MODEL_PRICING.
 */

import { Skill } from '../../core/skill.js';
import { hook, getPricingForTool } from '../../core/decorators.js';
import type { HookData, HookResult, Context, PricingConfig } from '../../core/types.js';
import type { PaymentVerifyResult, PaymentSettleResult } from './types.js';
import { PaymentRequiredError } from './x402.js';

// ============================================================================
// Helpers
// ============================================================================

function getEnv(name: string): string | undefined {
  if (typeof process !== 'undefined' && process.env) {
    return process.env[name];
  }
  return undefined;
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
  /** Default pricing per token (input+output) in credits */
  creditsPerToken?: number;
  /** Fixed agent fee charged per request during finalization */
  agentFee?: number;
  agentName?: string;
  agentId?: string;
  /** x402 accepted payment schemes (for Agent B) */
  acceptedSchemes?: Array<{ scheme: string; network: string }>;
  /** x402 facilitator URL (defaults to platformUrl) */
  facilitatorUrl?: string;
  /** Max x402 payment amount safety limit */
  maxPayment?: number;
}

export interface UsageRecord {
  type: 'llm' | 'tool';
  model?: string;
  promptTokens?: number;
  completionTokens?: number;
  cachedReadTokens?: number;
  pricing?: { credits: number; reason?: string; metadata?: Record<string, unknown> };
}

export class PaymentContext {
  paymentToken?: string;
  userId?: string;
  agentId?: string;
  lockId?: string;
  lockedAmountDollars: number = 0;
  paymentSuccessful: boolean = false;
  usageRecords: UsageRecord[] = [];
}

// ============================================================================
// PaymentSkill
// ============================================================================

export class PaymentSkill extends Skill {
  private enableBilling: boolean;
  private platformApiUrl: string;
  private apiKey: string | undefined;
  private minimumBalance: number;
  private perMessageLock: number;
  private defaultToolLock: number;
  private agentFee: number;
  private agentId: string | undefined;
  /** x402 accepted schemes (Agent B) */
  private acceptedSchemes: Array<{ scheme: string; network: string }>;
  /** x402 facilitator URL */
  private facilitatorUrl: string;
  /** Safety limit for x402 payments */
  private maxPayment: number;

  constructor(config: PaymentSkillConfig = {}) {
    super({ name: 'PaymentSkill' });

    this.enableBilling = config.enableBilling ?? true;
    this.platformApiUrl = (
      config.platformUrl
      || config.platformApiUrl
      || getEnv('ROBUTLER_PLATFORM_URL')
      || getEnv('ROBUTLER_INTERNAL_API_URL')
      || getEnv('ROBUTLER_API_URL')
      || 'http://localhost:3000'
    ).replace(/\/$/, '');
    this.apiKey = config.apiKey || getEnv('ROBUTLER_API_KEY');
    this.minimumBalance = config.minimumBalance ?? parseFloat(getEnv('MINIMUM_BALANCE') || '0.01');
    this.perMessageLock = config.perMessageLock ?? parseFloat(getEnv('PER_MESSAGE_LOCK') || '0.005');
    this.defaultToolLock = config.defaultToolLock ?? parseFloat(getEnv('DEFAULT_TOOL_LOCK') || '0.20');
    this.agentFee = config.agentFee ?? 0;
    this.agentId = config.agentId ?? config.agentName;
    this.acceptedSchemes = config.acceptedSchemes ?? [{ scheme: 'token', network: 'robutler' }];
    this.facilitatorUrl = (config.facilitatorUrl || this.platformApiUrl).replace(/\/$/, '');
    this.maxPayment = config.maxPayment ?? parseFloat(getEnv('X402_MAX_PAYMENT') || '10.0');
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
          try {
            const fallback = await this._lockBudget(token, 0);
            paymentCtx.lockId = fallback.lockId;
            paymentCtx.lockedAmountDollars = 0;
          } catch {
            // Proceed without lock
          }
        } else {
          throw err;
        }
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
    const lockAmount = pricingConfig?.lock
      ?? pricingConfig?.creditsPerCall
      ?? this.defaultToolLock;

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
   * Settle tool_fee charges after tool execution using @pricing metadata.
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

    // Settle tool_fee from @pricing metadata
    const pricingConfig = toolName ? this._findPricingForTool(toolName, context) : undefined;
    const toolFee = pricingConfig?.creditsPerCall;

    if (toolFee && toolFee > 0) {
      try {
        await this._settlePayment(paymentCtx.lockId, {
          amount: toolFee,
          chargeType: 'tool_fee',
          description: `Tool '${toolName}' execution`,
        });

        paymentCtx.usageRecords.push({
          type: 'tool',
          pricing: {
            credits: toolFee,
            reason: pricingConfig?.reason ?? `Tool '${toolName}' execution`,
          },
        });
      } catch {
        // Best-effort tool settlement; finalization will catch remainder
      }
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
          await this._settlePayment(lockId, {
            amount: this.agentFee,
            chargeType: 'agent_fee',
            description: 'Per-request agent fee',
          });
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
      await this._settlePayment(lockId, {
        usage: [...llmRecords, ...toolRecords],
        description: 'LLM + tool usage',
      });

      // Release remaining locked balance
      try {
        await this._settlePayment(lockId, { amount: 0, release: true });
      } catch {
        // Best-effort release
      }

      paymentCtx.paymentSuccessful = true;
      context.payment = { ...context.payment, settled: true };
    } catch {
      // Payment finalization is best-effort; don't crash the connection
    }
  }

  // ==========================================================================
  // x402 Protocol Support
  // ==========================================================================

  /**
   * Create x402 payment requirements for 402 responses (Agent B role).
   * Used by HTTP endpoints that require payment.
   */
  createX402Requirements(amount: number, resource: string = '/'): {
    accepts: Array<Record<string, unknown>>;
    version: string;
  } {
    return {
      version: '1.0',
      accepts: this.acceptedSchemes.map(s => ({
        scheme: s.scheme,
        network: s.network,
        maxAmountRequired: String(amount),
        resource,
        payTo: this.agentId ?? 'unknown',
        extra: s.scheme === 'token' ? { tokenType: 'jwt' } : undefined,
      })),
    };
  }

  /**
   * Verify an x402 payment header (Agent B role).
   * Tries local JWKS first, then facilitator verify endpoint.
   */
  async verifyX402Payment(
    paymentHeader: string,
    amount: number,
    resource: string = '/',
  ): Promise<{ valid: boolean; error?: string }> {
    if (amount > this.maxPayment) {
      return { valid: false, error: `Amount ${amount} exceeds max payment limit ${this.maxPayment}` };
    }
    try {
      const requirements = {
        scheme: 'token',
        network: 'robutler',
        maxAmountRequired: String(amount),
        payTo: this.agentId ?? 'unknown',
        resource,
      };

      const verifyRes = await fetch(`${this.facilitatorUrl}/api/payments/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify({ token: paymentHeader, requirements }),
      });
      const verification = (await verifyRes.json()) as PaymentVerifyResult;

      if (!verification.valid) {
        return { valid: false, error: verification.invalidReason ?? 'Verification failed' };
      }
      if ((verification.balance ?? 0) < amount) {
        return { valid: false, error: 'Insufficient token balance' };
      }

      // Settle
      const settleRes = await fetch(`${this.facilitatorUrl}/api/payments/settle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify({ token: paymentHeader, amount, requirements }),
      });
      const settlement = (await settleRes.json()) as PaymentSettleResult;

      if (!settlement.success) {
        return { valid: false, error: settlement.error ?? 'Settlement failed' };
      }

      return { valid: true };
    } catch (err) {
      return { valid: false, error: (err as Error).message };
    }
  }

  // ==========================================================================
  // Internal Methods
  // ==========================================================================

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
        body: JSON.stringify({ amount }),
      });
      if (!res.ok) {
        return { success: false, error: `HTTP ${res.status}` };
      }
      return { success: true };
    } catch (err: unknown) {
      return { success: false, error: String(err) };
    }
  }

  private async _settlePayment(
    lockId: string,
    options: {
      amount?: number;
      usage?: UsageRecord[];
      description?: string;
      chargeType?: string;
      release?: boolean;
    } = {},
  ): Promise<PaymentSettleResult> {
    const body: Record<string, unknown> = {
      lockId,
      description: options.description,
      chargeType: options.chargeType,
      release: options.release,
    };
    if (options.amount !== undefined) body.amount = options.amount;
    if (options.usage !== undefined) body.usage = options.usage;

    const res = await fetch(`${this.platformApiUrl}/api/payments/settle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify(body),
    });
    return (await res.json()) as PaymentSettleResult;
  }

  private _findPricingForTool(toolName: string, context: Context): PricingConfig | undefined {
    const skills = context.get<Array<{ constructor: Function }>>('_skills');
    if (skills) {
      return getPricingForTool(skills, toolName);
    }
    return undefined;
  }
}
