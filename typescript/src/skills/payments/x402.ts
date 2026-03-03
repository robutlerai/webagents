/**
 * Payment x402 Skill - JWT payment tokens and /api/payments/* support
 *
 * Verifies payment token from context (transport-agnostic) or X-PAYMENT header.
 * When payment is required but no token is provided, throws PaymentRequiredError (402)
 * so transports can negotiate (e.g. payment.required / 402 response).
 */

import { Skill } from '../../core/skill.js';
import { hook } from '../../core/decorators.js';
import type { HookData, HookResult } from '../../core/types.js';
import type { Context } from '../../core/types.js';
import { JWKSManager } from '../../crypto/jwks.js';
import type { PaymentVerifyResult, PaymentSettleResult } from './types.js';

/** Error thrown when payment is required but no valid token was provided. Transports catch and return 402 or payment.required. */
export class PaymentRequiredError extends Error {
  readonly status_code = 402;
  readonly accepts?: unknown[];
  constructor(
    message: string = 'Payment required',
    public readonly context: { accepts?: unknown[]; maxAmountRequired?: number } = {}
  ) {
    super(message);
    this.name = 'PaymentRequiredError';
    this.accepts = context.accepts;
    Object.setPrototypeOf(this, PaymentRequiredError.prototype);
  }
}

export interface PaymentX402Config {
  /** Base URL for payments API (e.g. https://robutler.ai) */
  facilitatorUrl?: string;
  /** JWKS manager for local JWT verification */
  jwksManager?: JWKSManager;
}

/**
 * Payment x402 skill: verifies and settles payment from X-PAYMENT header.
 * Uses local JWKS verification for JWTs when available; otherwise calls verify API.
 */
export class PaymentX402Skill extends Skill {
  private facilitatorUrl: string;
  private jwks: JWKSManager;

  constructor(config: PaymentX402Config = {}) {
    super();
    this.facilitatorUrl = (config.facilitatorUrl ?? 'https://robutler.ai').replace(/\/$/, '');
    this.jwks = config.jwksManager ?? new JWKSManager();
  }

  /**
   * Verify payment token - try local JWKS first, then API.
   * When expectedAudience is provided, token aud must match (recipient check).
   */
  async verifyPaymentToken(
    token: string,
    options?: { expectedAudience?: string | string[] }
  ): Promise<PaymentVerifyResult> {
    const local = await this.jwks.verifyPaymentToken(token, {
      expectedAudience: options?.expectedAudience,
    });
    if (local) return { valid: true, balance: local.balance };

    const res = await fetch(`${this.facilitatorUrl}/api/payments/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, expectedAudience: options?.expectedAudience }),
    });
    const data = (await res.json()) as PaymentVerifyResult;
    return data;
  }

  /**
   * Settle (charge) against a payment token
   */
  async settlePayment(
    token: string,
    amount: number,
    options: { recipientId?: string; description?: string; resource?: string } = {}
  ): Promise<PaymentSettleResult> {
    const res = await fetch(`${this.facilitatorUrl}/api/payments/settle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token,
        amount,
        recipientId: options.recipientId,
        description: options.description,
        resource: options.resource,
      }),
    });
    return (await res.json()) as PaymentSettleResult;
  }

  /**
   * Lock funds and create payment token (payer side)
   */
  async lockPayment(amount: number, options: { audience?: string[]; expiresIn?: number } = {}): Promise<{ token: string; expiresAt: string; lockedAmount: number } | null> {
    const res = await fetch(`${this.facilitatorUrl}/api/payments/lock`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        amount,
        audience: options.audience,
        expiresIn: options.expiresIn,
      }),
    });
    if (!res.ok) return null;
    return (await res.json()) as { token: string; expiresAt: string; lockedAmount: number };
  }

  @hook({ lifecycle: 'before_run', priority: 8 })
  async checkPayment(data: HookData, context: Context): Promise<HookResult | void> {
    // 1. Transport-agnostic: set by transport via context.set('payment_token', ...)
    let token = context.get<string>('payment_token');
    // 2. Fallback: HTTP header from metadata (backward compat)
    if (!token?.trim()) {
      token = (context.metadata?.['x-payment'] ?? context.metadata?.['X-PAYMENT']) as string | undefined;
    }
    if (!token?.trim()) {
      // When transport marks payment as required, throw so transport can return 402 / payment.required
      if (context.get<boolean>('payment_required')) {
        throw new PaymentRequiredError('This agent requires payment. Please provide a valid payment token.', {
          accepts: context.get<unknown[]>('payment_accepts') ?? [],
          maxAmountRequired: context.get<number>('payment_max_amount_required'),
        });
      }
      return;
    }

    const tokenTrimmed = token.trim();
    const result = await this.verifyPaymentToken(tokenTrimmed);
    if (result.valid && 'setPayment' in context && typeof (context as { setPayment: (p: unknown) => void }).setPayment === 'function') {
      (context as { setPayment: (p: unknown) => void }).setPayment({
        valid: true,
        token: tokenTrimmed,
        balance: result.balance,
      });
    }
  }
}
