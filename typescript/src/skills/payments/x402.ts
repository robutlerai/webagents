/**
 * Payment x402 Skill - JWT payment tokens and /api/payments/* support
 *
 * Verifies payment token from context (transport-agnostic) or X-PAYMENT header.
 * When payment is required but no token is provided, throws PaymentRequiredError (402)
 * so transports can negotiate (e.g. payment.required / 402 response).
 *
 * A SETTLE CARRIES THE AGENT'S KEY (2026-09-25): the settle route charges on
 * behalf of an authenticated agent, and this sent no credential, so every
 * `settlePayment` was refused with a 401. The key is `apiKey`, else the agent's
 * own key (`resolveAgentCredential`, as `PaymentSkill` finds it); the platform
 * is `facilitatorUrl`, else the one platform lookup (`../platform-url.ts`).
 * `lockPayment` is gone: it posted `{ amount, audience }` to the route that
 * locks against an EXISTING token, so it could never mint one; a payer's token
 * comes from a session or, for an agent, from `/api/payments/delegate`, which
 * the NLI skill uses.
 *
 * NOT A SELLER (2026-09-27, the final e2e re-run). This skill carries no
 * `paywall`, so a priced `@http` endpoint on an agent that has only it was
 * refused with the generic 503, which named nothing the developer could act
 * on. The seller is `PaymentSkill` (`./skill.ts`, its `paywall` getter), and
 * this skill says so: `paywallRefusal` is the sentence the servers answer
 * instead (`./paywall.ts` `serveThroughPaywall`). Chosen over growing a second
 * paywall here because the seller's configuration (`x402.credits`, `chain`,
 * `mpp`, `resource`, the platform URL and key the chat locks use) would have
 * to exist twice and stay identical, and the two doors would be the credential
 * floor's lesson again. Python has no twin of this sentence: its
 * `PaymentSkillX402` extends `PaymentSkill` and is the seller. Pinned by
 * `python/tests/fixtures/payments/paywall_x402_mpp.json` (`not_configured`).
 */

import { Skill } from '../../core/skill';
import { hook } from '../../core/decorators';
import type { HookData, HookResult } from '../../core/types';
import type { Context } from '../../core/types';
import { JWKSManager } from '../../crypto/jwks';
import type { PaymentVerifyResult, PaymentSettleResult } from './types';
import { readSettleResult } from './settle-result';
import { resolveAgentCredential } from '../../server/agent-credential';
import { DEFAULT_PLATFORM_URL, configuredPlatformUrl, resolveSkillPlatformUrl } from '../platform-url';
import { IDEMPOTENCY_KEY_BODY_FIELD, freshSettleIdempotencyKey, idempotencyHeaders } from './idempotency';

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

/**
 * What a priced `@http` endpoint answers (503, `payment_not_configured`) when
 * this skill is the agent's only payment skill (file comment). The words are
 * the fixture's `not_configured.typescript_x402_skill.message`.
 */
export const X402_SKILL_PAYWALL_REFUSAL =
  'This endpoint is priced, and PaymentX402Skill verifies payment tokens on chat turns but does not sell over x402; ' +
  'give the agent PaymentSkill (webagents/skills/payments) to take a payment for it.';

export interface PaymentX402Config {
  /** Base URL for payments API (e.g. https://robutler.ai). Default: the platform lookup. */
  facilitatorUrl?: string;
  /** JWKS manager for local JWT verification */
  jwksManager?: JWKSManager;
  /** The agent's platform key, sent with a settle. Default: the agent's own key. */
  apiKey?: string;
  /** The agent whose own key is looked up when `apiKey` is not given. */
  agentName?: string;
}

/**
 * Payment x402 skill: verifies and settles payment from X-PAYMENT header.
 * Uses local JWKS verification for JWTs when available; otherwise calls verify API.
 */
export class PaymentX402Skill extends Skill {
  /** Why a priced endpoint is not served through this skill (file comment); read by `serveThroughPaywall`. */
  readonly paywallRefusal = X402_SKILL_PAYWALL_REFUSAL;
  private facilitatorUrl: string;
  /** Whether `facilitatorUrl` was named (configured or by a variable) rather than defaulted. */
  private facilitatorNamed: boolean;
  private jwks: JWKSManager;
  private apiKey: string | undefined;
  private agentName: string | undefined;

  constructor(config: PaymentX402Config = {}) {
    super();
    const named = configuredPlatformUrl(config.facilitatorUrl);
    this.facilitatorUrl = named ?? DEFAULT_PLATFORM_URL;
    this.facilitatorNamed = named !== undefined;
    this.jwks = config.jwksManager ?? new JWKSManager();
    this.apiKey = config.apiKey;
    this.agentName = config.agentName;
  }

  /** The CLI's platform when none was named, and the agent's own key when none was given (file comment). */
  override async initialize(): Promise<void> {
    await super.initialize();
    if (!this.facilitatorNamed) this.facilitatorUrl = await resolveSkillPlatformUrl();
    if (!this.apiKey) this.apiKey = (await resolveAgentCredential(this.agentName))?.token;
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
    options: { recipientId?: string; description?: string; resource?: string; idempotencyKey?: string } = {}
  ): Promise<PaymentSettleResult> {
    // No lock is in hand (the platform resolves the token to one), so the
    // key is minted for this call unless the caller retries with its own:
    // a repeat of THIS call is a replay, the next call a new settle.
    const idempotencyKey = options.idempotencyKey ?? freshSettleIdempotencyKey('x402');
    const res = await fetch(`${this.facilitatorUrl}/api/payments/settle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...idempotencyHeaders(idempotencyKey),
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify({
        token,
        amount,
        recipientId: options.recipientId,
        description: options.description,
        resource: options.resource,
        [IDEMPOTENCY_KEY_BODY_FIELD]: idempotencyKey,
      }),
    });
    // 2026-09-18: `success` alone no longer means charged in full; the
    // reader keeps `partial`, `charged` and `unbilled` and warns on a partial.
    return readSettleResult(await res.json(), 'x402 settlePayment');
  }

  @hook({ lifecycle: 'before_run', priority: 8 })
  async checkPayment(_data: HookData, context: Context): Promise<HookResult | void> {
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
