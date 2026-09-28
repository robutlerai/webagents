/**
 * The Robutler credits scheme under x402 (webagents gap-closure plan 2.6,
 * 2026-09-26): `robutler-credits` on `robutler:1`, spec pack section 1.9.
 *
 * WHAT IT SELLS. Platform usage, priced in credits, paid from a Robutler
 * payment token the caller already holds (a session's token, or a child a
 * delegating agent minted for this hop). Robutler is the seller of record
 * (`payTo: robutler`); the agent that serves the endpoint is credited through
 * the existing settle, which is where Creator Rewards come from. No path
 * here moves credits from one user to another.
 *
 * BOUND TO THE REQUEST. The 402 carries a server nonce in the entry's
 * `extra.nonce`, `<uuid>.<expires>.<hmac>`, where the HMAC (SHA-256, a
 * per-server secret) covers the resource URL, the amount, the uuid, the
 * expiry and, since S-297 (2026-09-26), the request binding the paywall
 * computes (`requestBinding` in ./paywall.ts: the method, the path, the query
 * string and a SHA-256 of the body), so a retry that changes the inputs of
 * the request the 402 answered is `nonce_not_ours`. The retry echoes the
 * entry (v2 `accepted`) or names the nonce in its payload (v1), and the
 * paywall verifies the HMAC, the expiry and that the nonce was not used
 * before. A stateless server can therefore verify a challenge it does not
 * remember issuing, and a token captured from one request buys nothing on
 * another. The uuid doubles as the settle's Idempotency-Key
 * (`settle:x402:<uuid>`, the fixture's `fresh` shape).
 *
 * SINGLE USE ACROSS PROCESSES (S-297). The used-nonce set below is per
 * process, so a paid request replayed to another replica, or after a
 * restart, within the nonce's lifetime used to run the handler again and
 * reach the platform as a REPEAT of the same settle, which replayed instead
 * of charging: a second run for free. Two things close that. A client that
 * has a shared store claims the nonce there BEFORE the handler runs
 * (`CreditsClient.claimNonce`; the platform's in-process client claims with a
 * Redis `SET NX EX`, lib/payments/x402-nonce-claims.ts), and a client that has none
 * still answers nothing to a replay: the paywall treats a settle the platform
 * answered `replayed` as "already used" and withholds the answer.
 *
 * VERIFY, THEN SETTLE AFTER THE HANDLER. The token is verified before the
 * handler runs (locally against the platform's key set when a JWKS manager
 * is configured, else `POST /api/payments/verify`), and the charge is made
 * after the handler answered below 400, by `POST /api/payments/settle` by
 * token: the platform resolves the token to a lock and charges the agent's
 * price, commission included. A handler error settles nothing.
 *
 * The Python twin is `payments/x402_credits.py`; the fixture
 * `python/tests/fixtures/payments/x402_paywall.json` pins the nonce vectors.
 */

import type { JWKSManager } from '../../crypto/jwks';
import { readSettleResult } from './settle-result';
import type { PaymentSettleResult, PaymentVerifyResult } from './types';
import { freshSettleIdempotencyKey, IDEMPOTENCY_KEY_BODY_FIELD, idempotencyHeaders } from './idempotency';
import {
  CREDITS_ASSET,
  CREDITS_DECIMALS,
  CREDITS_MAX_TIMEOUT_SECONDS,
  CREDITS_NETWORK,
  CREDITS_PAY_TO,
  CREDITS_SCHEME,
  type PaymentRequirement,
} from './x402-wire';

// ── The platform, as the scheme talks to it ──────────────────────────────────

/** What a shared single-use store is asked to claim (S-297). */
export interface NonceClaim {
  /** The nonce's uuid, the settle's key. */
  nonceId: string;
  /** Unix seconds; a claim may be forgotten after this. */
  expires: number;
  resource: string;
  /** The request binding the nonce was minted with (`requestBinding`), empty when unbound. */
  binding: string;
}

/** What the credits scheme needs from the platform; the paywall's tests pass a fake. */
export interface CreditsClient {
  verifyToken(token: string, options?: { expectedAudience?: string | string[] }): Promise<PaymentVerifyResult>;
  settleToken(
    token: string,
    amountCredits: number,
    options: { idempotencyKey: string; description?: string; resource?: string },
  ): Promise<PaymentSettleResult>;
  /**
   * Claim the nonce in a store EVERY replica shares, before the handler runs
   * (S-297): true the first time, false when it was claimed before. A client
   * with no shared store leaves this out and the per-process set alone
   * applies; a throw (the store is unavailable) fails the request closed.
   */
  claimNonce?(claim: NonceClaim): Promise<boolean>;
}

export interface PlatformCreditsClientOptions {
  platformUrl: string;
  /** The agent's own platform key: the settle charges on behalf of an authenticated agent. */
  apiKey?: string;
  /** Local verification against the platform's key set, before the verify API. */
  jwks?: JWKSManager;
  fetch?: typeof fetch;
}

/**
 * A payment the platform could not be reached to verify (B12, 2026-09-28):
 * the reason the verify answers, which the paywall answers 503 with
 * `PLATFORM_UNREACHABLE_MESSAGE` rather than letting the connection error
 * escape as a 500 (the e2e run's paid retry against a platform that was
 * not there). The Python `x402_credits.PLATFORM_UNREACHABLE` is the same;
 * fixture `payments/final_sdk_x402_platform.json`.
 */
export const PLATFORM_UNREACHABLE = 'platform_unreachable';
export const PLATFORM_UNREACHABLE_MESSAGE =
  'The payment platform could not be reached to verify this payment. Try again in a moment.';

/** The platform over HTTP: `/api/payments/verify` and `/api/payments/settle`, as the payment skill calls them. */
export function platformCreditsClient(options: PlatformCreditsClientOptions): CreditsClient {
  const base = options.platformUrl.replace(/\/+$/, '');
  const fetchImpl = options.fetch ?? ((input: string | URL | Request, init?: RequestInit) => fetch(input, init));
  const auth = (): Record<string, string> => (options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {});
  return {
    async verifyToken(token, verifyOptions) {
      if (options.jwks) {
        try {
          const local = await options.jwks.verifyPaymentToken(token, { expectedAudience: verifyOptions?.expectedAudience });
          if (local) return { valid: true, balance: local.balance };
        } catch {
          // Fall through to the platform's verify.
        }
      }
      let res: Response;
      try {
        res = await fetchImpl(`${base}/api/payments/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...auth() },
          body: JSON.stringify({ token, expectedAudience: verifyOptions?.expectedAudience }),
        });
      } catch (err) {
        console.warn(`[payments] x402 credits verify: ${base} could not be reached: ${(err as Error).message}`);
        return { valid: false, invalidReason: PLATFORM_UNREACHABLE };
      }
      const data = (await res.json().catch(() => ({}))) as { valid?: boolean; balanceCredits?: number; balanceDollars?: number; error?: string };
      if (data.valid !== true) return { valid: false, invalidReason: data.error ?? `verify answered ${res.status}` };
      return { valid: true, balance: data.balanceCredits ?? data.balanceDollars };
    },
    async settleToken(token, amountCredits, settleOptions) {
      let res: Response;
      try {
        res = await fetchImpl(`${base}/api/payments/settle`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...idempotencyHeaders(settleOptions.idempotencyKey), ...auth() },
          body: JSON.stringify({
            token,
            amount: amountCredits,
            description: settleOptions.description,
            resource: settleOptions.resource,
            [IDEMPOTENCY_KEY_BODY_FIELD]: settleOptions.idempotencyKey,
          }),
        });
      } catch (err) {
        // A settle that never arrived charged nothing: a failed settle, whose
        // answer the paywall withholds, as for any other (B12).
        console.warn(`[payments] x402 credits settle: ${base} could not be reached: ${(err as Error).message}`);
        return readSettleResult({ success: false, error: PLATFORM_UNREACHABLE }, 'x402 credits settle');
      }
      return readSettleResult(await res.json().catch(() => ({})), 'x402 credits settle');
    },
  };
}

// ── The nonce ────────────────────────────────────────────────────────────────

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const NONCE_RE = /^([0-9a-f-]{36})\.(\d{1,12})\.([0-9a-f]{32})$/;

async function hmacHex(secret: string, message: string): Promise<string> {
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(message)));
  return Array.from(sig, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * The HMAC input: the resource URL, the amount, the uuid and the expiry,
 * `|`-joined, and the request binding as a fifth part when the nonce is
 * bound (S-297). An unbound nonce keeps the four-part message, so the
 * fixture's original vectors still hold.
 */
export function nonceMessage(resource: string, amount: string, id: string, expires: number, binding = ''): string {
  const base = `${resource}|${amount}|${id}|${expires}`;
  return binding ? `${base}|${binding}` : base;
}

/** `<uuid>.<expires>.<hmac16>`: a nonce bound to one resource, amount and (when given) request, verifiable without state. */
export async function mintCreditsNonce(
  secret: string,
  resource: string,
  amount: string,
  options: { ttlSeconds?: number; now?: Date; id?: string; binding?: string } = {},
): Promise<string> {
  const id = options.id ?? crypto.randomUUID();
  const expires = Math.floor((options.now ?? new Date()).getTime() / 1000) + (options.ttlSeconds ?? CREDITS_MAX_TIMEOUT_SECONDS);
  const mac = (await hmacHex(secret, nonceMessage(resource, amount, id, expires, options.binding))).slice(0, 32);
  return `${id}.${expires}.${mac}`;
}

export type NonceCheck = { ok: true; id: string; expires: number } | { ok: false; reason: string };

/** Verify a nonce's HMAC and expiry against THIS resource, amount and request binding. Single use is the caller's set. */
export async function verifyCreditsNonce(
  secret: string,
  nonce: unknown,
  resource: string,
  amount: string,
  now: Date = new Date(),
  binding = '',
): Promise<NonceCheck> {
  if (typeof nonce !== 'string') return { ok: false, reason: 'nonce_missing' };
  const m = NONCE_RE.exec(nonce);
  if (!m || !UUID_RE.test(m[1])) return { ok: false, reason: 'nonce_malformed' };
  const [, id, expiresText, mac] = m;
  const expires = Number(expiresText);
  const expected = (await hmacHex(secret, nonceMessage(resource, amount, id, expires, binding))).slice(0, 32);
  if (!constantTimeEqual(mac, expected)) return { ok: false, reason: 'nonce_not_ours' };
  if (Math.floor(now.getTime() / 1000) > expires) return { ok: false, reason: 'nonce_expired' };
  return { ok: true, id, expires };
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/**
 * The nonces this process has already settled, so a replayed request is
 * refused here before it reaches the platform (where its key would replay
 * rather than charge). Bounded: expired ids are swept as it grows, and past
 * the cap the oldest are dropped.
 */
export class UsedNonces {
  private readonly used = new Map<string, number>();
  constructor(private readonly cap = 20_000) {}

  /** True when `id` was not used before; records it until `expires`. */
  claim(id: string, expires: number, now: Date = new Date()): boolean {
    if (this.used.has(id)) return false;
    if (this.used.size >= 1_000) this.sweep(now);
    if (this.used.size >= this.cap) {
      const oldest = this.used.keys().next().value;
      if (oldest !== undefined) this.used.delete(oldest);
    }
    this.used.set(id, expires);
    return true;
  }

  private sweep(now: Date): void {
    const cutoff = Math.floor(now.getTime() / 1000);
    for (const [id, expires] of this.used) if (expires < cutoff) this.used.delete(id);
  }

  get size(): number {
    return this.used.size;
  }
}

// ── The scheme ───────────────────────────────────────────────────────────────

export interface CreditsSchemeOptions {
  client: CreditsClient;
  /** The HMAC secret the nonces are minted with; a random one per process when unset (a fleet configures one). */
  nonceSecret?: string;
  nonceTtlSeconds?: number;
  /** The platform URL, named in `extra.platform` so a client knows where the token comes from. */
  platformUrl?: string;
  /** The audience a token must carry, when the seller expects one (its own agent id). */
  expectedAudience?: string | string[];
  now?: () => Date;
}

export type CreditsVerification =
  | { ok: true; token: string; nonceId: string; balance?: number }
  | { ok: false; reason: string };

/** The credits scheme: mints the entry, verifies a payment, settles it. */
export class CreditsScheme {
  readonly scheme = CREDITS_SCHEME;
  readonly network = CREDITS_NETWORK;
  private readonly secret: string;
  private readonly used = new UsedNonces();
  private readonly now: () => Date;

  constructor(private readonly options: CreditsSchemeOptions) {
    this.secret = options.nonceSecret ?? crypto.randomUUID() + crypto.randomUUID();
    this.now = options.now ?? (() => new Date());
  }

  /** The `accepts[]` entry for `resource` at `amountNano`, nonce included; `binding` ties the nonce to the request (S-297). */
  async requirement(resource: string, amountNano: string, binding = ''): Promise<PaymentRequirement> {
    const nonce = await mintCreditsNonce(this.secret, resource, amountNano, { ttlSeconds: this.options.nonceTtlSeconds, now: this.now(), binding });
    return {
      scheme: CREDITS_SCHEME,
      network: CREDITS_NETWORK,
      amount: amountNano,
      asset: CREDITS_ASSET,
      payTo: CREDITS_PAY_TO,
      maxTimeoutSeconds: this.options.nonceTtlSeconds ?? CREDITS_MAX_TIMEOUT_SECONDS,
      extra: {
        nonce,
        decimals: CREDITS_DECIMALS,
        tokenType: 'jwt',
        ...(this.options.platformUrl ? { platform: this.options.platformUrl } : {}),
      },
    };
  }

  /**
   * Verify a credits payment: the nonce (HMAC over the resource, the amount
   * and the request binding; expiry; single use here and, when the client
   * has one, in the shared store) and the token (signature, audience, a
   * balance that covers the amount). `nonce` comes from the echoed entry (v2)
   * or the payload (v1). The shared claim comes LAST, after the token
   * verified, so an invalid payment does not burn the nonce.
   */
  async verify(payload: Record<string, unknown>, nonce: unknown, resource: string, amountNano: string, binding = ''): Promise<CreditsVerification> {
    const token = payload.token;
    if (typeof token !== 'string' || !token.trim()) return { ok: false, reason: 'token_missing' };
    const check = await verifyCreditsNonce(this.secret, nonce, resource, amountNano, this.now(), binding);
    if (!check.ok) return check;
    const verified = await this.options.client.verifyToken(token.trim(), { expectedAudience: this.options.expectedAudience });
    if (!verified.valid) return { ok: false, reason: verified.invalidReason ?? 'token_invalid' };
    const amountCredits = Number(amountNano) / 10 ** CREDITS_DECIMALS;
    if (typeof verified.balance === 'number' && verified.balance < amountCredits) return { ok: false, reason: 'insufficient_funds' };
    if (!this.used.claim(check.id, check.expires, this.now())) return { ok: false, reason: 'nonce_used' };
    if (this.options.client.claimNonce) {
      const claimed = await this.options.client.claimNonce({ nonceId: check.id, expires: check.expires, resource, binding });
      if (!claimed) return { ok: false, reason: 'nonce_used' };
    }
    return { ok: true, token: token.trim(), nonceId: check.id, balance: verified.balance };
  }

  /** The Idempotency-Key of the settle for a verified payment: the nonce's uuid, in the fixture's fresh shape. */
  static settleKey(nonceId: string): string {
    return `settle:x402:${nonceId}`;
  }

  /** Settle a verified payment for `amountNano` (the price, or a metered actual at or below it). */
  async settle(
    verified: Extract<CreditsVerification, { ok: true }>,
    amountNano: string,
    options: { description?: string; resource?: string } = {},
  ): Promise<PaymentSettleResult> {
    const amountCredits = Number(amountNano) / 10 ** CREDITS_DECIMALS;
    return this.options.client.settleToken(verified.token, amountCredits, {
      idempotencyKey: CreditsScheme.settleKey(verified.nonceId),
      description: options.description,
      resource: options.resource,
    });
  }
}

/** Kept for callers that settle by token outside a paywall; the same fresh key shape. */
export { freshSettleIdempotencyKey };
