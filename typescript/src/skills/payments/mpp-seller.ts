/**
 * MPP alongside x402, on the same priced endpoints (webagents gap-closure
 * plan 2.6, spec pack section 2: draft-httpauth-payment-01 with the method
 * drafts in tempoxyz/mpp-specs; 2026-09-26).
 *
 * The SELLER half of MPP. The SDK already carried the BUYER
 * (`./mpp-buyer.ts`: how an agent pays Robutler on a 402), and this module
 * borrows its pure helpers (RFC 8785 JCS, base64url without padding, the
 * `WWW-Authenticate: Payment` parser) so the two halves cannot disagree on
 * bytes. What is new is everything a server does:
 *
 *   - THE CHALLENGE (2.1). One `WWW-Authenticate: Payment …` per method on
 *     the same 402 that carries x402's `PAYMENT-REQUIRED` (mppx does this
 *     too, 2.5), `Cache-Control: no-store`. Required params `id`, `realm`,
 *     `method`, `intent`, `request`; `expires` (RFC 3339) and `opaque` are
 *     set here too. `request` is base64url-nopad of the JCS form of the
 *     method's request object.
 *   - STATELESS IDS. `id = base64url-nopad(HMAC-SHA256(secret, slots))`, the
 *     CURRENT slot order (mpp-specs #362): `realm|method|intent|request|
 *     expires|digest|opaque`, with `|Payment-Authorization` appended after
 *     `opaque` ONLY when the challenge names that header (this seller never
 *     does, so the seven-slot form is what it mints). A server that shares
 *     the secret can verify a challenge it did not issue. The resource URL
 *     is bound through `opaque` (`{"resource": url}`), so a credential
 *     answers one endpoint and no other.
 *   - THE CREDENTIAL (2.2). `Authorization: Payment <base64url-nopad(JSON)>`,
 *     JSON `{challenge, source?, payload}`; the echoed challenge must
 *     re-HMAC to its own id, not be expired, name our realm, and its id is
 *     SINGLE USE: a concurrent identical credential settles at most once.
 *   - METHODS. `stripe` (draft-stripe-charge-00): a Shared Payment Token,
 *     settled as a PaymentIntent created with `shared_payment_granted_token`,
 *     `confirm: true`, redirects never, under the idempotency key
 *     `${challenge.id}_${spt}`; 200 only when the intent `succeeded`. And
 *     `robutler` (method ids are lowercase letters only, so not
 *     `robutler-credits`): the credits scheme's token, settled through the
 *     paywall's credits scheme with the challenge's own nonce, so a
 *     credential is bound and single-use the same way an x402 payment is.
 *     "Servers MUST NOT require user accounts for payment" (11.7): the
 *     `robutler` challenge is only ever emitted BESIDE an account-less way
 *     to pay (Stripe here, or a chain scheme on the x402 side).
 *   - THE RECEIPT (2.3). `Payment-Receipt: base64url-nopad(JSON)` on 2xx
 *     only, `{status:"success", method, timestamp, reference, externalId?}`,
 *     with `Cache-Control: private`. The Stripe reference is the
 *     PaymentIntent id; the robutler reference is the settle's idempotency
 *     key, which an incident review can find on the platform.
 *   - PROBLEMS. RFC 9457 `application/problem+json` under
 *     `https://paymentauth.org/problems/`: `malformed-credential`,
 *     `invalid-challenge` (unknown or used id), `payment-expired`,
 *     `verification-failed`, `payment-insufficient`, all 402 with a fresh
 *     challenge; `method-unsupported` is 400; a credential AND an x402
 *     payment on one request is 400.
 *
 * No credential or receipt is ever logged. The Python twin is
 * `payments/mpp_seller.py`; the fixture `python/tests/fixtures/payments/
 * paywall_x402_mpp.json` pins the section 2.7 vectors for both.
 */

import {
  base64urlDecode,
  base64urlEncode,
  jcsCanonicalize,
  parseWwwAuthenticatePayment,
  MPP_INTENT_CHARGE,
  MPP_PROBLEM_BASE,
  MPP_RECEIPT_HEADER,
  type MppChallengeFields,
} from './mpp-buyer';
import { UsedNonces } from './x402-credits';
import { CREDITS_DECIMALS, creditsToNanocredits, isRecord } from './x402-wire';

// ── Names (the wire constants shared with the buyer come from `./mpp-buyer`) ─

export const MPP_AUTHORIZATION_HEADER = 'Authorization';
export const MPP_ALT_CREDENTIAL_HEADER = 'Payment-Authorization';
export const MPP_METHOD_STRIPE = 'stripe';
export const MPP_METHOD_ROBUTLER = 'robutler';
export const MPP_CHALLENGE_TTL_SECONDS = 300;
/** A method id is lowercase letters only (2.8). */
export const MPP_METHOD_RE = /^[a-z]+$/;

export type MppProblem =
  | 'payment-required'
  | 'malformed-credential'
  | 'invalid-challenge'
  | 'payment-expired'
  | 'verification-failed'
  | 'payment-insufficient'
  | 'method-unsupported';

// ── HMAC ids, the current slot order ─────────────────────────────────────────

/**
 * The HMAC input: seven positional slots, empty strings for absent ones;
 * with a `header` parameter, eight, the header BEFORE `opaque`. That is the
 * current order (mpp-specs #362 and mppx `Challenge.ts`); the IETF-archived
 * -01 text appends the header after `opaque` instead, and the fixture keeps
 * that older id as the one NOT to produce.
 */
export function challengeHmacInput(f: Omit<MppChallengeFields, 'id'>): string {
  const slots = [f.realm, f.method, f.intent, f.request, f.expires ?? '', f.digest ?? ''];
  if (f.header) slots.push(f.header);
  slots.push(f.opaque ?? '');
  return slots.join('|');
}

async function hmacSha256(secret: string, message: string): Promise<Uint8Array> {
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return new Uint8Array(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(message)));
}

/** `base64url-nopad(HMAC-SHA256(secret, slots))`: the stateless challenge id. */
export async function computeChallengeId(secret: string, f: Omit<MppChallengeFields, 'id'>): Promise<string> {
  return base64urlEncode(await hmacSha256(secret, challengeHmacInput(f)));
}

/** `request` and `opaque` on the wire: JCS, then base64url without padding. */
export function encodeJcsParam(value: unknown): string {
  return base64urlEncode(jcsCanonicalize(value));
}

export function decodeJcsParam(text: string | undefined): Record<string, unknown> | null {
  if (!text) return null;
  const bytes = base64urlDecode(text);
  if (!bytes) return null;
  try {
    const parsed = JSON.parse(new TextDecoder().decode(bytes)) as unknown;
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** One challenge as a `WWW-Authenticate` value, params quoted, in the order mppx writes them. */
export function formatChallenge(f: MppChallengeFields): string {
  const params: Array<[string, string]> = [['id', f.id], ['realm', f.realm], ['method', f.method], ['intent', f.intent]];
  if (f.expires) params.push(['expires', f.expires]);
  params.push(['request', f.request]);
  if (f.digest) params.push(['digest', f.digest]);
  if (f.opaque) params.push(['opaque', f.opaque]);
  if (f.header) params.push(['header', f.header]);
  return `Payment ${params.map(([k, v]) => `${k}="${v.replace(/(["\\])/g, '\\$1')}"`).join(', ')}`;
}

// ── Problems ─────────────────────────────────────────────────────────────────

export function problemBody(problem: MppProblem, status: number, detail: string): { body: string; contentType: string } {
  return {
    contentType: 'application/problem+json',
    body: JSON.stringify({ type: `${MPP_PROBLEM_BASE}${problem}`, title: problem.replace(/-/g, ' '), status, detail }),
  };
}

// ── Credentials ──────────────────────────────────────────────────────────────

export interface MppCredential {
  challenge: MppChallengeFields;
  source?: string;
  payload: Record<string, unknown>;
}

export type ReadMppCredential = { ok: true; credential: MppCredential } | { ok: false; detail: string } | null;

/**
 * The credential a request carries in `Authorization: Payment …` (or in
 * `Payment-Authorization`, which this seller accepts too since a client may
 * use either when the challenge names none), decoded without trusting it.
 */
export function readMppCredential(headers: { get(name: string): string | null }): ReadMppCredential {
  const auth = headers.get(MPP_AUTHORIZATION_HEADER);
  const alt = headers.get(MPP_ALT_CREDENTIAL_HEADER);
  const value = alt ?? (auth && /^\s*Payment\s/i.test(auth) ? auth : null);
  if (!value) return null;
  const m = /^\s*Payment\s+([A-Za-z0-9_-]+)\s*$/.exec(value);
  if (!m) return { ok: false, detail: 'the credential is not "Payment <base64url>"' };
  const bytes = base64urlDecode(m[1]);
  if (!bytes || bytes.length === 0) return { ok: false, detail: 'the credential is not base64url' };
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return { ok: false, detail: 'the credential does not decode to JSON' };
  }
  if (!isRecord(parsed) || !isRecord(parsed.challenge) || !isRecord(parsed.payload)) {
    return { ok: false, detail: 'the credential lacks a challenge or a payload object' };
  }
  const ch = parsed.challenge;
  for (const k of ['id', 'realm', 'method', 'intent', 'request'] as const) {
    if (typeof ch[k] !== 'string' || (ch[k] as string).length === 0) return { ok: false, detail: `challenge.${k} is missing` };
  }
  for (const k of ['expires', 'digest', 'opaque', 'header', 'description'] as const) {
    if (ch[k] !== undefined && typeof ch[k] !== 'string') return { ok: false, detail: `challenge.${k} is not a string` };
  }
  const challenge: MppChallengeFields = {
    id: ch.id as string,
    realm: ch.realm as string,
    method: ch.method as string,
    intent: ch.intent as string,
    request: ch.request as string,
  };
  if (ch.expires !== undefined) challenge.expires = ch.expires as string;
  if (ch.digest !== undefined) challenge.digest = ch.digest as string;
  if (ch.opaque !== undefined) challenge.opaque = ch.opaque as string;
  if (ch.header !== undefined) challenge.header = ch.header as string;
  return { ok: true, credential: { challenge, source: typeof parsed.source === 'string' ? parsed.source : undefined, payload: parsed.payload } };
}

// ── Stripe ───────────────────────────────────────────────────────────────────

/** The one Stripe call the `stripe` method makes; a test passes a double. */
export interface StripeSptClient {
  createPaymentIntent(
    params: {
      amount: number;
      currency: string;
      shared_payment_granted_token: string;
      confirm: true;
      automatic_payment_methods: { enabled: true; allow_redirects: 'never' };
      metadata: Record<string, string>;
    },
    options: { idempotencyKey: string },
  ): Promise<{ id: string; status: string }>;
}

export interface MppStripeConfig {
  /** Our Stripe Business Network profile, `profile_…` or `profile_test_…`. */
  profileId: string;
  client: StripeSptClient;
  /** Lowercase ISO 4217; `usd`. */
  currency?: string;
  paymentMethodTypes?: string[];
  /** Smallest currency units per credit: 100 cents at the settled 1 credit = 1 USD. */
  unitsPerCredit?: number;
}

// ── The seller ───────────────────────────────────────────────────────────────

export interface MppSellerConfig {
  /** The realm the challenge names: the seller's host. */
  realm: string;
  /** The HMAC secret behind the stateless ids; one per fleet. */
  secret: string;
  challengeTtlSeconds?: number;
  stripe?: MppStripeConfig;
  /** Offer the `robutler` (credits) method; only emitted beside an account-less method (11.7). Default true. */
  credits?: boolean;
  /**
   * Claim a `stripe` challenge id in a store EVERY replica shares, before the
   * handler runs (S-297, 2026-09-26): true the first time, false when it was
   * claimed before. The per-process set below is all a single process needs;
   * a fleet's Stripe idempotency key would otherwise answer a replayed
   * credential the original PaymentIntent, and the handler would run again.
   * The `robutler` method's single use is the credits scheme's (`claimNonce`).
   */
  claimChallenge?: (claim: { challengeId: string; expires: number; resource: string }) => Promise<boolean>;
  now?: () => Date;
}

/** The receipt a seller writes (the buyer's parsed form is `MppReceipt` in `./mpp-buyer`). */
export interface MppSellerReceipt {
  status: 'success';
  method: string;
  timestamp: string;
  reference: string;
  externalId?: string;
}

/** What a verified credential lets the paywall do once the handler answered. */
export interface MppAdmitted {
  ok: true;
  method: string;
  /** The credits token, for the `robutler` method; the paywall settles it through its credits scheme. */
  creditsToken?: string;
  /** The nonce the `robutler` challenge carried, for the credits scheme's binding. */
  creditsNonce?: string;
  /** Settle the `stripe` method; the receipt on success, the problem on failure. */
  settleStripe?: () => Promise<{ ok: true; receipt: MppSellerReceipt } | { ok: false; problem: MppProblem; detail: string }>;
  challengeId: string;
}

export interface MppRefused {
  ok: false;
  status: 400 | 402;
  problem: MppProblem;
  detail: string;
}

export class MppSeller {
  private readonly used = new UsedNonces();
  private readonly now: () => Date;
  readonly realm: string;

  constructor(private readonly config: MppSellerConfig) {
    if (!config.realm) throw new Error('mpp: a realm is required');
    if (!config.secret) throw new Error('mpp: a challenge secret is required');
    this.realm = config.realm;
    this.now = config.now ?? (() => new Date());
  }

  /** Whether the `robutler` method may be offered: beside Stripe, or beside a chain scheme on the x402 side. */
  creditsOffered(chainOffered: boolean): boolean {
    return this.config.credits !== false && (this.config.stripe !== undefined || chainOffered);
  }

  private expires(): string {
    const at = new Date(this.now().getTime() + (this.config.challengeTtlSeconds ?? MPP_CHALLENGE_TTL_SECONDS) * 1000);
    return at.toISOString().replace(/\.\d{3}Z$/, 'Z');
  }

  private async mint(method: string, request: unknown, url: string, description?: string): Promise<MppChallengeFields> {
    const unsigned: Omit<MppChallengeFields, 'id'> = {
      realm: this.realm,
      method,
      intent: MPP_INTENT_CHARGE,
      request: encodeJcsParam(request),
      expires: this.expires(),
      opaque: encodeJcsParam({ resource: url }),
    };
    const id = await computeChallengeId(this.config.secret, unsigned);
    const fields: MppChallengeFields = { id, ...unsigned };
    return description ? { ...fields, description } as MppChallengeFields : fields;
  }

  /**
   * The `WWW-Authenticate` values for a 402 at `url` for `maxCredits`.
   * `creditsNonce` is the nonce the paywall's credits scheme minted for this
   * request (the `robutler` challenge carries it, so its credential is bound
   * and single-use the way an x402 credits payment is).
   */
  async challenges(url: string, maxCredits: number, options: { chainOffered: boolean; creditsNonce?: string; platformUrl?: string; description?: string }): Promise<string[]> {
    const out: string[] = [];
    const stripe = this.config.stripe;
    if (stripe) {
      const amount = String(Math.round(maxCredits * (stripe.unitsPerCredit ?? 100)));
      const request = {
        amount,
        currency: stripe.currency ?? 'usd',
        ...(options.description ? { description: options.description } : {}),
        methodDetails: { networkId: stripe.profileId, paymentMethodTypes: stripe.paymentMethodTypes ?? ['card', 'link'] },
      };
      out.push(formatChallenge(await this.mint(MPP_METHOD_STRIPE, request, url)));
    }
    if (this.creditsOffered(options.chainOffered) && options.creditsNonce) {
      const request = {
        amount: creditsToNanocredits(maxCredits),
        currency: 'credits',
        methodDetails: { nonce: options.creditsNonce, decimals: CREDITS_DECIMALS, ...(options.platformUrl ? { platform: options.platformUrl } : {}) },
      };
      out.push(formatChallenge(await this.mint(MPP_METHOD_ROBUTLER, request, url)));
    }
    return out;
  }

  /**
   * Verify a credential for `url`: the id re-HMACs, the challenge is ours and
   * for this resource, not expired, not used before; the method is one we
   * serve and its payload is well-formed. What is answered is what the
   * paywall needs to settle after the handler.
   */
  async admit(credential: MppCredential, url: string, maxCredits: number): Promise<MppAdmitted | MppRefused> {
    const ch = credential.challenge;
    // What is served is decided first (a 400 the client can act on, 2.1);
    // whether the challenge is ours comes after.
    if (ch.intent !== MPP_INTENT_CHARGE) return { ok: false, status: 400, problem: 'method-unsupported', detail: `intent ${ch.intent} is not served` };
    if (!MPP_METHOD_RE.test(ch.method)) return { ok: false, status: 400, problem: 'method-unsupported', detail: 'method ids are lowercase letters' };
    if (ch.method !== MPP_METHOD_STRIPE && ch.method !== MPP_METHOD_ROBUTLER) {
      return { ok: false, status: 400, problem: 'method-unsupported', detail: `method ${ch.method} is not served` };
    }
    if (ch.method === MPP_METHOD_STRIPE && !this.config.stripe) return { ok: false, status: 400, problem: 'method-unsupported', detail: 'stripe is not configured' };
    if (ch.realm !== this.realm) return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the challenge is for another realm' };
    const { id, ...unsigned } = ch;
    const expected = await computeChallengeId(this.config.secret, unsigned);
    if (expected !== id) return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the challenge id does not verify' };
    if (ch.expires) {
      const at = Date.parse(ch.expires);
      if (!Number.isFinite(at)) return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'expires is not RFC 3339' };
      if (this.now().getTime() > at) return { ok: false, status: 402, problem: 'payment-expired', detail: 'the challenge has expired' };
    }
    const opaque = decodeJcsParam(ch.opaque);
    if (!opaque || opaque.resource !== url) return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the challenge is for another resource' };
    const request = decodeJcsParam(ch.request);
    if (!request || typeof request.amount !== 'string') return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the request is not readable' };

    if (ch.method === MPP_METHOD_STRIPE) {
      const stripe = this.config.stripe;
      if (!stripe) return { ok: false, status: 400, problem: 'method-unsupported', detail: 'stripe is not configured' };
      const spt = credential.payload.spt;
      if (typeof spt !== 'string' || !/^spt_[A-Za-z0-9]+$/.test(spt)) {
        return { ok: false, status: 402, problem: 'malformed-credential', detail: 'payload.spt is not a shared payment token' };
      }
      const amount = Number(request.amount);
      const expectedAmount = Math.round(maxCredits * (stripe.unitsPerCredit ?? 100));
      if (!Number.isSafeInteger(amount) || amount < expectedAmount) {
        return { ok: false, status: 402, problem: 'payment-insufficient', detail: 'the challenge amount is below the price' };
      }
      const expiresAt = ch.expires ? Date.parse(ch.expires) : this.now().getTime() + MPP_CHALLENGE_TTL_SECONDS * 1000;
      if (!this.used.claim(id, Math.floor(expiresAt / 1000), this.now())) {
        return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the challenge was already used' };
      }
      if (this.config.claimChallenge && !(await this.config.claimChallenge({ challengeId: id, expires: Math.floor(expiresAt / 1000), resource: url }))) {
        return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the challenge was already used' };
      }
      const externalId = typeof credential.payload.externalId === 'string' ? credential.payload.externalId : undefined;
      return {
        ok: true,
        method: MPP_METHOD_STRIPE,
        challengeId: id,
        settleStripe: async () => {
          let intent: { id: string; status: string };
          try {
            intent = await stripe.client.createPaymentIntent(
              {
                amount,
                currency: String(request.currency ?? stripe.currency ?? 'usd'),
                shared_payment_granted_token: spt,
                confirm: true,
                automatic_payment_methods: { enabled: true, allow_redirects: 'never' },
                metadata: { challenge_id: id },
              },
              { idempotencyKey: `${id}_${spt}` },
            );
          } catch (error) {
            return { ok: false, problem: 'verification-failed', detail: `the payment could not be taken: ${(error as Error).message}` };
          }
          if (intent.status !== 'succeeded') return { ok: false, problem: 'verification-failed', detail: `the payment did not succeed (${intent.status})` };
          const receipt: MppSellerReceipt = {
            status: 'success',
            method: MPP_METHOD_STRIPE,
            timestamp: this.now().toISOString().replace(/\.\d{3}Z$/, 'Z'),
            reference: intent.id,
            ...(externalId ? { externalId } : {}),
          };
          return { ok: true, receipt };
        },
      };
    }

    if (ch.method === MPP_METHOD_ROBUTLER) {
      const token = credential.payload.token;
      if (typeof token !== 'string' || !token.trim()) return { ok: false, status: 402, problem: 'malformed-credential', detail: 'payload.token is missing' };
      const details = isRecord(request.methodDetails) ? request.methodDetails : {};
      if (typeof details.nonce !== 'string') return { ok: false, status: 402, problem: 'invalid-challenge', detail: 'the challenge carries no nonce' };
      if (BigInt(request.amount) < BigInt(creditsToNanocredits(maxCredits))) {
        return { ok: false, status: 402, problem: 'payment-insufficient', detail: 'the challenge amount is below the price' };
      }
      // Single use is the credits scheme's nonce set: the same nonce backs both.
      return { ok: true, method: MPP_METHOD_ROBUTLER, challengeId: id, creditsToken: token.trim(), creditsNonce: details.nonce };
    }
    return { ok: false, status: 400, problem: 'method-unsupported', detail: `method ${ch.method} is not served` };
  }

  /** The receipt header for a paid 2xx. */
  static receiptHeader(receipt: MppSellerReceipt): [string, string] {
    return [MPP_RECEIPT_HEADER, base64urlEncode(JSON.stringify(receipt))];
  }

  /** A receipt for a `robutler` settle: the settle's key as the reference. */
  creditsReceipt(reference: string): MppSellerReceipt {
    return { status: 'success', method: MPP_METHOD_ROBUTLER, timestamp: this.now().toISOString().replace(/\.\d{3}Z$/, 'Z'), reference };
  }
}

export { parseWwwAuthenticatePayment };
