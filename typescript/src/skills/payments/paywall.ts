/**
 * The paywall: standard x402 (v2 and v1) on a priced `@http` endpoint
 * (webagents gap-closure plan 2.6, 2026-09-26).
 *
 * One object, built by the payment skill from its configuration and asked by
 * every server the SDK ships (`server/handler.ts`, `server/node.ts`) when a
 * request reaches an endpoint that carries `pricing` (a `@pricing` stacked on
 * `@http`, or a `price` on a no-code endpoint). What it does, in the spec's
 * order (spec pack section 1.5):
 *
 *   1. No payment on the request: answer 402 with `PAYMENT-REQUIRED` (v2, in
 *      the header) AND a v1 JSON body, `Cache-Control: no-store`. The offers
 *      are `accepts[]`: the credits scheme (`./x402-credits.ts`), and the
 *      chain schemes (`exact`, and `upto` for a metered endpoint) when a
 *      `chain` seller is configured. Every entry is well-formed for a strict
 *      parser, so a standard client that knows only the chain kinds simply
 *      drops the credits entry and pays on chain.
 *   2. A payment: read it (`PAYMENT-SIGNATURE` or `X-PAYMENT`), match it to an
 *      offer (a v2 `accepted` is matched field for field; a v1 payload names
 *      its scheme and network), VERIFY it (the credits scheme locally and
 *      against the platform, a chain scheme through the facilitator), and
 *      only then run the handler.
 *   3. The handler answered below 400: SETTLE, once, and answer with
 *      `PAYMENT-RESPONSE` (v2) or `X-PAYMENT-RESPONSE` (v1) and
 *      `Cache-Control: private`. A metered endpoint's actual amount is read
 *      from the handler's `Settlement-Overrides` header, which is stripped.
 *      The handler answered 400 or above: nothing is settled, the answer goes
 *      out as it is.
 *   4. The settle failed: 402 with the failed settle response in the header
 *      and `{}` as the body. `settlement_pending` is NOT a failure and NOT
 *      final: the resource is delivered, the pending response rides along
 *      with its transaction hash for reconciliation, and no second settle is
 *      ever attempted for the same payment. A settle the platform answered
 *      `replayed` (S-297, 2026-09-26) IS a failure here: the payment was
 *      spent by an earlier request (another replica, an earlier process),
 *      so this one is a replay and gets `{}`, never the handler's answer.
 *
 * BOUND TO THE REQUEST (S-297). The credits nonce a 402 mints covers, beside
 * the resource and the amount, `requestBinding(request)`: the method, the
 * path, the query string and a SHA-256 of the body. The retry must be the
 * request the 402 answered; one that changes the inputs is `nonce_not_ours`.
 * The body is read from a clone, so the handler still reads its own.
 *
 * WHO RECEIVES A CHAIN PAYMENT is the seller's configuration (the entry's
 * `payTo:`) and the counsel gate the brief sets: a self-hosted agent is the developer's
 * own software and names its own address; on the platform the address is
 * Robutler's and the path is OFF until counsel clears it (the portal's own
 * dispatcher decides that; this module only offers what it is configured
 * with). The credits scheme needs no such gate: Robutler is its seller of
 * record by construction.
 *
 * The Python twin is `payments/paywall.py`, wired into `server/core/app.py`.
 */

import type { HttpEndpoint, PricingConfig } from '../../core/types';
import { CreditsScheme, PLATFORM_UNREACHABLE, PLATFORM_UNREACHABLE_MESSAGE, type CreditsSchemeOptions } from './x402-credits';
import type { FacilitatorClient } from './x402-facilitator';
import { MppSeller, problemBody, readMppCredential, type MppProblem } from './mpp-seller';
import {
  buildPaymentRequired,
  creditsToAssetUnits,
  creditsToNanocredits,
  isAmountString,
  readPaymentPayload,
  readSettlementOverrides,
  requirementMatches,
  settleResponseHeader,
  v1NetworkToCaip2,
  X402_HEADERS,
  type BazaarDeclaration,
  type PaymentPayload,
  type PaymentRequirement,
  type ResourceInfo,
  type SettleResponse,
} from './x402-wire';

// ── Configuration ────────────────────────────────────────────────────────────

export interface ChainSellerConfig {
  /** The receiving address: the developer's own on a self-hosted agent, Robutler's on the platform. */
  payTo: string;
  /** CAIP-2, `eip155:84532` for Base Sepolia. */
  network: string;
  /** The token contract. */
  asset: string;
  /** The asset's decimals (USDC: 6). */
  decimals: number;
  /** Whole asset units per credit; 1 for a dollar stablecoin at the settled 1 credit = 1 USD rate. */
  unitsPerCredit?: number;
  /** `exact` always; add `upto` to meter (Permit2 assets only). */
  schemes?: Array<'exact' | 'upto'>;
  maxTimeoutSeconds?: number;
  /** The EIP-712 domain of the asset, `{ name, version }`; `upto` also needs `facilitatorAddress`. */
  extra?: Record<string, unknown>;
  facilitator: FacilitatorClient;
}

export interface PaywallConfig {
  /** The credits scheme; absent means the seller takes no credits (a self-hosted seller with no platform key). */
  credits?: CreditsSchemeOptions;
  chain?: ChainSellerConfig;
  /** MPP challenges beside the x402 offers on the same 402, and MPP credentials on the retry (`./mpp-seller.ts`). */
  mpp?: MppSeller;
  /** What the 402 says about the service (`resource.serviceName`, `description`, `tags`, `iconUrl`). */
  resource?: Omit<ResourceInfo, 'url'>;
  /** Called for every payment event; a host wires its trace here. */
  log?: (event: string, fields: Record<string, unknown>) => void;
}

/**
 * A chain seller from the environment: the receiving address in `addressVar`
 * (`X402_PAY_TO` for a self-hosted agent; the platform names its own
 * variable), with `X402_NETWORK`, `X402_ASSET`, `X402_ASSET_DECIMALS`,
 * `X402_ASSET_NAME` and `X402_ASSET_VERSION` beside it, Base Sepolia USDC
 * when unset. Undefined when no address is named: the seller takes no chain
 * payment. The address is written into the entry HERE, by the module that
 * owns the wire shape; a host hands over its environment and its
 * facilitator and never spells the field itself.
 */
export function chainSellerFromEnv(
  env: Record<string, string | undefined>,
  options: { addressVar?: string; facilitator: FacilitatorClient; schemes?: Array<'exact' | 'upto'> },
): ChainSellerConfig | undefined {
  const address = env[options.addressVar ?? 'X402_PAY_TO'];
  if (!address) return undefined;
  const decimals = Number(env.X402_ASSET_DECIMALS ?? '6');
  return {
    payTo: address,
    network: env.X402_NETWORK ?? 'eip155:84532',
    asset: env.X402_ASSET ?? '0x036CbD53842c5426634e7929541eC2318f3dCF7e',
    decimals: Number.isFinite(decimals) ? decimals : 6,
    extra: { name: env.X402_ASSET_NAME ?? 'USDC', version: env.X402_ASSET_VERSION ?? '2' },
    ...(options.schemes ? { schemes: options.schemes } : {}),
    facilitator: options.facilitator,
  };
}

/** What the paywall needs to know about the endpoint it guards. */
export interface PricedEndpoint {
  path: string;
  method: string;
  pricing: PricingConfig;
  description?: string;
  discovery?: BazaarDeclaration;
}

/** A pricing that names a maximum (`lock`) above its per-call price is metered: the handler settles the actual. */
export function endpointPrice(pricing: PricingConfig): { credits: number; maxCredits: number; metered: boolean } {
  const perCall = typeof pricing.creditsPerCall === 'number' ? pricing.creditsPerCall : 0;
  const lock = typeof pricing.lock === 'number' ? pricing.lock : undefined;
  const maxCredits = lock !== undefined && lock > perCall ? lock : perCall;
  if (!(maxCredits > 0)) throw new Error('x402: a priced endpoint needs creditsPerCall or lock above zero');
  return { credits: perCall > 0 ? perCall : maxCredits, maxCredits, metered: lock !== undefined && lock > perCall };
}

// ── Outcomes ─────────────────────────────────────────────────────────────────

interface Admitted {
  version: 1 | 2;
  scheme: 'credits' | 'chain';
  requirement: PaymentRequirement;
  payload: PaymentPayload;
  credits?: Awaited<ReturnType<CreditsScheme['verify']>>;
  maxCredits: number;
}

// ── The paywall ──────────────────────────────────────────────────────────────

export class Paywall {
  private readonly credits?: CreditsScheme;
  private readonly log: (event: string, fields: Record<string, unknown>) => void;

  constructor(private readonly config: PaywallConfig) {
    this.credits = config.credits ? new CreditsScheme(config.credits) : undefined;
    this.log = config.log ?? (() => undefined);
    if (!this.credits && !config.chain) throw new Error('x402: a paywall needs the credits scheme or a chain seller');
  }

  /** The offers for `endpoint` at `url`, one entry per scheme this seller takes; `binding` ties the credits nonce to the request (S-297). */
  async offers(endpoint: PricedEndpoint, url: string, binding = ''): Promise<{ accepts: PaymentRequirement[]; resource: ResourceInfo; maxCredits: number }> {
    const { maxCredits, metered } = endpointPrice(endpoint.pricing);
    const accepts: PaymentRequirement[] = [];
    if (this.credits) accepts.push(await this.credits.requirement(url, creditsToNanocredits(maxCredits), binding));
    const chain = this.config.chain;
    if (chain) {
      const amount = creditsToAssetUnits(maxCredits, chain.decimals, chain.unitsPerCredit ?? 1);
      const schemes = chain.schemes ?? ['exact'];
      const offered = metered && schemes.includes('upto') ? 'upto' : 'exact';
      accepts.push({
        scheme: offered,
        network: chain.network,
        amount,
        asset: chain.asset,
        payTo: chain.payTo,
        maxTimeoutSeconds: chain.maxTimeoutSeconds ?? 60,
        ...(chain.extra ? { extra: { ...chain.extra } } : {}),
      });
    }
    const resource: ResourceInfo = {
      url,
      description: endpoint.description ?? this.config.resource?.description ?? `${endpoint.method} ${endpoint.path}`,
      mimeType: this.config.resource?.mimeType ?? 'application/json',
      ...(this.config.resource?.serviceName ? { serviceName: this.config.resource.serviceName } : {}),
      ...(this.config.resource?.tags ? { tags: this.config.resource.tags } : {}),
      ...(this.config.resource?.iconUrl ? { iconUrl: this.config.resource.iconUrl } : {}),
    };
    return { accepts, resource, maxCredits };
  }

  /**
   * The 402: `PAYMENT-REQUIRED` (v2) in the header, the v1 document as the
   * body, one `WWW-Authenticate: Payment` per MPP method beside them,
   * `no-store`. An MPP problem (`problem`) replaces the v1 body with RFC 9457
   * problem details; the x402 offers still ride in the header.
   */
  async challenge(endpoint: PricedEndpoint, request: Request, error?: string, problem?: { problem: MppProblem; detail: string }): Promise<Response> {
    const url = resourceUrl(request);
    const { accepts, resource, maxCredits } = await this.offers(endpoint, url, await requestBinding(request));
    const { v2, v1 } = buildPaymentRequired(resource, accepts, { error, bazaar: endpoint.discovery });
    const headers = new Headers({
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
      [X402_HEADERS.required]: encodeBase64(v2),
    });
    if (this.config.mpp) {
      const creditsEntry = accepts.find((r) => r.scheme === this.credits?.scheme);
      const creditsNonce = typeof creditsEntry?.extra?.nonce === 'string' ? creditsEntry.extra.nonce : undefined;
      const mppChallenges = await this.config.mpp.challenges(url, maxCredits, {
        chainOffered: this.config.chain !== undefined,
        creditsNonce,
        platformUrl: this.config.credits?.platformUrl,
        description: resource.description,
      });
      for (const value of mppChallenges) headers.append('WWW-Authenticate', value);
    }
    if (problem) {
      const { body, contentType } = problemBody(problem.problem, 402, problem.detail);
      headers.set('Content-Type', contentType);
      return new Response(body, { status: 402, headers });
    }
    return new Response(JSON.stringify(v1), { status: 402, headers });
  }

  /**
   * Serve a priced endpoint: challenge, or verify, run, settle. `run` is the
   * handler, called only after a payment verified.
   */
  async handle(endpoint: PricedEndpoint, request: Request, run: () => Promise<Response>): Promise<Response> {
    const read = readPaymentPayload(request.headers);
    const mppRead = this.config.mpp ? readMppCredential(request.headers) : null;
    if (mppRead !== null && read !== null) {
      return json({ error: { code: 'invalid_payment', message: 'one payment per request: an MPP credential and an x402 payment were both sent' } }, 400);
    }
    if (mppRead !== null) return this.handleMpp(endpoint, request, mppRead, run);
    if (read === null) return this.challenge(endpoint, request);
    if (read.version === 0) return json({ error: { code: 'invalid_payment', message: read.error } }, 400);

    const url = resourceUrl(request);
    const binding = await requestBinding(request);
    const admitted = await this.admit(endpoint, url, read, binding);
    if ('refusal' in admitted) {
      this.log('x402.refused', { path: endpoint.path, reason: admitted.refusal });
      // The platform was not there to verify the payment (B12): not the
      // payer's fault, and not a crash. 503 with a sentence, nothing run.
      if (admitted.refusal === PLATFORM_UNREACHABLE) return platformUnreachable();
      return this.challenge(endpoint, request, admitted.refusal);
    }

    const response = await run();
    if (response.status >= 400) {
      // Nothing is settled for an error: the client keeps its authorization.
      this.log('x402.unsettled', { path: endpoint.path, status: response.status });
      return response;
    }

    const overrides = readSettlementOverrides(response.headers);
    const settled = await this.settle(admitted, endpoint, url, overrides);
    const headers = new Headers(response.headers);
    headers.delete(X402_HEADERS.settlementOverrides);
    const [name, value] = settleResponseHeader(admitted.version, settled);
    headers.set(name, value);
    headers.set('Cache-Control', 'private');

    if (!settled.success && settled.errorReason !== 'settlement_pending') {
      // A replayed settle is a replayed request (S-297): the answer is withheld.
      this.log(settled.errorReason === 'replayed' ? 'x402.replayed' : 'x402.settle_failed', { path: endpoint.path, reason: settled.errorReason });
      return new Response('{}', { status: 402, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', [name]: value } });
    }
    if (settled.errorReason === 'settlement_pending') {
      // Not final: delivered, recorded, never settled a second time.
      this.log('x402.settlement_pending', { path: endpoint.path, transaction: settled.transaction, network: settled.network });
    } else {
      this.log('x402.settled', { path: endpoint.path, scheme: admitted.requirement.scheme, network: settled.network, amount: settled.amount });
    }
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  }

  // ── MPP ───────────────────────────────────────────────────────────────────

  /**
   * An MPP credential: verify (the seller's rules), run, settle, receipt.
   * The `robutler` method settles through the credits scheme with the
   * challenge's nonce, the `stripe` method through the Stripe client. A
   * refusal answers a fresh challenge with problem details.
   */
  private async handleMpp(
    endpoint: PricedEndpoint,
    request: Request,
    mppRead: NonNullable<ReturnType<typeof readMppCredential>>,
    run: () => Promise<Response>,
  ): Promise<Response> {
    const mpp = this.config.mpp!;
    if (!mppRead.ok) return this.challenge(endpoint, request, undefined, { problem: 'malformed-credential', detail: mppRead.detail });
    const url = resourceUrl(request);
    const { maxCredits, credits: perCall, metered } = endpointPrice(endpoint.pricing);
    const admitted = await mpp.admit(mppRead.credential, url, maxCredits);
    if (!admitted.ok) {
      this.log('mpp.refused', { path: endpoint.path, problem: admitted.problem });
      if (admitted.status === 400) {
        const { body, contentType } = problemBody(admitted.problem, 400, admitted.detail);
        return new Response(body, { status: 400, headers: { 'Content-Type': contentType, 'Cache-Control': 'no-store' } });
      }
      return this.challenge(endpoint, request, undefined, { problem: admitted.problem, detail: admitted.detail });
    }

    let creditsVerified: Awaited<ReturnType<CreditsScheme['verify']>> | undefined;
    if (admitted.method === 'robutler') {
      if (!this.credits) return this.challenge(endpoint, request, undefined, { problem: 'method-unsupported', detail: 'credits are not taken here' });
      creditsVerified = await this.credits.verify({ token: admitted.creditsToken }, admitted.creditsNonce, url, creditsToNanocredits(maxCredits), await requestBinding(request));
      if (!creditsVerified.ok && creditsVerified.reason === PLATFORM_UNREACHABLE) return platformUnreachable();
      if (!creditsVerified.ok) {
        const problem: MppProblem = creditsVerified.reason === 'nonce_used' ? 'invalid-challenge'
          : creditsVerified.reason === 'nonce_expired' ? 'payment-expired'
          : creditsVerified.reason === 'insufficient_funds' ? 'payment-insufficient'
          : 'verification-failed';
        return this.challenge(endpoint, request, undefined, { problem, detail: creditsVerified.reason });
      }
    }

    const response = await run();
    if (response.status >= 400) {
      this.log('mpp.unsettled', { path: endpoint.path, status: response.status });
      return response;
    }

    const overrides = readSettlementOverrides(response.headers);
    const headers = new Headers(response.headers);
    headers.delete(X402_HEADERS.settlementOverrides);
    headers.set('Cache-Control', 'private');

    if (admitted.method === 'stripe' && admitted.settleStripe) {
      const settled = await admitted.settleStripe();
      if (!settled.ok) {
        this.log('mpp.settle_failed', { path: endpoint.path, method: 'stripe', problem: settled.problem });
        return this.challenge(endpoint, request, undefined, { problem: settled.problem, detail: settled.detail });
      }
      const [name, value] = MppSeller.receiptHeader(settled.receipt);
      headers.set(name, value);
      this.log('mpp.settled', { path: endpoint.path, method: 'stripe' });
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }

    if (creditsVerified?.ok && this.credits) {
      let actualNano = creditsToNanocredits(metered ? maxCredits : perCall);
      if (metered && overrides) {
        if (overrides.credits !== undefined) actualNano = creditsToNanocredits(overrides.credits);
        else if (overrides.amount !== undefined) actualNano = overrides.amount;
        if (BigInt(actualNano) > BigInt(creditsToNanocredits(maxCredits))) actualNano = creditsToNanocredits(maxCredits);
      }
      const result = await this.credits.settle(creditsVerified, actualNano, { description: `mpp ${endpoint.method} ${endpoint.path}`, resource: url });
      if (result.success && result.replayed) {
        // The payment was spent by an earlier request (S-297): a replay, answered nothing.
        this.log('mpp.replayed', { path: endpoint.path, method: 'robutler' });
        return this.challenge(endpoint, request, undefined, { problem: 'invalid-challenge', detail: 'the challenge was already used' });
      }
      if (!result.success) {
        this.log('mpp.settle_failed', { path: endpoint.path, method: 'robutler', reason: result.error });
        return this.challenge(endpoint, request, undefined, { problem: 'verification-failed', detail: result.error ?? 'the settle failed' });
      }
      const [name, value] = MppSeller.receiptHeader(mpp.creditsReceipt(CreditsScheme.settleKey(creditsVerified.nonceId)));
      headers.set(name, value);
      this.log('mpp.settled', { path: endpoint.path, method: 'robutler', amount: actualNano });
      return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
    }

    return this.challenge(endpoint, request, undefined, { problem: 'method-unsupported', detail: 'no way to settle this method' });
  }

  // ── Verify ────────────────────────────────────────────────────────────────

  private async admit(endpoint: PricedEndpoint, url: string, read: Exclude<ReadPaymentNonNull, { version: 0 }>, binding = ''): Promise<Admitted | { refusal: string }> {
    const { maxCredits, metered } = endpointPrice(endpoint.pricing);
    const amountNano = creditsToNanocredits(maxCredits);

    if (read.version === 2) {
      const accepted = read.payload.accepted;
      if (this.credits && accepted.scheme === this.credits.scheme && accepted.network === this.credits.network) {
        const offered = await this.credits.requirement(url, amountNano, binding);
        // The nonce is the client's echo; everything else must equal ours.
        const echoedNonce = accepted.extra?.nonce;
        if (!requirementMatches({ ...offered, extra: { ...offered.extra, nonce: echoedNonce } }, accepted)) {
          return { refusal: 'accepted requirement does not match an offer' };
        }
        const verified = await this.credits.verify(read.payload.payload, echoedNonce, url, amountNano, binding);
        if (!verified.ok) return { refusal: verified.reason };
        return { version: 2, scheme: 'credits', requirement: accepted, payload: read.payload, credits: verified, maxCredits };
      }
      const chain = this.config.chain;
      if (chain && accepted.network === chain.network) {
        const offered = (await this.offers(endpoint, url)).accepts.find((r) => r.network === chain.network);
        if (!offered || !requirementMatches(offered, accepted)) return { refusal: 'accepted requirement does not match an offer' };
        const verify = await chain.facilitator.verify(read.payload, accepted);
        if (!verify.isValid) return { refusal: verify.invalidReason ?? 'verification failed' };
        return { version: 2, scheme: 'chain', requirement: accepted, payload: read.payload, maxCredits };
      }
      return { refusal: 'unsupported scheme or network' };
    }

    // v1: the payload names its scheme and network; the requirement is rebuilt from the offer.
    const network = v1NetworkToCaip2(read.payload.network);
    if (this.credits && read.payload.scheme === this.credits.scheme && network === this.credits.network) {
      const verified = await this.credits.verify(read.payload.payload, read.payload.payload.nonce, url, amountNano, binding);
      if (!verified.ok) return { refusal: verified.reason };
      const requirement = await this.credits.requirement(url, amountNano, binding);
      return { version: 1, scheme: 'credits', requirement, payload: read.payload, credits: verified, maxCredits };
    }
    const chain = this.config.chain;
    if (chain && network === chain.network) {
      const requirement = (await this.offers(endpoint, url)).accepts.find((r) => r.network === chain.network);
      if (!requirement || requirement.scheme !== read.payload.scheme) return { refusal: 'unsupported scheme for this network' };
      const verify = await chain.facilitator.verify({ ...read.payload, network }, requirement);
      if (!verify.isValid) return { refusal: verify.invalidReason ?? 'verification failed' };
      return { version: 1, scheme: 'chain', requirement, payload: read.payload, maxCredits: metered ? maxCredits : maxCredits };
    }
    return { refusal: 'unsupported scheme or network' };
  }

  // ── Settle ────────────────────────────────────────────────────────────────

  private async settle(
    admitted: Admitted,
    endpoint: PricedEndpoint,
    url: string,
    overrides: { amount?: string; credits?: string } | null,
  ): Promise<SettleResponse> {
    const { credits: perCall, maxCredits, metered } = endpointPrice(endpoint.pricing);
    const description = `x402 ${endpoint.method} ${endpoint.path}`;

    if (admitted.scheme === 'credits' && admitted.credits?.ok && this.credits) {
      // The actual: a metered override in credits or nanocredits, clamped to
      // the authorized maximum; else the per-call price.
      let actualNano = creditsToNanocredits(metered ? maxCredits : perCall);
      if (metered && overrides) {
        if (overrides.credits !== undefined) actualNano = creditsToNanocredits(overrides.credits);
        else if (overrides.amount !== undefined) actualNano = overrides.amount;
        if (BigInt(actualNano) > BigInt(creditsToNanocredits(maxCredits))) actualNano = creditsToNanocredits(maxCredits);
      }
      const result = await this.credits.settle(admitted.credits, actualNano, { description, resource: url });
      // A settle the platform REPLAYED charged nothing now: an earlier
      // request spent this payment, so this one is a replay and fails (S-297).
      const success = result.success && !result.replayed;
      return {
        success,
        ...(success ? {} : { errorReason: result.replayed ? 'replayed' : result.error ?? 'settle_failed' }),
        transaction: '',
        network: admitted.requirement.network,
        amount: actualNano,
      };
    }

    const chain = this.config.chain!;
    let requirement = admitted.requirement;
    if (requirement.scheme === 'upto') {
      let actual = requirement.amount;
      if (overrides?.credits !== undefined) actual = creditsToAssetUnits(Number(overrides.credits), chain.decimals, chain.unitsPerCredit ?? 1);
      else if (isAmountString(overrides?.amount)) actual = overrides.amount;
      if (BigInt(actual) > BigInt(requirement.amount)) actual = requirement.amount;
      requirement = { ...requirement, amount: actual };
    }
    const payload = admitted.version === 1 ? { ...admitted.payload, network: chain.network } : admitted.payload;
    return chain.facilitator.settle(payload, requirement);
  }
}

type ReadPaymentNonNull = NonNullable<ReturnType<typeof readPaymentPayload>>;

// ── Helpers ──────────────────────────────────────────────────────────────────

function encodeBase64(value: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/** 503 when the platform could not verify a payment (B12; fixture `payments/final_sdk_x402_platform.json`). */
function platformUnreachable(): Response {
  return new Response(JSON.stringify({ error: { code: 'payment_platform_unreachable', message: PLATFORM_UNREACHABLE_MESSAGE } }), {
    status: 503,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Retry-After': '5' },
  });
}

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } });
}

/** The resource URL a challenge names and a nonce is bound to: the request URL without its query. */
export function resourceUrl(request: Request): string {
  const u = new URL(request.url);
  u.search = '';
  u.hash = '';
  return u.toString();
}

/**
 * The request binding a credits nonce covers (S-297): `METHOD|path|query|
 * sha256(body)`, the query without its `?` and empty when there is none,
 * the digest in lowercase hex (the empty body's for GET and HEAD). Read from
 * a clone of the request so the handler still reads its own body. The
 * fixture `python/tests/fixtures/payments/paywall_x402_mpp.json`
 * (`credits_nonce.bound_vectors`) pins the format for both SDKs.
 */
export async function requestBinding(request: Request): Promise<string> {
  const u = new URL(request.url);
  const method = request.method.toUpperCase();
  const body = method === 'GET' || method === 'HEAD' || request.body === null
    ? new Uint8Array(0)
    : new Uint8Array(await request.clone().arrayBuffer());
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', body)), (b) => b.toString(16).padStart(2, '0')).join('');
  return `${method}|${u.pathname}|${u.search.replace(/^\?/, '')}|${digest}`;
}

/** The paywall an agent's payment skill carries, or undefined when it has none. */
export function resolvePaywall(agent: unknown): Paywall | undefined {
  const skills = (agent as { skills?: unknown[] } | undefined)?.skills;
  if (!Array.isArray(skills)) return undefined;
  for (const skill of skills) {
    const paywall = (skill as { paywall?: unknown } | null)?.paywall;
    if (paywall instanceof Paywall) return paywall;
  }
  return undefined;
}

/** The generic refusal for a priced endpoint on an agent with no paywall (fixture `not_configured.message`). */
export const PAYMENT_NOT_CONFIGURED_MESSAGE = 'This endpoint is priced, and the agent has no payment skill to take a payment with.';

/**
 * Why the agent's payment skill cannot sell, when it says so: a skill that
 * carries a string `paywallRefusal` (`PaymentX402Skill`, 2026-09-27) rather
 * than a paywall names the seller to use. Undefined when none does.
 */
export function resolvePaywallRefusal(agent: unknown): string | undefined {
  const skills = (agent as { skills?: unknown[] } | undefined)?.skills;
  if (!Array.isArray(skills)) return undefined;
  for (const skill of skills) {
    const refusal = (skill as { paywallRefusal?: unknown } | null)?.paywallRefusal;
    if (typeof refusal === 'string' && refusal) return refusal;
  }
  return undefined;
}

/** A priced endpoint as the servers hand it over: the registry entry's own fields. */
export function pricedEndpointOf(endpoint: Pick<HttpEndpoint, 'path' | 'method' | 'pricing' | 'description' | 'discovery'>): PricedEndpoint | null {
  if (!endpoint.pricing) return null;
  return {
    path: endpoint.path,
    method: endpoint.method,
    pricing: endpoint.pricing,
    ...(endpoint.description ? { description: endpoint.description } : {}),
    ...(endpoint.discovery ? { discovery: endpoint.discovery } : {}),
  };
}

/**
 * Serve `endpoint` through the agent's paywall. A priced endpoint on an
 * agent with no paywall is refused with 503 rather than served free: the
 * price is the author's declared intent.
 */
export async function serveThroughPaywall(
  agent: unknown,
  endpoint: Pick<HttpEndpoint, 'path' | 'method' | 'pricing' | 'description' | 'discovery'>,
  request: Request,
  run: () => Promise<Response>,
): Promise<Response> {
  const priced = pricedEndpointOf(endpoint);
  if (!priced) return run();
  const paywall = resolvePaywall(agent);
  if (!paywall) {
    // A payment skill that cannot sell says which one can (`PaymentX402Skill`);
    // otherwise the generic sentence.
    const message = resolvePaywallRefusal(agent) ?? PAYMENT_NOT_CONFIGURED_MESSAGE;
    return json({ error: { code: 'payment_not_configured', message } }, 503);
  }
  return paywall.handle(priced, request, run);
}
