/**
 * The x402 facilitator client (webagents gap-closure plan 2.6, 2026-09-26).
 *
 * A chain scheme (`exact`, `upto`) is verified and settled by a facilitator
 * (spec pack section 1.6): `POST /verify` is read-only, `POST /settle` moves
 * the value, `GET /supported` lists the (scheme, network) kinds it serves.
 * The public one, `https://x402.org/facilitator`, takes no credentials and
 * serves TESTNETS only; Coinbase's CDP facilitator serves mainnets and wants
 * a per-request JWT minted from a CDP API key pair, bound to the method,
 * host and path of the call (`cdpAuthorizer`). Nothing here knows which one
 * it talks to: a URL, optional fixed headers and an optional per-request
 * authorizer are the whole configuration, and `facilitatorFromConfig` reads
 * them from config or the environment (`X402_FACILITATOR_URL`,
 * `CDP_API_KEY_ID`, `CDP_API_KEY_SECRET`).
 *
 * `FacilitatorClient` is an interface so a test can settle against a local
 * fake and a host can wire an in-process one; the reference server does the
 * same. The Python twin is `payments/x402_facilitator.py`.
 *
 * `settlement_pending` (section 1.11) is an answer, not an error: the
 * facilitator broadcast the transaction and has not seen it land. It is
 * handed back as it came, with its `transaction` hash, and the paywall never
 * settles the same payment twice on the strength of it.
 */

import type { PaymentPayload, PaymentRequirement, SettleResponse, VerifyResponse } from './x402-wire';
import { isRecord } from './x402-wire';

export const X402_ORG_FACILITATOR_URL = 'https://x402.org/facilitator';
export const CDP_FACILITATOR_URL = 'https://api.cdp.coinbase.com/platform/v2/x402';

export interface SupportedKind {
  x402Version: number;
  scheme: string;
  network: string;
}

export interface SupportedResponse {
  kinds: SupportedKind[];
  extensions?: unknown[];
  signers?: Record<string, string[]>;
}

export interface FacilitatorClient {
  verify(payload: PaymentPayload, requirement: PaymentRequirement): Promise<VerifyResponse>;
  settle(payload: PaymentPayload, requirement: PaymentRequirement): Promise<SettleResponse>;
  supported?(): Promise<SupportedResponse>;
}

/** Per-request authorization headers, for a facilitator that signs each call (CDP). */
export type FacilitatorAuthorizer = (method: string, url: string) => Promise<Record<string, string>>;

export interface HttpFacilitatorOptions {
  url: string;
  /** Fixed headers on every call (an API key a facilitator takes as a bearer). */
  headers?: Record<string, string>;
  authorize?: FacilitatorAuthorizer;
  fetch?: typeof fetch;
  timeoutMs?: number;
}

/**
 * A facilitator over HTTP. Errors from the facilitator itself (a 5xx, a body
 * that is not JSON) come back as a failed verify or settle with a reason
 * naming the status, never as a throw: the paywall answers the client the
 * same way for both, and a thrown fetch error would turn into a 500 that
 * says nothing.
 */
export class HttpFacilitatorClient implements FacilitatorClient {
  private readonly url: string;
  private readonly headers: Record<string, string>;
  private readonly authorize?: FacilitatorAuthorizer;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;

  constructor(options: HttpFacilitatorOptions) {
    this.url = options.url.replace(/\/+$/, '');
    this.headers = { ...(options.headers ?? {}) };
    this.authorize = options.authorize;
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  private async call(method: 'GET' | 'POST', path: string, body?: unknown): Promise<{ status: number; json: Record<string, unknown> | null }> {
    const url = `${this.url}${path}`;
    const headers: Record<string, string> = { Accept: 'application/json', ...this.headers };
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (this.authorize) Object.assign(headers, await this.authorize(method, url));
    const res = await this.fetchImpl(url, {
      method,
      headers,
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    const text = await res.text();
    let json: Record<string, unknown> | null = null;
    try {
      const parsed = JSON.parse(text) as unknown;
      json = isRecord(parsed) ? parsed : null;
    } catch {
      json = null;
    }
    return { status: res.status, json };
  }

  async verify(payload: PaymentPayload, requirement: PaymentRequirement): Promise<VerifyResponse> {
    const { status, json } = await this.call('POST', '/verify', {
      x402Version: payload.x402Version,
      paymentPayload: payload,
      paymentRequirements: requirement,
    });
    if (!json || typeof json.isValid !== 'boolean') {
      return { isValid: false, invalidReason: `facilitator_error:${status}` };
    }
    return json as unknown as VerifyResponse;
  }

  async settle(payload: PaymentPayload, requirement: PaymentRequirement): Promise<SettleResponse> {
    const { status, json } = await this.call('POST', '/settle', {
      x402Version: payload.x402Version,
      paymentPayload: payload,
      paymentRequirements: requirement,
    });
    if (!json || typeof json.success !== 'boolean') {
      return { success: false, errorReason: `facilitator_error:${status}`, transaction: '', network: requirement.network };
    }
    const out = json as unknown as SettleResponse;
    if (typeof out.transaction !== 'string') out.transaction = '';
    if (typeof out.network !== 'string') out.network = requirement.network;
    return out;
  }

  async supported(): Promise<SupportedResponse> {
    const { status, json } = await this.call('GET', '/supported');
    if (!json || !Array.isArray(json.kinds)) throw new Error(`x402 facilitator /supported answered ${status}`);
    return json as unknown as SupportedResponse;
  }
}

// ── CDP ──────────────────────────────────────────────────────────────────────

function base64url(bytes: Uint8Array | string): string {
  const buf = typeof bytes === 'string' ? new TextEncoder().encode(bytes) : bytes;
  let binary = '';
  for (let i = 0; i < buf.length; i += 1) binary += String.fromCharCode(buf[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** The PKCS#8 DER prefix for an Ed25519 private key (RFC 8410), followed by the 32-byte seed. */
const ED25519_PKCS8_PREFIX = Uint8Array.from([
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20,
]);

/**
 * A per-request CDP JWT: `ES256` for an EC key pair (the secret is a PEM),
 * `EdDSA` for an Ed25519 pair (the secret is base64 of seed plus public key,
 * as the CDP portal issues it). Claims: `sub` the key id, `iss` `cdp`, `aud`
 * `cdp_service`, a two-minute window, and `uris` naming exactly the call it
 * authorizes (`POST api.cdp.coinbase.com/platform/v2/x402/settle`), which is
 * what stops a captured token from paying for anything else. `node:crypto`
 * is imported lazily: this module is otherwise runtime-neutral, and a server
 * that never configures CDP never loads it.
 */
export function cdpAuthorizer(keyId: string, keySecret: string): FacilitatorAuthorizer {
  if (!keyId || !keySecret) throw new Error('x402: a CDP key id and secret are both required');
  return async (method, url) => {
    const crypto = await import('node:crypto');
    const u = new URL(url);
    const now = Math.floor(Date.now() / 1000);
    const isPem = keySecret.includes('-----BEGIN');
    const alg = isPem ? 'ES256' : 'EdDSA';
    const header = { alg, kid: keyId, typ: 'JWT', nonce: crypto.randomBytes(16).toString('hex') };
    const claims = {
      sub: keyId,
      iss: 'cdp',
      aud: ['cdp_service'],
      nbf: now,
      exp: now + 120,
      uris: [`${method.toUpperCase()} ${u.host}${u.pathname}`],
    };
    const signingInput = `${base64url(JSON.stringify(header))}.${base64url(JSON.stringify(claims))}`;
    let signature: Buffer;
    if (isPem) {
      const key = crypto.createPrivateKey(keySecret);
      signature = crypto.sign('sha256', Buffer.from(signingInput), { key, dsaEncoding: 'ieee-p1363' });
    } else {
      const raw = Buffer.from(keySecret, 'base64');
      if (raw.length !== 64 && raw.length !== 32) throw new Error('x402: a CDP Ed25519 secret is 32 or 64 base64 bytes');
      const der = Buffer.concat([Buffer.from(ED25519_PKCS8_PREFIX), raw.subarray(0, 32)]);
      const key = crypto.createPrivateKey({ key: der, format: 'der', type: 'pkcs8' });
      signature = crypto.sign(null, Buffer.from(signingInput), key);
    }
    return { Authorization: `Bearer ${signingInput}.${base64url(new Uint8Array(signature))}` };
  };
}

export interface FacilitatorConfig {
  /** Default: `https://x402.org/facilitator` (testnets only, no credentials). */
  url?: string;
  headers?: Record<string, string>;
  /** A CDP API key pair; `url` then defaults to the CDP facilitator. */
  cdp?: { keyId: string; keySecret: string };
  fetch?: typeof fetch;
  timeoutMs?: number;
}

/**
 * A client from config, with the environment as the fallback for each part:
 * `X402_FACILITATOR_URL`, and `CDP_API_KEY_ID` plus `CDP_API_KEY_SECRET` for
 * the CDP pair. With a pair and no URL the CDP facilitator is meant.
 */
export function facilitatorFromConfig(
  config: FacilitatorConfig = {},
  env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {},
): HttpFacilitatorClient {
  const keyId = config.cdp?.keyId ?? env.CDP_API_KEY_ID;
  const keySecret = config.cdp?.keySecret ?? env.CDP_API_KEY_SECRET;
  const cdp = keyId && keySecret ? { keyId, keySecret } : undefined;
  const url = config.url ?? env.X402_FACILITATOR_URL ?? (cdp ? CDP_FACILITATOR_URL : X402_ORG_FACILITATOR_URL);
  return new HttpFacilitatorClient({
    url,
    headers: config.headers,
    authorize: cdp ? cdpAuthorizer(cdp.keyId, cdp.keySecret) : undefined,
    fetch: config.fetch,
    timeoutMs: config.timeoutMs,
  });
}
