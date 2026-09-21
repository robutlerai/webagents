/**
 * The MPP buyer: how an agent built on this SDK buys platform usage from
 * Robutler when a paid resource answers 402 (machine-purchase design,
 * sections 2.1, 6.3 and 6.4; pass P9a, 2026-09-18).
 *
 * WHAT IT DOES. `payingFetch` sends a signed request; when the answer is a
 * 402 carrying one `WWW-Authenticate: Payment` challenge whose `realm` is
 * the host it asked, and that host is on the policy's `realms` allowlist,
 * it checks the operator's policy, obtains a payment from a source the
 * operator configured, builds the credential, adds the terms acceptance
 * and re-sends THE SAME request once with both headers among the
 * signature's covered components. A fresh challenge on a 402 that charged
 * nothing (an expired challenge, a stale terms version) is paid again, a
 * bounded number of times. Everything else is returned to the caller as it
 * came. `purchase` is the in-band variant: a UAMP `payment.required`
 * carries the challenge and a purchase URL, and the buyer pays it there
 * with the same loop, after which the UAMP client resumes on the socket.
 * `upgradeHeaders` signs that socket's upgrade, so the platform's socket
 * door can tell who is asking and send the in-band challenge at all.
 *
 * WHO IT PAYS (S-147, fixed 2026-09-18). A challenge names its own seller:
 * "realm equals the host asked" held for ANY host, so a delegate to a
 * model-chosen URL, or a UAMP peer naming its own purchase URL, could
 * collect the operator's card or stablecoin payment. `policy.realms` is
 * now required (it defaults to the host of `platformUrl` when that is
 * set, and construction throws with neither): a challenge is paid only
 * when its realm AND the host it was fetched from, or the purchase URL,
 * are on it. The check runs before any source call and before any held
 * credential is re-presented, so an off-list peer never sees either.
 *
 * WHAT THE ANSWER AFTER A CREDENTIAL MEANS (2026-09-18; the 503 contract
 * shared with the portal's door and purchase URL, read by body CODE and not
 * by the flag alone). SETTLED: a 2xx; any answer whose body carries a
 * `purchase` block (the grant happened: the door's 402 after a grant, its
 * 503 `funding_failed`, its 409 `purchase_already_granted` after a lost
 * 200); a 503 `funding_failed` even without the block. A settled answer
 * with `retry.sameCredential: false` means "send the call again WITHOUT a
 * credential", done once. RE-PRESENT: a 503 with `retry.sameCredential:
 * true` (`sale_serve_pending` among them), and nothing else: `Retry-After`
 * alone is not read as that any more. SERVED ELSEWHERE: 409
 * `call_already_served` or `call_in_progress` (a per-call sale served, or
 * being served, on an earlier presentation; the door never serves it
 * twice): paid, terminal, never paid again and never re-presented, since a
 * re-presentation can only answer 409 again. REFUNDED: 409 `call_refunded`:
 * the reservation is freed. UNKNOWN: any other 5xx, or an exception once
 * the credential left: money may have moved. REFUSED: anything else,
 * including a 503 whose `sameCredential: false` names no payment
 * (`mpp_not_configured`, `metered_not_ready`) and the velocity 429s; nothing
 * was charged, the reservation is freed, and a 429 or 503 is reported
 * through `onRefusal` as `platform_refused`, never retried by the buyer.
 *
 * NEVER DISCARDING A CREDENTIAL (S-150, fixed 2026-09-18). A re-present
 * answer is followed with backoff (`Retry-After`, at least 5 s doubling,
 * at most `maxRetryAfterSeconds`) until the challenge's expiry plus
 * `settlementGraceSeconds`, because the platform completes a broadcast
 * Tempo transfer by its hash even after expiry. When that runs out, or the
 * outcome is unknown, the credential is HELD, not dropped: it is exposed in
 * `onRefusal`, in `purchase`'s outcome and by `pendingCredentials()`, kept
 * counted in the daily ledger, saved through `policy.persist` when one is
 * given, and re-presented before this buyer pays again for the same
 * resource (a pack credential also at a purchase URL, which honours a pack
 * challenge from any door). Only a settled or a refused answer clears it.
 * It used to live only in a local variable of a loop that gave up after
 * about 15 s, so the next call signed a second transfer.
 *
 * WHICH SELLER IT PAYS (S-154 residual, fixed 2026-09-18). The realm
 * allowlist says which HOST may challenge; it cannot say who is paid, and
 * frames on an external agent's socket are relayed raw by the platform, so
 * an in-band challenge can read "realm = the platform" while naming an
 * attacker's Stripe profile (the SPT is issued to it before the platform
 * refuses the HMAC) or wallet. So a card challenge is paid only when its
 * `methodDetails.networkId` equals the pinned Robutler profile, and a Tempo
 * challenge only when its `recipient` equals the pinned deposit address,
 * else `seller_not_pinned` before any source call. The pin is
 * `policy.stripeProfileId` / `policy.tempoDepositAddress`, else read per
 * allowlisted host from `https://<host>/openapi.json` (unsigned, https only,
 * no redirects, 10 s, and 1 MB counted WHILE the body streams: the bound used
 * to be applied after `response.text()` had buffered all of it), believed for
 * an hour (until 2026-09-19 a positive pin was cached for the life of the
 * process, so a rotated deposit address was never re-read and every Tempo
 * challenge was refused until restart), from the `x-payment-info`
 * offers: `payTo` on the stripe offer, `recipient` on the tempo offer, the
 * names the platform's own `accepts` entries use. A document naming none,
 * or two different ones, pins nothing, and nothing is paid: fail closed. The
 * platform's discovery offers carry both fields (lib/openapi/mpp-discovery.ts
 * in the portal, since 2026-09-18), so no explicit pin is needed against it.
 *
 * ONE PURCHASE PER CALL (S-151, fixed 2026-09-18). A door that granted a
 * pack and still could not fund the call answers 402 WITH a `purchase`
 * block. That pack is PAID: it stays in the ledger and `onPurchase` fires
 * for it. A fresh challenge after a settled purchase is paid only while
 * `policy.maxPurchasesPerCall` (default 1) allows, and it is counted
 * against the daily cap like any other.
 *
 * NO REDIRECT IS EVER FOLLOWED (S-180, fixed 2026-09-19). Every request this
 * buyer sends, the first one, the paid retry and the retry without a
 * credential alike, is sent with `redirect: 'error'`, whatever the caller
 * passed. `purchase()` used to set `follow`, `send()` reused the seed's mode
 * and `payingFetch` inherited the caller's, which defaults to follow; fetch
 * strips only `Authorization` and `Cookie` on a cross-origin redirect, so
 * `Payment-Authorization` and `Robutler-Terms-Accepted` travelled to whatever
 * the `Location` named, and that host's 2xx was read as a settled purchase,
 * past the realm allowlist and the seller pin, which only ever saw the first
 * URL. A redirect is now a typed refusal, `redirect_refused`: `payingFetch`
 * throws `MppRedirectError` (there is no answer to return) and `purchase`
 * reports it as its outcome. It spends nothing: the reservation of a fresh
 * credential is released; a credential ALREADY in doubt stays held, since a
 * redirect says nothing about whether it settled. A runtime whose `fetch`
 * cannot say WHY it failed leaves the credential held as an unknown outcome,
 * the conservative reading, and a `fetch` that followed anyway is never read
 * as settled either. The Python twin refuses the same way.
 *
 * EVERY SEND IS BOUNDED (2026-09-19). `policy.purchaseTimeoutSeconds` (default
 * 60) is how long ANY request the buyer sends may wait for its answer to
 * begin, combined with the caller's own `AbortSignal` when there is one.
 * Until that day only the in-band purchase had it: `payingFetch` sent with no
 * timeout at all, so a caller without a signal hung forever holding its
 * reserved entry in the daily cap. The bound is on the wait for the response
 * HEAD, not on the whole exchange, which is how the Python twin's httpx
 * timeout behaves: a long streamed answer is the caller's to bound. The
 * bodies the buyer reads for itself (a 402's, and the answer to a credential)
 * are read under the same bound.
 *
 * A STREAMED ANSWER IS NEVER READ (finding sdk-1, fixed 2026-09-19). Whether
 * a purchase settled is decided from what arrives with the HEAD: the status,
 * and the `Payment-Receipt` header for the record. The buyer reads a body for
 * itself only when it is a small JSON document (`buyerReadsBody`: a JSON
 * content type, never more than `BUYER_BODY_MAX_BYTES` declared or counted,
 * and on a 2xx only with a declared length), which is every problem document
 * and purchase document the platform sends and nothing a caller streams. It
 * used to `clone().json()` every answer to a credential, so a streamed
 * completion after a paid retry was teed and read to its end before the
 * caller saw the first chunk, and buffered whole. A streamed 2xx is now
 * handed over untouched, as it arrived. The Python twin sends with
 * `stream=True` and reads by the same rule.
 *
 * THE PURCHASE POINTER (2026-09-19). A rail that has not verified its caller
 * cannot mint a challenge for it: a token holder whose token ran dry on the
 * `/llm` socket or on `POST /api/llm/chat/completions` is told where to buy
 * instead, an `mpp` entry in `requirements` that carries `purchase_url` and
 * NO `challenge` (the portal's lib/payments/purchase-pointer.ts). Both SDKs
 * ignored such an entry. `purchaseAt` acts on it: the buyer sends its own
 * signed `POST` to the purchase URL, is answered 402 with the one challenge
 * minted for ITS identity, and pays that under the whole policy above
 * (realm allowlist, seller pin, caps, Terms, held credentials first). Two
 * things are checked BEFORE that first request leaves, because a pointer,
 * unlike a challenge, carries no secret and anyone can write one: the
 * purchase URL's host AND the host that named it (`from`: the resource that
 * answered 402, or the socket) must both be on `policy.realms`, so a
 * delegate at a host the operator never named cannot make this agent buy,
 * and cannot make it send a signed request anywhere. `payingFetch` does the
 * same on an HTTP 402 whose JSON body carries `requirements` and whose
 * `WWW-Authenticate` carries no challenge, then re-sends the original request
 * once WITHOUT its payment token (`X-Payment-Token`, `X-PAYMENT`,
 * `?payment_token=`): the purchase funded the signing identity's balance,
 * and the platform's door serves from that balance only a signed request
 * that names no token (`machineDoorEligible`); re-sending the dry token could
 * only answer 402 again, a pack bought and nothing served. A purchase made
 * through a pointer counts against `maxPurchasesPerCall` and the daily cap
 * like any other, and `call` (any object the caller keeps for one call, as
 * the UAMP client does per response) carries the per-call count across
 * separate `purchase` and `purchaseAt` invocations.
 *
 * A POINTER IS FOLLOWED ONLY UNDER A DAILY CAP (2026-09-19, the same day). The
 * two host checks say where a pointer may come from and where it may lead;
 * they do not bound HOW MUCH it can make this agent buy. A pointer carries no
 * secret, so any peer reachable THROUGH an allowlisted host can write one: an
 * external agent's UAMP frames are relayed raw by the platform, so they arrive
 * from the platform's own host. Nobody but the pinned seller is paid and the
 * usage lands on this agent's OWN balance, but it is the operator's card or
 * wallet that is spent, on a peer's say-so. `maxPurchasesPerCall` bounds that
 * per call and `dailyCapCents` per day; with no daily cap configured the total
 * across calls had no bound at all, because a peer starts a new call whenever
 * it likes. So `purchaseAt`, and `payingFetch` for any purchase URL a 402 BODY
 * names, refuse with `pointer_needs_daily_cap` before anything is sent when
 * `policy.dailyCapCents` is not set. An entry in a 402 body that does carry a
 * challenge is held to the same rule: its purchase URL is still one the caller
 * did not choose, and no platform rail sends that form over HTTP (a verified
 * HTTP caller is challenged in `WWW-Authenticate`). An ordinary challenge, on
 * the URL the caller itself chose to fetch, and an in-band `purchase`, are
 * paid exactly as before, with or without a daily cap. The Python twin refuses
 * the same way.
 *
 * THE CREDENTIAL HEADER IS CHECKED WHEN THE CHALLENGE IS READ (2026-09-19). A
 * challenge names the field its credential goes in (`header`). A name the
 * signer refuses to cover (`content-digest`, `signature`, ...), one of the
 * buyer's own signed fields, or a framing field the HTTP client owns
 * (`content-length`, `connection`, ...) made the paid retry throw AFTER
 * `obtain()` had the card source issue a token or the wallet sign, and the
 * credential was then held against the daily cap although nothing was sent.
 * Such a challenge is now unreadable (`credentialHeaderRefusal`), so it is
 * refused before any source call. And a retry that fails BEFORE it is sent
 * (the signer refusing, for any reason) releases a fresh credential's
 * reservation instead of holding it.
 *
 * THE LEDGER is per process unless `policy.persist` is supplied. The daily
 * cap is checked and the amount reserved in one synchronous step before
 * any await (the Terms callback and the source are awaited after it), so
 * concurrent calls on one buyer, or a burst of UAMP `payment.required`
 * events, cannot all pass the check before any of them reserves. `persist`
 * carries the ledger and the held credentials across restarts; it does not
 * make two processes share one cap atomically, so replicas that share a
 * store should each be given their own share of the cap.
 *
 * WHAT IT NEVER HOLDS. The SDK holds no card and no key it was not given:
 * a card pays through `CardPaymentSource.getSpt`, which returns a Stripe
 * shared payment token issued to Robutler's network profile for exactly
 * the challenge amount, and a stablecoin pays through
 * `StablecoinPaymentSource.signTempoTransfer`, which returns a signed,
 * UNBROADCAST transaction; the platform broadcasts it. Both are
 * interfaces the operator implements; nothing here reads an environment
 * variable for a secret.
 *
 * WHAT THE POLICY DECIDES, and only the policy: `maxPerPurchaseCents` (no
 * single challenge above it is paid), `dailyCapCents` (a rolling 24 hour
 * sum of what was presented), `preferPurchase` (`pack` or `exact`, sent as
 * the signed `Robutler-Purchase` hint the door reads, design D2),
 * `methods` (the order the buyer will pay in, sent as the signed
 * `Robutler-Payment-Methods` hint), and `acceptTerms`: a pinned version,
 * or a callback the operator sets once. A challenge names the Terms
 * version it is minted under and the paid retry must carry
 * `Robutler-Terms-Accepted` equal to it, signed; if the operator's policy
 * does not accept that version, nothing is paid and the 402 is returned.
 * The platform records the assent against the agent row and the signing
 * key, so accepting is a legal act of the operator's, which is why it is
 * never defaulted here.
 *
 * THE CHALLENGE PARSER is a port of the platform's pure `challenge.ts`
 * (`parseWwwAuthenticatePayment`, JCS, base64url) with no Node dependency:
 * this file runs wherever the SDK runs. The credential is
 * `Payment <base64url of JCS {challenge, source?, payload}>`, the
 * challenge echoed field for field as parsed, which is what the platform
 * re-HMACs.
 *
 * Copy rule: every string here is machine-facing or operator-facing. The
 * buyer buys platform usage from Robutler; it never pays an agent or a
 * creator, and nothing here states a rate between usage and money.
 */

import { signMessage, signRequest, type SigningIdentity, type SignRequestOptions } from '../../crypto/http-signature';

// ---------------------------------------------------------------------------
// Wire constants (design sections 2.3 and 6.3)
// ---------------------------------------------------------------------------

/** The core spec's alternate credential field, which every platform challenge names in `header`. */
export const MPP_CREDENTIAL_HEADER = 'Payment-Authorization';
export const MPP_RECEIPT_HEADER = 'Payment-Receipt';
/** The paid retry's assent header; among the covered components on a signed request (design section 6.3). */
export const TERMS_ACCEPTED_HEADER = 'Robutler-Terms-Accepted';
/** The version every payment 402 names beside its `Link`. */
export const TERMS_VERSION_HEADER = 'Robutler-Terms-Version';
/** D2: the buyer's signed choice between a pack and an exact top-up. */
export const PURCHASE_HINT_HEADER = 'Robutler-Purchase';
/** Design 6.1: the buyer's signed method order, `tempo, stripe`, a comma and space between names. */
export const PAYMENT_METHODS_HINT_HEADER = 'Robutler-Payment-Methods';
export const MPP_PROBLEM_BASE = 'https://paymentauth.org/problems/';
export const ROBUTLER_PROBLEM_BASE = 'https://robutler.ai/problems/';
export const MPP_INTENT_CHARGE = 'charge';

export type MppMethod = 'stripe' | 'tempo';
export const MPP_METHODS: ReadonlyArray<MppMethod> = ['stripe', 'tempo'];
export type MppPurchaseKind = 'pack' | 'exact';

// ---------------------------------------------------------------------------
// RFC 8785 JCS and base64url, browser-safe
// ---------------------------------------------------------------------------

/**
 * RFC 8785 canonical JSON: keys sorted by UTF-16 code units, no whitespace,
 * `JSON.stringify` number and string forms. `undefined` members are dropped
 * and `undefined` array elements become `null`, as `JSON.stringify` does.
 * Byte-identical to the platform's `canonicalize`, which is what makes the
 * echoed challenge re-HMAC on the server.
 */
export function jcsCanonicalize(value: unknown): string {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('JCS: non-finite numbers are not JSON');
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map((v) => jcsCanonicalize(v === undefined ? null : v)).join(',')}]`;
  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj)
      .filter((k) => obj[k] !== undefined)
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${keys.map((k) => `${JSON.stringify(k)}:${jcsCanonicalize(obj[k])}`).join(',')}}`;
  }
  throw new Error(`JCS: cannot serialise a ${typeof value}`);
}

export function base64urlEncode(input: string | Uint8Array): string {
  const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** Strict: the base64url alphabet only, no padding, else null. */
export function base64urlDecode(input: string): Uint8Array | null {
  if (typeof input !== 'string' || !/^[A-Za-z0-9_-]+$/.test(input)) return null;
  const padded = input.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (input.length % 4)) % 4);
  let binary: string;
  try {
    binary = atob(padded);
  } catch {
    return null;
  }
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

function decodeJcsJson(b64: string | undefined): unknown {
  if (!b64) return undefined;
  const bytes = base64urlDecode(b64);
  if (!bytes) return undefined;
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return undefined;
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

// ---------------------------------------------------------------------------
// The challenge, as the platform's pure module defines it
// ---------------------------------------------------------------------------

/** The auth-params of one challenge in their wire (base64url) forms; the object the credential echoes. */
export interface MppChallengeFields {
  id: string;
  realm: string;
  method: string;
  intent: string;
  request: string;
  expires?: string;
  digest?: string;
  opaque?: string;
  header?: string;
}

/**
 * Parse the `Payment` challenge out of a `WWW-Authenticate` value. The
 * value may list other schemes before or after it (a Fetch `Headers` joins
 * repeated fields with `, `), so the scan starts at the `Payment` scheme
 * and stops at the first thing that is not an `auth-param`, which is where
 * another scheme would begin. Null when there is no `Payment` challenge or
 * it lacks a required parameter (`id`, `realm`, `method`, `intent`,
 * `request`). Values are unquoted and unescaped once, as the platform's
 * `parseWwwAuthenticatePayment` does.
 */
export function parseWwwAuthenticatePayment(header: string | null | undefined): MppChallengeFields | null {
  if (typeof header !== 'string') return null;
  const start = /(?:^|,)\s*Payment\s+/i.exec(header);
  if (!start) return null;
  const params = /([A-Za-z0-9_-]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,"]+))\s*(?:,\s*|$)/y;
  params.lastIndex = start.index + start[0].length;
  const out: Record<string, string> = {};
  let m: RegExpExecArray | null;
  while (params.lastIndex < header.length && (m = params.exec(header)) !== null) {
    out[m[1]] = m[2] !== undefined ? m[2].replace(/\\(.)/g, '$1') : m[3];
  }
  if (!out.id || !out.realm || !out.method || !out.intent || !out.request) return null;
  const fields: MppChallengeFields = {
    id: out.id,
    realm: out.realm,
    method: out.method,
    intent: out.intent,
    request: out.request,
  };
  if (out.expires !== undefined) fields.expires = out.expires;
  if (out.digest !== undefined) fields.digest = out.digest;
  if (out.opaque !== undefined) fields.opaque = out.opaque;
  if (out.header !== undefined) fields.header = out.header;
  return fields;
}

/**
 * The decoded `request` of a charge challenge. The Stripe method carries
 * `methodDetails.networkId` and `paymentMethodTypes`; the Tempo method
 * carries `recipient`, a token contract as `currency`, and `methodDetails
 * .chainId`, `memo` and `supportedModes` (design section 6.6). `amount` is
 * a decimal integer string in the method's unit: cents for Stripe, cents
 * times 10000 for Tempo.
 */
export interface MppChargeRequest {
  amount: string;
  currency: string;
  methodDetails: Record<string, unknown>;
  recipient?: string;
  externalId?: string;
}

export function decodeChallengeRequest(fields: Pick<MppChallengeFields, 'request'>): MppChargeRequest | null {
  const decoded = decodeJcsJson(fields.request);
  if (!isRecord(decoded)) return null;
  if (typeof decoded.amount !== 'string' || !/^\d+$/.test(decoded.amount)) return null;
  if (typeof decoded.currency !== 'string' || decoded.currency.length === 0) return null;
  if (!isRecord(decoded.methodDetails)) return null;
  const out: MppChargeRequest = { amount: decoded.amount, currency: decoded.currency, methodDetails: decoded.methodDetails };
  if (typeof decoded.recipient === 'string') out.recipient = decoded.recipient;
  if (typeof decoded.externalId === 'string') out.externalId = decoded.externalId;
  return out;
}

/** The Tempo unit: cents times 10000 (design section 6.6). */
export const TEMPO_UNITS_PER_CENT = 10_000;

/** A challenge's amount in whole cents whatever its method, or null when the unit does not divide. */
export function challengeAmountCents(method: string, request: Pick<MppChargeRequest, 'amount'>): number | null {
  const units = Number(request.amount);
  if (!Number.isSafeInteger(units) || units <= 0) return null;
  if (method === 'tempo') return units % TEMPO_UNITS_PER_CENT === 0 ? units / TEMPO_UNITS_PER_CENT : null;
  return units;
}

/** One challenge, parsed and decoded, as the policy and the sources see it. */
export interface MppChallenge {
  fields: MppChallengeFields;
  request: MppChargeRequest;
  /** Whole cents, whatever the method's own unit; null when unreadable. */
  amountCents: number | null;
  expiresAt: Date | null;
  /** The header the credential goes in: the challenge's `header` parameter, else `Authorization` (core spec). */
  credentialHeader: string;
}

// RFC 9110 section 5.1 token, lowercased: the field-name alphabet (the signer's own rule).
const CREDENTIAL_FIELD_NAME_RE = /^[a-z0-9!#$%&'*+\-.^_`|~]+$/;

/**
 * Field names a challenge may NOT put its credential in (file comment, "THE
 * CREDENTIAL HEADER IS CHECKED WHEN THE CHALLENGE IS READ"). Three groups,
 * identical in the Python twin: the fields the signer refuses to cover (the
 * signature's own, and `content-digest`, which the body rule covers); the
 * buyer's own signed fields, where the credential would overwrite one or be
 * overwritten by it; and message framing and connection management, which
 * the HTTP client owns and refuses (`content-length`) or silently rewrites
 * (`host`). `authorization` is NOT here: it is the core spec's own default
 * when a challenge names no `header`, and the signer covers it like any
 * other field.
 */
export const CREDENTIAL_HEADERS_REFUSED: ReadonlySet<string> = new Set([
  'content-digest',
  'signature-agent',
  'signature-input',
  'signature',
  'robutler-terms-accepted',
  'robutler-purchase',
  'robutler-payment-methods',
  'host',
  'content-length',
  'content-type',
  'content-encoding',
  'transfer-encoding',
  'connection',
  'keep-alive',
  'upgrade',
  'te',
  'trailer',
  'expect',
  'cookie',
  'proxy-authorization',
  'proxy-connection',
]);

/** Why a challenge's credential header cannot be used, or null when it can. */
export function credentialHeaderRefusal(name: string): string | null {
  const lower = name.trim().toLowerCase();
  if (!CREDENTIAL_FIELD_NAME_RE.test(lower)) return `${JSON.stringify(name)} is not a header field name`;
  if (CREDENTIAL_HEADERS_REFUSED.has(lower)) {
    return `${lower} is a field the signer, the buyer or the HTTP client owns; a credential cannot be sent in it`;
  }
  return null;
}

/**
 * Parse and decode a `WWW-Authenticate` value, or null when it carries no
 * readable `Payment` challenge. A challenge whose `header` parameter names a
 * field the credential cannot be sent in (`credentialHeaderRefusal`) is not
 * readable: it is refused HERE, before the policy reserves anything and
 * before any payment source is asked.
 */
export function readMppChallenge(header: string | null | undefined): MppChallenge | null {
  const fields = parseWwwAuthenticatePayment(header);
  if (!fields) return null;
  const request = decodeChallengeRequest(fields);
  if (!request) return null;
  const credentialHeader = fields.header && fields.header.trim() ? fields.header.trim() : 'Authorization';
  if (credentialHeaderRefusal(credentialHeader) !== null) return null;
  let expiresAt: Date | null = null;
  if (fields.expires) {
    const at = new Date(fields.expires);
    expiresAt = Number.isNaN(at.getTime()) ? null : at;
  }
  return {
    fields,
    request,
    amountCents: challengeAmountCents(fields.method, request),
    expiresAt,
    credentialHeader,
  };
}

// ---------------------------------------------------------------------------
// Credential and receipt
// ---------------------------------------------------------------------------

/**
 * `Payment <token68>`: base64url of JCS `{ challenge, source?, payload }`,
 * the challenge echoed exactly as parsed (absent optional fields stay
 * absent, so the platform's HMAC input is rebuilt from the same slots).
 */
export function encodeMppCredential(
  fields: MppChallengeFields,
  payload: Record<string, unknown>,
  source?: string,
): string {
  const challenge: Record<string, string> = {
    id: fields.id,
    realm: fields.realm,
    method: fields.method,
    intent: fields.intent,
    request: fields.request,
  };
  if (fields.expires !== undefined) challenge.expires = fields.expires;
  if (fields.digest !== undefined) challenge.digest = fields.digest;
  if (fields.opaque !== undefined) challenge.opaque = fields.opaque;
  if (fields.header !== undefined) challenge.header = fields.header;
  const body: Record<string, unknown> = { challenge, payload };
  if (source !== undefined) body.source = source;
  return `Payment ${base64urlEncode(jcsCanonicalize(body))}`;
}

/**
 * The Tempo credential payload around a signed, unbroadcast transaction.
 * Checked against the platform's verifier on 2026-09-18
 * (`verifyMppTempoCredential`, lib/payments/mpp/challenge.ts): pull mode is
 * `type: "transaction"` and `signature` is the hex type 0x76 transaction,
 * which `TEMPO_SIGNED_TRANSACTION_RE` below mirrors.
 */
export function tempoCredentialPayload(serializedTransaction: string): Record<string, unknown> {
  return { type: 'transaction', signature: serializedTransaction };
}

/** The platform's pull-mode shape: `0x76` (the Tempo transaction type) then whole bytes. */
export const TEMPO_SIGNED_TRANSACTION_RE = /^0x76(?:[0-9a-fA-F]{2})+$/;
const TEMPO_ADDRESS_RE = /^0x[0-9a-fA-F]{40}$/;
const TEMPO_PAYER_DID_RE = /^did:pkh:eip155:([1-9][0-9]*):(0x[0-9a-fA-F]{40})$/;

/**
 * The credential `source` for a Tempo payer: `did:pkh:eip155:<chainId>:<address>`,
 * the only form the platform's `parseTempoPayerDid` accepts
 * (lib/payments/mpp/tempo.ts). Both SDKs sent the bare address until
 * 2026-09-18; the platform refused it as `malformed-credential` and answered
 * a fresh challenge, so every stablecoin purchase from a wallet exposing an
 * address failed after the wallet had signed twice. A DID passes through
 * when it names the challenge's chain. Throws on anything else, and the
 * caller asks this BEFORE the wallet signs, so nothing is signed for a
 * credential the platform would refuse.
 */
export function tempoPayerDid(chainId: number, address: string): string {
  const did = TEMPO_PAYER_DID_RE.exec(address);
  if (did) {
    if (Number(did[1]) !== chainId) {
      throw new Error(`the stablecoin source's DID names chain ${did[1]} and the challenge names chain ${chainId}`);
    }
    return address;
  }
  if (!TEMPO_ADDRESS_RE.test(address)) {
    throw new Error('the stablecoin source address is neither a 0x address of 40 hex digits nor a did:pkh:eip155 DID');
  }
  return `did:pkh:eip155:${chainId}:${address}`;
}

export interface MppReceipt {
  status: string;
  method: string;
  timestamp: string;
  reference: string;
}

/** The `Payment-Receipt` value: base64url JCS `{status, method, timestamp, reference}`, or null. */
export function parsePaymentReceipt(value: string | null | undefined): MppReceipt | null {
  if (!value) return null;
  const decoded = decodeJcsJson(value.trim());
  if (!isRecord(decoded)) return null;
  const { status, method, timestamp, reference } = decoded;
  if (typeof status !== 'string' || typeof method !== 'string' || typeof timestamp !== 'string' || typeof reference !== 'string') return null;
  return { status, method, timestamp, reference };
}

// ---------------------------------------------------------------------------
// Sources and policy
// ---------------------------------------------------------------------------

export interface SptRequest {
  /** Robutler's Stripe Business Network Profile id, the challenge's `methodDetails.networkId`. */
  networkId: string;
  amountCents: number;
  currency: string;
  expiresAt: Date | null;
  challengeId: string;
  paymentMethodTypes: string[];
}

/**
 * A card source: returns a Stripe shared payment token (`spt_...`) issued to
 * `networkId` for exactly `amountCents`. In Stripe's sandbox that is
 * `test_helpers/shared_payment/granted_tokens`; a Link Agent Wallet source
 * has no documented API as of 2026-09-18 (design section 6.4, UNVERIFIED).
 */
export interface CardPaymentSource {
  getSpt(request: SptRequest): Promise<string>;
}

export interface TempoTransferRequest {
  chainId: number;
  /** The token contract the challenge names as `currency`. */
  currency: string;
  recipient: string;
  /** Token units as the challenge states them (cents times 10000). */
  amount: string;
  memo: string;
  validBefore: Date | null;
  challengeId: string;
}

/**
 * A stablecoin source: signs a Tempo `transferWithMemo` of exactly `amount`
 * to `recipient` with the challenge's memo and returns the serialized,
 * UNBROADCAST transaction (`0x76...`, the platform's pull-mode shape). The
 * platform broadcasts it (pull mode, design section 6.6). `address` is the
 * payer: a `0x` address of 40 hex digits, or already its DID. The
 * credential's `source` carries it as `did:pkh:eip155:<chainId>:<address>`
 * (`tempoPayerDid`) so the platform can check it against the transaction's
 * sender.
 */
export interface StablecoinPaymentSource {
  signTempoTransfer(request: TempoTransferRequest): Promise<string>;
  readonly address?: string;
}

export interface MppTermsRequest {
  version: string;
  url: string | null;
}

/**
 * A credential the buyer presented and cannot yet call settled or refused
 * (S-150): the platform asked for it again past the settlement budget, or
 * answered in a way that says nothing about whether money moved. Everything
 * an operator needs to re-present it by hand is here: send any request with
 * `httpMethod` to `url` (a door serves THAT request against the purchase),
 * carrying `headerName: value` and, when `termsVersion` is set,
 * `Robutler-Terms-Accepted: termsVersion`, both covered by the agent's
 * signature. The buyer does the same itself before it pays again. JSON-safe
 * and identical in shape in both SDKs, so one store serves either.
 */
export interface MppPendingCredential {
  /** The challenge id; also the id of its daily-ledger entry. */
  id: string;
  realm: string;
  httpMethod: string;
  url: string;
  headerName: string;
  value: string;
  paymentMethod: string;
  /** The challenge opaque's `kind` (`pack`, `topup`), null when unreadable. A pack is also redeemable at a purchase URL. */
  kind: string | null;
  amountCents: number;
  currency: string;
  termsVersion: string | null;
  expiresAt: string | null;
  /** When the buyer's own re-presentation within a call stops: the expiry (or first presentation plus 300 s) plus the grace. */
  deadline: string;
  /** The Tempo transfer hash the platform named in its 503, when it did. */
  transactionHash: string | null;
  heldAt: string;
  /** `settlement_retries_exhausted` or `settlement_outcome_unknown`. */
  reason: string;
}

/** What `policy.persist` saves and loads: the daily ledger and the held credentials. */
export interface MppBuyerState {
  version: 1;
  /** `at` is epoch milliseconds; `id` is the challenge id. */
  ledger: Array<{ id: string; at: number; cents: number }>;
  pending: MppPendingCredential[];
}

/**
 * Where the ledger and the held credentials outlive the process. `load` is
 * asked once, before the first purchase decision, and merged into what the
 * process already holds; `save` is handed the whole state after every
 * change, in order. A `save` that throws is logged and does not stop a
 * purchase; a `load` that throws does, since a cap without its history is
 * not the cap the operator set.
 */
export interface MppBuyerPersistence {
  load(): MppBuyerState | null | undefined | Promise<MppBuyerState | null | undefined>;
  save(state: MppBuyerState): void | Promise<void>;
}

export interface MppBuyerPolicy {
  /** No single challenge above this is paid. */
  maxPerPurchaseCents: number;
  /**
   * A rolling 24 hour cap on what is presented for payment. Absent means no
   * daily cap, and then NO PURCHASE POINTER IS FOLLOWED (`pointer_needs_daily_cap`;
   * file comment, "A POINTER IS FOLLOWED ONLY UNDER A DAILY CAP"): a challenge
   * on a URL the caller chose is still paid. Per process unless `persist` is set.
   */
  dailyCapCents?: number;
  /** D2: the signed `Robutler-Purchase` hint. Absent means the door's default (packs on cards, exact top-ups on stablecoin). */
  preferPurchase?: MppPurchaseKind;
  /**
   * The order the buyer pays in, sent as the signed `Robutler-Payment-Methods`
   * hint. Default: every method with a configured source, stablecoin first;
   * the hint is then sent only when exactly one source is configured.
   */
  methods?: readonly MppMethod[];
  /**
   * The Terms acceptance, the operator's act: a pinned version (paid only
   * when the challenge names exactly it) or a callback asked once per
   * version seen, which returns true to accept. There is no default.
   */
  acceptTerms: string | ((terms: MppTermsRequest) => boolean | Promise<boolean>);
  /**
   * S-147: the hosts this buyer pays, as `host`, `host:port` or a URL. A
   * challenge is paid only when its `realm` and the host of the URL it came
   * from (or the in-band purchase URL) are both here. Default: the host of
   * `MppBuyerConfig.platformUrl`; required when that is not set.
   */
  realms?: readonly string[];
  /** How many challenges one call may present before giving up. Default 2: the first and one fresh one. */
  maxChallenges?: number;
  /** S-151: how many purchases that SETTLED one call may make. Default 1: a second pack needs the operator to say so. */
  maxPurchasesPerCall?: number;
  /** An optional hard cap on re-presentations within one call. No default: the time budget alone bounds them. */
  maxSettlementRetries?: number;
  /** S-150: how long past the challenge's expiry the buyer keeps re-presenting within one call, in seconds. Default 120. */
  settlementGraceSeconds?: number;
  /** The longest `Retry-After` (and backoff step) honoured, in seconds. Default 30. */
  maxRetryAfterSeconds?: number;
  /**
   * How long EVERY request the buyer sends (`payingFetch` and the in-band
   * `purchase` alike) may wait for its answer to begin, in seconds, combined
   * with the caller's own signal when there is one. Default 60; the Python
   * twin uses the same. Not a bound on a streamed body the caller reads.
   */
  purchaseTimeoutSeconds?: number;
  /** Carries the ledger and the held credentials across restarts. Without it both are per process. */
  persist?: MppBuyerPersistence;
  /** S-154: Robutler's Stripe profile (`profile_...`); a card challenge naming another `networkId` is never paid. Default: read from discovery. */
  stripeProfileId?: string;
  /** S-154: Robutler's Tempo deposit address; a Tempo challenge naming another `recipient` is never paid. Default: read from discovery. */
  tempoDepositAddress?: string;
}

export type MppBuyerRefusalReason =
  | 'no_challenge'
  | 'realm_not_allowed'
  | 'realm_mismatch'
  | 'unsupported_intent'
  | 'amount_unreadable'
  | 'challenge_expired'
  | 'method_unavailable'
  | 'over_max_per_purchase'
  | 'over_daily_cap'
  | 'terms_refused'
  | 'challenges_exhausted'
  | 'purchase_limit_per_call'
  /** A purchase pointer is followed only when `policy.dailyCapCents` is set: with none, nothing bounds what a peer's pointers make this agent buy across calls. Nothing was sent. */
  | 'pointer_needs_daily_cap'
  | 'settlement_retries_exhausted'
  | 'settlement_outcome_unknown'
  | 'pending_credential_refused'
  /** A 429 or a no-payment 503 answered to a presented credential: nothing charged, not retried. */
  | 'platform_refused'
  /** S-154: the challenge's seller (card `networkId`, Tempo `recipient`) is not the pinned Robutler one, or none could be pinned. */
  | 'seller_not_pinned'
  /** S-180: the host answered a request with a redirect. Never followed; nothing spent. */
  | 'redirect_refused';

/** The redirect statuses of the Fetch standard. A 304 or a 300 is an answer, not a redirect. */
const REDIRECT_STATUSES: ReadonlySet<number> = new Set([301, 302, 303, 307, 308]);

/**
 * A request the buyer sent was answered with a redirect (S-180; file
 * comment, "NO REDIRECT IS EVER FOLLOWED"). Thrown by `payingFetch`, which
 * has no answer to return; `purchase` reports it as its outcome instead.
 * `status` and `location` are set when the runtime's `fetch` handed the 3xx
 * back, null when it could only say that it refused one. `followed` is true
 * only for a custom `fetch` that ignored `redirect: 'error'` and followed:
 * a credential on that request went to a host nobody named, so it is held as
 * an unknown outcome rather than released.
 */
export class MppRedirectError extends Error {
  readonly code = 'redirect_refused';
  readonly url: string;
  readonly status: number | null;
  readonly location: string | null;
  readonly followed: boolean;
  /** The credential that stays held because it was already in doubt when the redirect came (S-150); null when nothing is held. */
  pendingCredential: MppPendingCredential | null = null;

  constructor(init: { url: string; status?: number | null; location?: string | null; followed?: boolean; cause?: unknown }) {
    super(
      init.followed
        ? `the request to ${init.url} was redirected and the redirect was FOLLOWED by the configured fetch, against redirect: 'error'; its answer is not read`
        : `${init.url} answered with a redirect${init.status ? ` (${init.status})` : ''}${init.location ? ` to ${init.location}` : ''}; ` +
            'the buyer never follows one, because a followed request carries its payment credential to another host',
    );
    this.name = 'MppRedirectError';
    this.url = init.url;
    this.status = init.status ?? null;
    this.location = init.location ?? null;
    this.followed = init.followed ?? false;
    if (init.cause !== undefined) (this as { cause?: unknown }).cause = init.cause;
  }
}

/** True when a `fetch` rejection says it refused a redirect (undici: `fetch failed`, cause `unexpected redirect`). */
function isRedirectFailure(err: unknown): boolean {
  if (!(err instanceof Error) || err.name === 'AbortError' || err.name === 'TimeoutError') return false;
  const cause = (err as { cause?: unknown }).cause;
  return /redirect/i.test(`${err.message} ${cause instanceof Error ? cause.message : ''}`);
}

/**
 * Terminal answers to a presented credential that are not refusals
 * (2026-09-18, the portal's S-149): the per-call sale was served, is being
 * served, or was refunded, on an earlier presentation. Nothing is paid again.
 */
export type MppTerminalOutcome = 'call_already_served' | 'call_in_progress' | 'call_refunded';

export interface MppBuyerRefusal {
  reason: MppBuyerRefusalReason;
  url: string;
  challengeId: string | null;
  amountCents: number | null;
  method: string | null;
  termsVersion: string | null;
  detail: string;
  /** The credential the refusal is about when its outcome is in doubt (S-150): held and re-presentable, or just refused after doubt. */
  pendingCredential: MppPendingCredential | null;
}

export interface MppPurchaseRecord {
  url: string;
  challengeId: string;
  method: string;
  amountCents: number;
  currency: string;
  termsVersion: string | null;
  /** The `Payment-Receipt` of the answer, when one was attached. */
  receipt: MppReceipt | null;
  /** The answer's status: 200 as a rule, 402 or 503 when the door granted the pack and still could not fund the call (S-151). */
  status: number;
}

export interface MppBuyerConfig {
  /** The agent identity every request is signed with. The platform binds the challenge to it. */
  identity: SigningIdentity;
  policy: MppBuyerPolicy;
  sources: {
    card?: CardPaymentSource;
    stablecoin?: StablecoinPaymentSource;
  };
  /** The platform this buyer buys from (`https://robutler.ai`); its host is the default `policy.realms`. */
  platformUrl?: string;
  /** Signer options passed through (`form`, `lifetimeSeconds`, `allowHttp`). `coveredHeaders` here are covered on every request beside the buyer's own. */
  sign?: SignRequestOptions;
  fetch?: typeof globalThis.fetch;
  now?: () => Date;
  sleep?: (ms: number) => Promise<void>;
  onPurchase?: (record: MppPurchaseRecord) => void;
  onRefusal?: (refusal: MppBuyerRefusal) => void;
}

export type MppPurchaseOutcome =
  | { ok: true; status: number; response: Response; record: MppPurchaseRecord }
  | {
      ok: false;
      status: number | null;
      response: Response | null;
      /** `redirect_refused` comes with `status: null` and `response: null`: a redirect is never followed, so there is no answer. */
      reason: MppBuyerRefusalReason | MppTerminalOutcome | 'served_without_purchase' | 'unexpected_status';
      detail: string;
      /** S-150: the credential whose outcome is in doubt, for the operator to re-present; null when none is. */
      pendingCredential?: MppPendingCredential | null;
      /** The receipt a terminal 409 carries in its body (`call_already_served`, `call_in_progress`, `call_refunded`). */
      receipt?: MppReceipt | null;
    };

// ---------------------------------------------------------------------------
// The buyer
// ---------------------------------------------------------------------------

interface SeedRequest {
  method: string;
  url: string;
  headers: Headers;
  body: Uint8Array | null;
  /** The caller's own signal, combined with the buyer's timeout on every send. The caller's redirect mode is NOT kept: every send is `redirect: 'error'` (S-180). */
  signal: AbortSignal | null;
  /** How long each send may wait for its answer to begin: `policy.purchaseTimeoutSeconds`, on EVERY send. */
  timeoutMs: number;
}

/** A request built and signed, not yet sent; `deadline` aborts it when the buyer's timeout fires. */
interface PreparedSend {
  request: Request;
  deadline: AbortController;
}

interface Credential {
  headerName: string;
  value: string;
  challengeId: string;
  method: string;
  amountCents: number;
  currency: string;
  termsVersion: string | null;
  realm: string;
  kind: string | null;
  expiresAt: Date | null;
  /** Epoch ms: this call's re-presentation stops here. */
  deadline: number;
  transactionHash: string | null;
  /** The platform asked for it again, or it came out of the held set: its outcome has been in doubt. */
  inDoubt: boolean;
}

interface LoopResult {
  response: Response;
  /** The last credential sent, if any. */
  presented: Credential | null;
  /** The last credential that settled in this call, if any. */
  settled: Credential | null;
  refusal: MppBuyerRefusalReason | MppTerminalOutcome | null;
  detail: string;
  pending: MppPendingCredential | null;
  receipt: MppReceipt | null;
}

type AfterCredential = 'settled' | 'retry_same' | 'served_elsewhere' | 'refunded' | 'unknown' | 'refused';

const DEFAULT_MAX_CHALLENGES = 2;
const DEFAULT_MAX_PURCHASES_PER_CALL = 1;
const DEFAULT_MAX_RETRY_AFTER_SECONDS = 30;
const DEFAULT_RETRY_AFTER_SECONDS = 5;
const DEFAULT_SETTLEMENT_GRACE_SECONDS = 120;
/** A challenge with no `expires` (the platform always sends one) is given this long from its first presentation. */
const DEFAULT_SETTLEMENT_WINDOW_SECONDS = 300;
const DEFAULT_PURCHASE_TIMEOUT_SECONDS = 60;
const DAY_MS = 24 * 60 * 60 * 1000;
/** S-154 discovery: the document path, its size and time bounds, and how long a document naming no seller is believed. */
export const DISCOVERY_PATH = '/openapi.json';
const DISCOVERY_MAX_BYTES = 1024 * 1024;
const DISCOVERY_TIMEOUT_MS = 10_000;
const DISCOVERY_NEGATIVE_TTL_MS = 300_000;
/** How long a pin read from discovery is believed before it is read again: a rotated deposit address is picked up within the hour. */
export const DISCOVERY_PIN_TTL_MS = 3_600_000;

/** The Robutler seller a discovery document names, per method; null where it names none (or two). */
export interface MppSellerPins {
  stripeProfileId: string | null;
  tempoDepositAddress: string | null;
}

/**
 * The sellers a discovery document names (S-154): every operation's
 * `x-payment-info.offers`, `payTo` on a `stripe` offer and `recipient` (or
 * `extra.recipient`) on a `tempo` offer. Two different values for one method
 * pin nothing, so an ambiguous document can never widen who is paid.
 */
export function sellerPinsFromDiscovery(doc: unknown): MppSellerPins {
  const stripe = new Set<string>();
  const tempo = new Set<string>();
  const paths = isRecord(doc) && isRecord(doc.paths) ? doc.paths : {};
  for (const item of Object.values(paths)) {
    if (!isRecord(item)) continue;
    for (const op of Object.values(item)) {
      const info = isRecord(op) ? op['x-payment-info'] : undefined;
      const offers = isRecord(info) && Array.isArray(info.offers) ? info.offers : [];
      for (const offer of offers) {
        if (!isRecord(offer)) continue;
        if (offer.method === 'stripe' && typeof offer.payTo === 'string' && offer.payTo) stripe.add(offer.payTo);
        if (offer.method === 'tempo') {
          const extra = isRecord(offer.extra) ? offer.extra : {};
          const recipient = typeof offer.recipient === 'string' ? offer.recipient : typeof extra.recipient === 'string' ? extra.recipient : '';
          if (recipient) tempo.add(recipient.toLowerCase());
        }
      }
    }
  }
  return {
    stripeProfileId: stripe.size === 1 ? [...stripe][0] : null,
    tempoDepositAddress: tempo.size === 1 ? [...tempo][0] : null,
  };
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** `Retry-After` as seconds: a delta, or an HTTP date relative to `now`; `fallback` when absent or unreadable. */
export function retryAfterSeconds(value: string | null | undefined, now: Date, fallback = DEFAULT_RETRY_AFTER_SECONDS): number {
  if (!value) return fallback;
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) return Number(trimmed);
  const at = Date.parse(trimmed);
  if (Number.isNaN(at)) return fallback;
  return Math.max(0, Math.ceil((at - now.getTime()) / 1000));
}

/** The most the buyer reads of any answer for itself: a problem or purchase document is a few kilobytes. */
export const BUYER_BODY_MAX_BYTES = 64 * 1024;

/** `application/json` and every `application/*+json` (`application/problem+json`), parameters ignored. */
export function isJsonMediaType(contentType: string | null | undefined): boolean {
  const type = (contentType ?? '').split(';')[0].trim().toLowerCase();
  return /^application\/(?:[a-z0-9!#$&^_.+-]+\+)?json$/.test(type);
}

/**
 * Whether the buyer reads this answer's body for itself, decided from the
 * HEAD alone (file comment, "A STREAMED ANSWER IS NEVER READ"). Only a JSON
 * document, never one that DECLARES more than `BUYER_BODY_MAX_BYTES`, and a
 * 2xx only when it declares its length: a 2xx with no declared length is a
 * stream the caller is waiting on (a streamed completion), and it is handed
 * over untouched. `purchaseDocument` lifts that last rule for the answer of a
 * purchase URL the buyer itself called, which is a purchase document whatever
 * its framing and which no caller is streaming. An error answer needs no
 * declared length (the platform's own 402s and 503s are sent chunked); it is
 * counted while it is read instead.
 */
export function buyerReadsBody(response: Response, opts: { purchaseDocument?: boolean } = {}): boolean {
  if (response.body === null || !isJsonMediaType(response.headers.get('content-type'))) return false;
  const header = response.headers.get('content-length');
  const declared = header !== null && /^\d+$/.test(header.trim()) ? Number(header.trim()) : null;
  if (declared !== null && declared > BUYER_BODY_MAX_BYTES) return false;
  if (response.status >= 200 && response.status < 300) return declared !== null || opts.purchaseDocument === true;
  return true;
}

/**
 * The JSON body of an answer the caller still gets to read, or null. Read
 * only when `buyerReadsBody` says so, from a clone, counted WHILE it streams
 * (more than `BUYER_BODY_MAX_BYTES` is an unreadable body, and the clone is
 * cancelled, so nothing past the cap is buffered) and under `timeoutMs` (file
 * comment, "EVERY SEND IS BOUNDED"): a body that stalls is an unreadable
 * body, never a hang. The response itself is never aborted or consumed, so
 * the caller reads the whole answer whatever happened here.
 *
 * Until 2026-09-19 this was `response.clone().json()` on EVERY answer to a
 * credential (finding sdk-1): a streamed completion was teed and read to its
 * end before the caller saw its first byte, and when the timer won the race
 * the clone kept buffering the rest of the stream behind the caller's back.
 */
async function readSmallJson(
  response: Response,
  timeoutMs: number,
  opts: { purchaseDocument?: boolean } = {},
): Promise<Record<string, unknown> | null> {
  if (!buyerReadsBody(response, opts)) return null;
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    reader = response.clone().body?.getReader();
    if (!reader) return null;
    const source = reader;
    const read = (async (): Promise<Uint8Array | null> => {
      const chunks: Uint8Array[] = [];
      let size = 0;
      for (;;) {
        const { done, value } = await source.read();
        if (done) break;
        size += value.byteLength;
        if (size > BUYER_BODY_MAX_BYTES) return null;
        chunks.push(value);
      }
      const bytes = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return bytes;
    })();
    const bytes = await Promise.race([
      read,
      new Promise<null>((resolve) => {
        timer = setTimeout(() => resolve(null), timeoutMs);
      }),
    ]);
    if (bytes === null) {
      // Over the cap, or too slow: stop the clone, so the tee buffers no more.
      void source.cancel().catch(() => undefined);
      read.catch(() => undefined);
      return null;
    }
    const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes));
    return isRecord(parsed) ? parsed : null;
  } catch {
    void reader?.cancel().catch(() => undefined);
    return null;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/**
 * The `mpp` entry of a `requirements` member (a UAMP `payment.required`, or
 * the body of an HTTP 402 from a rail with no verified caller): the purchase
 * URL, the challenge when the sender had verified its caller and minted one,
 * and the Terms notice that came with it. An entry WITHOUT a `challenge`
 * member is the platform's purchase pointer (file comment, "THE PURCHASE
 * POINTER"). An entry whose `challenge` is present and not a non-empty string
 * is malformed, and is not read as a pointer.
 */
export interface MppRequirementEntry {
  purchaseUrl: string;
  challenge: string | null;
  terms: { url: string | null; version: string } | null;
}

export function mppRequirementOf(requirements: unknown): MppRequirementEntry | null {
  const schemes = isRecord(requirements) && Array.isArray(requirements.schemes) ? requirements.schemes : [];
  for (const entry of schemes) {
    if (!isRecord(entry) || entry.scheme !== 'mpp') continue;
    if (typeof entry.purchase_url !== 'string' || !entry.purchase_url.trim()) continue;
    const hasChallenge = entry.challenge !== undefined && entry.challenge !== null;
    if (hasChallenge && (typeof entry.challenge !== 'string' || !entry.challenge.trim())) continue;
    const terms = isRecord(entry.terms) && typeof entry.terms.version === 'string' && entry.terms.version.trim()
      ? { url: typeof entry.terms.url === 'string' ? entry.terms.url : null, version: entry.terms.version.trim() }
      : null;
    return { purchaseUrl: entry.purchase_url.trim(), challenge: hasChallenge ? (entry.challenge as string) : null, terms };
  }
  return null;
}

/** How a caller names a payment token; the machine door never serves a request that carries one (the portal's `presentedCredentials`). */
const PAYMENT_TOKEN_HEADERS = ['x-payment-token', 'x-payment'] as const;
const PAYMENT_TOKEN_QUERY = 'payment_token';

/** The terms the 402 names: the `Robutler-Terms-Version` header first, the problem body's `terms` object second. */
export function readTermsNotice(response: Response, body: Record<string, unknown> | null): MppTermsRequest | null {
  const headerVersion = response.headers.get(TERMS_VERSION_HEADER)?.trim();
  const terms = body && isRecord(body.terms) ? body.terms : null;
  const version = headerVersion || (terms && typeof terms.version === 'string' ? terms.version.trim() : '');
  if (!version) return null;
  let url: string | null = terms && typeof terms.url === 'string' ? terms.url : null;
  if (!url) {
    const link = response.headers.get('link');
    const m = link ? /<([^>]+)>\s*;[^,]*rel="?terms-of-service"?/i.exec(link) : null;
    if (m) url = m[1];
  }
  return { version, url };
}

/** `retry.sameCredential` of a problem body: true, false, or null when the body says neither. */
export function sameCredentialOf(body: Record<string, unknown> | null): boolean | null {
  if (!body || !isRecord(body.retry)) return null;
  const same = body.retry.sameCredential;
  return typeof same === 'boolean' ? same : null;
}

/** The problem body's `error` code, or null. */
export function problemCodeOf(body: Record<string, unknown> | null): string | null {
  return body && typeof body.error === 'string' ? body.error : null;
}

/**
 * The receipt of an answer: the `Payment-Receipt` header on a success, else
 * the `receipt` a 409 carries in its body (an object, or the header form),
 * since the platform never attaches the header to an error.
 */
export function receiptOf(response: Response, body: Record<string, unknown> | null): MppReceipt | null {
  const fromHeader = parsePaymentReceipt(response.headers.get(MPP_RECEIPT_HEADER));
  if (fromHeader || !body) return fromHeader;
  const r = body.receipt;
  if (typeof r === 'string') return parsePaymentReceipt(r);
  if (isRecord(r) && typeof r.status === 'string' && typeof r.method === 'string' && typeof r.timestamp === 'string' && typeof r.reference === 'string') {
    return { status: r.status, method: r.method, timestamp: r.timestamp, reference: r.reference };
  }
  return null;
}

/**
 * The file comment's readings of an answer to a request that carried a
 * credential. By CODE as well as flag (2026-09-18): `sameCredential: false`
 * on a 503 is a paid purchase only for `funding_failed` (or a body with a
 * `purchase` block); `mpp_not_configured` and `metered_not_ready` carry the
 * same flag and charged nothing.
 */
export function afterCredential(status: number, body: Record<string, unknown> | null): AfterCredential {
  if (status >= 200 && status < 300) return 'settled';
  if (body && isRecord(body.purchase)) return 'settled';
  const code = problemCodeOf(body);
  const same = sameCredentialOf(body);
  if (status === 503 && same === true) return 'retry_same';
  if (status === 503 && code === 'funding_failed') return 'settled';
  if (status === 409 && (code === 'call_already_served' || code === 'call_in_progress')) return 'served_elsewhere';
  if (status === 409 && code === 'call_refunded') return 'refunded';
  if (status === 503 && same === false) return 'refused';
  if (status >= 500) return 'unknown';
  return 'refused';
}

/** One allowlist entry: the exact `host` a URL must have, and the `hostname` a realm may be. */
interface RealmEntry {
  host: string;
  hostname: string;
}

/** A `policy.realms` entry (`host`, `host:port` or a URL) as WHATWG spells its host and hostname. */
export function parseRealmEntry(entry: string): RealmEntry {
  if (typeof entry !== 'string' || !entry.trim()) throw new Error('policy.realms entries must be non-empty hosts');
  const raw = entry.trim();
  let url: URL;
  try {
    url = new URL(raw.includes('://') ? raw : `https://${raw}`);
  } catch {
    throw new Error(`policy.realms entry ${JSON.stringify(entry)} is not a host`);
  }
  if (!raw.includes('://') && (url.pathname !== '/' || url.search || url.hash || url.username || url.password || raw.includes('/'))) {
    throw new Error(`policy.realms entry ${JSON.stringify(entry)} is not a host`);
  }
  return { host: url.host.toLowerCase(), hostname: url.hostname.toLowerCase() };
}

/** `METHOD origin+path`: what one door is, whatever the query or body. */
function resourceKey(method: string, url: string): string {
  const u = new URL(url);
  return `${method.toUpperCase()} ${u.origin}${u.pathname}`;
}

/** The challenge opaque's `kind`; an opaque minted before kinds existed is a pack. Null when unreadable. */
function opaqueKind(fields: MppChallengeFields): string | null {
  const decoded = decodeJcsJson(fields.opaque);
  if (!isRecord(decoded)) return null;
  return typeof decoded.kind === 'string' ? decoded.kind : 'pack';
}

function isPendingRecord(v: unknown): v is MppPendingCredential {
  if (!isRecord(v)) return false;
  for (const k of ['id', 'realm', 'httpMethod', 'url', 'headerName', 'value', 'paymentMethod', 'currency', 'deadline', 'heldAt', 'reason']) {
    if (typeof v[k] !== 'string' || !(v[k] as string)) return false;
  }
  for (const k of ['kind', 'termsVersion', 'expiresAt', 'transactionHash']) {
    if (v[k] !== null && v[k] !== undefined && typeof v[k] !== 'string') return false;
  }
  return Number.isSafeInteger(v.amountCents) && (v.amountCents as number) > 0 && !Number.isNaN(Date.parse(v.deadline as string));
}

export class MppBuyer {
  private readonly config: MppBuyerConfig;
  private readonly realms: readonly RealmEntry[];
  private ledger: Array<{ id: string; at: number; cents: number }> = [];
  private readonly held = new Map<string, MppPendingCredential>();
  /** Held ids a call is re-presenting right now, so a concurrent call does not present the same one. */
  private readonly claimed = new Set<string>();
  private loaded: Promise<void> | null = null;
  private saving: Promise<void> = Promise.resolve();
  /** Challenge ids `onPurchase` already fired for: a lost 200 answered later by a 409 is still ONE purchase. */
  private readonly reported = new Set<string>();
  /** S-154: seller pins read from discovery, per allowlisted host; re-read after an hour, or after five minutes when the document names no seller for the method asked. */
  private readonly discovered = new Map<string, { pins: MppSellerPins; at: number } | Promise<MppSellerPins>>();
  /** Purchases that settled per `call` key, so `maxPurchasesPerCall` holds across separate `purchase`/`purchaseAt` invocations of one call. */
  private readonly calls = new WeakMap<object, number>();

  constructor(config: MppBuyerConfig) {
    if (!config.identity) throw new Error('MppBuyer needs the agent identity it signs with');
    if (!config.policy || typeof config.policy !== 'object') throw new Error('MppBuyer needs a policy');
    const { maxPerPurchaseCents, dailyCapCents, acceptTerms } = config.policy;
    if (!Number.isSafeInteger(maxPerPurchaseCents) || maxPerPurchaseCents <= 0) {
      throw new Error('policy.maxPerPurchaseCents must be a positive whole number of cents');
    }
    if (dailyCapCents !== undefined && (!Number.isSafeInteger(dailyCapCents) || dailyCapCents <= 0)) {
      throw new Error('policy.dailyCapCents must be a positive whole number of cents when set');
    }
    if (typeof acceptTerms !== 'string' && typeof acceptTerms !== 'function') {
      throw new Error('policy.acceptTerms must be a pinned Terms version or a callback; the SDK never accepts the Terms on its own');
    }
    if (typeof acceptTerms === 'string' && acceptTerms.trim().length === 0) {
      throw new Error('policy.acceptTerms cannot be an empty version');
    }
    for (const method of config.policy.methods ?? []) {
      if (!MPP_METHODS.includes(method)) throw new Error(`policy.methods names an unknown method ${String(method)}`);
    }
    const positive: Array<[string, number | undefined, boolean]> = [
      ['maxPurchasesPerCall', config.policy.maxPurchasesPerCall, false],
      ['purchaseTimeoutSeconds', config.policy.purchaseTimeoutSeconds, false],
      ['maxSettlementRetries', config.policy.maxSettlementRetries, true],
      ['settlementGraceSeconds', config.policy.settlementGraceSeconds, true],
    ];
    for (const [name, value, zeroOk] of positive) {
      if (value !== undefined && (!Number.isSafeInteger(value) || value < (zeroOk ? 0 : 1))) {
        throw new Error(`policy.${name} must be a whole number${zeroOk ? '' : ' above zero'} when set`);
      }
    }
    if (!config.sources.card && !config.sources.stablecoin) {
      throw new Error('MppBuyer needs at least one payment source (sources.card or sources.stablecoin)');
    }
    const entries = config.policy.realms ?? (config.platformUrl ? [config.platformUrl] : null);
    if (!entries || entries.length === 0) {
      throw new Error(
        'policy.realms must name the hosts this buyer pays (S-147: a challenge names its own seller), ' +
          'or set platformUrl to default it to the platform host',
      );
    }
    this.realms = entries.map(parseRealmEntry);
    this.config = config;
  }

  /** The methods this buyer will pay with, in order: the policy's order, else stablecoin first, each only with a source. */
  methodOrder(): MppMethod[] {
    const wanted = this.config.policy.methods ?? (['tempo', 'stripe'] as const);
    return wanted.filter((m) => this.hasSource(m));
  }

  /** True when `url`'s host is on the realm allowlist: the only hosts this buyer pays, or signs a socket upgrade for. */
  allowsUrl(url: string | URL): boolean {
    let u: URL;
    try {
      u = new URL(url);
    } catch {
      return false;
    }
    const host = u.host.toLowerCase();
    return this.realms.some((r) => r.host === host);
  }

  private realmAllowed(realm: string): boolean {
    const r = realm.toLowerCase();
    return this.realms.some((e) => e.hostname === r || e.host === r);
  }

  /** The credentials held because their outcome is in doubt (S-150), oldest first. */
  pendingCredentials(): MppPendingCredential[] {
    return [...this.held.values()].sort((a, b) => a.heldAt.localeCompare(b.heldAt)).map((p) => ({ ...p }));
  }

  private hasSource(method: string): boolean {
    if (method === 'stripe') return !!this.config.sources.card;
    if (method === 'tempo') return !!this.config.sources.stablecoin;
    return false;
  }

  private now(): Date {
    return this.config.now ? this.config.now() : new Date();
  }

  /** Cents presented for payment in the last 24 hours, in this process (and what `persist` loaded). */
  spentTodayCents(): number {
    const cutoff = this.now().getTime() - DAY_MS;
    let sum = 0;
    for (const entry of this.ledger) if (entry.at > cutoff) sum += entry.cents;
    return sum;
  }

  private reserve(id: string, cents: number): void {
    const now = this.now().getTime();
    this.ledger = this.ledger.filter((e) => e.at > now - DAY_MS && e.id !== id);
    this.ledger.push({ id, at: now, cents });
  }

  private release(id: string): void {
    this.ledger = this.ledger.filter((e) => e.id !== id);
  }

  // -- persistence ----------------------------------------------------------

  private ready(): Promise<void> {
    if (!this.loaded) {
      this.loaded = this.load().catch((err: unknown) => {
        this.loaded = null;
        throw err;
      });
    }
    return this.loaded;
  }

  private async load(): Promise<void> {
    const persist = this.config.policy.persist;
    if (!persist) return;
    const state = await persist.load();
    if (!state || !isRecord(state)) return;
    const known = new Set(this.ledger.map((e) => e.id));
    for (const e of Array.isArray(state.ledger) ? state.ledger : []) {
      if (isRecord(e) && typeof e.id === 'string' && Number.isFinite(e.at) && Number.isSafeInteger(e.cents) && !known.has(e.id)) {
        this.ledger.push({ id: e.id, at: e.at as number, cents: e.cents as number });
        known.add(e.id);
      }
    }
    this.ledger.sort((a, b) => a.at - b.at);
    for (const p of Array.isArray(state.pending) ? state.pending : []) {
      if (isPendingRecord(p) && !this.held.has(p.id)) this.held.set(p.id, { ...p });
    }
  }

  private snapshot(): MppBuyerState {
    return { version: 1, ledger: this.ledger.map((e) => ({ ...e })), pending: this.pendingCredentials() };
  }

  /** Hand the state to `persist.save`, in order; the snapshot is taken now, synchronously. */
  private save(): Promise<void> {
    const persist = this.config.policy.persist;
    if (!persist) return Promise.resolve();
    const state = this.snapshot();
    this.saving = this.saving
      .then(() => persist.save(state))
      .catch((err: unknown) => {
        console.warn(`[mpp-buyer] policy.persist.save failed: ${(err as Error)?.message ?? String(err)}`);
      });
    return this.saving;
  }

  private pendingOf(credential: Credential, seed: SeedRequest, reason: string): MppPendingCredential {
    const existing = this.held.get(credential.challengeId);
    return {
      id: credential.challengeId,
      realm: credential.realm,
      httpMethod: existing?.httpMethod ?? seed.method.toUpperCase(),
      url: existing?.url ?? seed.url,
      headerName: credential.headerName,
      value: credential.value,
      paymentMethod: credential.method,
      kind: credential.kind,
      amountCents: credential.amountCents,
      currency: credential.currency,
      termsVersion: credential.termsVersion,
      expiresAt: credential.expiresAt ? credential.expiresAt.toISOString() : null,
      deadline: new Date(credential.deadline).toISOString(),
      transactionHash: credential.transactionHash,
      heldAt: existing?.heldAt ?? this.now().toISOString(),
      reason,
    };
  }

  private hold(credential: Credential, seed: SeedRequest, reason: string): MppPendingCredential {
    const record = this.pendingOf(credential, seed, reason);
    this.held.set(record.id, record);
    return { ...record };
  }

  /** The oldest held credential this call may re-present on `seed`: the same resource, or a pack at a purchase URL. */
  private heldFor(seed: SeedRequest, realm: string, inBand: boolean): MppPendingCredential | null {
    const key = resourceKey(seed.method, seed.url);
    const r = realm.toLowerCase();
    for (const record of this.pendingCredentials()) {
      if (this.claimed.has(record.id) || record.realm.toLowerCase() !== r) continue;
      if (resourceKey(record.httpMethod, record.url) === key || (inBand && record.kind === 'pack')) return record;
    }
    return null;
  }

  private credentialOfHeld(record: MppPendingCredential): Credential {
    return {
      headerName: record.headerName,
      value: record.value,
      challengeId: record.id,
      method: record.paymentMethod,
      amountCents: record.amountCents,
      currency: record.currency,
      termsVersion: record.termsVersion,
      realm: record.realm,
      kind: record.kind,
      expiresAt: record.expiresAt ? new Date(record.expiresAt) : null,
      deadline: Date.parse(record.deadline),
      transactionHash: record.transactionHash,
      inDoubt: true,
    };
  }

  /**
   * The signed hint headers, by name; covered on every request the buyer
   * sends. The door mints ONE challenge, in the first method of its own
   * order the hint allows (design 6.1), and that order is stablecoin first.
   * A card-only buyer sending no hint met a Tempo challenge whenever
   * stablecoin was on and refused it as `method_unavailable` (review,
   * 2026-09-18), so a buyer that can pay only one way always says so; one
   * that can pay both says nothing unless the policy orders them.
   */
  private hintHeaders(): Array<[string, string]> {
    const out: Array<[string, string]> = [];
    if (this.config.policy.preferPurchase) out.push([PURCHASE_HINT_HEADER, this.config.policy.preferPurchase]);
    const order = this.methodOrder();
    if (this.config.policy.methods || order.length === 1) out.push([PAYMENT_METHODS_HINT_HEADER, order.join(', ')]);
    return out;
  }

  /**
   * The headers that sign a UAMP socket's upgrade: a GET on the http(s)
   * form of `url` (`wss:` is signed as `https:`, same host, path and
   * query), which is exactly the request the platform's socket door
   * rebuilds from the upgrade's Host header and path
   * (`upgradeRequest`, lib/payments/machine-door-socket.ts) and verifies
   * before it hands the socket off in-process. Nothing signed the upgrade
   * until 2026-09-18, so the door never saw a signed agent and the in-band
   * purchase could never start (review, "the TS SDK buyer never reaches any
   * door"). The hints ride the upgrade, covered, as on every other request.
   * Empty for a host off the realm allowlist: the platform is the only
   * place this upgrade identifies the agent for payment.
   */
  async upgradeHeaders(url: string): Promise<Record<string, string>> {
    const target = new URL(url);
    if (target.protocol === 'wss:') target.protocol = 'https:';
    else if (target.protocol === 'ws:') target.protocol = 'http:';
    else if (target.protocol !== 'https:' && target.protocol !== 'http:') throw new Error(`cannot sign an upgrade to ${target.protocol} URL`);
    if (!this.allowsUrl(target)) return {};
    const headers = new Headers();
    const hints = this.hintHeaders();
    for (const [name, value] of hints) headers.set(name, value);
    const signed = await signMessage(
      this.config.identity,
      { method: 'GET', url: target.toString(), headers },
      { ...this.config.sign, coveredHeaders: hints.map(([name]) => name) },
    );
    const out: Record<string, string> = {};
    for (const [name, value] of hints) out[name] = value;
    for (const [name, value] of Object.entries(signed.headers)) if (value !== undefined) out[name] = value;
    return out;
  }

  /**
   * `fetch`, signed, that buys platform usage on a 402 and re-sends the
   * same request once (file comment). The returned `Response` is the last
   * answer: the resource on success, or the 402, 403 or 503 the loop
   * stopped at, unread, so a caller can read the problem body itself. A
   * credential whose outcome is in doubt is reported through `onRefusal`
   * and `pendingCredentials()`.
   */
  async payingFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
    const seed = await captureRequest(input, init, this.timeoutMs());
    const result = await this.loop(seed, null, false, {});
    return result.response;
  }

  /** `policy.purchaseTimeoutSeconds` in milliseconds: the bound on every send (file comment, "EVERY SEND IS BOUNDED"). */
  private timeoutMs(): number {
    return (this.config.policy.purchaseTimeoutSeconds ?? DEFAULT_PURCHASE_TIMEOUT_SECONDS) * 1000;
  }

  /**
   * Pay a challenge received out of band (a UAMP `payment.required` with
   * scheme `mpp`) at `url`, the purchase URL the same message named, by
   * `POST` with no body. The challenge's origin admits the purchase URL
   * (design section 6.1), so the credential redeems there. The purchase
   * URL's host must be on the realm allowlist (S-147), and a held pack
   * credential is re-presented here before a new one is paid (S-150).
   */
  async purchase(request: {
    url: string;
    challenge: string;
    terms?: { url?: string | null; version?: string | null } | null;
    method?: string;
    headers?: HeadersInit;
    /** Any object kept for ONE call (the UAMP client keeps one per response): `maxPurchasesPerCall` then holds across every purchase made for it. */
    call?: object;
  }): Promise<MppPurchaseOutcome> {
    const challenge = readMppChallenge(request.challenge);
    if (!challenge) {
      return { ok: false, status: null, response: null, reason: 'no_challenge', detail: 'the challenge is not a readable Payment challenge', pendingCredential: null };
    }
    const version = request.terms?.version?.trim();
    const terms: MppTermsRequest | null = version ? { version, url: request.terms?.url ?? null } : null;
    return this.buy(this.purchaseSeed(request), { challenge, terms }, request.call ?? {});
  }

  /**
   * Act on a PURCHASE POINTER (file comment): an `mpp` requirement that names
   * `url`, the purchase URL, and carries no challenge, because its sender had
   * not verified who was asking. The buyer asks the purchase URL itself, by a
   * signed `POST` with no body, is challenged there as its own identity, and
   * pays that challenge under the same policy as any other. `from` is the URL
   * of whatever named the pointer (the resource that answered 402, or the
   * socket; `wss:` is read as `https:`). NOTHING is sent unless the hosts of
   * BOTH `url` and `from` are on `policy.realms` (S-147): a pointer carries
   * no secret, so any peer can write one, and an off-list peer must not be
   * able to make this agent buy or aim its signed request at a host. And
   * nothing is sent unless `policy.dailyCapCents` is set
   * (`pointer_needs_daily_cap`): a peer BEHIND an allowlisted host can write a
   * pointer too, and without a daily cap the total it could make this agent
   * buy across calls has no bound.
   */
  async purchaseAt(request: {
    url: string;
    from: string;
    method?: string;
    headers?: HeadersInit;
    /** As on `purchase`. */
    call?: object;
  }): Promise<MppPurchaseOutcome> {
    const call = request.call ?? {};
    const refusal = this.pointerVerdict(request.url, request.from, call);
    if (refusal) {
      this.refuse(request.url, null, null, refusal.reason, refusal.detail);
      return { ok: false, status: null, response: null, reason: refusal.reason, detail: refusal.detail, pendingCredential: null };
    }
    return this.buy(this.purchaseSeed(request), null, call);
  }

  private purchaseSeed(request: { url: string; method?: string; headers?: HeadersInit }): SeedRequest {
    return {
      method: request.method ?? 'POST',
      url: request.url,
      headers: new Headers(request.headers),
      body: null,
      signal: null,
      timeoutMs: this.timeoutMs(),
    };
  }

  /**
   * What is checked before a pointer is followed, all of it before any
   * request leaves: both hosts on the allowlist (S-147); a configured daily
   * cap (file comment, "A POINTER IS FOLLOWED ONLY UNDER A DAILY CAP"); and
   * room left under `maxPurchasesPerCall`, so a call that already bought is
   * not even asked for another challenge.
   */
  private pointerVerdict(url: string, from: string, call: object): { reason: MppBuyerRefusalReason; detail: string } | null {
    let named: URL | null = null;
    try {
      named = new URL(from);
      if (named.protocol === 'wss:') named.protocol = 'https:';
      else if (named.protocol === 'ws:') named.protocol = 'http:';
    } catch {
      named = null;
    }
    if (!this.allowsUrl(url) || !named || !this.allowsUrl(named)) {
      return {
        reason: 'realm_not_allowed',
        detail: `the purchase pointer to ${url}, named by ${from}, is not followed: both hosts must be on policy.realms; nothing was sent`,
      };
    }
    // The ceiling. `maxPurchasesPerCall` below bounds one call, and a peer
    // behind an allowlisted host starts as many calls as it likes: only the
    // daily cap bounds the total, so without one no pointer is followed.
    if (this.config.policy.dailyCapCents === undefined) {
      return {
        reason: 'pointer_needs_daily_cap',
        detail:
          `the purchase pointer to ${url} is not followed: policy.dailyCapCents is not set. A pointer carries no secret, so any peer ` +
          'reachable through an allowlisted host can write one, and without a daily cap nothing bounds what such pointers make this agent ' +
          'buy across calls. Set policy.dailyCapCents to follow purchase pointers; a challenge on a URL the caller chose is paid as before. Nothing was sent',
      };
    }
    const max = this.config.policy.maxPurchasesPerCall ?? DEFAULT_MAX_PURCHASES_PER_CALL;
    const made = this.calls.get(call) ?? 0;
    if (made >= max) {
      return {
        reason: 'purchase_limit_per_call',
        detail: `${made} purchase${made === 1 ? '' : 's'} settled for this call and policy.maxPurchasesPerCall is ${max}; the purchase pointer would buy another`,
      };
    }
    return null;
  }

  /**
   * One purchase at a purchase URL, as an outcome: the loop with a challenge
   * already in hand (`initial`), or with none, in which case the first signed
   * request is what asks for it (`purchaseAt`).
   */
  private async buy(
    seed: SeedRequest,
    initial: { challenge: MppChallenge; terms: MppTermsRequest | null } | null,
    call: object,
  ): Promise<MppPurchaseOutcome> {
    let result: LoopResult;
    try {
      result = await this.loop(seed, initial, true, call);
    } catch (err) {
      // S-180: a redirect is a refusal with no answer. `onRefusal` has
      // already fired, and the ledger is already settled (`present`).
      if (!(err instanceof MppRedirectError)) throw err;
      return { ok: false, status: null, response: null, reason: 'redirect_refused', detail: err.message, pendingCredential: err.pendingCredential };
    }
    if (result.refusal) {
      return {
        ok: false,
        status: result.response.status,
        response: result.response,
        reason: result.refusal,
        detail: result.detail,
        pendingCredential: result.pending,
        receipt: result.receipt,
      };
    }
    if (!result.presented && !result.settled) {
      return { ok: false, status: result.response.status, response: result.response, reason: 'served_without_purchase', detail: 'the purchase URL answered without a challenge and nothing was bought', pendingCredential: null };
    }
    if (!result.settled || !result.response.ok) {
      return { ok: false, status: result.response.status, response: result.response, reason: 'unexpected_status', detail: result.detail, pendingCredential: result.pending };
    }
    return {
      ok: true,
      status: result.response.status,
      response: result.response,
      record: this.recordOf(seed.url, result.settled, result.response, await readSmallJson(result.response, seed.timeoutMs, { purchaseDocument: true })),
    };
  }

  private recordOf(url: string, credential: Credential, response: Response, body: Record<string, unknown> | null = null): MppPurchaseRecord {
    return {
      url,
      challengeId: credential.challengeId,
      method: credential.method,
      amountCents: credential.amountCents,
      currency: credential.currency,
      termsVersion: credential.termsVersion,
      receipt: receiptOf(response, body),
      status: response.status,
    };
  }

  /** `onPurchase`, at most once per challenge id however many answers say the purchase settled. */
  private reportPurchase(url: string, credential: Credential, response: Response, body: Record<string, unknown> | null): void {
    if (this.reported.has(credential.challengeId)) return;
    this.reported.add(credential.challengeId);
    if (this.reported.size > 1024) this.reported.delete(this.reported.values().next().value as string);
    this.config.onPurchase?.(this.recordOf(url, credential, response, body));
  }

  /**
   * The loop of the file comment. `initial` is a challenge already in hand
   * (in-band purchase); null means send the request first and read the
   * challenge off its 402. `inBand` says `seed` is a purchase URL the buyer
   * itself is calling, not a resource a caller asked for. `call` is where the
   * count of settled purchases lives, so it outlasts this one loop (file
   * comment, "THE PURCHASE POINTER").
   */
  private async loop(
    seedRequest: SeedRequest,
    initial: { challenge: MppChallenge; terms: MppTermsRequest | null } | null,
    inBand: boolean,
    call: object,
  ): Promise<LoopResult> {
    const policy = this.config.policy;
    const maxChallenges = policy.maxChallenges ?? DEFAULT_MAX_CHALLENGES;
    const maxPurchases = policy.maxPurchasesPerCall ?? DEFAULT_MAX_PURCHASES_PER_CALL;
    const maxRetryAfter = policy.maxRetryAfterSeconds ?? DEFAULT_MAX_RETRY_AFTER_SECONDS;
    const hints = this.hintHeaders();
    // Reassigned once at most: the re-send after a pointer purchase drops the payment token.
    let seed = seedRequest;
    const bodyOf = (answer: Response) => readSmallJson(answer, seedRequest.timeoutMs, { purchaseDocument: inBand });

    let credential: Credential | null = null;
    let presented: Credential | null = null;
    let settled: Credential | null = null;
    let challengesPaid = 0;
    let purchases = this.calls.get(call) ?? 0;
    let settlementRetries = 0;
    // The settlement budget is measured by the clock AND by what was slept,
    // so a frozen test clock still reaches the deadline.
    let budgetFrom = 0;
    let waitedMs = 0;
    let heldChecked = false;
    let retriedWithout = false;
    let pending = initial;
    let response: Response | null = null;
    const done = (
      refusal: MppBuyerRefusalReason | MppTerminalOutcome | null,
      detail: string,
      record: MppPendingCredential | null = null,
      receipt: MppReceipt | null = null,
    ): LoopResult => ({
      response: response ?? this.syntheticRefusal(),
      presented,
      settled,
      refusal,
      detail,
      pending: record,
      receipt,
    });

    if (!pending) response = await this.sendPlain(seed, hints);

    try {
      for (;;) {
        if (pending) {
          await this.ready();
          const realmRefusal = this.realmVerdict(seed.url, pending.challenge);
          if (realmRefusal) {
            this.refuse(seed.url, pending.challenge, pending.terms, realmRefusal.reason, realmRefusal.detail);
            return done(realmRefusal.reason, realmRefusal.detail);
          }
          // S-150: a held credential for this resource (or a pack one, at a
          // purchase URL) goes first. Paying the new challenge while it is in
          // doubt is how the second transfer used to be signed.
          if (!heldChecked) {
            heldChecked = true;
            const held = this.heldFor(seed, pending.challenge.fields.realm, inBand);
            if (held) {
              this.claimed.add(held.id);
              const again = this.credentialOfHeld(held);
              credential = again;
              presented = again;
              settlementRetries = 0;
              budgetFrom = this.now().getTime();
              waitedMs = 0;
              pending = null;
              response = await this.present(seed, hints, again, 0);
              continue;
            }
          }
          if (purchases >= maxPurchases) {
            const detail = `${purchases} purchase${purchases === 1 ? '' : 's'} settled for this call and policy.maxPurchasesPerCall is ${maxPurchases}; the fresh challenge would buy another`;
            this.refuse(seed.url, pending.challenge, pending.terms, 'purchase_limit_per_call', detail);
            return done('purchase_limit_per_call', detail);
          }
          if (challengesPaid >= maxChallenges) {
            const detail = `${challengesPaid} challenges were paid for one request and the policy allows no more`;
            this.refuse(seed.url, pending.challenge, pending.terms, 'challenges_exhausted', detail);
            return done('challenges_exhausted', detail);
          }
          // S-154: who is paid, before anything is reserved or issued (the
          // pin may need one fetch, so it sits before the synchronous admit).
          const sellerRefusal = await this.sellerVerdict(pending.challenge);
          if (sellerRefusal) {
            this.refuse(seed.url, pending.challenge, pending.terms, 'seller_not_pinned', sellerRefusal);
            return done('seller_not_pinned', sellerRefusal);
          }
          // Checked and reserved in one synchronous step (file comment, THE LEDGER).
          const verdict = this.admit(pending.challenge);
          if (!verdict.ok) {
            this.refuse(seed.url, pending.challenge, pending.terms, verdict.reason, verdict.detail);
            return done(verdict.reason, verdict.detail);
          }
          const id = pending.challenge.fields.id;
          await this.save();
          let fresh: Credential;
          try {
            if (!(await this.termsAccepted(pending.terms))) {
              this.release(id);
              await this.save();
              const terms = pending.terms as MppTermsRequest;
              const detail = `the policy does not accept Terms version ${terms.version}${terms.url ? ` (${terms.url})` : ''}`;
              this.refuse(seed.url, pending.challenge, pending.terms, 'terms_refused', detail);
              return done('terms_refused', detail);
            }
            fresh = await this.obtain(pending.challenge, verdict.method, pending.terms);
          } catch (err) {
            // Nothing was presented: a source that failed charged nothing.
            this.release(id);
            await this.save();
            throw err;
          }
          challengesPaid += 1;
          credential = fresh;
          presented = fresh;
          settlementRetries = 0;
          budgetFrom = this.now().getTime();
          waitedMs = 0;
          pending = null;
          response = await this.present(seed, hints, fresh, 0);
          continue;
        }

        const current = response as Response;
        const sent = credential as Credential | null;
        if (!sent) {
          if (current.status === 402) {
            const challenge = readMppChallenge(current.headers.get('www-authenticate'));
            if (challenge) {
              pending = { challenge, terms: readTermsNotice(current, await bodyOf(current)) };
              continue;
            }
            // THE PURCHASE POINTER (file comment): a 402 with no challenge
            // whose body names where to buy. Never on a purchase URL's own
            // answer: a pointer there would be a pointer to a pointer.
            const entry = inBand ? null : mppRequirementOf((await bodyOf(current))?.requirements);
            if (entry) {
              const refusal = this.pointerVerdict(entry.purchaseUrl, seed.url, call);
              if (refusal) {
                this.refuse(seed.url, null, null, refusal.reason, refusal.detail);
                return done(refusal.reason, refusal.detail);
              }
              // An entry that does carry a challenge is paid as `purchase` pays one.
              let inHand: { challenge: MppChallenge; terms: MppTermsRequest | null } | null = null;
              if (entry.challenge !== null) {
                const parsed = readMppChallenge(entry.challenge);
                if (!parsed) {
                  const detail = 'the 402 names a purchase URL with a challenge that is not a readable Payment challenge';
                  this.refuse(seed.url, null, null, 'no_challenge', detail);
                  return done('no_challenge', detail);
                }
                inHand = { challenge: parsed, terms: entry.terms };
              }
              const bought = await this.buy(this.purchaseSeed({ url: entry.purchaseUrl }), inHand, call);
              // The purchase URL's own answer is nobody's to read.
              await bought.response?.body?.cancel().catch(() => undefined);
              if (!bought.ok) {
                // `onRefusal` already fired inside; the caller gets the 402 it was answered.
                const reason = bought.reason === 'served_without_purchase' || bought.reason === 'unexpected_status' ? null : bought.reason;
                return done(reason, bought.detail, bought.pendingCredential ?? null, bought.receipt ?? null);
              }
              purchases = this.calls.get(call) ?? purchases + 1;
              // The purchase funded THIS identity's balance, which the door
              // serves only to a signed request naming no payment token.
              seed = withoutPaymentToken(seed);
              await current.body?.cancel().catch(() => undefined);
              response = await this.sendPlain(seed, hints);
              continue;
            }
          }
          return done(null, `answered ${current.status}`);
        }

        const body = await bodyOf(current);
        const reading = afterCredential(current.status, body);

        if (reading === 'retry_same') {
          sent.inDoubt = true;
          const txHash = body ? body.transactionHash : undefined;
          if (typeof txHash === 'string' && txHash) sent.transactionHash = txHash;
          const remaining = sent.deadline - Math.max(this.now().getTime(), budgetFrom + waitedMs);
          const capped = policy.maxSettlementRetries !== undefined && settlementRetries >= policy.maxSettlementRetries;
          if (remaining <= 0 || capped) {
            const record = this.hold(sent, seed, 'settlement_retries_exhausted');
            this.claimed.delete(sent.challengeId);
            credential = null;
            await this.save();
            const detail =
              `the platform asked ${settlementRetries + 1} times to re-present the same credential and ` +
              `${capped ? 'policy.maxSettlementRetries allows no more' : 'the settlement budget (the challenge expiry plus the grace) has run out'}; ` +
              `the charge may have settled, so the credential is held and re-presented before this buyer pays again for ${record.httpMethod} ${record.url}`;
            this.refuse(seed.url, null, null, 'settlement_retries_exhausted', detail, record);
            return done('settlement_retries_exhausted', detail, record);
          }
          settlementRetries += 1;
          const backoff = DEFAULT_RETRY_AFTER_SECONDS * 2 ** (settlementRetries - 1);
          const seconds = Math.min(Math.max(retryAfterSeconds(current.headers.get('retry-after'), this.now()), backoff), maxRetryAfter);
          const ms = Math.max(0, Math.min(seconds * 1000, Math.ceil(remaining)));
          waitedMs += ms;
          response = await this.present(seed, hints, sent, ms);
          continue;
        }

        if (reading === 'settled') {
          this.held.delete(sent.challengeId);
          this.claimed.delete(sent.challengeId);
          credential = null;
          settled = sent;
          purchases += 1;
          this.calls.set(call, purchases);
          await this.save();
          this.reportPurchase(seed.url, sent, current, body);
          if (current.status === 402) {
            // S-151: the pack was granted and the door still could not fund
            // the call. It is paid and counted; the fresh challenge is a
            // SECOND purchase, which the per-call limit decides.
            const challenge = readMppChallenge(current.headers.get('www-authenticate'));
            if (challenge) {
              pending = { challenge, terms: readTermsNotice(current, body) };
              continue;
            }
          } else if ((current.status === 503 || current.status === 409) && sameCredentialOf(body) === false && !retriedWithout) {
            // The contract: the purchase is done (503 `funding_failed`, or
            // 409 `purchase_already_granted` after a lost 200), so the call
            // goes again WITHOUT the credential, once, after `Retry-After`
            // (a 409 names none and goes at once). Never paid again.
            retriedWithout = true;
            const fallback = current.status === 409 ? 0 : DEFAULT_RETRY_AFTER_SECONDS;
            const seconds = Math.min(retryAfterSeconds(current.headers.get('retry-after'), this.now(), fallback), maxRetryAfter);
            if (seconds > 0) await this.pause(seconds * 1000, seed.signal);
            response = await this.sendPlain(seed, hints);
            continue;
          }
          return done(null, `answered ${current.status}`);
        }

        const code = problemCodeOf(body);
        if (reading === 'served_elsewhere') {
          // A per-call sale served, or being served, on an earlier
          // presentation (the portal's S-149). It is paid; the door never
          // serves it twice, so re-presenting can only answer 409 again and
          // is not done, and nothing is paid again. Terminal.
          this.held.delete(sent.challengeId);
          this.claimed.delete(sent.challengeId);
          credential = null;
          settled = sent;
          await this.save();
          this.reportPurchase(seed.url, sent, current, body);
          const detail =
            code === 'call_in_progress'
              ? 'the call this credential paid for is being served on an earlier presentation; it is not served again and nothing is paid again'
              : 'the call this credential paid for was already served on an earlier presentation whose response was lost; nothing is paid again';
          return done(code as MppTerminalOutcome, detail, null, receiptOf(current, body));
        }

        if (reading === 'refunded') {
          // Refunded: nothing is owed, so the reservation is freed. A new
          // purchase is a new call under the normal policy.
          this.held.delete(sent.challengeId);
          this.claimed.delete(sent.challengeId);
          credential = null;
          this.release(sent.challengeId);
          await this.save();
          return done('call_refunded', 'the per-call purchase this credential paid for was refunded; its reservation is freed and nothing is paid again', null, receiptOf(current, body));
        }

        if (reading === 'unknown') {
          const record = this.hold(sent, seed, 'settlement_outcome_unknown');
          this.claimed.delete(sent.challengeId);
          credential = null;
          await this.save();
          const detail =
            `the platform answered ${current.status} to the request carrying the credential without saying whether it settled; ` +
            `the credential is held and re-presented before this buyer pays again for ${record.httpMethod} ${record.url}`;
          this.refuse(seed.url, null, null, 'settlement_outcome_unknown', detail, record);
          return done('settlement_outcome_unknown', detail, record);
        }

        // Refused: nothing was charged by this presentation.
        this.claimed.delete(sent.challengeId);
        credential = null;
        if (sent.inDoubt) {
          // It had been in doubt and the platform now refuses it: it will
          // never complete it. Cleared, left counted (money may have moved
          // before the doubt began), reported with the credential for the
          // operator, and nothing new is paid in this call.
          const record = this.pendingOf(sent, seed, 'refused');
          this.held.delete(sent.challengeId);
          await this.save();
          const detail = `the platform refused a credential whose outcome had been in doubt (answered ${current.status}); it is no longer held and is reported here for reconciliation`;
          this.refuse(seed.url, null, null, 'pending_credential_refused', detail, record);
          return done('pending_credential_refused', detail, record);
        }
        this.release(sent.challengeId);
        await this.save();
        if (current.status === 402) {
          const challenge = readMppChallenge(current.headers.get('www-authenticate'));
          if (!challenge) {
            const detail = 'the 402 carries no readable Payment challenge';
            this.refuse(seed.url, null, null, 'no_challenge', detail);
            return done('no_challenge', detail);
          }
          pending = { challenge, terms: readTermsNotice(current, body) };
          continue;
        }
        if (current.status === 429 || current.status === 503) {
          // A velocity cap or a sale the deployment cannot make: nothing was
          // charged, the reservation is freed, and the buyer does not retry.
          const detail = `the platform refused the purchase (${current.status}${code ? ` ${code}` : ''}); nothing was charged and it is not retried`;
          this.refuse(seed.url, null, null, 'platform_refused', detail, null, sent);
          return done('platform_refused', detail);
        }
        return done(null, `answered ${current.status}`);
      }
    } finally {
      const last = credential as Credential | null;
      if (last) this.claimed.delete(last.challengeId);
    }
  }

  private syntheticRefusal(): Response {
    // Only reached on the in-band path, where no response exists yet: the
    // caller of `purchase` reads the outcome object, never this body.
    return new Response(null, { status: 402 });
  }

  private refuse(
    url: string,
    challenge: MppChallenge | null,
    terms: MppTermsRequest | null,
    reason: MppBuyerRefusalReason,
    detail: string,
    pendingCredential: MppPendingCredential | null = null,
    presented: Credential | null = null,
  ): void {
    this.config.onRefusal?.({
      reason,
      url,
      challengeId: challenge?.fields.id ?? pendingCredential?.id ?? presented?.challengeId ?? null,
      amountCents: challenge?.amountCents ?? pendingCredential?.amountCents ?? presented?.amountCents ?? null,
      method: challenge?.fields.method ?? pendingCredential?.paymentMethod ?? presented?.method ?? null,
      termsVersion: terms?.version ?? pendingCredential?.termsVersion ?? presented?.termsVersion ?? null,
      detail,
      pendingCredential,
    });
  }

  /**
   * S-154: null when the challenge's seller is the pinned Robutler one, else
   * why not. A method other than stripe or tempo is left to `admit`, which
   * refuses it as `method_unavailable`.
   */
  private async sellerVerdict(challenge: MppChallenge): Promise<string | null> {
    const method = challenge.fields.method;
    if (method !== 'stripe' && method !== 'tempo') return null;
    const policy = this.config.policy;
    const named =
      method === 'stripe'
        ? typeof challenge.request.methodDetails.networkId === 'string'
          ? challenge.request.methodDetails.networkId
          : ''
        : (challenge.request.recipient ?? '');
    let pinned = method === 'stripe' ? policy.stripeProfileId : policy.tempoDepositAddress;
    let from = method === 'stripe' ? 'policy.stripeProfileId' : 'policy.tempoDepositAddress';
    if (!pinned) {
      const pins = await this.discoveredPins(challenge.fields.realm, method);
      pinned = (method === 'stripe' ? pins?.stripeProfileId : pins?.tempoDepositAddress) ?? undefined;
      from = 'the platform discovery document';
    }
    if (!pinned) {
      return (
        `no Robutler ${method === 'stripe' ? 'Stripe profile' : 'deposit address'} is pinned for ${challenge.fields.realm}: ` +
        `set ${method === 'stripe' ? 'policy.stripeProfileId' : 'policy.tempoDepositAddress'}, the discovery document names none; nothing is paid to an unpinned seller`
      );
    }
    const same = method === 'stripe' ? named === pinned : named.toLowerCase() === pinned.toLowerCase();
    if (same) return null;
    return `the challenge names ${method === 'stripe' ? 'Stripe profile' : 'recipient'} ${named || '(none)'} and the Robutler seller pinned by ${from} is ${pinned}; nothing is paid to another seller`;
  }

  /**
   * The pins discovery names for the allowlisted host `realm` resolves to
   * (file comment, WHICH SELLER IT PAYS). A document that names the seller
   * for `method` is believed for `DISCOVERY_PIN_TTL_MS`, one that does not
   * for five minutes; then it is read again, and what the NEW read says is
   * the whole answer: a re-read that fails, names no seller or names two
   * pins nothing, and nothing is paid. The stale pin is never a fallback.
   */
  private async discoveredPins(realm: string, method: 'stripe' | 'tempo'): Promise<MppSellerPins | null> {
    const r = realm.toLowerCase();
    const entry = this.realms.find((e) => e.hostname === r || e.host === r);
    if (!entry) return null;
    const cached = this.discovered.get(entry.host);
    if (cached instanceof Promise) return cached;
    const now = this.now().getTime();
    if (cached) {
      const named = method === 'stripe' ? cached.pins.stripeProfileId : cached.pins.tempoDepositAddress;
      if (now - cached.at < (named ? DISCOVERY_PIN_TTL_MS : DISCOVERY_NEGATIVE_TTL_MS)) return cached.pins;
    }
    const pending = this.fetchPins(`https://${entry.host}${DISCOVERY_PATH}`);
    this.discovered.set(entry.host, pending);
    const pins = await pending;
    this.discovered.set(entry.host, { pins, at: this.now().getTime() });
    return pins;
  }

  /**
   * A body as text, counted WHILE it streams: more than `maxBytes` cancels
   * the read and answers null. `response.text()` first and a length check
   * after (the code until 2026-09-19) buffers whatever the host sends.
   */
  private static async readBounded(response: Response, maxBytes: number): Promise<string | null> {
    const reader = response.body?.getReader();
    if (!reader) return '';
    const chunks: Uint8Array[] = [];
    let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) {
        await reader.cancel().catch(() => undefined);
        return null;
      }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return new TextDecoder().decode(bytes);
  }

  private async fetchPins(url: string): Promise<MppSellerPins> {
    const none: MppSellerPins = { stripeProfileId: null, tempoDepositAddress: null };
    if (!this.allowsUrl(url) || new URL(url).protocol !== 'https:') return none;
    try {
      const response = await (this.config.fetch ?? globalThis.fetch)(
        new Request(url, { method: 'GET', headers: { accept: 'application/json' }, redirect: 'error', signal: AbortSignal.timeout(DISCOVERY_TIMEOUT_MS) }),
      );
      if (!response.ok) return none;
      const declared = Number(response.headers.get('content-length'));
      if (Number.isFinite(declared) && declared > DISCOVERY_MAX_BYTES) {
        await response.body?.cancel().catch(() => undefined);
        return none;
      }
      const text = await MppBuyer.readBounded(response, DISCOVERY_MAX_BYTES);
      if (text === null) return none;
      return sellerPinsFromDiscovery(JSON.parse(text));
    } catch {
      return none;
    }
  }

  /** S-147 first, then the realm-is-the-host rule. Runs before any held credential is re-presented and before any source call. */
  private realmVerdict(url: string, challenge: MppChallenge): { reason: MppBuyerRefusalReason; detail: string } | null {
    const target = new URL(url);
    const realm = challenge.fields.realm.toLowerCase();
    if (!this.realmAllowed(realm) || !this.allowsUrl(target)) {
      return {
        reason: 'realm_not_allowed',
        detail: `the challenge realm ${challenge.fields.realm} (asked at ${target.host}) is not on policy.realms; nothing is paid to a host the operator did not name`,
      };
    }
    if (realm !== target.hostname.toLowerCase() && realm !== target.host.toLowerCase()) {
      return { reason: 'realm_mismatch', detail: `the challenge realm ${challenge.fields.realm} is not the host asked (${target.host}); nothing is paid to a third party` };
    }
    return null;
  }

  /**
   * The rest of the policy, in the order a refusal is cheapest, and on
   * success the reservation, all synchronous: no await may sit between the
   * daily-cap check and the reservation (file comment, THE LEDGER).
   */
  private admit(challenge: MppChallenge): { ok: true; method: MppMethod } | { ok: false; reason: MppBuyerRefusalReason; detail: string } {
    const policy = this.config.policy;
    if (challenge.fields.intent !== MPP_INTENT_CHARGE) {
      return { ok: false, reason: 'unsupported_intent', detail: `intent ${challenge.fields.intent} is not charge` };
    }
    if (challenge.amountCents === null) {
      return { ok: false, reason: 'amount_unreadable', detail: 'the challenge amount is not a whole number of cents in its method unit' };
    }
    if (challenge.expiresAt && challenge.expiresAt.getTime() <= this.now().getTime()) {
      return { ok: false, reason: 'challenge_expired', detail: 'the challenge had expired before it could be paid' };
    }
    const method = challenge.fields.method;
    if (!this.methodOrder().includes(method as MppMethod)) {
      return { ok: false, reason: 'method_unavailable', detail: `the challenge asks for method ${method} and the policy has no source for it` };
    }
    if (challenge.amountCents > policy.maxPerPurchaseCents) {
      return { ok: false, reason: 'over_max_per_purchase', detail: `${challenge.amountCents} cents is above policy.maxPerPurchaseCents (${policy.maxPerPurchaseCents})` };
    }
    const spent = this.spentTodayCents();
    if (policy.dailyCapCents !== undefined && spent + challenge.amountCents > policy.dailyCapCents) {
      return { ok: false, reason: 'over_daily_cap', detail: `${spent} cents presented in the last 24 hours plus ${challenge.amountCents} would exceed policy.dailyCapCents (${policy.dailyCapCents})` };
    }
    this.reserve(challenge.fields.id, challenge.amountCents);
    return { ok: true, method: method as MppMethod };
  }

  /** The Terms, last: the operator's callback runs only once everything cheaper has passed. */
  private async termsAccepted(terms: MppTermsRequest | null): Promise<boolean> {
    if (!terms) return true;
    const accept = this.config.policy.acceptTerms;
    return typeof accept === 'string' ? accept === terms.version : !!(await accept({ version: terms.version, url: terms.url }));
  }

  /** Obtain the payment from the source for the method and build the credential. A source error propagates: nothing is retried against a source that failed. */
  private async obtain(challenge: MppChallenge, method: MppMethod, terms: MppTermsRequest | null): Promise<Credential> {
    const amountCents = challenge.amountCents as number;
    let payload: Record<string, unknown>;
    let source: string | undefined;
    if (method === 'stripe') {
      const card = this.config.sources.card as CardPaymentSource;
      const details = challenge.request.methodDetails;
      const networkId = typeof details.networkId === 'string' ? details.networkId : '';
      if (!networkId) throw new Error('the Stripe challenge names no networkId');
      const types = Array.isArray(details.paymentMethodTypes) ? details.paymentMethodTypes.filter((t): t is string => typeof t === 'string') : ['card'];
      const spt = await card.getSpt({
        networkId,
        amountCents,
        currency: challenge.request.currency,
        expiresAt: challenge.expiresAt,
        challengeId: challenge.fields.id,
        paymentMethodTypes: types,
      });
      if (typeof spt !== 'string' || !/^spt_[A-Za-z0-9_]+$/.test(spt)) throw new Error('the card source did not return a shared payment token id (spt_...)');
      payload = { spt };
    } else {
      const wallet = this.config.sources.stablecoin as StablecoinPaymentSource;
      const details = challenge.request.methodDetails;
      const chainId = typeof details.chainId === 'number' ? details.chainId : Number(details.chainId);
      const memo = typeof details.memo === 'string' ? details.memo : '';
      const recipient = challenge.request.recipient ?? '';
      if (!Number.isSafeInteger(chainId) || !memo || !recipient) throw new Error('the Tempo challenge lacks chainId, memo or recipient');
      // The DID is built before the wallet signs: an address the platform
      // would refuse must not cost a signature (tempoPayerDid).
      source = wallet.address === undefined ? undefined : tempoPayerDid(chainId, wallet.address);
      const serialized = await wallet.signTempoTransfer({
        chainId,
        currency: challenge.request.currency,
        recipient,
        amount: challenge.request.amount,
        memo,
        validBefore: challenge.expiresAt,
        challengeId: challenge.fields.id,
      });
      if (typeof serialized !== 'string' || !TEMPO_SIGNED_TRANSACTION_RE.test(serialized)) {
        throw new Error('the stablecoin source did not return a serialized Tempo transaction (0x76 followed by hex bytes)');
      }
      payload = tempoCredentialPayload(serialized);
    }
    const graceMs = (this.config.policy.settlementGraceSeconds ?? DEFAULT_SETTLEMENT_GRACE_SECONDS) * 1000;
    const base = challenge.expiresAt ? challenge.expiresAt.getTime() : this.now().getTime() + DEFAULT_SETTLEMENT_WINDOW_SECONDS * 1000;
    return {
      headerName: challenge.credentialHeader,
      value: encodeMppCredential(challenge.fields, payload, source),
      challengeId: challenge.fields.id,
      method,
      amountCents,
      currency: challenge.request.currency,
      termsVersion: terms?.version ?? null,
      realm: challenge.fields.realm,
      kind: opaqueKind(challenge.fields),
      expiresAt: challenge.expiresAt,
      deadline: base + graceMs,
      transactionHash: null,
      inDoubt: false,
    };
  }

  /**
   * Send the request carrying `credential`, after `delayMs`. Three ways it
   * can fail, read differently (file comment):
   *
   *  * BEFORE ANYTHING WAS SENT (the wait was aborted, or the signer refused
   *    the request): a fresh credential never left this process, so its
   *    reservation is released, exactly as when a payment source fails. One
   *    already in doubt stays held.
   *  * A REDIRECT (S-180): never followed. Nothing is spent: a fresh
   *    credential's reservation is released; one already in doubt stays
   *    held, since a redirect says nothing about whether it settled.
   *    Reported as `redirect_refused`, then thrown.
   *  * ANYTHING ELSE once the request was handed to `fetch` (a network error,
   *    a timeout, an abort): the outcome is unknown, so the credential is
   *    HELD and reported before the error propagates (S-150): the caller's
   *    retry, or the operator, can then complete it instead of paying again.
   */
  private async present(seed: SeedRequest, hints: Array<[string, string]>, credential: Credential, delayMs: number): Promise<Response> {
    let dispatched = false;
    try {
      if (delayMs > 0) await this.pause(delayMs, seed.signal);
      const prepared = await this.prepare(seed, hints, credential);
      dispatched = true;
      return await this.dispatch(seed, prepared);
    } catch (err) {
      this.claimed.delete(credential.challengeId);
      const redirect = err instanceof MppRedirectError && !err.followed ? err : null;
      if ((!dispatched || redirect) && !credential.inDoubt) {
        this.release(credential.challengeId);
        await this.save();
        if (redirect) this.refuse(seed.url, null, null, 'redirect_refused', `${redirect.message}; nothing was spent and the reservation is released`, null, credential);
        throw err;
      }
      const record = this.hold(credential, seed, 'settlement_outcome_unknown');
      await this.save();
      if (err instanceof MppRedirectError) err.pendingCredential = record;
      if (redirect) {
        const detail = `${redirect.message}; the credential was already in doubt, so it stays held and is re-presented before this buyer pays again for ${record.httpMethod} ${record.url}`;
        this.refuse(seed.url, null, null, 'redirect_refused', detail, record);
        throw err;
      }
      const detail =
        `the request carrying the credential failed (${(err as Error)?.message ?? String(err)}); ` +
        `the credential is held and re-presented before this buyer pays again for ${record.httpMethod} ${record.url}`;
      this.refuse(seed.url, null, null, 'settlement_outcome_unknown', detail, record);
      throw err;
    }
  }

  /** A send that carries no credential: the first request, and the retry after a settled purchase. A redirect is reported, then thrown (S-180). */
  private async sendPlain(seed: SeedRequest, hints: Array<[string, string]>): Promise<Response> {
    try {
      return await this.dispatch(seed, await this.prepare(seed, hints, null));
    } catch (err) {
      if (err instanceof MppRedirectError) this.refuse(seed.url, null, null, 'redirect_refused', err.message);
      throw err;
    }
  }

  /** A wait that ends early, as an abort, when the request's own signal fires. */
  private async pause(ms: number, signal: AbortSignal | null): Promise<void> {
    signal?.throwIfAborted();
    const sleep = this.config.sleep ?? defaultSleep;
    if (!signal) return sleep(ms);
    await new Promise<void>((resolve, reject) => {
      const onAbort = () => reject(signal.reason ?? new Error('aborted'));
      signal.addEventListener('abort', onAbort, { once: true });
      sleep(ms).then(
        () => {
          signal.removeEventListener('abort', onAbort);
          resolve();
        },
        (err: unknown) => {
          signal.removeEventListener('abort', onAbort);
          reject(err);
        },
      );
    });
  }

  /**
   * Build and sign one send: a fresh `Request` from the seed (a body can be
   * read once), the hints and, on a paid retry, the credential and the
   * assent, all covered, freshly signed. Nothing is sent here, so a throw
   * means nothing left this process. ALWAYS `redirect: 'error'` (S-180), and
   * always abortable by the buyer's own deadline beside the caller's signal.
   */
  private async prepare(seed: SeedRequest, hints: Array<[string, string]>, credential: Credential | null): Promise<PreparedSend> {
    const headers = new Headers(seed.headers);
    const covered: string[] = [...(this.config.sign?.coveredHeaders ?? [])];
    for (const [name, value] of hints) {
      headers.set(name, value);
      covered.push(name);
    }
    if (credential) {
      headers.set(credential.headerName, credential.value);
      covered.push(credential.headerName);
      if (credential.termsVersion) {
        headers.set(TERMS_ACCEPTED_HEADER, credential.termsVersion);
        covered.push(TERMS_ACCEPTED_HEADER);
      }
    } else {
      headers.delete(TERMS_ACCEPTED_HEADER);
    }
    const deadline = new AbortController();
    // The bytes came out of `Request.arrayBuffer()`, so they are backed by a
    // plain ArrayBuffer; the cast only bridges the TS 5.7 `Uint8Array<T>`
    // generic to `BodyInit` in the lib version each consumer compiles with.
    const request = new Request(seed.url, {
      method: seed.method,
      headers,
      body: seed.body && seed.body.byteLength > 0 ? (seed.body as unknown as BodyInit) : undefined,
      redirect: 'error',
      signal: seed.signal ? AbortSignal.any([seed.signal, deadline.signal]) : deadline.signal,
    });
    // `signRequest` carries the redirect mode and the signal onto the signed Request.
    const signed = await signRequest(this.config.identity, request, { ...this.config.sign, coveredHeaders: covered });
    return { request: signed, deadline };
  }

  /**
   * Hand a prepared request to `fetch`, bounded by `seed.timeoutMs` until the
   * answer's head arrives (file comment, "EVERY SEND IS BOUNDED"), and refuse
   * a redirect however the runtime reports one (S-180): a rejection that
   * names a redirect, an opaque redirect, a 3xx handed back as it came by a
   * `fetch` with no redirect modes, or an answer that says it was redirected.
   */
  private async dispatch(seed: SeedRequest, prepared: PreparedSend): Promise<Response> {
    const timer = setTimeout(() => {
      prepared.deadline.abort(new DOMException(`${seed.method.toUpperCase()} ${seed.url} did not answer within ${seed.timeoutMs} ms`, 'TimeoutError'));
    }, seed.timeoutMs);
    let response: Response;
    try {
      response = await (this.config.fetch ?? globalThis.fetch)(prepared.request);
    } catch (err) {
      if (isRedirectFailure(err)) throw new MppRedirectError({ url: seed.url, cause: err });
      throw err;
    } finally {
      clearTimeout(timer);
    }
    if (response.redirected) throw new MppRedirectError({ url: seed.url, followed: true });
    if (response.type === 'opaqueredirect' || REDIRECT_STATUSES.has(response.status)) {
      await response.body?.cancel().catch(() => undefined);
      throw new MppRedirectError({ url: seed.url, status: response.status || null, location: response.headers.get('location') });
    }
    return response;
  }
}

/**
 * `seed` with every way of naming a payment token removed (file comment, "THE
 * PURCHASE POINTER"): the token is what ran dry, and the platform's door
 * serves the balance a purchase funded only to a signed request that names
 * none. Everything else, the caller's other headers and the body, is kept.
 */
function withoutPaymentToken(seed: SeedRequest): SeedRequest {
  const headers = new Headers(seed.headers);
  for (const name of PAYMENT_TOKEN_HEADERS) headers.delete(name);
  const url = new URL(seed.url);
  url.searchParams.delete(PAYMENT_TOKEN_QUERY);
  // `URL.toString()` re-serialises the query; leave a URL that named no token byte-identical.
  return { ...seed, headers, url: new URL(seed.url).searchParams.has(PAYMENT_TOKEN_QUERY) ? url.toString() : seed.url };
}

async function captureRequest(input: string | URL | Request, init: RequestInit | undefined, timeoutMs: number): Promise<SeedRequest> {
  const request = new Request(input, init);
  const body = request.body !== null ? new Uint8Array(await request.arrayBuffer()) : null;
  return {
    method: request.method,
    url: request.url,
    headers: new Headers(request.headers),
    body,
    // `request.redirect` is deliberately not read: every send refuses redirects (S-180).
    signal: request.signal,
    timeoutMs,
  };
}
