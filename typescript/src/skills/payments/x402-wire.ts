/**
 * x402 on the wire, v2 and v1 (webagents gap-closure plan 2.6, 2026-09-26).
 *
 * The pure half of the paywall (`./paywall.ts`): header names, the standard
 * base64 codec, the `PaymentRequired`, `PaymentPayload` and `SettleResponse`
 * shapes, the v1 projection of a v2 requirement, the strict-parser
 * well-formedness rule and the requirement matching rule, all as the spec
 * pack pins them (`docs/internal/architecture/webagents-gap-closure/
 * spec-payments-identity.txt` section 1). The Python twin is
 * `python/webagents/agents/skills/robutler/payments/x402_wire.py`; the
 * fixture `python/tests/fixtures/payments/x402_paywall.json` holds the names
 * and the section 1.10 vectors both suites read, so the two cannot drift.
 *
 * WHAT CHANGED. The SDKs used to answer a 402 in a private shape: `scheme:
 * 'token'` on `network: 'robutler'`, `maxAmountRequired`, a `version: '1.0'`
 * field no x402 client reads, and `payTo: <agent id>`. A standard x402
 * client dropped every entry as unknown and gave up; a strict parser refused
 * the whole body (`network` without a `:`). Now:
 *
 *   - v2 (`x402Version: 2`) travels in the HEADERS: `PAYMENT-REQUIRED` on the
 *     402, `PAYMENT-SIGNATURE` on the retry, `PAYMENT-RESPONSE` on the answer.
 *     Standard base64 WITH padding; a TS reference decoder rejects anything
 *     outside `^[A-Za-z0-9+/]*={0,2}$`, so base64url would be a silent no.
 *   - v1 (`x402Version: 1`) travels in the BODY of the 402 and in `X-PAYMENT`
 *     and `X-PAYMENT-RESPONSE`. One 402 carries both: the reference client
 *     reads the header first and falls back to a `x402Version: 1` body.
 *   - Every entry is well-formed for a strict parser (section 1.9): `network`
 *     contains `:`, `amount:`, `asset:` and `payTo:` are non-empty strings,
 *     `maxTimeoutSeconds` is a positive number. A custom entry is harmless to
 *     a standard client only if every entry is well-formed, because the
 *     client validates each and drops the failures.
 *
 * THE CREDITS SCHEME (section 1.9, "custom non-chain scheme next to chain
 * schemes: allowed"): scheme `robutler-credits` on network `robutler:1`, the
 * asset named `credits` (never a currency: credits are named as credits,
 * CLAUDE.md), `payTo:` the role constant `robutler` (Robutler is the seller of
 * record; a creator's address here would read as one user paying another).
 * The amount is in nanocredits, the atomic unit (9 decimals, `extra.decimals`).
 * The payload is a Robutler payment token, bound to THIS request by the
 * server nonce in `extra.nonce` (`./x402-credits.ts`).
 */

// ── Names ────────────────────────────────────────────────────────────────────

export const X402_HEADERS = {
  /** v2: the 402's `PaymentRequired`, base64 JSON. */
  required: 'PAYMENT-REQUIRED',
  /** v2: the retry's `PaymentPayload`, base64 JSON. */
  signature: 'PAYMENT-SIGNATURE',
  /** v2: the answer's `SettleResponse`, base64 JSON. */
  response: 'PAYMENT-RESPONSE',
  /** v1: the retry's payload. */
  v1Payment: 'X-PAYMENT',
  /** v1: the answer's settle response. */
  v1Response: 'X-PAYMENT-RESPONSE',
  /** A metered (`upto`) handler's actual amount, read and stripped by the paywall. */
  settlementOverrides: 'Settlement-Overrides',
} as const;

/** What a browser client must be allowed to READ (`Access-Control-Expose-Headers`). */
export const X402_CORS_EXPOSE_HEADERS: readonly string[] = [
  'PAYMENT-REQUIRED',
  'PAYMENT-RESPONSE',
  'X-PAYMENT-RESPONSE',
  'WWW-Authenticate',
  'Payment-Receipt',
];

/**
 * What a browser client must be allowed to SEND (`Access-Control-Allow-Headers`).
 * `Access-Control-Expose-Headers` is here because the official fetch wrapper
 * sends a REQUEST header of that name (section 1.11).
 */
export const X402_CORS_ALLOW_HEADERS: readonly string[] = [
  'Content-Type',
  'Authorization',
  'PAYMENT-SIGNATURE',
  'X-PAYMENT',
  'Access-Control-Expose-Headers',
  'Payment-Authorization',
];

export const CREDITS_SCHEME = 'robutler-credits';
export const CREDITS_NETWORK = 'robutler:1';
export const CREDITS_ASSET = 'credits';
export const CREDITS_PAY_TO = 'robutler';
/** One credit is 1e9 nanocredits, the atomic unit every amount is quoted in. */
export const CREDITS_DECIMALS = 9;
export const CREDITS_MAX_TIMEOUT_SECONDS = 300;

/** The v1 network names the reference SDKs map from CAIP-2 (mechanisms/evm/src/constants.ts). */
export const V1_NETWORK_NAMES: Readonly<Record<string, string>> = {
  'eip155:84532': 'base-sepolia',
  'eip155:8453': 'base',
  'eip155:43113': 'avalanche-fuji',
  'eip155:43114': 'avalanche',
  'eip155:137': 'polygon',
  'eip155:80002': 'polygon-amoy',
};

// ── Shapes ───────────────────────────────────────────────────────────────────

/** One `accepts[]` entry, v2 (section 1.2). */
export interface PaymentRequirement {
  scheme: string;
  /** CAIP-2, or a namespaced non-chain network (`robutler:1`); always contains `:`. */
  network: string;
  /** Atomic units, as a decimal string. */
  amount: string;
  /** A token contract, an ISO 4217 code, or a named non-chain asset (`credits`). */
  asset: string;
  /** A wallet address or a role constant (`robutler`). */
  payTo: string;
  maxTimeoutSeconds: number;
  extra?: Record<string, unknown>;
}

export interface ResourceInfo {
  url: string;
  description?: string;
  mimeType?: string;
  /** Printable ASCII, at most 32 characters. */
  serviceName?: string;
  /** At most 5, each at most 32 characters. */
  tags?: string[];
  iconUrl?: string;
}

export interface PaymentRequiredV2 {
  x402Version: 2;
  error?: string;
  resource: ResourceInfo;
  accepts: PaymentRequirement[];
  extensions?: Record<string, unknown>;
}

/** One `accepts[]` entry, v1: the v2 entry with the resource folded in and `maxAmountRequired`. */
export interface PaymentRequirementV1 {
  scheme: string;
  network: string;
  maxAmountRequired: string;
  asset: string;
  payTo: string;
  resource: string;
  description: string;
  mimeType?: string;
  maxTimeoutSeconds: number;
  outputSchema?: unknown;
  extra?: Record<string, unknown>;
}

export interface PaymentRequiredV1 {
  x402Version: 1;
  /** Required in v1. */
  error: string;
  accepts: PaymentRequirementV1[];
}

export interface PaymentPayloadV2 {
  x402Version: 2;
  resource?: ResourceInfo;
  /** A copy of the chosen requirement, matched field for field (`requirementMatches`). */
  accepted: PaymentRequirement;
  payload: Record<string, unknown>;
  extensions?: Record<string, unknown>;
}

export interface PaymentPayloadV1 {
  x402Version: 1;
  scheme: string;
  network: string;
  payload: Record<string, unknown>;
}

export type PaymentPayload = PaymentPayloadV2 | PaymentPayloadV1;

/** A facilitator's (or the credits scheme's) answer to /settle (section 1.4). */
export interface SettleResponse {
  success: boolean;
  errorReason?: string;
  payer?: string;
  /** Required; `""` when nothing was broadcast; non-empty beside `settlement_pending`. */
  transaction: string;
  network: string;
  /** Required for `upto`; what was actually taken. */
  amount?: string;
  extensions?: Record<string, unknown>;
}

/** A facilitator's answer to /verify. The reference also emits `invalidMessage`; unknown fields are tolerated. */
export interface VerifyResponse {
  isValid: boolean;
  invalidReason?: string;
  payer?: string;
  extensions?: Record<string, unknown>;
  extra?: Record<string, unknown>;
  [key: string]: unknown;
}

/** Bazaar discovery metadata a seller may declare on an endpoint (section 1.8). */
export interface BazaarDeclaration {
  input:
    | { type: 'http'; method: 'GET'; queryParams?: Record<string, unknown>; routeTemplate?: string }
    | { type: 'http'; method: 'POST'; bodyType: 'json'; body?: Record<string, unknown>; routeTemplate?: string }
    | { type: 'mcp'; toolName: string; inputSchema: unknown; description?: string; transport?: string; example?: unknown };
  output?: { type: string; example?: unknown };
  /** JSON Schema 2020-12 validating `info`; no external `$ref`. */
  schema?: Record<string, unknown>;
}

// ── Standard base64, with padding ────────────────────────────────────────────

const STANDARD_BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

function utf8Encode(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function utf8Decode(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

/** `base64(JSON)` as x402 transports carry it: standard alphabet, `=` padding, `JSON.stringify` bytes. */
export function encodeBase64Json(value: unknown): string {
  const bytes = utf8Encode(JSON.stringify(value));
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/** Strict: the standard alphabet with padding, decoding to a JSON object; null otherwise. */
export function decodeBase64Json(text: string | null | undefined): Record<string, unknown> | null {
  if (typeof text !== 'string' || text.length === 0 || text.length % 4 !== 0 || !STANDARD_BASE64.test(text)) return null;
  let binary: string;
  try {
    binary = atob(text);
  } catch {
    return null;
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  let parsed: unknown;
  try {
    parsed = JSON.parse(utf8Decode(bytes));
  } catch {
    return null;
  }
  return isRecord(parsed) ? parsed : null;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// ── Amounts ──────────────────────────────────────────────────────────────────

const AMOUNT_RE = /^[0-9]+$/;

/** A decimal credit amount as nanocredits, the atomic unit, as a decimal string. */
export function creditsToNanocredits(credits: number | string): string {
  const n = typeof credits === 'string' ? Number(credits) : credits;
  if (!Number.isFinite(n) || n < 0) throw new Error(`x402: not a credit amount: ${JSON.stringify(credits)}`);
  // Round to whole nanocredits; `Math.round` on the scaled value, never on the
  // float alone, so 0.1 credits is exactly 100000000.
  return BigInt(Math.round(n * 10 ** CREDITS_DECIMALS)).toString();
}

/** A credit amount as atomic units of a chain asset with `decimals` decimals, at `unitsPerCredit` whole units per credit. */
export function creditsToAssetUnits(credits: number, decimals: number, unitsPerCredit = 1): string {
  if (!Number.isFinite(credits) || credits < 0) throw new Error(`x402: not a credit amount: ${credits}`);
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 30) throw new Error(`x402: bad decimals ${decimals}`);
  // Scale in two steps to keep the float short of precision loss for the
  // decimals every stablecoin uses (6 or 18).
  const whole = credits * unitsPerCredit;
  return (BigInt(Math.round(whole * 10 ** Math.min(decimals, 9))) * BigInt(10) ** BigInt(Math.max(decimals - 9, 0))).toString();
}

/** Whether `text` is a decimal integer amount string (`amount`, `maxAmountRequired`, overrides). */
export function isAmountString(text: unknown): text is string {
  return typeof text === 'string' && AMOUNT_RE.test(text);
}

// ── Well-formedness and matching (sections 1.3 and 1.9) ──────────────────────

/** The entry's required strings, one rule each: a non-empty string, and for the network one with a `:`. */
const REQUIRED_STRINGS: Readonly<Record<string, (text: string) => boolean>> = {
  scheme: (text) => text.length > 0,
  network: (text) => text.includes(':'),
  amount: (text) => text.length > 0,
  asset: (text) => text.length > 0,
  payTo: (text) => text.length > 0,
};

/** Well-formed for a strict parser: `network` contains `:`, the strings are non-empty, the timeout is positive. */
export function isWellFormedRequirement(value: unknown): value is PaymentRequirement {
  if (!isRecord(value)) return false;
  const r = value as Record<string, unknown>;
  for (const [key, rule] of Object.entries(REQUIRED_STRINGS)) {
    const text = r[key];
    if (typeof text !== 'string' || !rule(text)) return false;
  }
  return (
    typeof r.maxTimeoutSeconds === 'number' && Number.isFinite(r.maxTimeoutSeconds) && r.maxTimeoutSeconds > 0 &&
    (r.extra === undefined || isRecord(r.extra))
  );
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null) return false;
  if (Array.isArray(a)) {
    return Array.isArray(b) && a.length === b.length && a.every((v, i) => deepEqual(v, b[i]));
  }
  if (typeof a === 'object') {
    if (Array.isArray(b)) return false;
    const ra = a as Record<string, unknown>;
    const rb = b as Record<string, unknown>;
    const ka = Object.keys(ra).filter((k) => ra[k] !== undefined);
    const kb = Object.keys(rb).filter((k) => rb[k] !== undefined);
    return ka.length === kb.length && ka.every((k) => deepEqual(ra[k], rb[k]));
  }
  return false;
}

/**
 * Whether a client's `accepted` is one of the offered requirements: every
 * field but `extra` deep-equal, TYPES INCLUDED (`amount` a string,
 * `maxTimeoutSeconds` a number), and the offered `extra` contained in the
 * accepted one (a client may add extension data, never drop the server's).
 */
export function requirementMatches(offered: PaymentRequirement, accepted: unknown): boolean {
  if (!isWellFormedRequirement(accepted)) return false;
  const { extra: offeredExtra, ...offeredRest } = offered;
  const { extra: acceptedExtra, ...acceptedRest } = accepted;
  if (!deepEqual(offeredRest, acceptedRest)) return false;
  if (!offeredExtra) return true;
  if (!acceptedExtra) return false;
  return Object.keys(offeredExtra).every((k) => deepEqual(offeredExtra[k], acceptedExtra[k]));
}

// ── The 402, v2 header and v1 body ───────────────────────────────────────────

/** The v1 projection of a v2 requirement: the resource folded in, `amount` renamed. */
export function toV1Requirement(
  requirement: PaymentRequirement,
  resource: ResourceInfo,
  outputSchema?: unknown,
): PaymentRequirementV1 {
  const v1: PaymentRequirementV1 = {
    scheme: requirement.scheme,
    network: V1_NETWORK_NAMES[requirement.network] ?? requirement.network,
    maxAmountRequired: requirement.amount,
    asset: requirement.asset,
    payTo: requirement.payTo,
    resource: resource.url,
    description: resource.description ?? '',
    mimeType: resource.mimeType ?? 'application/json',
    maxTimeoutSeconds: requirement.maxTimeoutSeconds,
  };
  if (outputSchema !== undefined) v1.outputSchema = outputSchema;
  if (requirement.extra) v1.extra = requirement.extra;
  return v1;
}

export interface PaymentRequiredPair {
  v2: PaymentRequiredV2;
  v1: PaymentRequiredV1;
}

/**
 * The 402's two documents from one set of offers. `error` is what the client
 * shows (or, on a verify failure, the facilitator's `invalidReason`).
 * `extensions.bazaar` rides on v2; its `info.output` becomes v1's
 * `outputSchema`.
 */
export function buildPaymentRequired(
  resource: ResourceInfo,
  accepts: PaymentRequirement[],
  options: { error?: string; extensions?: Record<string, unknown>; bazaar?: BazaarDeclaration } = {},
): PaymentRequiredPair {
  for (const entry of accepts) {
    if (!isWellFormedRequirement(entry)) {
      const named = entry as { scheme?: string; network?: string } | undefined;
      throw new Error(`x402: malformed requirement for ${named?.scheme ?? '?'} on ${named?.network ?? '?'}`);
    }
  }
  const error = options.error ?? `${X402_HEADERS.signature} header is required`;
  const extensions: Record<string, unknown> = { ...(options.extensions ?? {}) };
  if (options.bazaar) {
    extensions.bazaar = {
      info: { input: options.bazaar.input, ...(options.bazaar.output ? { output: options.bazaar.output } : {}) },
      ...(options.bazaar.schema ? { schema: options.bazaar.schema } : {}),
    };
  }
  const v2: PaymentRequiredV2 = {
    x402Version: 2,
    error,
    resource,
    accepts,
    ...(Object.keys(extensions).length > 0 ? { extensions } : {}),
  };
  const outputSchema = options.bazaar?.output;
  const v1: PaymentRequiredV1 = {
    x402Version: 1,
    error,
    accepts: accepts.map((r) => toV1Requirement(r, resource, outputSchema)),
  };
  return { v2, v1 };
}

// ── The retry's payload ──────────────────────────────────────────────────────

export type ReadPayment =
  | { version: 2; payload: PaymentPayloadV2; raw: string }
  | { version: 1; payload: PaymentPayloadV1; raw: string }
  | { version: 0; error: string }
  | null;

/**
 * The payment a request carries: `PAYMENT-SIGNATURE` (v2) first, else
 * `X-PAYMENT` (v1). Null when neither is present; `version: 0` with a
 * reason when one is present and malformed (a 400, never a 402: the client
 * paid attention to the challenge and got the encoding wrong).
 */
export function readPaymentPayload(headers: { get(name: string): string | null }): ReadPayment {
  const v2 = headers.get(X402_HEADERS.signature);
  if (v2) {
    const decoded = decodeBase64Json(v2.trim());
    if (!decoded) return { version: 0, error: `${X402_HEADERS.signature} is not base64 JSON` };
    if (decoded.x402Version !== 2) return { version: 0, error: `${X402_HEADERS.signature} is not x402Version 2` };
    if (!isRecord(decoded.accepted) || !isRecord(decoded.payload)) {
      return { version: 0, error: `${X402_HEADERS.signature} lacks accepted or payload` };
    }
    return { version: 2, payload: decoded as unknown as PaymentPayloadV2, raw: v2.trim() };
  }
  const v1 = headers.get(X402_HEADERS.v1Payment);
  if (v1) {
    const decoded = decodeBase64Json(v1.trim());
    if (!decoded) return { version: 0, error: `${X402_HEADERS.v1Payment} is not base64 JSON` };
    if (decoded.x402Version !== 1) return { version: 0, error: `${X402_HEADERS.v1Payment} is not x402Version 1` };
    if (typeof decoded.scheme !== 'string' || typeof decoded.network !== 'string' || !isRecord(decoded.payload)) {
      return { version: 0, error: `${X402_HEADERS.v1Payment} lacks scheme, network or payload` };
    }
    return { version: 1, payload: decoded as unknown as PaymentPayloadV1, raw: v1.trim() };
  }
  return null;
}

/** The CAIP-2 form of a v1 network name, so a v1 payload can be matched against v2 offers. */
export function v1NetworkToCaip2(name: string): string {
  if (name.includes(':')) return name;
  for (const [caip2, v1] of Object.entries(V1_NETWORK_NAMES)) if (v1 === name) return caip2;
  return name;
}

// ── The answer's settle response ─────────────────────────────────────────────

/** The header a paid answer carries: v2 `PAYMENT-RESPONSE`, v1 `X-PAYMENT-RESPONSE`, the same base64 JSON. */
export function settleResponseHeader(version: 1 | 2, settle: SettleResponse): [string, string] {
  return [version === 2 ? X402_HEADERS.response : X402_HEADERS.v1Response, encodeBase64Json(settle)];
}

/** A metered handler's `Settlement-Overrides: {"amount":"500"}` (or `{"credits":"0.002"}`), or null. */
export function readSettlementOverrides(headers: { get(name: string): string | null }): { amount?: string; credits?: string } | null {
  const raw = headers.get(X402_HEADERS.settlementOverrides);
  if (!raw) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const out: { amount?: string; credits?: string } = {};
  if (isAmountString(parsed.amount)) out.amount = parsed.amount;
  if (typeof parsed.credits === 'string' || typeof parsed.credits === 'number') out.credits = String(parsed.credits);
  return out.amount !== undefined || out.credits !== undefined ? out : null;
}
