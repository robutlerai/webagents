/**
 * Provider signature verification helpers.
 *
 * Each helper is intentionally framework-agnostic — they consume `Request`
 * objects (the Web Fetch API standard, supported by Node 18+, Bun, Deno,
 * Cloudflare Workers, and Next.js Edge / Node runtimes). Returns true when
 * the signature is valid.
 */
import { createHmac, timingSafeEqual } from 'node:crypto';

/**
 * Slack request signing — see https://api.slack.com/authentication/verifying-requests-from-slack
 *
 * Computes `v0=` + HMAC-SHA256 over `v0:${timestamp}:${rawBody}` using the
 * signing secret and constant-time-compares to `X-Slack-Signature`. Reject
 * timestamps older than 5 minutes to prevent replay.
 */
export function verifySlackSignature(args: {
  signingSecret: string;
  timestamp: string;
  signature: string;
  rawBody: string;
}): boolean {
  if (!args.signingSecret || !args.timestamp || !args.signature) return false;
  const ts = Number(args.timestamp);
  if (!Number.isFinite(ts)) return false;
  if (Math.abs(Date.now() / 1000 - ts) > 60 * 5) return false;
  const base = `v0:${args.timestamp}:${args.rawBody}`;
  const mac = createHmac('sha256', args.signingSecret).update(base).digest('hex');
  const expected = `v0=${mac}`;
  return constantTimeEqual(expected, args.signature);
}

/**
 * Twilio signature verification — `X-Twilio-Signature` HMAC-SHA1 over the
 * full URL + sorted form params, using the auth token as key.
 *
 * Spec: https://www.twilio.com/docs/usage/security#validating-requests
 */
export function verifyTwilioSignature(args: {
  authToken: string;
  url: string;
  params: Record<string, string>;
  signature: string;
}): boolean {
  if (!args.authToken || !args.signature) return false;
  const sortedKeys = Object.keys(args.params).sort();
  const concatenated = args.url + sortedKeys.map((k) => k + args.params[k]).join('');
  const mac = createHmac('sha1', args.authToken).update(concatenated).digest('base64');
  return constantTimeEqual(mac, args.signature);
}

/**
 * Meta `X-Hub-Signature-256` verification — HMAC-SHA256 over the raw body
 * using the app secret. Supports dual-secret rotation: pass the previous
 * secret as the second array element and any match passes.
 *
 * Spec: https://developers.facebook.com/docs/graph-api/webhooks/getting-started/#validating-payloads
 */
export function verifyMetaSignature(args: {
  appSecrets: string[];
  signatureHeader: string;
  rawBody: string;
}): boolean {
  if (!args.signatureHeader || !args.signatureHeader.startsWith('sha256=')) return false;
  const provided = args.signatureHeader.slice('sha256='.length);
  for (const secret of args.appSecrets) {
    if (!secret) continue;
    const mac = createHmac('sha256', secret).update(args.rawBody).digest('hex');
    if (constantTimeEqual(mac, provided)) return true;
  }
  return false;
}

/**
 * Discord interactions endpoint requires Ed25519 verification using
 * X-Signature-Ed25519 + X-Signature-Timestamp + raw body, signed by the
 * application's public key.
 *
 * Spec: https://discord.com/developers/docs/interactions/receiving-and-responding#security-and-authorization
 *
 * Uses Web Crypto's SubtleCrypto so the helper works in Edge runtimes.
 */
export async function verifyDiscordSignature(args: {
  publicKey: string;
  signatureHex: string;
  timestamp: string;
  rawBody: string;
}): Promise<boolean> {
  if (!args.publicKey || !args.signatureHex || !args.timestamp) return false;
  try {
    const subtle = (globalThis.crypto as Crypto | undefined)?.subtle;
    if (!subtle) return false;
    const keyBytes = hexToBytes(args.publicKey);
    const sig = hexToBytes(args.signatureHex);
    const data = new TextEncoder().encode(args.timestamp + args.rawBody);
    const key = await subtle.importKey(
      'raw',
      keyBytes,
      { name: 'Ed25519' } as unknown as AlgorithmIdentifier,
      false,
      ['verify'],
    );
    return await subtle.verify({ name: 'Ed25519' } as AlgorithmIdentifier, key, sig, data);
  } catch {
    return false;
  }
}

/**
 * X (Twitter) Account Activity webhook signature verification.
 *
 * X computes `sha256=` + base64(HMAC-SHA256(consumerSecret, rawBody)) and
 * sends it as `X-Twitter-Webhooks-Signature`. Same algorithm is used to
 * generate the CRC challenge response, so both share `xHmacSha256Base64`.
 *
 * Spec: https://developer.twitter.com/en/docs/twitter-api/enterprise/account-activity-api/guides/securing-webhooks
 */
export function verifyXWebhookSignature(args: {
  consumerSecret: string;
  signatureHeader: string;
  rawBody: string;
}): boolean {
  if (!args.signatureHeader || !args.signatureHeader.startsWith('sha256=')) return false;
  const provided = args.signatureHeader.slice('sha256='.length);
  const expected = xHmacSha256Base64(args.consumerSecret, args.rawBody);
  return constantTimeEqual(expected, provided);
}

/**
 * Build the X CRC (Challenge-Response Check) challenge response. X pings the
 * webhook URL with `?crc_token=<token>` and expects
 * `{ "response_token": "sha256=<base64-hmac>" }` in return.
 */
export function buildXCrcResponse(consumerSecret: string, crcToken: string): string {
  return `sha256=${xHmacSha256Base64(consumerSecret, crcToken)}`;
}

function xHmacSha256Base64(secret: string, payload: string): string {
  return createHmac('sha256', secret).update(payload).digest('base64');
}

/**
 * Google Chat App event endpoints are JWT-verified. Google signs every
 * request with `chat@system.gserviceaccount.com`'s private key; we verify
 * against Google's published JWKS at:
 *   https://www.googleapis.com/service_accounts/v1/jwk/chat@system.gserviceaccount.com
 *
 * Caller passes the project number / project id as `expectedAudience` so a
 * stolen JWT for another Workspace project can't be replayed against ours.
 *
 * Returns true when the JWT validates AND the `aud` claim matches.
 *
 * The JWKS fetch is cached for 5 minutes per JWKS URL via a small in-process
 * Map so high-throughput hosts don't hit Google for every event.
 */
export async function verifyGoogleChatJwt(args: {
  jwt: string;
  expectedAudience: string;
  jwksUrl?: string;
  now?: number;
}): Promise<boolean> {
  const jwt = args.jwt?.trim();
  if (!jwt || jwt.split('.').length !== 3) return false;
  const url = args.jwksUrl ?? GOOGLE_CHAT_JWKS_URL;
  try {
    const subtle = (globalThis.crypto as Crypto | undefined)?.subtle;
    if (!subtle) return false;
    const [headerB64, payloadB64, signatureB64] = jwt.split('.');
    const header = JSON.parse(b64UrlDecodeToString(headerB64)) as { kid?: string; alg?: string };
    const payload = JSON.parse(b64UrlDecodeToString(payloadB64)) as {
      iss?: string;
      aud?: string | string[];
      exp?: number;
      email?: string;
    };
    if (header.alg !== 'RS256') return false;
    if (payload.iss !== 'chat@system.gserviceaccount.com' && payload.email !== 'chat@system.gserviceaccount.com') {
      return false;
    }
    const aud = Array.isArray(payload.aud) ? payload.aud : payload.aud ? [payload.aud] : [];
    if (!aud.includes(args.expectedAudience)) return false;
    const now = Math.floor((args.now ?? Date.now()) / 1000);
    if (typeof payload.exp !== 'number' || payload.exp + 60 < now) return false;
    const jwks = await loadJwks(url);
    const key = jwks.keys.find((k) => k.kid === header.kid) ?? jwks.keys[0];
    if (!key) return false;
    const cryptoKey = await subtle.importKey(
      'jwk',
      key as JsonWebKey,
      { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
      false,
      ['verify'],
    );
    const signed = new TextEncoder().encode(`${headerB64}.${payloadB64}`);
    const sig = b64UrlDecodeToBytes(signatureB64);
    return await subtle.verify('RSASSA-PKCS1-v1_5', cryptoKey, sig, signed);
  } catch {
    return false;
  }
}

const GOOGLE_CHAT_JWKS_URL =
  'https://www.googleapis.com/service_accounts/v1/jwk/chat@system.gserviceaccount.com';
const JWKS_TTL_MS = 5 * 60 * 1000;

interface CachedJwks {
  fetchedAt: number;
  jwks: { keys: Array<JsonWebKey & { kid?: string }> };
}
const jwksCache = new Map<string, CachedJwks>();

async function loadJwks(url: string): Promise<{ keys: Array<JsonWebKey & { kid?: string }> }> {
  const cached = jwksCache.get(url);
  if (cached && Date.now() - cached.fetchedAt < JWKS_TTL_MS) return cached.jwks;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`jwks_fetch_failed_${r.status}`);
  const jwks = (await r.json()) as { keys: Array<JsonWebKey & { kid?: string }> };
  jwksCache.set(url, { fetchedAt: Date.now(), jwks });
  return jwks;
}

function b64UrlDecodeToString(b64: string): string {
  return Buffer.from(b64.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8');
}

function b64UrlDecodeToBytes(b64: string): Uint8Array<ArrayBuffer> {
  // Allocate a fresh ArrayBuffer-backed Uint8Array so WebCrypto's BufferSource
  // typing is satisfied on TS5+ (SharedArrayBuffer-backed views are rejected).
  const src = Buffer.from(b64.replace(/-/g, '+').replace(/_/g, '/'), 'base64');
  const out = new Uint8Array(new ArrayBuffer(src.length));
  out.set(src);
  return out as Uint8Array<ArrayBuffer>;
}

function hexToBytes(hex: string): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(new ArrayBuffer(hex.length / 2));
  for (let i = 0; i < out.length; i += 1) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out as Uint8Array<ArrayBuffer>;
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}
