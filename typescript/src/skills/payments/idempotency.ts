/**
 * Idempotency keys for `POST /api/payments/settle` (2026-09-26), the twin of
 * `python/webagents/agents/skills/robutler/payments/idempotency.py`. The
 * shared fixture `python/tests/fixtures/payments/settle_idempotency.json`
 * pins the names and the derivation for both SDKs and for the platform.
 *
 * WHY. The platform charged again for every repeat of a settle against a
 * lock that still held more than the charge, so a lost answer could charge a
 * payer several times when a client retried (S-254). The platform now records
 * each settle under the caller's `Idempotency-Key` and answers a repeat with
 * the first result, charging nothing, provided the key is the same. So every
 * settle this SDK sends carries one, and the value must be STABLE for a retry
 * of the same settle and NEW for a genuinely new one:
 *
 *   - `settleIdempotencyKey(lockId, purpose)` names a settle the skill's
 *     lifecycle can identify: one lock, one purpose (`usage`, `agent_fee`,
 *     `release`). Deriving it from those two, with no counter, is what makes
 *     a second finalize on the same context (the HTTP client's retry, or the
 *     Python run loop's error paths, which finalize twice) a replay rather
 *     than a second charge. The invariant: a lock is settled for `usage` and
 *     for `agent_fee` at most once each, at finalize. A transport that must
 *     settle one lock twice for the same purpose widens the purpose; it never
 *     reuses one.
 *   - `freshSettleIdempotencyKey(scope)` is for a settle nothing can name: an
 *     x402 settle by token (no lock in hand) or a raw client call. Minted once
 *     per call, so the call's own retry reuses it and two calls are two
 *     settles.
 *   - A PER-CALL settle (wave-0 review finding 5, 2026-09-26) names the tool
 *     call it charges for: `settle:<lockId>:<purpose>:<callId>`. Two settles
 *     for one purpose on one lock, made for two tool calls, derived the SAME
 *     key before, and the platform answered the second with the first's
 *     numbers, dropping its charge. With the call id in the purpose they are
 *     two keys; a retry of either still replays.
 */

/** The request header a settle's key travels in. */
export const IDEMPOTENCY_KEY_HEADER = 'Idempotency-Key';
/** The body field carrying the same value, for a client that cannot set headers. */
export const IDEMPOTENCY_KEY_BODY_FIELD = 'idempotencyKey';
/** Set by the platform on the answer to a repeated settle: nothing was charged by that call. */
export const IDEMPOTENT_REPLAYED_HEADER = 'Idempotent-Replayed';
/** The platform's limit; a derived key is far shorter. */
export const IDEMPOTENCY_KEY_MAX_LENGTH = 255;

/** The settles the payment skill's lifecycle names, one of each per lock. */
export type SettlePurpose = 'usage' | 'agent_fee' | 'release';
/** A purpose is one word: the three above, or a charge type for an amount settle. */
const PURPOSE_PATTERN = /^[a-z0-9_]+$/;
/** A tool-call id as the providers mint them (`call_…`, `toolu_…`): one token, no separators the key uses. */
const CALL_ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

/**
 * `settle:<lockId>:<purpose>`: the same settle derives the same key. With a
 * `callId`, `settle:<lockId>:<purpose>:<callId>`: the same settle for the
 * same tool call derives the same key, and another call's never does.
 */
export function settleIdempotencyKey(lockId: string, purpose: SettlePurpose | string, callId?: string): string {
  if (!lockId) throw new Error('settleIdempotencyKey: a lock id is required');
  if (!PURPOSE_PATTERN.test(purpose)) throw new Error(`settleIdempotencyKey: invalid purpose ${JSON.stringify(purpose)}`);
  if (callId === undefined) return `settle:${lockId}:${purpose}`;
  if (!CALL_ID_PATTERN.test(callId)) throw new Error(`settleIdempotencyKey: invalid call id ${JSON.stringify(callId)}`);
  return `settle:${lockId}:${purpose}:${callId}`;
}

/** `settle:<scope>:<uuid4>`: a key for a settle nothing can name, minted once per call. */
export function freshSettleIdempotencyKey(scope: 'x402' | 'redeem' | 'client'): string {
  return `settle:${scope}:${crypto.randomUUID()}`;
}

/** The headers a settle request carries for its key. */
export function idempotencyHeaders(key: string): Record<string, string> {
  return { [IDEMPOTENCY_KEY_HEADER]: key };
}
