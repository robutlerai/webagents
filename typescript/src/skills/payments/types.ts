/**
 * Payment types for x402 / Robutler payments
 */

export interface PaymentVerifyResult {
  valid: boolean;
  balance?: number;
  invalidReason?: string;
}

/**
 * `POST /api/payments/settle`, as `readSettleResult` normalises it
 * (2026-09-18). `success: true` means the charge COMMITTED, not that it was
 * charged in full: since the portal's S-148 fix a short lock is charged what
 * it holds and the rest is unbilled, and the answer then says `partial:
 * true` with `unbilled` and `requested`. `charged + unbilled = requested`.
 * The amounts come as nanocent decimal strings beside a `...Dollars` number.
 */
export interface PaymentSettleResult {
  success: boolean;
  /** True when LESS than requested was charged. Never read a partial success as charged in full. */
  partial?: boolean;
  /** What was charged: a nanocent decimal string on the current platform (a number on older ones). */
  charged?: number | string;
  /** What was charged, in credits. */
  chargedCredits?: number;
  /** @deprecated Same value as `chargedCredits` (the platform's older name). */
  chargedDollars?: number;
  /** What was not charged, on a partial success. */
  unbilled?: string;
  unbilledCredits?: number;
  /** @deprecated Same value as `unbilledCredits`. */
  unbilledDollars?: number;
  /** What the settle asked for, on a partial success. */
  requested?: string;
  requestedCredits?: number;
  /** @deprecated Same value as `requestedCredits`. */
  requestedDollars?: number;
  remaining?: number | string;
  remainingCredits?: number;
  /** @deprecated Same value as `remainingCredits`. */
  remainingDollars?: number;
  error?: string;
  /**
   * True when the platform answered a repeat under this settle's
   * Idempotency-Key (2026-09-26): the numbers are the FIRST settle's and this
   * call charged nothing.
   */
  replayed?: boolean;
}

export interface PaymentLockResult {
  token: string;
  expiresAt: string;
  lockedAmount: number;
}
