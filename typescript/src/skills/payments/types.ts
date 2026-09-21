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
  chargedDollars?: number;
  /** What was not charged, on a partial success. */
  unbilled?: string;
  unbilledDollars?: number;
  /** What the settle asked for, on a partial success. */
  requested?: string;
  requestedDollars?: number;
  remaining?: number | string;
  remainingDollars?: number;
  error?: string;
}

export interface PaymentLockResult {
  token: string;
  expiresAt: string;
  lockedAmount: number;
}
