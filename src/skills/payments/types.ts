/**
 * Payment types for x402 / Roborum payments
 */

export interface PaymentVerifyResult {
  valid: boolean;
  balance?: number;
  invalidReason?: string;
}

export interface PaymentSettleResult {
  success: boolean;
  charged?: number;
  remaining?: number;
  error?: string;
}

export interface PaymentLockResult {
  token: string;
  expiresAt: string;
  lockedAmount: number;
}
