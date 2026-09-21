/**
 * Reading `POST /api/payments/settle` (2026-09-18).
 *
 * WHY. Since the portal's S-148 fix `settlePaymentToken` charges a short
 * lock what it holds instead of nothing, and the route answers `success:
 * true` together with `partial`, and, when partial, `unbilled` and
 * `requested` (app/api/payments/settle/route.ts). The SDK callers read only
 * `success`, so a partial settle read as charged in full: the payment skill
 * marked the run paid, and an x402 settle reported a valid payment for an
 * amount it did not collect. Every caller now goes through this reader,
 * which keeps the platform's fields, logs a partial at warn, and leaves it to
 * the caller to report it as partial. The Python twin is
 * `webagents/agents/skills/robutler/payments/settle_result.py`.
 */

import type { PaymentSettleResult } from './types';

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

const numberOrUndefined = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const amountOrUndefined = (v: unknown): number | string | undefined =>
  typeof v === 'string' || (typeof v === 'number' && Number.isFinite(v)) ? v : undefined;

/** The settle answer as the SDK reads it; `where` names the caller in the warn line. */
export function readSettleResult(raw: unknown, where: string): PaymentSettleResult {
  const body = isRecord(raw) ? raw : {};
  const success = body.success === true;
  const out: PaymentSettleResult = { success, partial: success && body.partial === true };
  const charged = amountOrUndefined(body.charged);
  if (charged !== undefined) out.charged = charged;
  const remaining = amountOrUndefined(body.remaining);
  if (remaining !== undefined) out.remaining = remaining;
  for (const k of ['chargedDollars', 'unbilledDollars', 'requestedDollars', 'remainingDollars'] as const) {
    const v = numberOrUndefined(body[k]);
    if (v !== undefined) out[k] = v;
  }
  if (typeof body.unbilled === 'string') out.unbilled = body.unbilled;
  if (typeof body.requested === 'string') out.requested = body.requested;
  if (typeof body.error === 'string') out.error = body.error;
  if (out.partial) {
    console.warn(
      `[payments] ${where}: PARTIAL settle, charged ${out.chargedDollars ?? out.charged ?? '?'} of ` +
        `${out.requestedDollars ?? out.requested ?? '?'} (unbilled ${out.unbilledDollars ?? out.unbilled ?? '?'})`,
    );
  }
  return out;
}
