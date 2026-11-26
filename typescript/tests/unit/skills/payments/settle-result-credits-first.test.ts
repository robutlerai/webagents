/**
 * Credit amounts are read under their `...Credits` names first. The platform answers both
 * `<name>Credits` and the older `<name>Dollars` (the same value); an older platform answers only
 * the second. The reader prefers the first and fills both, so code reading either name keeps
 * working and the SDK no longer depends on the older names.
 */
import { describe, it, expect, vi } from 'vitest';
import { readSettleResult } from '../../../../src/skills/payments/settle-result';

describe('readSettleResult reads credit amounts under their credits names first', () => {
  it('a credits-only answer fills both names', () => {
    const r = readSettleResult({ success: true, charged: '5000000', chargedCredits: 0.005, remainingCredits: 1.5 }, 't');
    expect(r.chargedCredits).toBe(0.005);
    expect(r.chargedDollars).toBe(0.005);
    expect(r.remainingCredits).toBe(1.5);
    expect(r.remainingDollars).toBe(1.5);
  });

  it('an older answer with only the dollars names still reads, under both names', () => {
    const r = readSettleResult({ success: true, chargedDollars: 0.004, remainingDollars: 0 }, 't');
    expect(r.chargedCredits).toBe(0.004);
    expect(r.chargedDollars).toBe(0.004);
    expect(r.remainingCredits).toBe(0);
  });

  it('when both are present the credits name wins', () => {
    const r = readSettleResult({ success: true, chargedCredits: 0.003, chargedDollars: 99 }, 't');
    expect(r.chargedCredits).toBe(0.003);
    expect(r.chargedDollars).toBe(0.003);
  });

  it('a partial settle is reported from the credits names', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const r = readSettleResult({ success: true, partial: true, chargedCredits: 0.003, unbilledCredits: 0.002, requestedCredits: 0.005 }, 'finalize');
    expect(r.partial).toBe(true);
    expect(r.unbilledCredits).toBe(0.002);
    expect(r.unbilledDollars).toBe(0.002);
    expect(warn.mock.calls[0][0]).toContain('charged 0.003 of 0.005 (unbilled 0.002)');
    warn.mockRestore();
  });
});
