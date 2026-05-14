/**
 * Cron expression evaluator + humanizer tests.
 *
 * `isCronExpressionDue` is called once per minute for every configured
 * schedule by the portal `cron-tick` route. A regression here silently
 * stops all crons across the cluster, so this file is the canonical
 * gating-correctness surface for function-cron. Every grammar branch
 * covered here mirrors the parser branches in `skill.ts` so adding a
 * grammar feature ALWAYS comes with a row in the table.
 *
 * `humanizeCron` is the prompt + binding-badge rendering helper that
 * keeps owners from ever seeing 5-field cron syntax. Tests pin the
 * common output shapes so a prompt/UX regression is caught here, not
 * by users.
 */

import { describe, it, expect } from 'vitest';
import {
  isCronExpressionDue,
  validateCronExpression,
  humanizeCron,
} from '../../../../src/skills/cron/skill.js';

const at = (iso: string) => new Date(iso);

describe('validateCronExpression', () => {
  it.each([
    ['* * * * *', true],
    ['*/5 * * * *', true],
    ['0 9 * * *', true],
    ['30 9 * * *', true],
    ['0 0 1 * *', true],
    ['0 12 * * 1', true],
    ['0,15,30,45 * * * *', true],
    ['0-30/10 * * * *', true],
    // Boundary checks
    ['59 23 31 12 6', true],
    ['0 0 1 1 0', true],
    // 6-field (with seconds) is NOT supported — must be 5-field.
    ['0 0 9 * * *', false],
    // 4-field is rejected.
    ['0 9 * *', false],
    // Out-of-range parts.
    ['60 * * * *', false],
    ['* 24 * * *', false],
    ['* * 32 * *', false],
    ['* * * 13 *', false],
    ['* * * * 7', false],
    // Garbage tokens.
    ['not a cron', false],
    ['', false],
    ['*/0 * * * *', false],
  ])('validateCronExpression(%j) ok=%j', (expr, ok) => {
    expect(validateCronExpression(expr).ok).toBe(ok);
  });
});

describe('isCronExpressionDue', () => {
  it('matches every-minute wildcard', () => {
    expect(isCronExpressionDue('* * * * *', at('2026-05-05T12:00:00Z'))).toBe(true);
    expect(isCronExpressionDue('* * * * *', at('2026-05-05T12:37:00Z'))).toBe(true);
  });

  it('hourly at minute 0', () => {
    expect(isCronExpressionDue('0 * * * *', at('2026-05-05T12:00:00Z'))).toBe(true);
    expect(isCronExpressionDue('0 * * * *', at('2026-05-05T12:01:00Z'))).toBe(false);
  });

  it('every 5 minutes via step', () => {
    expect(isCronExpressionDue('*/5 * * * *', at('2026-05-05T12:00:00Z'))).toBe(true);
    expect(isCronExpressionDue('*/5 * * * *', at('2026-05-05T12:05:00Z'))).toBe(true);
    expect(isCronExpressionDue('*/5 * * * *', at('2026-05-05T12:07:00Z'))).toBe(false);
  });

  it('daily at 09:30 UTC', () => {
    expect(isCronExpressionDue('30 9 * * *', at('2026-05-05T09:30:00Z'))).toBe(true);
    expect(isCronExpressionDue('30 9 * * *', at('2026-05-05T09:31:00Z'))).toBe(false);
    expect(isCronExpressionDue('30 9 * * *', at('2026-05-05T10:30:00Z'))).toBe(false);
  });

  it('weekly Mondays at 12:00 UTC (dow=1)', () => {
    // 2026-05-04 is a Monday.
    expect(isCronExpressionDue('0 12 * * 1', at('2026-05-04T12:00:00Z'))).toBe(true);
    expect(isCronExpressionDue('0 12 * * 1', at('2026-05-05T12:00:00Z'))).toBe(false);
  });

  it('comma list of minutes', () => {
    const e = '0,15,30,45 * * * *';
    expect(isCronExpressionDue(e, at('2026-05-05T12:00:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:15:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:14:00Z'))).toBe(false);
  });

  it('range with step', () => {
    const e = '0-30/10 * * * *';
    expect(isCronExpressionDue(e, at('2026-05-05T12:10:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:20:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:30:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:40:00Z'))).toBe(false);
  });

  it('monthly on day 1 at 00:00', () => {
    expect(isCronExpressionDue('0 0 1 * *', at('2026-06-01T00:00:00Z'))).toBe(true);
    expect(isCronExpressionDue('0 0 1 * *', at('2026-06-02T00:00:00Z'))).toBe(false);
  });

  it('specific month + day-of-week', () => {
    // 31 Dec 2026 at 23:59 UTC — Thursday.
    expect(isCronExpressionDue('59 23 31 12 4', at('2026-12-31T23:59:00Z'))).toBe(true);
    // Wrong year-day combo (Jan 1 is Friday in 2027).
    expect(isCronExpressionDue('59 23 31 12 4', at('2027-12-31T23:59:00Z'))).toBe(false);
  });

  it('rejects invalid expressions', () => {
    expect(isCronExpressionDue('not a cron', at('2026-05-05T12:00:00Z'))).toBe(false);
    expect(isCronExpressionDue('* * *', at('2026-05-05T12:00:00Z'))).toBe(false);
    expect(isCronExpressionDue('', at('2026-05-05T12:00:00Z'))).toBe(false);
  });

  it('seconds are ignored — only minute-precision matches', () => {
    expect(isCronExpressionDue('30 9 * * *', at('2026-05-05T09:30:45Z'))).toBe(true);
  });
});

describe('humanizeCron', () => {
  it.each([
    // Every-minute / every-N-minute steps
    ['* * * * *', 'every minute'],
    ['*/5 * * * *', 'every 5 minutes'],
    ['*/15 * * * *', 'every 15 minutes'],
    // Hourly variants
    ['0 * * * *', 'hourly'],
    ['30 * * * *', 'hourly at :30'],
    ['0 */2 * * *', 'every 2 hours'],
    // Daily variants
    ['0 9 * * *', 'daily at 09:00 UTC'],
    ['30 14 * * *', 'daily at 14:30 UTC'],
    // Weekly variants
    ['0 9 * * 1', 'weekly on Monday at 09:00 UTC'],
    ['0 9 * * 0', 'weekly on Sunday at 09:00 UTC'],
    ['0 9 * * 6', 'weekly on Saturday at 09:00 UTC'],
    // Monthly variants
    ['0 0 1 * *', 'monthly on day 1 at 00:00 UTC'],
    ['30 9 15 * *', 'monthly on day 15 at 09:30 UTC'],
    // Unrenderable expressions fall back to the literal — the function
    // never lies about the schedule. (Note that this preserves the
    // "never throw" contract; the badge / prompt will still display
    // raw syntax for exotic cases, but those cases never originate
    // from our UI/factory.)
    ['0 9 1-5 * *', '0 9 1-5 * *'],
    ['0,30 9-17 * * 1-5', '0,30 9-17 * * 1-5'],
  ])('humanizeCron(%j) === %j', (expr, expected) => {
    expect(humanizeCron(expr)).toBe(expected);
  });

  it('returns input unchanged for non-cron strings', () => {
    expect(humanizeCron('not a cron')).toBe('not a cron');
    expect(humanizeCron('')).toBe('');
  });
});
