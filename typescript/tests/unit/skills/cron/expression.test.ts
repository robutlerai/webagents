/**
 * Cron expression "is due now" evaluator tests.
 * The portal cron-tick CronJob calls `isCronExpressionDue` once per minute
 * for every configured schedule, so accuracy here translates directly to
 * "scheduled functions actually run".
 */

import { describe, it, expect } from 'vitest';
import {
  isCronExpressionDue,
  validateCronExpression,
} from '../../../../src/skills/cron/skill.js';

const at = (iso: string) => new Date(iso);

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
    // 0-30/10 → 0, 10, 20, 30
    const e = '0-30/10 * * * *';
    expect(isCronExpressionDue(e, at('2026-05-05T12:10:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:20:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:30:00Z'))).toBe(true);
    expect(isCronExpressionDue(e, at('2026-05-05T12:40:00Z'))).toBe(false);
  });

  it('returns false on invalid expression', () => {
    expect(validateCronExpression('not a cron').ok).toBe(false);
    expect(isCronExpressionDue('not a cron', at('2026-05-05T12:00:00Z'))).toBe(false);
    expect(isCronExpressionDue('* * *', at('2026-05-05T12:00:00Z'))).toBe(false);
  });
});
