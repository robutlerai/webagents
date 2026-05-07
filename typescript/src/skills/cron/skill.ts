/**
 * `CronSkill`
 *
 * Reads `agent_configs.skills.cron.schedules[]` and exposes the schedule
 * list to the cloud cron dispatcher (`/api/internal/agents/scheduled`) and
 * to the localhost `webagentsd` 1-min loop. The skill itself does NOT run
 * cron-triggered code in-process — the dispatcher fans out via BullMQ
 * (cloud) or the daemon's setInterval (local) and ultimately calls
 * `FunctionRuntimeSkill.invoke(use, ctx)`.
 *
 * Each schedule entry has:
 *   - `id`     — stable id (used as Redis dedupe bucket).
 *   - `cron`   — 5-field cron string (UTC).
 *   - `use?`   — function name (omit / null → run host agent main loop).
 *   - `enabled?` — when false, dispatcher skips this entry.
 *
 * Legacy `enabledTools.schedule` is removed via a one-shot pre-launch
 * migration script; this skill only reads `skills.cron.schedules[]`.
 */

import { Skill } from '../../core/skill';
import { prompt } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';

/** A single cron schedule entry. */
export interface CronScheduleEntry {
  id: string;
  cron: string;
  /** Function name to invoke; omit / null to run the host agent main loop. */
  use?: string | null;
  enabled?: boolean;
  /** Free-form description shown in the Cron skill pane. */
  description?: string;
}

export interface CronSkillConfig extends SkillConfig {
  schedules?: CronScheduleEntry[];
}

/** Minimum interval between fires per schedule id (60s in cloud free tier). */
const MIN_INTERVAL_SEC = 60;

/** Re-exported util for dispatcher-side validation. */
export function validateCronExpression(expr: string): { ok: boolean; error?: string } {
  if (typeof expr !== 'string') return { ok: false, error: 'cron must be a string' };
  const fields = expr.trim().split(/\s+/);
  if (fields.length !== 5) {
    return { ok: false, error: `cron must have 5 fields (got ${fields.length})` };
  }
  const ranges = [
    { min: 0, max: 59 }, // minute
    { min: 0, max: 23 }, // hour
    { min: 1, max: 31 }, // day of month
    { min: 1, max: 12 }, // month
    { min: 0, max: 6 },  // day of week
  ];
  for (let i = 0; i < 5; i++) {
    const f = fields[i];
    const { min, max } = ranges[i];
    if (!isCronField(f, min, max)) {
      return { ok: false, error: `cron field ${i + 1} ("${f}") is invalid` };
    }
  }
  return { ok: true };
}

/**
 * Returns true when the 5-field cron expression matches the given moment
 * at minute granularity (UTC). Used by the cron-tick CronJob to decide
 * which schedules to fire each minute. Mirrors the validator's grammar
 * (`*`, `*\/n`, `a-b`, `a,b,c`, plain integers).
 */
export function isCronExpressionDue(expr: string, at: Date = new Date()): boolean {
  const v = validateCronExpression(expr);
  if (!v.ok) return false;
  const fields = expr.trim().split(/\s+/);
  const m = at.getUTCMinutes();
  const h = at.getUTCHours();
  const dom = at.getUTCDate();
  const mon = at.getUTCMonth() + 1;
  const dow = at.getUTCDay();
  return (
    matchCronField(fields[0], m, 0, 59) &&
    matchCronField(fields[1], h, 0, 23) &&
    matchCronField(fields[2], dom, 1, 31) &&
    matchCronField(fields[3], mon, 1, 12) &&
    matchCronField(fields[4], dow, 0, 6)
  );
}

function matchCronField(f: string, value: number, min: number, max: number): boolean {
  return f.split(',').some((part) => {
    const stepMatch = /^([\d*\-]+)\/(\d+)$/.exec(part);
    const base = stepMatch ? stepMatch[1] : part;
    const step = stepMatch ? parseInt(stepMatch[2], 10) : 1;
    let lo = min;
    let hi = max;
    if (base === '*') {
      // full range
    } else {
      const rangeMatch = /^(\d+)-(\d+)$/.exec(base);
      if (rangeMatch) {
        lo = parseInt(rangeMatch[1], 10);
        hi = parseInt(rangeMatch[2], 10);
      } else {
        const n = parseInt(base, 10);
        if (!Number.isFinite(n)) return false;
        lo = n;
        hi = n;
      }
    }
    if (value < lo || value > hi) return false;
    return (value - lo) % step === 0;
  });
}

function isCronField(f: string, min: number, max: number): boolean {
  if (f === '*') return true;
  // Allow commas, ranges, steps (`*/5`, `1-30/2`, `0,15,30`).
  return f.split(',').every((part) => {
    const stepMatch = /^([\d*\-]+)\/(\d+)$/.exec(part);
    const base = stepMatch ? stepMatch[1] : part;
    const step = stepMatch ? parseInt(stepMatch[2], 10) : null;
    if (step !== null && (!Number.isFinite(step) || step <= 0)) return false;
    if (base === '*') return true;
    const rangeMatch = /^(\d+)-(\d+)$/.exec(base);
    if (rangeMatch) {
      const a = parseInt(rangeMatch[1], 10);
      const b = parseInt(rangeMatch[2], 10);
      return a >= min && b <= max && a <= b;
    }
    const n = parseInt(base, 10);
    return Number.isFinite(n) && n >= min && n <= max;
  });
}

export class CronSkill extends Skill {
  readonly name = 'cron';
  readonly dependencies = ['function-runtime'] as const;

  private readonly schedules: CronScheduleEntry[];

  constructor(config: CronSkillConfig = {}) {
    super(config);
    this.schedules = (config.schedules ?? []).filter((s) => s.enabled !== false);
  }

  /** All registered schedule entries. */
  list(): readonly CronScheduleEntry[] {
    return this.schedules;
  }

  /** Schedules grouped by their function-name target (for `Used By` UI). */
  groupedByFunction(): Map<string | null, CronScheduleEntry[]> {
    const out = new Map<string | null, CronScheduleEntry[]>();
    for (const s of this.schedules) {
      const key = s.use ?? null;
      const list = out.get(key) ?? [];
      list.push(s);
      out.set(key, list);
    }
    return out;
  }

  /**
   * Validates the configured schedules. Returns errors so the save route
   * can reject malformed input before persisting `agent_configs`.
   */
  validate(): Array<{ id: string; error: string }> {
    const errors: Array<{ id: string; error: string }> = [];
    const seen = new Set<string>();
    for (const s of this.schedules) {
      if (!s.id) {
        errors.push({ id: '', error: 'schedule.id is required' });
        continue;
      }
      if (seen.has(s.id)) {
        errors.push({ id: s.id, error: 'duplicate schedule id' });
        continue;
      }
      seen.add(s.id);
      const v = validateCronExpression(s.cron);
      if (!v.ok) errors.push({ id: s.id, error: v.error ?? 'invalid cron' });
    }
    return errors;
  }

  /** Minimum allowed period across schedules (used by per-tier rate caps). */
  static readonly MIN_INTERVAL_SEC = MIN_INTERVAL_SEC;

  @prompt({ priority: 70, name: 'cronRuntime', scope: 'all' })
  cronRuntime(_ctx: Context): string {
    if (this.schedules.length === 0) return '';
    const lines = this.schedules.map(
      (s) => `- ${s.id} ("${s.cron}") -> ${s.use ?? 'host agent'}`,
    );
    return `Active cron schedules on this agent:\n${lines.join('\n')}`;
  }
}
