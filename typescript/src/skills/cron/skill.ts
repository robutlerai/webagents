/**
 * `CronSkill`
 *
 * Exposes operator guidance for **function-cron**: the cluster
 * `portal-functions-cron-tick` CronJob hits
 * `/api/internal/functions/cron-tick` once per minute (UTC), which scans
 * every active agent's `skills.cron.schedules[]`, picks out schedules due
 * this minute, and synchronously calls `FunctionRuntimeSkill.invoke(use, ctx)`
 * — same code path as a manual replay or a `custom_tools` call. The local
 * `webagentsd` daemon runs the equivalent loop for `webagents dev`.
 *
 * Robutler ships TWO first-class scheduling systems. Neither is legacy.
 * Pick the right one for the job:
 *   - **Function-cron** (this skill) — many bindings per agent, sandboxed
 *     functions, headless. Use for ETL, reports, notifications, polling.
 *   - **Agent-cron** ("Automatic Runs") — one schedule per agent, fires a
 *     full LLM run in a background chat. Use for conversational summaries
 *     the owner reads in their DMs. Configured via `enabledTools.schedule`
 *     and dispatched by `/api/internal/agents/scheduled` → `/run`.
 *
 * Each schedule entry has:
 *   - `id`       — stable id (used in metrics, audit, dedupe key).
 *   - `cron`     — 5-field cron string (UTC). USERS NEVER SEE THIS — agents
 *                  convert from plain-English frequency before saving.
 *   - `use`      — function name (REQUIRED). Function-cron always invokes
 *                  a user-declared function; there is no "host agent main
 *                  loop" shorthand — that's what agent-cron is for.
 *   - `enabled?` — when false, dispatcher skips this entry.
 *   - `description?` — free-form human description shown in the UI.
 */

import { Skill } from '../../core/skill';
import { prompt } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';

/** A single cron schedule entry. */
export interface CronStructuredSchedule {
  frequency: 'hourly' | 'every_6h' | 'every_12h' | 'daily' | 'weekly';
  /** Local-tz time in 'HH:MM' (24h). Only meaningful for daily/weekly. */
  preferredTime?: string;
  /** 0=Sunday … 6=Saturday. Only meaningful for weekly. */
  preferredDay?: number;
  /** IANA timezone (e.g. `America/Los_Angeles`). */
  timeZone?: string;
}

export interface CronScheduleEntry {
  id: string;
  /**
   * Structured schedule shape (§P2). Preferred for ALL new rows; the
   * dispatcher uses this directly. Mutually exclusive with `cron` in
   * practice — the portal's structured-shape migration moves every
   * legacy row to this field.
   */
  schedule?: CronStructuredSchedule;
  /**
   * Legacy raw 5-field UTC cron string. Still accepted by the dispatcher
   * for backwards-compat with un-migrated rows.
   */
  cron?: string;
  /**
   * Function name to invoke. REQUIRED — function-cron always runs a
   * user-declared function. Use agent-cron (`enabledTools.schedule`) for
   * host-agent-main-loop scheduling.
   */
  use: string;
  enabled?: boolean;
  /** Free-form description shown in the Functions pane binding badge. */
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
      // §P2: structured `schedule` is preferred; the dispatcher validates
      // it at the Zod layer in `update-agent-capabilities.ts`. We only
      // need to re-check the legacy raw-cron path here.
      if (s.schedule) continue;
      if (!s.cron) {
        errors.push({ id: s.id, error: 'either `schedule` or `cron` is required' });
        continue;
      }
      const v = validateCronExpression(s.cron);
      if (!v.ok) errors.push({ id: s.id, error: v.error ?? 'invalid cron' });
    }
    return errors;
  }

  /** Minimum allowed period across schedules (used by per-tier rate caps). */
  static readonly MIN_INTERVAL_SEC = MIN_INTERVAL_SEC;

  @prompt({ priority: 70, name: 'cronRuntime', scope: 'all' })
  cronRuntime(_ctx: Context): string {
    const lines: string[] = ['## Cron schedules (function-cron)'];

    if (this.schedules.length === 0) {
      lines.push(
        'No cron schedules are configured on this agent. To add one, call `update_agent_capabilities` (or `add_to_skill` with `skill: "cron"`).',
      );
    } else {
      lines.push('Active schedules on this agent:');
      for (const s of this.schedules) {
        // Structured `schedule` wins over raw `cron` (§P2). Either way
        // the rendered label is plain English — owners and the LLM
        // both see "daily 9 AM UTC", never asterisks.
        const label = s.schedule
          ? `${s.schedule.frequency}${s.schedule.preferredTime ? `@${s.schedule.preferredTime}` : ''}${s.schedule.timeZone ? ` ${s.schedule.timeZone}` : ''}`
          : s.cron
            ? humanizeCron(s.cron)
            : 'unscheduled';
        lines.push(
          `- ${s.id} — ${label} → ${s.use}${s.description ? ` — ${s.description}` : ''}`,
        );
      }
    }

    lines.push(
      '',
      '### Behavior',
      '- **function-cron** runs **sandboxed functions** (headless, no LLM). For a recurring LLM run in a background chat, use **agent-cron** ("Automatic Runs", `enabledTools.schedule`).',
      '- **Never expose raw cron syntax to the owner.** Gather plain-English frequency ("every 15 min", "weekdays at 9 AM PT") and emit the structured `schedule: { frequency, preferredTime?, preferredDay?, timeZone? }` block. When confirming, render the human label, never asterisks.',
      `- Evaluated every minute; min interval is **${MIN_INTERVAL_SEC}s**. Sub-minute requests silently coarsen.`,
      '- `use` is REQUIRED — function-cron always invokes a declared function.',
      '- **Headless**: no chat user, no `paymentToken` from a caller. Design the function to be self-contained and idempotent; user-facing tools (search/delegate/notify) only fire if owner pre-authorized them.',
      '- **Idempotent**: same `(scheduleId, UTC minute)` is deduped via an idempotency key; an in-flight lock prevents a long-running job from overlapping itself.',
      '- Failures land in the function-invocations log (Functions pane → per-function Activity). Point owners there when debugging — you cannot read it from inside the chat.',
    );

    return lines.join('\n');
  }
}

/**
 * Best-effort plain-English rendering of a 5-field UTC cron expression for
 * the system prompt and the binding-badge UI. Returns the original string
 * unchanged when the pattern doesn't match one of the common shapes — never
 * throws, never lies about what the schedule means.
 *
 * Covers the shapes the factory is allowed to emit (and the friendly picker
 * produces post-§P2): minute steps, hourly, daily at HH:MM, weekly on a
 * named day at HH:MM, monthly on day N at HH:MM, every N minutes/hours.
 */
export function humanizeCron(expr: string): string {
  if (typeof expr !== 'string') return String(expr);
  const fields = expr.trim().split(/\s+/);
  if (fields.length !== 5) return expr;
  const [m, h, dom, mon, dow] = fields;

  const isStar = (f: string) => f === '*';
  const stepOf = (f: string): number | null => {
    const r = /^\*\/(\d+)$/.exec(f);
    return r ? parseInt(r[1], 10) : null;
  };
  const intOf = (f: string): number | null => {
    if (!/^\d+$/.test(f)) return null;
    const n = parseInt(f, 10);
    return Number.isFinite(n) ? n : null;
  };
  const pad = (n: number) => String(n).padStart(2, '0');

  const DOW_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

  const mStep = stepOf(m);
  const hStep = stepOf(h);
  const mInt = intOf(m);
  const hInt = intOf(h);
  const domInt = intOf(dom);
  const dowInt = intOf(dow);
  const monAny = isStar(mon);

  // every minute (literal `* * * * *`)
  if (isStar(m) && isStar(h) && isStar(dom) && monAny && isStar(dow)) {
    return 'every minute';
  }
  // every N minutes (e.g. `*/5 * * * *`)
  if (mStep && isStar(h) && isStar(dom) && monAny && isStar(dow)) {
    return mStep === 1 ? 'every minute' : `every ${mStep} minutes`;
  }
  // every N hours on the minute
  if (mInt === 0 && hStep && isStar(dom) && monAny && isStar(dow)) {
    return hStep === 1 ? 'hourly' : `every ${hStep} hours`;
  }
  // hourly at minute X
  if (mInt !== null && isStar(h) && isStar(dom) && monAny && isStar(dow)) {
    return mInt === 0 ? 'hourly' : `hourly at :${pad(mInt)}`;
  }
  // daily at HH:MM
  if (mInt !== null && hInt !== null && isStar(dom) && monAny && isStar(dow)) {
    return `daily at ${pad(hInt)}:${pad(mInt)} UTC`;
  }
  // weekly on <day> at HH:MM
  if (mInt !== null && hInt !== null && isStar(dom) && monAny && dowInt !== null && dowInt >= 0 && dowInt <= 6) {
    return `weekly on ${DOW_NAMES[dowInt]} at ${pad(hInt)}:${pad(mInt)} UTC`;
  }
  // monthly on day N at HH:MM
  if (mInt !== null && hInt !== null && domInt !== null && monAny && isStar(dow)) {
    return `monthly on day ${domInt} at ${pad(hInt)}:${pad(mInt)} UTC`;
  }
  return expr;
}
