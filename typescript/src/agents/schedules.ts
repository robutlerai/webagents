/**
 * The `cron:` block of an agent file (plan item 1.7, 2026-09-26): what the
 * daemon runs on a schedule, as the agent's owner, and where the result goes.
 *
 *     cron:
 *       - name: daily-report
 *         schedule: "0 9 * * 1-5"      # or `every: 30m`
 *         timezone: Europe/Berlin      # an IANA name; UTC when absent
 *         prompt: Summarize yesterday's activity.
 *         deliver:
 *           file: reports/daily.md     # or `webhook: https://...`, or `chat: owner`
 *       - name: watch
 *         every: 1h
 *         heartbeat: true              # the standing instructions, delivered only when there is something to report
 *         deliver:
 *           chat: owner
 *
 * THE STRING FORM IS REFUSED. `cron: "0 9 * * *"` parsed for a year and ran
 * nowhere: this daemon never read the key, and the Python daemon's job loop
 * was a sleep under a TODO. It named no prompt and no delivery target, so
 * there was nothing it could have run and nowhere the result could have gone.
 * A file that still says it is told the shape that works.
 *
 * ONE GRAMMAR IN BOTH SDKS. The five cron fields take `*`, `n`, `a-b`, `*\/n`,
 * `a-b/n` and comma lists, the platform's function-cron grammar
 * (`skills/cron/skill.ts`): no names, no `@daily`, no sixth field, and no step
 * on a single number (`5/10`), which Vixie cron and croniter read differently.
 * So a schedule means the same thing to the stepper here and to croniter in
 * Python. `every` is a whole number of s, m, h or d, at least one minute: a
 * heartbeat that wakes the model every few seconds is a bill, not a schedule.
 * `timezone` is checked against the zone database, UTC when absent (the
 * platform's function-cron is UTC too). `deliver` names exactly one target;
 * `channel` is reserved for the channel relay and refused with its own
 * sentence until then, so a file written for it fails now rather than
 * delivering nowhere.
 *
 * Every sentence here is the shared fixture's
 * (`python/tests/fixtures/daemon/cron.json`), word for word with
 * `python/webagents/cli/loader/schedules.py`; the access block's style
 * (`access: unknown key "x". It takes ...`), which both loaders already speak.
 */

import { AgentFileError } from './index';

/** The keys a schedule takes, in the order the sentence lists them. */
export const SCHEDULE_KEYS = ['deliver', 'enabled', 'every', 'heartbeat', 'name', 'prompt', 'schedule', 'timezone'] as const;
/** The delivery targets, in the order the sentence lists them. */
export const DELIVER_KINDS = ['chat', 'file', 'webhook'] as const;
/** The keys a webhook mapping takes. */
export const WEBHOOK_KEYS = ['retries', 'timeout', 'url'] as const;
/** `deliver: channel: ...` is the channel relay's, not yet available. */
export const RESERVED_DELIVER_KINDS = ['channel'] as const;

export const DEFAULT_TIMEZONE = 'UTC';
export const WEBHOOK_TIMEOUT_DEFAULT = 15;
export const WEBHOOK_RETRIES_DEFAULT = 3;
export const EVERY_MIN_SECONDS = 60;

const SHAPE = 'Each takes name, schedule or every, prompt or heartbeat, and deliver.';
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const EVERY_RE = /^(\d+)([smhd])$/;
const TZ_RE = /^[A-Za-z_][A-Za-z0-9_+-]*(\/[A-Za-z0-9_+-]+)*$/;
const EVERY_UNITS: Record<string, number> = { s: 1, m: 60, h: 3600, d: 86400 };
/** (low, high) of each cron field: minute, hour, day of month, month, day of week. */
const FIELD_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0, 59],
  [0, 23],
  [1, 31],
  [1, 12],
  [0, 6],
];

/** Where a schedule's result goes: one of `file`, `webhook` or `chat`. */
export type DeliverTarget =
  /** A path relative to the agent's folder. */
  | { kind: 'file'; path: string }
  /** The URL, the timeout in seconds, the retries after the first attempt. */
  | { kind: 'webhook'; url: string; timeout: number; retries: number }
  /** Whose chat: `owner`. */
  | { kind: 'chat'; to: 'owner' };

/** One entry of the block, as the daemon runs it. */
export interface CronSchedule {
  name: string;
  /** `cron` (a 5-field expression) or `every` (a fixed interval). */
  kind: 'cron' | 'every';
  expression?: string;
  everySeconds?: number;
  timezone: string;
  /** The turn's message; absent for a heartbeat, which runs the standing instructions. */
  prompt?: string;
  heartbeat: boolean;
  enabled: boolean;
  deliver: DeliverTarget;
}

/** The schedule as the shared fixture spells it (the Python `CronSchedule.to_dict`). */
export function describeSchedule(schedule: CronSchedule): Record<string, unknown> {
  const deliver: Record<string, unknown> =
    schedule.deliver.kind === 'file'
      ? { kind: 'file', path: schedule.deliver.path }
      : schedule.deliver.kind === 'webhook'
        ? { kind: 'webhook', url: schedule.deliver.url, timeout: schedule.deliver.timeout, retries: schedule.deliver.retries }
        : { kind: 'chat', to: schedule.deliver.to };
  return {
    name: schedule.name,
    kind: schedule.kind,
    expression: schedule.expression ?? null,
    every_seconds: schedule.everySeconds ?? null,
    timezone: schedule.timezone,
    prompt: schedule.prompt ?? null,
    heartbeat: schedule.heartbeat,
    enabled: schedule.enabled,
    deliver,
  };
}

/** One field of the grammar in the file comment (the Python `_is_cron_field`). */
function isCronField(field: string, low: number, high: number): boolean {
  if (field === '*') return true;
  return field.split(',').every((part) => {
    const stepMatch = /^([\d*-]+)\/(\d+)$/.exec(part);
    const base = stepMatch ? stepMatch[1] : part;
    if (stepMatch) {
      if (parseInt(stepMatch[2], 10) <= 0) return false;
      // A step walks `*` or a range; `5/10` means different things to
      // different crons, so it is not a schedule here.
      if (base !== '*' && !/^\d+-\d+$/.test(base)) return false;
    }
    if (base === '*') return true;
    const rangeMatch = /^(\d+)-(\d+)$/.exec(base);
    if (rangeMatch) {
      const a = parseInt(rangeMatch[1], 10);
      const b = parseInt(rangeMatch[2], 10);
      return low <= a && b <= high && a <= b;
    }
    if (!/^\d+$/.test(base)) return false;
    const n = parseInt(base, 10);
    return low <= n && n <= high;
  });
}

/**
 * The expression with single spaces between its five fields, or `undefined`
 * when `value` is not a schedule in the grammar above.
 */
export function cronExpression(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const fields = value.split(/\s+/).filter((field) => field !== '');
  if (fields.length !== 5) return undefined;
  if (!fields.every((field, i) => isCronField(field, FIELD_RANGES[i][0], FIELD_RANGES[i][1]))) return undefined;
  return fields.join(' ');
}

/** The seconds `every` names (`15m`, `2h`, `1d`, `90s`), or `undefined`: not a duration, or under a minute. */
export function everySeconds(value: unknown): number | undefined {
  if (typeof value !== 'string') return undefined;
  const match = EVERY_RE.exec(value);
  if (!match) return undefined;
  const seconds = parseInt(match[1], 10) * EVERY_UNITS[match[2]];
  return seconds >= EVERY_MIN_SECONDS ? seconds : undefined;
}

/** Whether `value` names a zone in the zone database, in its own spelling. */
export function isTimezone(value: unknown): value is string {
  if (typeof value !== 'string' || !TZ_RE.test(value)) return false;
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: value });
  } catch {
    return false;
  }
  return true;
}

/**
 * Whether a `file` target stays inside the agent's folder, by its spelling:
 * relative, and never a `..` segment. The deliverer checks the real path again
 * when it writes, in case a link on the way out points elsewhere.
 */
export function isInsideFolder(path: unknown): path is string {
  if (typeof path !== 'string' || path.trim() === '') return false;
  if (path.startsWith('/') || path.startsWith('\\') || path.startsWith('~') || /^[A-Za-z]:/.test(path)) return false;
  return path.split(/[\\/]/).every((segment) => segment !== '..');
}

function isHttpUrl(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  try {
    const url = new URL(value);
    return (url.protocol === 'http:' || url.protocol === 'https:') && url.host !== '';
  } catch {
    return false;
  }
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isMapping(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function deliverTarget(value: unknown, where: string): DeliverTarget {
  if (!isMapping(value) || Object.keys(value).length !== 1) {
    throw new AgentFileError(`${where}: deliver must name one target: chat, file or webhook.`);
  }
  const [kind, config] = Object.entries(value)[0];
  if ((RESERVED_DELIVER_KINDS as readonly string[]).includes(kind)) {
    throw new AgentFileError(`${where}: deliver: ${kind} targets are not available yet.`);
  }
  if (!(DELIVER_KINDS as readonly string[]).includes(kind)) {
    throw new AgentFileError(`${where}: deliver: unknown target "${kind}". It takes chat, file or webhook.`);
  }
  if (kind === 'file') {
    if (!isInsideFolder(config)) {
      throw new AgentFileError(`${where}: deliver: file must be a relative path inside the agent's folder.`);
    }
    return { kind: 'file', path: config };
  }
  if (kind === 'chat') {
    if (config !== 'owner') throw new AgentFileError(`${where}: deliver: chat must be owner.`);
    return { kind: 'chat', to: 'owner' };
  }
  // webhook: a URL, or a mapping with url, timeout and retries.
  const shape = `${where}: deliver: webhook must be an http(s) URL, or a mapping with url, timeout and retries.`;
  if (typeof config === 'string') {
    if (!isHttpUrl(config)) throw new AgentFileError(shape);
    return { kind: 'webhook', url: config, timeout: WEBHOOK_TIMEOUT_DEFAULT, retries: WEBHOOK_RETRIES_DEFAULT };
  }
  if (!isMapping(config) || !isHttpUrl(config.url)) throw new AgentFileError(shape);
  for (const key of Object.keys(config)) {
    if (!(WEBHOOK_KEYS as readonly string[]).includes(key)) {
      throw new AgentFileError(`${where}: deliver: webhook: unknown key "${key}". It takes retries, timeout and url.`);
    }
  }
  const timeout = 'timeout' in config ? config.timeout : WEBHOOK_TIMEOUT_DEFAULT;
  if (!isNumber(timeout) || timeout < 1 || timeout > 300) {
    throw new AgentFileError(`${where}: deliver: webhook: timeout must be a number of seconds from 1 to 300.`);
  }
  const retries = 'retries' in config ? config.retries : WEBHOOK_RETRIES_DEFAULT;
  if (!isNumber(retries) || !Number.isInteger(retries) || retries < 0 || retries > 10) {
    throw new AgentFileError(`${where}: deliver: webhook: retries must be a whole number from 0 to 10.`);
  }
  return { kind: 'webhook', url: config.url as string, timeout, retries };
}

function schedule(index: number, entry: unknown, seen: Map<string, number>): CronSchedule {
  if (!isMapping(entry)) throw new AgentFileError(`cron: schedule ${index} must be a mapping. ${SHAPE}`);
  const name = entry.name;
  if (typeof name !== 'string' || !NAME_RE.test(name)) {
    throw new AgentFileError(`cron: schedule ${index}: name must be a non-empty string of letters, digits, - and _.`);
  }
  const where = `cron: schedule ${index} (${name})`;
  const earlier = seen.get(name);
  if (earlier !== undefined) throw new AgentFileError(`${where}: name is already used by schedule ${earlier}.`);
  seen.set(name, index);
  for (const key of Object.keys(entry)) {
    if (!(SCHEDULE_KEYS as readonly string[]).includes(key)) {
      throw new AgentFileError(
        `${where}: unknown key "${key}". It takes deliver, enabled, every, heartbeat, name, prompt, schedule and timezone.`,
      );
    }
  }

  const hasSchedule = 'schedule' in entry;
  const hasEvery = 'every' in entry;
  if (hasSchedule === hasEvery) throw new AgentFileError(`${where}: exactly one of schedule or every is required.`);
  let expression: string | undefined;
  let seconds: number | undefined;
  if (hasSchedule) {
    expression = cronExpression(entry.schedule);
    if (expression === undefined) {
      throw new AgentFileError(
        `${where}: schedule must be a 5-field cron expression (minute hour day-of-month month day-of-week).`,
      );
    }
  } else {
    seconds = everySeconds(entry.every);
    if (seconds === undefined) throw new AgentFileError(`${where}: every must be a duration like 15m, 2h or 1d, at least 1m.`);
  }

  const timezone = 'timezone' in entry ? entry.timezone : DEFAULT_TIMEZONE;
  if (!isTimezone(timezone)) throw new AgentFileError(`${where}: timezone must be an IANA zone name like Europe/Berlin.`);

  const hasPrompt = 'prompt' in entry;
  const hasHeartbeat = 'heartbeat' in entry;
  if (hasHeartbeat && entry.heartbeat !== true) {
    throw new AgentFileError(`${where}: heartbeat must be true; leave it out for a prompt schedule.`);
  }
  if (hasPrompt === hasHeartbeat) throw new AgentFileError(`${where}: exactly one of prompt or heartbeat is required.`);
  let prompt: string | undefined;
  if (hasPrompt) {
    if (typeof entry.prompt !== 'string' || entry.prompt.trim() === '') {
      throw new AgentFileError(`${where}: prompt must be a non-empty string.`);
    }
    prompt = entry.prompt;
  }

  const enabled = 'enabled' in entry ? entry.enabled : true;
  if (typeof enabled !== 'boolean') throw new AgentFileError(`${where}: enabled must be true or false.`);

  const deliver = deliverTarget(entry.deliver, where);
  return {
    name,
    kind: hasSchedule ? 'cron' : 'every',
    ...(expression !== undefined ? { expression } : {}),
    ...(seconds !== undefined ? { everySeconds: seconds } : {}),
    timezone,
    ...(prompt !== undefined ? { prompt } : {}),
    heartbeat: hasHeartbeat,
    enabled,
    deliver,
  };
}

/**
 * The schedules a `cron:` value declares, or an `AgentFileError` whose
 * sentence is written for the file's author (the shared fixture's).
 */
export function parseCronBlock(value: unknown): CronSchedule[] {
  if (!Array.isArray(value)) throw new AgentFileError(`cron: must be a list of schedules. ${SHAPE}`);
  const seen = new Map<string, number>();
  return value.map((entry, i) => schedule(i + 1, entry, seen));
}
