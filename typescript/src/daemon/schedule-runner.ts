/**
 * Runs the `cron:` schedules of the agents the daemon serves (plan item 1.7,
 * 2026-09-26): the same runner, minute for minute, as
 * `python/webagents/cli/daemon/schedule_runner.py`.
 *
 * WHAT RAN BEFORE. `daemon/cron.ts` held jobs added over HTTP, each a `task`
 * string the daemon ran as a prompt, and ignored `cron:` in agent files
 * altogether; the Python daemon's job loop slept under a TODO. Nothing an
 * agent file scheduled ever ran, and the docs said it did.
 *
 * WHAT RUNS NOW. Each schedule is a turn of the agent's normal runtime, as
 * its OWNER (`access/caller.ts` `LOCAL_OWNER`, the caller `webagents -p`
 * gives a turn), with the schedule's `prompt` as the one user message; the
 * reply goes where the schedule says (`deliver.ts`). Schedules come from
 * agent files only: the HTTP routes list them and can no longer add one
 * (S-273), because an added job with a prompt of the caller's choosing was a
 * way to run the owner's agent on the owner's key.
 *
 * TIME. A cron schedule fires at the next whole minute that matches its
 * expression in its zone, strictly after the previous fire; day-of-month and
 * day-of-week are OR'd when both are restricted, as Vixie cron does. A
 * wall-clock minute that does not exist in the zone that day (a
 * spring-forward gap) is skipped; one that exists twice (a fall-back hour)
 * fires at its first occurrence. An `every` schedule fires one interval after
 * the daemon first sees it, then one interval after each fire. Zone
 * arithmetic is `Intl`'s, so no dependency; the fixture's vectors
 * (`python/tests/fixtures/daemon/cron.json`, `next_run`) hold in both SDKs.
 *
 * STATE, so a restart neither double-runs nor drops a schedule
 * (`.webagents/cron/<agent>.json` in the agent's folder). The next fire is
 * written BEFORE the turn runs, so a daemon that dies mid-run does not run
 * the same slot again when it comes back; a fire the daemon slept through is
 * run once, as soon as it is back, and the schedule then resumes from now
 * rather than replaying every missed slot. A schedule whose expression,
 * interval or zone changed is rescheduled from now. One run per schedule at
 * a time: a slot that comes due while the previous run is still going is
 * skipped, and said.
 *
 * HEARTBEAT. A `heartbeat: true` schedule has no prompt of its own: the turn
 * is `HEARTBEAT_PROMPT`, which asks the agent to follow its standing
 * instructions and answer `HEARTBEAT_SENTINEL` when there is nothing to say
 * (OpenClaw's heartbeat, Hermes' monitor mode). That reply, or an empty one,
 * is the run's `nothing` and is delivered nowhere; anything else is the
 * report and goes where the schedule says. The words are the fixture's
 * (`heartbeat`), the same in `schedule_runner.py`.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { LOCAL_OWNER } from '../access/caller';
import type { CronSchedule } from '../agents/schedules';
import { describeSchedule, parseCronBlock } from '../agents/schedules';
import type { IAgent } from '../core/types';
import { deliver as deliverDefault, type DeliveryContext, type Outcome, type RunResult } from './deliver';
import { agentFinishOf, isAgentFinish } from '../core/tool-budget';

/** Where an agent's schedule state lives, under the agent's folder. */
export const STATE_DIR = path.join('.webagents', 'cron');
/** The one user message of a heartbeat turn (the fixture's `heartbeat.prompt`). */
export const HEARTBEAT_PROMPT =
  'Heartbeat. Follow your standing instructions and check on whatever they ask you to keep an eye on. ' +
  'If there is nothing to report, reply with exactly HEARTBEAT_OK and nothing else. ' +
  'Otherwise reply with the report itself, written for the person who reads it.';
/** The reply that means "nothing to report". */
export const HEARTBEAT_SENTINEL = 'HEARTBEAT_OK';
/** What a model wraps the sentinel in: whitespace, emphasis, quotes, end punctuation. */
const SENTINEL_WRAPPING = /^[\s*_`"'.!]+|[\s*_`"'.!]+$/g;

/** Whether a heartbeat reply says nothing: empty, or the sentinel however it is dressed. */
export function isQuietHeartbeat(text: string): boolean {
  const bare = text.replace(SENTINEL_WRAPPING, '');
  return bare === '' || bare.toUpperCase() === HEARTBEAT_SENTINEL;
}
/** How far ahead a cron expression is searched for its next slot (`0 0 29 2 *` needs four years). */
export const SEARCH_DAYS = 366 * 5;
const MINUTE_MS = 60_000;
/** How often the daemon looks for due schedules. */
export const TICK_MS = 5_000;
/** (low, high) of each cron field: minute, hour, day of month, month, day of week. */
const FIELD_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0, 59],
  [0, 23],
  [1, 31],
  [1, 12],
  [0, 6],
];

/** `2026-09-28T07:00:00Z`: the instant to the second, as both SDKs write it. */
export function isoUtc(ms: number): string {
  return new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');
}

/** The milliseconds an ISO 8601 instant names, or `undefined`. */
export function parseIso(text: unknown): number | undefined {
  if (typeof text !== 'string' || !text) return undefined;
  const ms = Date.parse(text);
  return Number.isNaN(ms) ? undefined : ms;
}

/** The values one validated cron field allows. */
function fieldValues(field: string, low: number, high: number): Set<number> {
  const values = new Set<number>();
  for (const part of field.split(',')) {
    const [base, stepText] = part.split('/');
    const step = stepText ? parseInt(stepText, 10) : 1;
    let start: number;
    let end: number;
    if (base === '*') {
      start = low;
      end = high;
    } else if (base.includes('-')) {
      const [a, b] = base.split('-');
      start = parseInt(a, 10);
      end = parseInt(b, 10);
    } else {
      start = end = parseInt(base, 10);
    }
    for (let v = start; v <= end; v += step) values.add(v);
  }
  return values;
}

interface LocalParts {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
}

const formatters = new Map<string, Intl.DateTimeFormat>();

function formatterFor(tz: string): Intl.DateTimeFormat {
  let formatter = formatters.get(tz);
  if (!formatter) {
    formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
    formatters.set(tz, formatter);
  }
  return formatter;
}

/** The wall clock in `tz` at `ms`. */
function localParts(ms: number, tz: string): LocalParts {
  const parts: Record<string, number> = {};
  for (const part of formatterFor(tz).formatToParts(new Date(ms))) {
    if (part.type !== 'literal') parts[part.type] = parseInt(part.value, 10);
  }
  return { year: parts.year, month: parts.month, day: parts.day, hour: parts.hour === 24 ? 0 : parts.hour, minute: parts.minute };
}

function wallMs(p: LocalParts): number {
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute);
}

/** `tz`'s offset from UTC at `ms` (a whole minute), in ms. */
function tzOffsetMs(ms: number, tz: string): number {
  return wallMs(localParts(ms, tz)) - ms;
}

/**
 * The instant a wall-clock minute names in `tz`: its first occurrence when it
 * exists twice, `undefined` when it does not exist that day.
 */
export function localToUtcMs(year: number, month: number, day: number, hour: number, minute: number, tz: string): number | undefined {
  const wall = Date.UTC(year, month - 1, day, hour, minute);
  const candidates = new Set<number>();
  const off1 = tzOffsetMs(wall, tz);
  candidates.add(wall - off1);
  const off2 = tzOffsetMs(wall - off1, tz);
  candidates.add(wall - off2);
  const off3 = tzOffsetMs(wall - off2, tz);
  candidates.add(wall - off3);
  const valid = [...candidates].filter((t) => {
    const p = localParts(t, tz);
    return p.year === year && p.month === month && p.day === day && p.hour === hour && p.minute === minute;
  });
  return valid.length ? Math.min(...valid) : undefined;
}

/**
 * The first whole minute strictly after `afterMs` that `expression` matches
 * in `tz` (file comment, TIME), or `undefined` within `SEARCH_DAYS`.
 */
export function nextCronRun(expression: string, tz: string, afterMs: number): number | undefined {
  const fields = expression.split(/\s+/);
  const [minutes, hours, doms, months, dows] = fields.map((field, i) => fieldValues(field, FIELD_RANGES[i][0], FIELD_RANGES[i][1]));
  const domAny = fields[2] === '*';
  const dowAny = fields[4] === '*';
  const dayMatches = (dom: number, month: number, dow: number): boolean => {
    if (!months.has(month)) return false;
    if (domAny && dowAny) return true;
    if (!domAny && !dowAny) return doms.has(dom) || dows.has(dow);
    return domAny ? dows.has(dow) : doms.has(dom);
  };
  const startMs = Math.floor(afterMs / MINUTE_MS) * MINUTE_MS + MINUTE_MS;
  const start = localParts(startMs, tz);
  const sortedHours = [...hours].sort((a, b) => a - b);
  const sortedMinutes = [...minutes].sort((a, b) => a - b);
  for (let offset = 0; offset <= SEARCH_DAYS; offset += 1) {
    // Calendar arithmetic on the local date, done in UTC space where it is exact.
    const day = new Date(Date.UTC(start.year, start.month - 1, start.day + offset));
    const y = day.getUTCFullYear();
    const m = day.getUTCMonth() + 1;
    const d = day.getUTCDate();
    if (!dayMatches(d, m, day.getUTCDay())) continue;
    for (const hour of sortedHours) {
      if (offset === 0 && hour < start.hour) continue;
      for (const minute of sortedMinutes) {
        if (offset === 0 && hour === start.hour && minute < start.minute) continue;
        const instant = localToUtcMs(y, m, d, hour, minute, tz);
        if (instant === undefined || instant <= afterMs) continue;
        return instant;
      }
    }
  }
  return undefined;
}

/** When `schedule` fires next, strictly after `afterMs`. */
export function nextRunFor(schedule: CronSchedule, afterMs: number): number | undefined {
  if (schedule.kind === 'cron') return nextCronRun(schedule.expression ?? '', schedule.timezone, afterMs);
  return afterMs + (schedule.everySeconds ?? 0) * 1000;
}

/** What a schedule's timing is, so a changed one is rescheduled from now. */
export function scheduleSpec(schedule: CronSchedule): string {
  return schedule.kind === 'cron'
    ? `cron ${schedule.expression} ${schedule.timezone}`
    : `every ${schedule.everySeconds} ${schedule.timezone}`;
}

export function statePath(agentDir: string, agentName: string): string {
  return path.join(agentDir, STATE_DIR, `${agentName}.json`);
}

/** `{at, outcome, detail}` of a run. */
export interface RunRecord {
  at: string;
  outcome: 'delivered' | 'nothing' | 'failed';
  detail: string;
  /** When the agent's tool budget ended the turn (2026-09-28, `core/tool-budget.ts`). */
  finish?: { reason: string; rounds?: number; tool?: string };
}

/** One schedule the runner holds, with its persisted state. */
export interface ScheduleEntry {
  agentName: string;
  agentDir: string;
  schedule: CronSchedule;
  /** When it fires next, ms; `undefined` while disabled or when no slot exists. */
  nextRun?: number;
  /** When it last fired, ms. */
  lastFire?: number;
  lastRun?: RunRecord;
  running: boolean;
  /** The running turn's finish, for `lastRun`. */
  turnFinish?: RunRecord['finish'];
}

/** The reply's text from what `agent.run` returns. */
export function replyText(response: unknown): string {
  const content = (response as { content?: unknown } | undefined)?.content;
  return typeof content === 'string' ? content : '';
}

function describeError(error: unknown): string {
  const message = (error as { message?: unknown } | undefined)?.message;
  return typeof message === 'string' && message ? message : String(error);
}

export interface ScheduleRunnerOptions {
  /** The served agent by name, or nothing when it is not served. */
  agentFor: (name: string) => IAgent | undefined | Promise<IAgent | undefined>;
  /** Milliseconds since the epoch; injected by tests. */
  clock?: () => number;
  /** The deliverer; `deliver.ts` by default. */
  deliver?: (target: CronSchedule['deliver'], result: RunResult, ctx: DeliveryContext) => Promise<Outcome>;
  /** Where the runner's lines go; `console` by default. */
  log?: (line: string) => void;
  /**
   * Write state files (default true). `webagents cron list` reads the
   * daemon's state and must not write any: a listing that started the
   * clock on an `every` schedule would have the daemon run it early.
   */
  persist?: boolean;
}

/** Holds the served agents' schedules and runs the due ones (file comment). */
export class ScheduleRunner {
  private readonly agentFor: ScheduleRunnerOptions['agentFor'];
  private readonly clock: () => number;
  private readonly deliverFn: NonNullable<ScheduleRunnerOptions['deliver']>;
  private readonly log: (line: string) => void;
  private readonly persist: boolean;
  private readonly entriesByKey: Map<string, ScheduleEntry> = new Map();
  private timer: NodeJS.Timeout | null = null;
  private ticking: Promise<void> | null = null;

  constructor(options: ScheduleRunnerOptions) {
    this.agentFor = options.agentFor;
    this.clock = options.clock ?? (() => Date.now());
    this.deliverFn = options.deliver ?? deliverDefault;
    this.log = options.log ?? ((line) => console.log(line));
    this.persist = options.persist ?? true;
  }

  // -- what is scheduled ----------------------------------------------------------------

  /**
   * The schedules of `agentName`, replacing what it had: state on disk is
   * kept for a schedule whose timing is unchanged, so a restart resumes it.
   */
  setSchedules(agentName: string, agentDir: string, schedules: readonly CronSchedule[]): void {
    const saved = this.loadState(agentDir, agentName);
    const now = this.clock();
    const kept = new Map<string, ScheduleEntry>();
    for (const schedule of schedules) {
      const key = `${agentName}\u0000${schedule.name}`;
      const before = this.entriesByKey.get(key);
      const entry: ScheduleEntry = { agentName, agentDir, schedule, running: before?.running ?? false };
      const record = saved[schedule.name];
      const sameTiming = record !== undefined && record.spec === scheduleSpec(schedule);
      if (record) {
        const lastFire = parseIso(record.lastFire);
        if (lastFire !== undefined) entry.lastFire = lastFire;
        if (record.lastRun && typeof record.lastRun === 'object') entry.lastRun = record.lastRun as RunRecord;
      }
      const savedNext = record ? parseIso(record.nextRun) : undefined;
      if (!schedule.enabled) {
        entry.nextRun = undefined;
      } else if (sameTiming && savedNext !== undefined) {
        // A fire the daemon slept through stays due, and runs once.
        entry.nextRun = savedNext;
      } else {
        entry.nextRun = nextRunFor(schedule, now);
      }
      kept.set(key, entry);
    }
    for (const key of [...this.entriesByKey.keys()]) if (key.startsWith(`${agentName}\u0000`)) this.entriesByKey.delete(key);
    for (const [key, entry] of kept) this.entriesByKey.set(key, entry);
    this.saveState(agentName, agentDir);
  }

  removeAgent(agentName: string): void {
    for (const key of [...this.entriesByKey.keys()]) if (key.startsWith(`${agentName}\u0000`)) this.entriesByKey.delete(key);
  }

  /** The schedules a file's `cron:` block declares, for an agent served from `filePath`. */
  setFromBlock(agentName: string, filePath: string, block: unknown): void {
    this.setSchedules(agentName, path.dirname(filePath), block === undefined ? [] : parseCronBlock(block));
  }

  /** Every schedule, by agent name, each agent's in the order its file declares them. */
  entries(): ScheduleEntry[] {
    // A stable sort on the agent alone: `setSchedules` re-inserts an agent's
    // entries in file order, and the Map keeps insertion order.
    return [...this.entriesByKey.values()].sort((a, b) => (a.agentName < b.agentName ? -1 : a.agentName > b.agentName ? 1 : 0));
  }

  /** Every schedule with its state, for `GET /agents/cron` and `webagents cron list`. */
  list(): Record<string, unknown>[] {
    return this.entries().map((entry) => this.describe(entry));
  }

  find(agentName: string, scheduleName: string): ScheduleEntry | undefined {
    return this.entriesByKey.get(`${agentName}\u0000${scheduleName}`);
  }

  describe(entry: ScheduleEntry): Record<string, unknown> {
    return {
      agent: entry.agentName,
      ...describeSchedule(entry.schedule),
      next_run: entry.nextRun !== undefined ? isoUtc(entry.nextRun) : null,
      last_fire: entry.lastFire !== undefined ? isoUtc(entry.lastFire) : null,
      last_run: entry.lastRun ?? null,
      running: entry.running,
    };
  }

  // -- running ----------------------------------------------------------------------------

  /** Run every schedule that is due, each at most once, concurrently, and wait for them. */
  async tick(): Promise<void> {
    const now = this.clock();
    const runs: Promise<void>[] = [];
    for (const entry of this.entries()) {
      if (!entry.schedule.enabled || entry.nextRun === undefined || entry.nextRun > now) continue;
      if (entry.running) {
        this.log(`[daemon] ${entry.agentName}/${entry.schedule.name}: due, but the previous run is still going`);
        continue;
      }
      // The next fire, written before the turn: a daemon that dies mid-run
      // does not run this slot again.
      entry.lastFire = now;
      entry.nextRun = nextRunFor(entry.schedule, now);
      this.saveState(entry.agentName, entry.agentDir);
      runs.push(this.execute(entry, now));
    }
    if (runs.length) await Promise.all(runs);
  }

  /** Run one schedule now (`webagents cron run`), leaving its next fire as it was. */
  async runNow(agentName: string, scheduleName: string): Promise<RunRecord> {
    const entry = this.find(agentName, scheduleName);
    if (!entry) throw new Error(`${agentName}/${scheduleName}: no such schedule`);
    if (entry.running) return { at: isoUtc(this.clock()), outcome: 'failed', detail: 'the previous run is still going' };
    await this.execute(entry, this.clock());
    return { ...(entry.lastRun as RunRecord) };
  }

  private async execute(entry: ScheduleEntry, firedAt: number): Promise<void> {
    entry.running = true;
    entry.turnFinish = undefined;
    let outcome: Outcome;
    try {
      outcome = await this.turnAndDeliver(entry, firedAt);
    } finally {
      entry.running = false;
    }
    entry.lastRun = { at: isoUtc(firedAt), outcome: outcome[0], detail: outcome[1], ...(entry.turnFinish ? { finish: entry.turnFinish } : {}) };
    this.saveState(entry.agentName, entry.agentDir);
    this.log(`[daemon] ${entry.agentName}/${entry.schedule.name}: ${outcome[0]} (${outcome[1]})`);
  }

  private async turnAndDeliver(entry: ScheduleEntry, firedAt: number): Promise<Outcome> {
    const agent = await this.agentFor(entry.agentName);
    if (!agent) return ['failed', 'the agent is not served'];
    const heartbeat = entry.schedule.heartbeat;
    const prompt = heartbeat ? HEARTBEAT_PROMPT : (entry.schedule.prompt ?? '');
    let content: string;
    try {
      // The owner's own turn, as `webagents -p` runs one.
      const response = await agent.run([{ role: 'user', content: prompt }], { auth: { ...LOCAL_OWNER } });
      content = replyText(response);
      // The agent's tool budget ended the turn (2026-09-28): the record says
      // so, and the answer its last, tool-less call gave is delivered.
      const finish = response.finish;
      if (finish && isAgentFinish(finish.reason)) {
        entry.turnFinish = { reason: finish.reason, ...(finish.rounds !== undefined ? { rounds: finish.rounds } : {}), ...(finish.tool ? { tool: finish.tool } : {}) };
      }
    } catch (err) {
      // Its last call brought no answer: an empty reply, as the Python daemon records it.
      const ended = agentFinishOf(err);
      if (ended) {
        entry.turnFinish = ended;
        return ['nothing', 'the reply was empty'];
      }
      return ['failed', `turn failed: ${describeError(err)}`];
    }
    // Nothing to deliver: a quiet heartbeat, or a reply with no words in it.
    if (heartbeat && isQuietHeartbeat(content)) return ['nothing', 'nothing to report'];
    if (content.trim() === '') return ['nothing', 'the reply was empty'];
    const result: RunResult = {
      agent: entry.agentName,
      schedule: entry.schedule.name,
      kind: entry.schedule.kind,
      // A heartbeat's prompt is the runner's, not the file's: the record carries none.
      ...(heartbeat ? {} : { prompt }),
      content,
      ranAt: isoUtc(firedAt),
    };
    try {
      return await this.deliverFn(entry.schedule.deliver, result, { agentDir: entry.agentDir, agent });
    } catch (err) {
      return ['failed', `delivery failed: ${describeError(err)}`];
    }
  }

  /** The daemon's loop: a tick every `intervalMs` until `stop()`; ticks never overlap. */
  start(intervalMs = TICK_MS): void {
    if (this.timer) return;
    this.timer = setInterval(() => {
      if (this.ticking) return;
      this.ticking = this.tick()
        .catch((err) => this.log(`[daemon] schedule tick failed: ${describeError(err)}`))
        .finally(() => {
          this.ticking = null;
        });
    }, intervalMs);
    this.timer.unref?.();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  // -- state ------------------------------------------------------------------------------

  private loadState(agentDir: string, agentName: string): Record<string, { spec?: unknown; nextRun?: unknown; lastFire?: unknown; lastRun?: unknown }> {
    try {
      const data = JSON.parse(fs.readFileSync(statePath(agentDir, agentName), 'utf-8')) as { schedules?: unknown };
      const schedules = data?.schedules;
      return schedules && typeof schedules === 'object' ? (schedules as Record<string, Record<string, unknown>>) : {};
    } catch {
      return {};
    }
  }

  private saveState(agentName: string, agentDir: string): void {
    if (!this.persist) return;
    const schedules: Record<string, unknown> = {};
    for (const entry of this.entries()) {
      if (entry.agentName !== agentName) continue;
      schedules[entry.schedule.name] = {
        spec: scheduleSpec(entry.schedule),
        nextRun: entry.nextRun !== undefined ? isoUtc(entry.nextRun) : null,
        lastFire: entry.lastFire !== undefined ? isoUtc(entry.lastFire) : null,
        lastRun: entry.lastRun ?? null,
      };
    }
    const file = statePath(agentDir, agentName);
    try {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, `${JSON.stringify({ schedules }, null, 2)}\n`, 'utf-8');
    } catch (err) {
      this.log(`[daemon] ${file}: could not write schedule state: ${describeError(err)}`);
    }
  }
}
