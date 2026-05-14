/**
 * `TimeSkill`
 *
 * Optional helper that exposes a `current_time` tool for sub-day precision
 * and timezone conversion. The primary time signal in Robutler is the
 * `[<iso-utc-ts>, took <duration>]` prefix on every tool result and the
 * one-time `[chat started <iso-utc-ts>]` synthetic prefix on the first user
 * message of a chat (both injected by the portal proxy). This skill is for
 * the smaller set of cases the prefix doesn't cover:
 *
 *   - Pure-text Q&A turns where the LLM hasn't called any tool yet and the
 *     `chat started` anchor is hours stale.
 *   - "What is the current time in Tokyo?" / timezone conversion.
 *   - Relative date math ("what was the date 5 days ago", "what day of week
 *     is 2026-12-25") that doesn't depend on the most-recent tool timestamp.
 *
 * Returns ISO 8601 UTC by default, with optional IANA timezone formatting.
 */

import { Skill } from '../../core/skill';
import { tool, prompt } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';

export interface TimeSkillConfig extends SkillConfig {
  /**
   * Default IANA timezone for `current_time` when the caller omits `tz`.
   * Defaults to `'UTC'`. The portal usually threads the caller's preferred
   * timezone in here at construction time so the LLM gets local-time
   * formatting without having to ask.
   */
  defaultTz?: string;
}

export interface CurrentTimeResult {
  /** Always ISO 8601 in UTC (e.g. `2026-05-13T15:42:07.123Z`). */
  utc: string;
  /** The same instant formatted in the requested timezone (or default). */
  local: string;
  /** IANA tz name actually used. */
  tz: string;
  /** ISO 8601 weekday (1 = Monday, 7 = Sunday). */
  weekday: number;
  /** Unix epoch milliseconds. */
  epochMs: number;
}

export class TimeSkill extends Skill {
  private readonly defaultTz: string;

  constructor(config: TimeSkillConfig = {}) {
    super({ ...config, name: config.name || 'time' });
    this.defaultTz = config.defaultTz || 'UTC';
  }

  @prompt({ priority: 40, name: 'timeGuide', scope: 'all' })
  timeGuide(_ctx: Context): string {
    return [
      '## Time skill',
      '',
      `\`current_time\` returns the current instant in UTC and the requested IANA timezone (default: \`${this.defaultTz}\`).`,
      '',
      '### When to call',
      '- Prefer the inline time signal: tool results carry `[<iso-utc-ts>, took <duration>]` and the first user message may carry `[chat started <iso-utc-ts>]`. For "what date is it today" and most recency reasoning, READ the most recent timestamp instead of calling this tool.',
      '- Call `current_time` when (a) you need precision the prefix can\'t give you (sub-second, or no recent tool call), (b) you need a different IANA timezone than the prefix carries, or (c) you need to compute relative dates ("5 days ago", "next Monday") and want the system clock as a reference.',
      '- DO NOT call `current_time` reflexively at the start of every turn — that\'s a wasted tool call.',
      '',
      '### Output',
      '- `utc` — ISO 8601 UTC, the canonical form. Use this when persisting times or comparing across timezones.',
      '- `local` — the same instant in the requested IANA `tz` (24-hour, no abbreviation). Use for user-facing strings.',
      '- `weekday` — ISO 8601 (1 = Monday, 7 = Sunday). Useful for "is today a weekday" checks without parsing strings.',
      '',
      'Pass `tz` only when you need a non-default zone (e.g. `tz: "America/Los_Angeles"`). Invalid IANA names fall back to UTC silently — verify before relying on the local string.',
    ].join('\n');
  }

  @tool({
    name: 'current_time',
    description:
      'Return the current time in UTC and the given (or default) IANA timezone. Use for sub-second precision or when you need a non-default timezone — for the common "what date is today" case, read the most recent `[<iso-utc-ts>, took ...]` prefix on a tool result instead.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        tz: {
          type: 'string',
          description: 'IANA timezone (e.g. "America/Los_Angeles", "Asia/Tokyo"). Defaults to the skill\'s configured default. Invalid names fall back to UTC.',
        },
      },
    },
  })
  async currentTime(
    params: { tz?: string },
    _context: Context,
  ): Promise<CurrentTimeResult> {
    const requested = (params.tz || this.defaultTz || 'UTC').trim();
    const now = new Date();
    const tz = isValidTimeZone(requested) ? requested : 'UTC';
    const local = formatInTz(now, tz);
    const weekday = isoWeekday(now);
    return {
      utc: now.toISOString(),
      local,
      tz,
      weekday,
      epochMs: now.getTime(),
    };
  }
}

function isValidTimeZone(tz: string): boolean {
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: tz }).format(new Date());
    return true;
  } catch {
    return false;
  }
}

function formatInTz(date: Date, tz: string): string {
  const fmt = new Intl.DateTimeFormat('en-CA', {
    timeZone: tz,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
  const parts = fmt.formatToParts(date).reduce<Record<string, string>>((acc, p) => {
    if (p.type !== 'literal') acc[p.type] = p.value;
    return acc;
  }, {});
  const hour = parts.hour === '24' ? '00' : parts.hour;
  return `${parts.year}-${parts.month}-${parts.day} ${hour}:${parts.minute}:${parts.second} ${tz}`;
}

function isoWeekday(date: Date): number {
  const day = date.getUTCDay();
  return day === 0 ? 7 : day;
}
