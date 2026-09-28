/**
 * `webagents cron list` and `webagents cron run <agent> <name>` (plan item
 * 1.7, 2026-09-26): the `cron:` schedules the agent files in a folder
 * declare, as the daemon runs them (`daemon/schedule-runner.ts`).
 *
 * `list` reads the folder's agent files exactly as the daemon would
 * (`daemon/watcher.ts`, `findAgentFiles` and `agentDefinitionFrom`) and the
 * runner's state under each agent's `.webagents/cron/`, and writes nothing:
 * the runner is built with `persist: false`, because a listing that started
 * the clock on an `every` schedule the daemon has not seen would have the
 * daemon run it early. `run` builds the one agent the schedule names the way
 * the daemon does (`buildDefinedAgent`), runs the schedule now as the owner,
 * delivers as configured, and records the run in that state, where the
 * daemon's next listing shows it.
 *
 * The lines are the Python CLI's, word for word
 * (`python/webagents/cli/cron_command.py`), held by
 * `python/tests/fixtures/cli/cron.json`. The bodies live here rather than in
 * `index.ts` so they can be unit-tested: that module parses argv when it
 * loads.
 */

import * as path from 'node:path';

import { AgentFileError, parseAgentMarkdown, readAgentFile } from '../agents/index';
import { parseCronBlock, type CronSchedule } from '../agents/schedules';
import type { IAgent } from '../core/types';
import { ScheduleRunner, type RunRecord } from '../daemon/schedule-runner';
import { agentDefinitionFrom, findAgentFiles, type AgentDefinition } from '../daemon/watcher';

/** The table's columns (the fixture's `list.columns`). */
export const COLUMNS = ['AGENT', 'SCHEDULE', 'WHEN', 'NEXT', 'LAST'] as const;
const NONE = 'No schedules: no agent file in {folder} declares cron:.';
/**
 * The folder held a file the loader refused (2026-09-28, the e2e pass): "no
 * agent file declares cron:" was untrue when the refused one did. The
 * fixture's `list.none_refused`.
 */
const NONE_REFUSED = 'No schedules: no agent file in {folder} that loads declares cron:.';
/**
 * A refused file's line, `cron`'s own words (the fixture's
 * `list.refused_line`): it read "[daemon] ... The agent is not served.",
 * the daemon's words, in a command that serves nothing.
 */
const REFUSED_LINE = '{file}: {reason} Its schedules do not run until the file loads.';
const NO_AGENT = 'No agent named {agent} in {folder}.';
const NO_SCHEDULE = 'No schedule named {name} for agent {agent} in {folder}.';
/** The process's exit code per outcome (the fixture's `run.exit`). */
export const EXIT_CODES: Record<string, number> = { delivered: 0, nothing: 0, failed: 1 };
/** `list`'s exit code when a file was refused (the fixture's `list.refused_exit`, 2026-09-26). */
export const LIST_REFUSED_EXIT = 1;

export interface CronCommandOptions {
  /** The folder whose agents to read; the working directory by default. */
  watch?: string;
  /**
   * Only this agent's schedules (the chat's `/cron`, interactive-mode spec
   * 3.7, 2026-09-26); every agent in the folder when unset, as the CLI lists.
   */
  agent?: string;
}

/** Seams for the tests: the agent builder, the clock, and where lines go. */
export interface CronCommandDeps {
  buildAgent?: (definition: AgentDefinition) => Promise<IAgent | null>;
  clock?: () => number;
  json?: boolean;
  log?: (line: string) => void;
  error?: (line: string) => void;
  emit?: (data: unknown) => void;
}

/** `1d`, `2h`, `30m` or `90s`: a whole number of the largest unit that divides it. */
export function humanDuration(seconds: number): string {
  for (const [unit, size] of [['d', 86400], ['h', 3600], ['m', 60]] as const) {
    if (seconds % size === 0) return `${seconds / size}${unit}`;
  }
  return `${seconds}s`;
}

type Row = Record<string, unknown>;

function whenColumn(row: Row): string {
  const when = row.kind === 'cron' ? `${row.expression} ${row.timezone}` : `every ${humanDuration(Number(row.every_seconds))}`;
  return row.heartbeat ? `${when} (heartbeat)` : when;
}

function nextColumn(row: Row): string {
  if (row.enabled === false) return 'off';
  return typeof row.next_run === 'string' ? row.next_run : '-';
}

function lastColumn(row: Row): string {
  if (row.running) return `running since ${row.last_fire}`;
  const last = row.last_run as { at: string; outcome: string; detail: string } | null | undefined;
  if (!last) return 'never';
  if (last.outcome === 'failed') return `failed ${last.at}: ${last.detail}`;
  return `${last.outcome} ${last.at}`;
}

/** The table for the runner's `list()` rows (the fixture's `list.cases`). */
export function renderScheduleTable(rows: readonly Row[]): string[] {
  const table: string[][] = [[...COLUMNS], ...rows.map((row) => [String(row.agent), String(row.name), whenColumn(row), nextColumn(row), lastColumn(row)])];
  const widths = COLUMNS.slice(0, -1).map((_, i) => Math.max(...table.map((row) => row[i].length)));
  return table.map((row) => [...widths.map((width, i) => row[i].padEnd(width)), row[row.length - 1]].join('  ').trimEnd());
}

/** One run's line (the fixture's `run.line`). */
export function runLine(record: { agent: string; name: string; outcome: string; detail: string }): string {
  return `${record.agent}/${record.name}: ${record.outcome} (${record.detail})`;
}

/** One agent file the daemon would serve, with the schedules its `cron:` declares. */
export interface FolderSchedules {
  definition: AgentDefinition;
  schedules: CronSchedule[];
}

/**
 * The folder's agents as the daemon would serve them: a file that does not
 * load, or whose `cron:` block is wrong, is said on stderr and skipped, as the
 * daemon says it and does not serve it.
 */
export function folderSchedules(
  folder: string,
  error: (line: string) => void = (line) => console.error(line),
  /** Counts the files that were refused, for `list` to exit non-zero on (2026-09-26). */
  refused: { count: number } = { count: 0 },
): FolderSchedules[] {
  const out: FolderSchedules[] = [];
  const refuse = (line: string) => {
    refused.count += 1;
    error(line);
  };
  for (const filePath of findAgentFiles(folder).keys()) {
    // Never through a symbolic link (S-290); refuse a file the loader will
    // not take (a link, bad YAML, an unknown key, the string cron form,
    // S-270/D7) and route the sentence through this command's own error, as
    // the daemon reports it. `agentDefinitionFrom` re-parses a file that has
    // already passed, so it never adds a second line.
    let content: string;
    try {
      content = readAgentFile(filePath);
      parseAgentMarkdown(content, filePath);
    } catch (err) {
      if (!(err instanceof AgentFileError)) throw err;
      // The loader's message names the file already (`<file>: <reason>`).
      refuse(`${err.message} Its schedules do not run until the file loads.`);
      continue;
    }
    const definition = agentDefinitionFrom(content, filePath);
    if (!definition) continue;
    try {
      out.push({ definition, schedules: definition.cron === undefined ? [] : parseCronBlock(definition.cron) });
    } catch (err) {
      if (!(err instanceof AgentFileError)) throw err;
      refuse(REFUSED_LINE.replace('{file}', filePath).replace('{reason}', err.message));
    }
  }
  return out;
}

function folderOf(options: CronCommandOptions): string {
  return path.resolve(options.watch ?? process.cwd());
}

/** `webagents cron list`. */
export async function cronListAction(options: CronCommandOptions = {}, deps: CronCommandDeps = {}): Promise<number> {
  const log = deps.log ?? ((line: string) => console.log(line));
  const folder = folderOf(options);
  // A file the daemon would refuse (the string `cron:` form, S-270/D7) is
  // said, the listing goes on, and the command exits 1 (2026-09-26, the e2e
  // run): it exited 0 after "No schedules", as if the file were fine.
  const refused = { count: 0 };
  const runner = new ScheduleRunner({ agentFor: () => undefined, clock: deps.clock, log: () => {}, persist: false });
  for (const { definition, schedules } of folderSchedules(folder, deps.error, refused)) {
    if (options.agent !== undefined && definition.name !== options.agent) continue;
    runner.setSchedules(definition.name, path.dirname(definition.filePath), schedules);
  }
  const rows = runner.list();
  const code = refused.count ? LIST_REFUSED_EXIT : 0;
  if (deps.json) {
    const { emit } = await import('./output.js');
    (deps.emit ?? emit)({ schedules: rows, refused: refused.count });
    return code;
  }
  if (rows.length === 0) {
    log((refused.count ? NONE_REFUSED : NONE).replace('{folder}', folder));
    return code;
  }
  for (const line of renderScheduleTable(rows)) log(line);
  return code;
}

/**
 * The agent a schedule names, built as the daemon builds it: with the
 * signing identity the daemon gives it (`daemon/agent-identity.ts`, the key
 * in the agent folder's `.webagents/keys`, the issuer from
 * `WEBAGENTS_PUBLIC_URL` or the configured daemon address), so a webhook run
 * here is signed exactly as the daemon signs it (2026-09-27). A key file that
 * cannot be used is said and the run goes out unsigned; it is never replaced.
 */
export async function buildScheduledAgent(definition: AgentDefinition, error: (line: string) => void = (line) => console.error(line)): Promise<IAgent | null> {
  const { buildDefinedAgent } = await import('../daemon/server.js');
  const { daemonAgentIdentity, daemonPublicUrl } = await import('../daemon/agent-identity.js');
  const { ConfigStore } = await import('./config-store.js');
  const store = new ConfigStore();
  const publicUrl = daemonPublicUrl({ hostname: String(store.get('daemon.host', '127.0.0.1')), port: Number(store.get('daemon.port', 8765)) });
  let identity: Awaited<ReturnType<typeof daemonAgentIdentity>> | undefined;
  try {
    identity = await daemonAgentIdentity(definition, publicUrl);
  } catch (err) {
    error(`[daemon] ${definition.filePath}: ${(err as Error).message} The run goes out without a signing identity.`);
  }
  return buildDefinedAgent(definition, { identity });
}

/** `webagents cron run <agent> <name>`: the exit code. */
export async function cronRunAction(agentName: string, scheduleName: string, options: CronCommandOptions = {}, deps: CronCommandDeps = {}): Promise<number> {
  const log = deps.log ?? ((line: string) => console.log(line));
  const error = deps.error ?? ((line: string) => console.error(line));
  const folder = folderOf(options);
  const found = folderSchedules(folder, deps.error).find(({ definition }) => definition.name === agentName);
  if (!found) {
    error(NO_AGENT.replace('{agent}', agentName).replace('{folder}', folder));
    return 1;
  }
  if (!found.schedules.some((schedule) => schedule.name === scheduleName)) {
    error(NO_SCHEDULE.replace('{name}', scheduleName).replace('{agent}', agentName).replace('{folder}', folder));
    return 1;
  }
  const build = deps.buildAgent ?? ((definition: AgentDefinition) => buildScheduledAgent(definition, error));
  const agent = await build(found.definition);
  const runner = new ScheduleRunner({ agentFor: () => agent ?? undefined, clock: deps.clock, log: () => {} });
  runner.setSchedules(agentName, path.dirname(found.definition.filePath), found.schedules);
  const record: RunRecord = await runner.runNow(agentName, scheduleName);
  const full = { agent: agentName, name: scheduleName, ...record };
  if (deps.json) {
    const { emit } = await import('./output.js');
    (deps.emit ?? emit)(full);
  } else {
    log(runLine(full));
  }
  return EXIT_CODES[record.outcome] ?? 1;
}
