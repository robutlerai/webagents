/**
 * `webagents cron list` and `webagents cron run` (plan item 1.7, 2026-09-26):
 * the lines are `python/tests/fixtures/cli/cron.json`, the same the Python
 * CLI prints (`tests/cli/test_cron_cli_w1daemon.py`); the help words are held
 * to these declarations by the Python `test_cli_parity.py`.
 */

import { describe, expect, it } from 'vitest';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { IAgent } from '../../../src/core/types';
import {
  EXIT_CODES,
  cronListAction,
  cronRunAction,
  folderSchedules,
  humanDuration,
  renderScheduleTable,
  runLine,
} from '../../../src/cli/cron-action';
import { parseIso, statePath } from '../../../src/daemon/schedule-runner';
import { TSX_CLI, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const FIXTURE = JSON.parse(readFileSync(path.join(FIXTURES, 'cli', 'cron.json'), 'utf8')) as {
  help: Record<string, string>;
  list: { columns: string[]; none: string; durations: Record<string, string>; cases: { name: string; schedules: Record<string, unknown>[]; lines: string[] }[] };
  run: {
    line: string;
    no_agent: string;
    no_schedule: string;
    exit: Record<string, number>;
    cases: { record: { agent: string; name: string; at: string; outcome: string; detail: string }; line: string; exit: number }[];
  };
};
const DAEMON = JSON.parse(readFileSync(path.join(FIXTURES, 'daemon', 'cron.json'), 'utf8')) as {
  agent_file: string;
  parsed: { name: string; deliver: unknown }[];
  string_form_refused: string;
  file_entry: string;
};
const CLI_SOURCE = path.resolve(HERE, '../../../src/cli/index.ts');
const T0 = parseIso('2026-09-26T10:00:30Z')!;

const tempDir = tempDirs();

function project(): string {
  const dir = tempDir('wa-cron-cli-');
  const folder = path.join(dir, 'project');
  mkdirSync(folder);
  writeFileSync(path.join(folder, 'AGENT.md'), DAEMON.agent_file);
  return folder;
}

function fakeAgent(reply: string | Error) {
  const calls: unknown[] = [];
  const agent = {
    name: 'reporter',
    async run(messages: unknown) {
      calls.push(messages);
      if (reply instanceof Error) throw reply;
      return { content: reply };
    },
  } as unknown as IAgent;
  return { agent, calls, set: (next: string | Error) => { reply = next; } };
}

function runCli(args: string[], cwd: string): Promise<{ code: number | null; out: string; err: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [TSX_CLI, CLI_SOURCE, ...args], {
      cwd,
      env: { ...process.env, HOME: path.join(cwd, '.home'), WEBAGENTS_PROFILE: '', WEBAGENTS_SECRETS_BACKEND: 'file' },
    });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (err += d));
    child.on('close', (code) => resolve({ code, out, err }));
  });
}

describe('the words', () => {
  it('reads durations as the fixture spells them', () => {
    for (const [seconds, text] of Object.entries(FIXTURE.list.durations)) expect(humanDuration(Number(seconds)), seconds).toBe(text);
  });

  it.each(FIXTURE.list.cases.map((c) => [c.name, c] as const))('the table: %s', (_name, { schedules, lines }) => {
    expect(renderScheduleTable(schedules)).toEqual(lines);
  });

  it.each(FIXTURE.run.cases.map((c) => [c.line, c] as const))('the run line: %s', (_line, { record, line, exit }) => {
    expect(runLine(record)).toBe(line);
    expect(EXIT_CODES[record.outcome]).toBe(exit);
    expect(EXIT_CODES).toEqual(FIXTURE.run.exit);
  });
});

describe('cron list', () => {
  it('reads the folder as the daemon would and writes nothing', async () => {
    const folder = project();
    writeFileSync(path.join(folder, 'AGENT-broken.md'), '---\nname: broken\ncron: "0 9 * * *"\n---\nReport.\n');
    writeFileSync(path.join(folder, 'AGENT-plain.md'), '---\nname: plain\n---\nNo schedules here.\n');
    const lines: string[] = [];
    const errors: string[] = [];
    await cronListAction({ watch: folder }, { clock: () => T0, log: (l) => lines.push(l), error: (l) => errors.push(l) });
    expect(lines).toEqual([
      'AGENT     SCHEDULE      WHEN                       NEXT                  LAST',
      'reporter  daily-report  0 9 * * 1-5 Europe/Berlin  2026-09-28T07:00:00Z  never',
      'reporter  queue         every 30m                  off                   never',
      'reporter  watch         every 1h (heartbeat)       2026-09-26T11:00:30Z  never',
      'reporter  ping          */5 * * * * UTC            2026-09-26T10:05:00Z  never',
    ]);
    // `cron`'s own words since 2026-09-28 (`cli/cron.json`, `list.refused_line`).
    expect(errors).toEqual([`${path.join(folder, 'AGENT-broken.md')}: ${DAEMON.string_form_refused} Its schedules do not run until the file loads.`]);
    // Nothing written: the daemon's clock is the daemon's.
    expect(existsSync(path.join(folder, '.webagents'))).toBe(false);
    // The daemon's walk order: by file name, `AGENT-plain.md` before `AGENT.md`.
    expect(folderSchedules(folder, () => {}).map((f) => f.definition.name)).toEqual(['plain', 'reporter']);
  });

  it("shows the daemon's state and says when there is none", async () => {
    const folder = project();
    const state = statePath(folder, 'reporter');
    mkdirSync(path.dirname(state), { recursive: true });
    writeFileSync(
      state,
      JSON.stringify({
        schedules: {
          'daily-report': {
            spec: 'cron 0 9 * * 1-5 Europe/Berlin',
            nextRun: '2026-09-28T07:00:00Z',
            lastFire: '2026-09-25T07:00:00Z',
            lastRun: { at: '2026-09-25T07:00:00Z', outcome: 'delivered', detail: 'file reports/daily.md' },
          },
        },
      }),
    );
    const lines: string[] = [];
    await cronListAction({ watch: folder }, { clock: () => T0, log: (l) => lines.push(l) });
    expect(lines[1]).toBe('reporter  daily-report  0 9 * * 1-5 Europe/Berlin  2026-09-28T07:00:00Z  delivered 2026-09-25T07:00:00Z');

    const empty = tempDir('wa-cron-empty-');
    writeFileSync(path.join(empty, 'AGENT.md'), '---\nname: quiet\n---\nNothing scheduled.\n');
    lines.length = 0;
    await cronListAction({ watch: empty }, { log: (l) => lines.push(l) });
    expect(lines).toEqual([FIXTURE.list.none.replace('{folder}', path.resolve(empty))]);

    const emitted: unknown[] = [];
    await cronListAction({ watch: folder }, { clock: () => T0, json: true, emit: (d) => emitted.push(d) });
    const schedules = (emitted[0] as { schedules: { name: string; deliver: unknown }[] }).schedules;
    expect(schedules.map((s) => s.name)).toEqual(DAEMON.parsed.map((s) => s.name));
    expect(schedules[0].deliver).toEqual(DAEMON.parsed[0].deliver);
  });

  it('through the CLI', async () => {
    const folder = project();
    const { code, out } = await runCli(['cron', 'list', '-w', folder], path.dirname(folder));
    expect(code).toBe(0);
    const lines = out.trimEnd().split('\n');
    expect(lines[0].split(/\s+/)).toEqual(FIXTURE.list.columns);
    expect(lines[1].startsWith('reporter  daily-report  0 9 * * 1-5 Europe/Berlin  ')).toBe(true);
    expect(lines.slice(1).map((l) => l.split(/\s+/)[1])).toEqual(DAEMON.parsed.map((s) => s.name));

    const asJson = await runCli(['--json', 'cron', 'list', '-w', folder], path.dirname(folder));
    expect(asJson.code).toBe(0);
    expect((JSON.parse(asJson.out) as { data: { schedules: { name: string }[] } }).data.schedules.map((s) => s.name)).toEqual(
      DAEMON.parsed.map((s) => s.name),
    );

    const help = await runCli(['cron', '-h'], path.dirname(folder));
    expect(help.out).toContain(FIXTURE.help.group);
    expect(help.out).toContain(FIXTURE.help.list);
    expect(help.out).toContain(FIXTURE.help.run);
  }, 60_000);
});

describe('cron run', () => {
  it('builds the agent, runs the schedule now and records it', async () => {
    const folder = project();
    const { agent, calls, set } = fakeAgent('Nothing happened.');
    const built: string[] = [];
    const buildAgent = async (definition: { filePath: string }) => {
      built.push(definition.filePath);
      return agent;
    };
    const lines: string[] = [];
    expect(await cronRunAction('reporter', 'daily-report', { watch: folder }, { buildAgent, clock: () => T0, log: (l) => lines.push(l) })).toBe(0);
    expect(built).toEqual([path.join(folder, 'AGENT.md')]);
    expect(calls).toEqual([[{ role: 'user', content: "Summarize yesterday's activity." }]]);
    expect(lines).toEqual(['reporter/daily-report: delivered (file reports/daily.md)']);
    expect(readFileSync(path.join(folder, 'reports', 'daily.md'), 'utf8')).toBe(
      DAEMON.file_entry.replace('{schedule}', 'daily-report').replace('{ran_at}', '2026-09-26T10:00:30Z').replace('{content}', 'Nothing happened.'),
    );
    // Recorded where the daemon's listing reads it; the next fire is the schedule's own.
    const state = JSON.parse(readFileSync(statePath(folder, 'reporter'), 'utf8')) as { schedules: Record<string, Record<string, unknown>> };
    expect(state.schedules['daily-report'].lastRun).toEqual({ at: '2026-09-26T10:00:30Z', outcome: 'delivered', detail: 'file reports/daily.md' });
    expect(state.schedules['daily-report'].nextRun).toBe('2026-09-28T07:00:00Z');
    const listed: string[] = [];
    await cronListAction({ watch: folder }, { clock: () => T0, log: (l) => listed.push(l) });
    expect(listed[1].endsWith('  delivered 2026-09-26T10:00:30Z')).toBe(true);

    // A failing turn: the record, and exit 1.
    set(new Error('no model key'));
    lines.length = 0;
    expect(await cronRunAction('reporter', 'daily-report', { watch: folder }, { buildAgent, clock: () => T0, log: (l) => lines.push(l) })).toBe(1);
    expect(lines).toEqual(['reporter/daily-report: failed (turn failed: no model key)']);

    // As JSON.
    set('Fine.');
    const emitted: unknown[] = [];
    expect(await cronRunAction('reporter', 'daily-report', { watch: folder }, { buildAgent, clock: () => T0, json: true, emit: (d) => emitted.push(d) })).toBe(0);
    expect(emitted).toEqual([{ agent: 'reporter', name: 'daily-report', at: '2026-09-26T10:00:30Z', outcome: 'delivered', detail: 'file reports/daily.md' }]);
  });

  it('names what it cannot find', async () => {
    const folder = project();
    const errors: string[] = [];
    const deps = { error: (l: string) => errors.push(l), buildAgent: async () => null };
    expect(await cronRunAction('nobody', 'daily-report', { watch: folder }, deps)).toBe(1);
    expect(await cronRunAction('reporter', 'nightly', { watch: folder }, deps)).toBe(1);
    expect(errors).toEqual([
      FIXTURE.run.no_agent.replace('{agent}', 'nobody').replace('{folder}', path.resolve(folder)),
      FIXTURE.run.no_schedule.replace('{name}', 'nightly').replace('{agent}', 'reporter').replace('{folder}', path.resolve(folder)),
    ]);

    const { code, err } = await runCli(['cron', 'run', 'nobody', 'daily-report', '-w', folder], path.dirname(folder));
    expect(code).toBe(1);
    expect(err).toContain(FIXTURE.run.no_agent.replace('{agent}', 'nobody').replace('{folder}', path.resolve(folder)));
  }, 60_000);
});
