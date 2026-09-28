/**
 * The `cron:` block of an agent file (plan item 1.7, 2026-09-26): the same
 * block parses to the same schedules, and a wrong block gets the same
 * sentence, in both SDKs. The cases are
 * `python/tests/fixtures/daemon/cron.json`; the Python suite runs them in
 * `tests/cli/test_cron_schema_w1daemon.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgentFileError, parseAgentMarkdown } from '../../../src/agents/index';
import { cronExpression, describeSchedule, everySeconds, isTimezone, parseCronBlock } from '../../../src/agents/schedules';
import { WebAgentsDaemon } from '../../../src/daemon/server';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/daemon/cron.json'), 'utf8')) as {
  defaults: { timezone: string };
  agent_file: string;
  parsed: Record<string, unknown>[];
  string_form_refused: string;
  expressions: { valid: [string, string][]; invalid: string[] };
  every: { valid: Record<string, number>; invalid: string[] };
  errors: { cron: unknown; message: string }[];
};

const tempDir = tempDirs();

// The daemon serves an agent only with a model its callers can run on
// (S-327, 2026-09-28: a provider key, or the agent's own platform
// credential, never the owner's sign-in); these tests are about the rest.
const MODEL_KEY_SAVED = process.env.OPENAI_API_KEY;
beforeEach(() => {
  process.env.OPENAI_API_KEY = 'sk-test-daemon-model';
});
afterEach(() => {
  if (MODEL_KEY_SAVED === undefined) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = MODEL_KEY_SAVED;
});

/** Through JSON, so `undefined` and `null` compare as the fixture spells them. */
const plain = (value: unknown): unknown => JSON.parse(JSON.stringify(value));

describe('the block, as the fixture and the Python loader read it', () => {
  it('parses the agent file to the fixture schedules', () => {
    const parsed = parseAgentMarkdown(FIXTURE.agent_file, '/x/AGENT.md');
    expect(parsed.name).toBe('reporter');
    expect(plain(parseCronBlock(parsed.cron).map(describeSchedule))).toEqual(FIXTURE.parsed);
  });

  it('keeps the block as written, out of extra', () => {
    const parsed = parseAgentMarkdown(FIXTURE.agent_file);
    expect(Array.isArray(parsed.cron)).toBe(true);
    expect(parsed.extra).not.toHaveProperty('cron');
  });

  it('refuses the string form', () => {
    expect(() => parseCronBlock('0 9 * * *')).toThrow(FIXTURE.string_form_refused);
  });

  it('an empty list is no schedules', () => {
    expect(parseCronBlock([])).toEqual([]);
  });

  it.each(FIXTURE.errors.map((c) => [c.message.slice(0, 60), c] as const))('%s', (_label, { cron, message }) => {
    let thrown: unknown;
    try {
      parseCronBlock(cron);
    } catch (err) {
      thrown = err;
    }
    expect(thrown).toBeInstanceOf(AgentFileError);
    expect((thrown as Error).message).toBe(message);
  });

  it.each(FIXTURE.expressions.valid)('accepts %s', (written, normalized) => {
    expect(cronExpression(written)).toBe(normalized);
  });

  it.each(FIXTURE.expressions.invalid)('rejects %j', (written) => {
    expect(cronExpression(written)).toBeUndefined();
  });

  it('reads every durations', () => {
    for (const [written, seconds] of Object.entries(FIXTURE.every.valid)) expect(everySeconds(written), written).toBe(seconds);
    for (const written of FIXTURE.every.invalid) expect(everySeconds(written), written).toBeUndefined();
  });

  it('checks timezones against the zone database', () => {
    expect(isTimezone(FIXTURE.defaults.timezone)).toBe(true);
    expect(isTimezone('Europe/Berlin')).toBe(true);
    expect(isTimezone('America/New_York')).toBe(true);
    expect(isTimezone('Mars/Olympus')).toBe(false);
    expect(isTimezone('')).toBe(false);
    expect(isTimezone(5)).toBe(false);
  });
});

describe('the daemon', () => {
  afterEach(() => vi.restoreAllMocks());

  it('keeps a served agent schedules and refuses a file whose block is wrong', async () => {
    const dir = tempDir('wa-cron-schema-');
    writeFileSync(path.join(dir, 'AGENT.md'), FIXTURE.agent_file);
    writeFileSync(path.join(dir, 'AGENT-broken.md'), '---\nname: broken\ncron: "0 9 * * *"\n---\nReport.\n');
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    const daemon = new WebAgentsDaemon({ port: 0, watchDir: dir, cron: false, healthChecks: false });
    try {
      await daemon.discover();
      expect(plain(daemon.schedulesOf('reporter').map(describeSchedule))).toEqual(FIXTURE.parsed);
      expect(daemon.schedulesOf('broken')).toEqual([]);
      expect(daemon.getRegistry().get('broken')).toBeUndefined();
      const said = errors.mock.calls.map((call) => String(call[0]));
      expect(said).toContain(`[daemon] ${path.join(dir, 'AGENT-broken.md')}: ${FIXTURE.string_form_refused} The agent is not served.`);
      const info = await daemon.app.fetch(new Request('http://localhost/agents/reporter'));
      expect(info.status).toBe(200);
      expect((await info.json()).schedules).toEqual(FIXTURE.parsed);
    } finally {
      daemon.stop();
    }
  });
});
