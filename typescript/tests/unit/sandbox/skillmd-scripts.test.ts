/**
 * A SKILL.md skill's scripts run through the kernel sandbox (gap-closure
 * plan item 1.4, 2026-09-26), under real srt (the tests skip, with the
 * reason, where it cannot run). The Python suite runs the same probes in
 * `tests/sandbox/test_skillmd_scripts.py`, against the same fixture
 * repository and `python/tests/fixtures/skillmd/skillmd.json`.
 *
 * What is proved: the sample script runs and its output comes back as the
 * fixture says; a script cannot write outside the agent's allowed folders,
 * not even into its own skill folder; it cannot reach the network unless the
 * agent's `network:` lists the host; an agent with no `sandbox:` still
 * confines scripts (a synthesised `strict` policy); a timeout is reported as
 * one; secrets are withheld.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ENV_CLI, backendStatus, resetBackendStatus } from '../../../src/sandbox/index';
import { discoverSkills } from '../../../src/skills/skillmd/skillmd-loader';
import { SkillMdSkill } from '../../../src/skills/skillmd/skillmd-skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/skillmd');
const FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'skillmd.json'), 'utf8'));
const REPO = path.join(FIXTURES, FIXTURE.sample.repo);
const PROBE = FIXTURE.scripts.probe as Record<string, string>;
const tempDir = tempDirs();

const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`srt skill-script tests skipped: ${status.reason}`);

const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };
const context = { auth: OWNER, get: () => undefined } as never;

/** An agent folder with the sample repository's skills installed under `.agents/skills`, and a folder outside it. */
function agentFolder(): { agentDir: string; outside: string } {
  const base = tempDir('wa-skillmd-run-');
  const agentDir = path.join(base, 'agent');
  for (const name of ['pdf', 'xlsx']) {
    fs.cpSync(path.join(REPO, 'skills', name), path.join(agentDir, '.agents', 'skills', name), { recursive: true });
  }
  const outside = path.join(base, 'outside');
  fs.mkdirSync(outside);
  return { agentDir: fs.realpathSync(agentDir), outside: fs.realpathSync(outside) };
}

function skillFor(agentDir: string, sandbox: Record<string, unknown> | null): SkillMdSkill {
  const found = discoverSkills(agentDir);
  expect(found.skills.map((s) => s.name)).toEqual(['pdf', 'xlsx']);
  return new SkillMdSkill({ skills: found.skills, agentDir, sandbox });
}

async function run(skill: SkillMdSkill, params: Record<string, unknown>): Promise<string> {
  const found = skill.tools.find((t) => t.name === 'run_skill_script')!;
  return (await found.handler(params, context)) as string;
}

describe('scripts run confined', () => {
  let server: http.Server;
  let port = 0;

  beforeAll(async () => {
    server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end('HELLO-FROM-SITE\n');
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    port = (server.address() as AddressInfo).port;
  });

  afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  forReal('runs the sample script and answers', async () => {
    const { agentDir } = agentFolder();
    const out = await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/fill_form.py', args: ['input.pdf', 'name=Ada'] });
    expect(out.trim()).toBe(FIXTURE.sample.fill_form_output);
  });

  forReal('says when there is no output', async () => {
    const { agentDir } = agentFolder();
    fs.writeFileSync(path.join(agentDir, '.agents', 'skills', 'pdf', 'scripts', 'quiet.sh'), 'exit 0\n');
    expect(await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/quiet.sh' })).toBe(FIXTURE.scripts.no_output);
  });

  forReal('returns stderr and the exit code', async () => {
    const { agentDir } = agentFolder();
    fs.writeFileSync(path.join(agentDir, '.agents', 'skills', 'pdf', 'scripts', 'fail.sh'), 'echo out; echo err >&2; exit 3\n');
    const out = await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/fail.sh' });
    expect(out.startsWith('out\n')).toBe(true);
    expect(out).toContain((FIXTURE.scripts.stderr_line as string).replace('{stderr}', 'err\n'));
    expect(out.endsWith((FIXTURE.scripts.exit_line as string).replace('{code}', '3'))).toBe(true);
  });

  forReal('cannot write outside the allowed folders, nor into its own folder', async () => {
    const { agentDir, outside } = agentFolder();
    const outsideFile = path.join(outside, 'x.txt');
    const own = path.join(agentDir, '.agents', 'skills', 'pdf', 'planted.md');
    const inside = path.join(agentDir, 'work.txt');
    const out = await run(skillFor(agentDir, { preset: 'development' }), {
      skill: 'pdf',
      script: 'scripts/probe.py',
      args: ['--write', outsideFile, '--write', own, '--write', inside],
    });
    const lines = out.trim().split('\n');
    expect(lines[0]).toBe(PROBE.ok);
    expect(lines[1].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[2].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[3].startsWith(PROBE.write_ok_starts_with), out).toBe(true);
    expect(fs.existsSync(outsideFile)).toBe(false);
    expect(fs.existsSync(own)).toBe(false);
    expect(fs.existsSync(inside)).toBe(true);
  });

  forReal('cannot reach the network unless the agent allows the host', async () => {
    const { agentDir } = agentFolder();
    const url = `http://127.0.0.1:${port}/`;
    const closed = await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/probe.py', args: ['--fetch', url] });
    expect(closed.trim().split('\n')[1].startsWith(PROBE.fetch_refused_starts_with), closed).toBe(true);
    const opened = await run(skillFor(agentDir, { preset: 'development', network: [`127.0.0.1:${port}`] }), {
      skill: 'pdf',
      script: 'scripts/probe.py',
      args: ['--fetch', url],
    });
    expect(opened.trim().split('\n')[1], opened).toBe(`${PROBE.fetch_ok_starts_with}HELLO-FROM-SITE`);
  });

  forReal('confines scripts of an agent with no sandbox', async () => {
    const { agentDir, outside } = agentFolder();
    const outsideFile = path.join(outside, 'x.txt');
    const inside = path.join(agentDir, 'work.txt');
    const url = `http://127.0.0.1:${port}/`;
    const out = await run(skillFor(agentDir, null), {
      skill: 'pdf',
      script: 'scripts/probe.py',
      args: ['--write', outsideFile, '--write', inside, '--fetch', url],
    });
    const lines = out.trim().split('\n');
    expect(lines[0]).toBe(PROBE.ok);
    // A synthesised strict policy: no writes anywhere, not even the agent folder, and no network.
    expect(lines[1].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[2].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[3].startsWith(PROBE.fetch_refused_starts_with), out).toBe(true);
    expect(fs.existsSync(outsideFile)).toBe(false);
    expect(fs.existsSync(inside)).toBe(false);
  });

  forReal('reports a timeout as one', async () => {
    const { agentDir } = agentFolder();
    fs.writeFileSync(path.join(agentDir, '.agents', 'skills', 'pdf', 'scripts', 'slow.sh'), 'sleep 30\n');
    const out = await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/slow.sh', timeout: 1 });
    expect(out).toBe((FIXTURE.scripts.timed_out as string).replace('{timeout}', '1'));
  }, 20_000);

  forReal('withholds secrets from scripts', async () => {
    const { agentDir } = agentFolder();
    const saved = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = 'sk-should-not-leak';
    try {
      fs.writeFileSync(path.join(agentDir, '.agents', 'skills', 'pdf', 'scripts', 'env.sh'), 'echo "key=${OPENAI_API_KEY:-withheld}"\n');
      expect((await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/env.sh' })).trim()).toBe('key=withheld');
    } finally {
      if (saved === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = saved;
    }
  });
});

describe('without srt', () => {
  it('refuses the script of an agent whose declared sandbox cannot run', async () => {
    const { agentDir } = agentFolder();
    const saved = process.env[ENV_CLI];
    process.env[ENV_CLI] = path.join(agentDir, 'nowhere', 'cli.js');
    resetBackendStatus();
    try {
      const out = await run(skillFor(agentDir, { preset: 'development' }), { skill: 'pdf', script: 'scripts/fill_form.py' });
      expect(out.startsWith(FIXTURE.scripts.refusals.unavailable_starts_with)).toBe(true);
      expect(out).toContain('was not run');
    } finally {
      if (saved === undefined) delete process.env[ENV_CLI];
      else process.env[ENV_CLI] = saved;
      resetBackendStatus();
    }
  });
});
