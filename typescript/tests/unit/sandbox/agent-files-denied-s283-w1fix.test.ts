/**
 * S-283 (2026-09-26): a confined command could rewrite the agent's own
 * definition and context files (`AGENT.md`, `AGENT-*.md`, `WEBAGENTS.md`,
 * `mcp.json`) and other skills' `SKILL.md`, and the daemon reloaded them.
 * The agent's folder is a write root under `development` and under the
 * default `strict`, and only `.webagents/*` was denied inside it; a
 * SKILL.md script protected only its own folder.
 *
 * Pinned here, against the shared fixture's `agent_file_deny` section
 * (`python/tests/fixtures/sandbox/srt.json`; the Python twin is
 * `python/tests/sandbox/test_agent_files_denied_s283_w1fix.py`): the
 * escalation set carries the four literal names and the `AGENT-*.md`
 * pattern, resolved per engine (enumerated by name on both, the glob itself
 * on macOS); under real srt, both presets refuse every write, creation and
 * rename of those files while a plain file in the agent folder still
 * writes; and a SKILL.md script cannot write a sibling skill's `SKILL.md`
 * or `AGENT.md`, because every skill folder is read-only for it now.
 *
 * Enforcement tests run srt for real and skip, with the reason, where it
 * cannot run. The one expectation that differs by engine, a matching file
 * that does not exist when the command starts, is asserted on macOS and
 * recorded as the Linux residual the fixture states.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  AGENT_FILE_PATTERNS,
  ESCALATION_DENY,
  agentFileDenies,
  backendStatus,
  buildSettings,
  denyWrites,
  matchesAgentFilePattern,
  parseSandboxDeclaration,
  policyFromDeclaration,
  runSandboxed,
} from '../../../src/sandbox/index';
import { discoverSkills } from '../../../src/skills/skillmd/skillmd-loader';
import { SkillMdSkill } from '../../../src/skills/skillmd/skillmd-skill';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures');
const FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'sandbox', 'srt.json'), 'utf8'));
const SKILLMD = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'skillmd', 'skillmd.json'), 'utf8'));
const REPO = path.join(FIXTURES, 'skillmd', SKILLMD.sample.repo);
const PROBE = SKILLMD.scripts.probe as Record<string, string>;
const tempDir = tempDirs();

const status = backendStatus();
const forReal = status.available ? it : it.skip;
if (!status.available) console.warn(`srt S-283 tests skipped: ${status.reason}`);

const OWNER = { authenticated: true, scope: 'owner', provider: 'local' };
const context = { auth: OWNER, get: () => undefined } as never;

/** The four literal names S-283 added to the escalation set. */
const LITERAL_AGENT_FILES = ['AGENT.md', 'WEBAGENTS.md', 'mcp.json', '.agents/skills'];

/**
 * An agent folder as the daemon serves one: its definition, a named
 * sibling agent file, the inherited context, an MCP config, and the sample
 * repository's two skills installed under `.agents/skills`.
 */
function agentFolder(): string {
  const agentDir = path.join(tempDir('wa-s283-'), 'agent');
  for (const name of ['pdf', 'xlsx']) {
    fs.cpSync(path.join(REPO, 'skills', name), path.join(agentDir, '.agents', 'skills', name), { recursive: true });
  }
  for (const name of ['AGENT.md', 'AGENT-helper.md', 'WEBAGENTS.md', 'mcp.json']) {
    fs.writeFileSync(path.join(agentDir, name), 'original\n');
  }
  return fs.realpathSync(agentDir);
}

function policy(declared: Record<string, unknown>, cwd: string) {
  return policyFromDeclaration(parseSandboxDeclaration(declared), { cwd });
}

describe('the fixture is the contract', () => {
  it('names the agent files in the escalation set and the patterns beside it', () => {
    for (const name of LITERAL_AGENT_FILES) expect([...ESCALATION_DENY]).toContain(name);
    expect([...ESCALATION_DENY]).toEqual(FIXTURE.escalation_deny);
    expect([...AGENT_FILE_PATTERNS]).toEqual(FIXTURE.agent_file_deny.patterns);
    expect(FIXTURE.agent_file_deny.glob_on).toEqual(['darwin']);
  });

  it("matches the daemon's agent file names and nothing else", () => {
    for (const name of ['AGENT-helper.md', 'AGENT-evil.md', 'AGENT-.md']) expect(matchesAgentFilePattern(name), name).toBe(true);
    for (const name of ['AGENT.md', 'AGENTS.md', 'agent-x.md', 'AGENT-x.md.bak', 'AGENT_x.md', 'notes/AGENT-x.md']) {
      expect(matchesAgentFilePattern(name), name).toBe(false);
    }
  });

  it('enumerates the existing matches by name on both engines, and passes the glob on macOS only', () => {
    const agentDir = agentFolder();
    expect(agentFileDenies(agentDir, 'linux')).toEqual([path.join(agentDir, 'AGENT-helper.md')]);
    expect(agentFileDenies(agentDir, 'darwin')).toEqual([path.join(agentDir, 'AGENT-helper.md'), path.join(agentDir, 'AGENT-*.md')]);
    // A root that cannot be listed still gets the pattern on macOS, and nothing on Linux.
    const missing = path.join(agentDir, 'nowhere');
    expect(agentFileDenies(missing, 'linux')).toEqual([]);
    expect(agentFileDenies(missing, 'darwin')).toEqual([path.join(missing, 'AGENT-*.md')]);
  });

  it('denies them in every write root under development and under the default strict', () => {
    const agentDir = agentFolder();
    for (const declared of [{ preset: 'development' }, { preset: 'strict' }]) {
      const built = policy(declared, agentDir);
      expect(built.writeRoots, JSON.stringify(declared)).toContain(agentDir);
      const darwin = denyWrites(built, 'darwin');
      const linux = denyWrites(built, 'linux');
      for (const name of [...LITERAL_AGENT_FILES, 'AGENT-helper.md']) {
        expect(darwin, `${name} under ${declared.preset} (darwin)`).toContain(path.join(agentDir, name));
        expect(linux, `${name} under ${declared.preset} (linux)`).toContain(path.join(agentDir, name));
      }
      expect(darwin).toContain(path.join(agentDir, 'AGENT-*.md'));
      expect(linux).not.toContain(path.join(agentDir, 'AGENT-*.md'));
    }
  });

  it('writes them into the settings srt reads, and srt accepts the shape', async () => {
    const agentDir = agentFolder();
    const built = policy({ preset: 'development' }, agentDir);
    const settings = buildSettings(built, {}) as { filesystem: { denyWrite: string[] } };
    for (const name of [...LITERAL_AGENT_FILES, 'AGENT-helper.md']) {
      expect(settings.filesystem.denyWrite, name).toContain(path.join(agentDir, name));
    }
    if (process.platform === 'darwin') expect(settings.filesystem.denyWrite).toContain(path.join(agentDir, 'AGENT-*.md'));
    const srt = await import(/* @vite-ignore */ '@anthropic-ai/sandbox-runtime' as string);
    const parsed = srt.SandboxRuntimeConfigSchema.safeParse(settings);
    expect(parsed.success, JSON.stringify(parsed.error?.issues)).toBe(true);
    expect(parsed.data.filesystem.denyWrite).toEqual(settings.filesystem.denyWrite);
  });
});

/**
 * One shell line per attempt, each reporting `WROTE <label>` or
 * `denied <label>`, so a single confined command probes every file.
 */
function probeCommand(agentDir: string): string {
  const write = (relative: string) =>
    `if (echo pwned > "${path.join(agentDir, relative)}") 2>/dev/null; then echo "WROTE ${relative}"; else echo "denied ${relative}"; fi`;
  return [
    write('AGENT.md'),
    write('AGENT-helper.md'),
    write('AGENT-evil.md'),
    write('WEBAGENTS.md'),
    write('mcp.json'),
    write('.agents/skills/xlsx/SKILL.md'),
    `if mkdir -p "${path.join(agentDir, '.agents/skills/new')}" 2>/dev/null; then echo "WROTE .agents/skills/new"; else echo "denied .agents/skills/new"; fi`,
    `if mv "${path.join(agentDir, 'AGENT-helper.md')}" "${path.join(agentDir, 'AGENT-moved.md')}" 2>/dev/null; then echo "WROTE rename"; else echo "denied rename"; fi`,
    `if rm "${path.join(agentDir, 'AGENT.md')}" 2>/dev/null; then echo "WROTE unlink"; else echo "denied unlink"; fi`,
    write('work.txt'),
  ].join('\n');
}

function verdicts(stdout: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of stdout.split('\n')) {
    const match = /^(WROTE|denied) (.+)$/.exec(line.trim());
    if (match) out[match[2]] = match[1];
  }
  return out;
}

describe('a confined command cannot write the agent files (real srt)', () => {
  for (const declared of [{ preset: 'development' }, { preset: 'strict' }]) {
    forReal(`under ${declared.preset}`, async () => {
      const agentDir = agentFolder();
      const built = policy(declared, agentDir);
      const result = await runSandboxed(probeCommand(agentDir), built, { timeout: 30 });
      const seen = verdicts(result.stdout);
      const denied = [
        'AGENT.md',
        'AGENT-helper.md',
        'WEBAGENTS.md',
        'mcp.json',
        '.agents/skills/xlsx/SKILL.md',
        '.agents/skills/new',
        'rename',
        'unlink',
        // A matching file that does not exist yet: the glob covers it on
        // macOS; on Linux srt takes no write glob (fixture `agent_file_deny.linux`).
        ...(process.platform === 'darwin' ? ['AGENT-evil.md'] : []),
      ];
      for (const label of denied) expect(seen[label], `${label}: ${result.stdout}\n${result.stderr}`).toBe('denied');
      expect(seen['work.txt'], result.stdout).toBe('WROTE');
      // The host agrees: nothing of the agent's changed, and the plain file landed.
      for (const name of ['AGENT.md', 'AGENT-helper.md', 'WEBAGENTS.md', 'mcp.json']) {
        expect(fs.readFileSync(path.join(agentDir, name), 'utf8'), name).toBe('original\n');
      }
      expect(fs.readFileSync(path.join(agentDir, '.agents/skills/xlsx/SKILL.md'), 'utf8')).toBe(
        fs.readFileSync(path.join(REPO, 'skills', 'xlsx', 'SKILL.md'), 'utf8'),
      );
      expect(fs.existsSync(path.join(agentDir, '.agents/skills/new'))).toBe(false);
      expect(fs.existsSync(path.join(agentDir, 'AGENT-moved.md'))).toBe(false);
      if (process.platform === 'darwin') expect(fs.existsSync(path.join(agentDir, 'AGENT-evil.md'))).toBe(false);
      expect(fs.readFileSync(path.join(agentDir, 'work.txt'), 'utf8')).toBe('pwned\n');
    }, 60_000);
  }
});

describe('a SKILL.md script cannot write a sibling skill or the agent file', () => {
  function skillFor(agentDir: string, sandbox: Record<string, unknown> | null): SkillMdSkill {
    const found = discoverSkills(agentDir);
    expect(found.skills.map((s) => s.name)).toEqual(['pdf', 'xlsx']);
    return new SkillMdSkill({ skills: found.skills, agentDir, sandbox });
  }

  it('makes every skill folder read-only for a script, not only its own', () => {
    const agentDir = agentFolder();
    const skill = skillFor(agentDir, { preset: 'development' });
    const found = discoverSkills(agentDir);
    const pdf = found.skills.find((s) => s.name === 'pdf')!;
    const built = skill.scriptPolicy(pdf);
    expect(typeof built).not.toBe('string');
    const readOnly = (built as { readOnly?: string[] }).readOnly ?? [];
    for (const each of found.skills) expect(readOnly, each.name).toContain(each.directory);
    // And the agent's own files are in the denies the script's settings get.
    const denied = denyWrites(built as never);
    for (const name of [...LITERAL_AGENT_FILES, 'AGENT-helper.md']) expect(denied, name).toContain(path.join(agentDir, name));
  });

  forReal('refuses the sibling SKILL.md and AGENT.md under development, and still writes a plain file', async () => {
    const agentDir = agentFolder();
    const skill = skillFor(agentDir, { preset: 'development' });
    const run = skill.tools.find((t) => t.name === 'run_skill_script')!;
    const sibling = path.join(agentDir, '.agents', 'skills', 'xlsx', 'SKILL.md');
    const own = path.join(agentDir, '.agents', 'skills', 'pdf', 'planted.md');
    const definition = path.join(agentDir, 'AGENT.md');
    const plain = path.join(agentDir, 'work.txt');
    const out = (await run.handler(
      { skill: 'pdf', script: 'scripts/probe.py', args: ['--write', sibling, '--write', own, '--write', definition, '--write', plain] },
      context,
    )) as string;
    const lines = out.trim().split('\n');
    expect(lines[0]).toBe(PROBE.ok);
    expect(lines[1].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[2].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[3].startsWith(PROBE.write_refused_starts_with), out).toBe(true);
    expect(lines[4].startsWith(PROBE.write_ok_starts_with), out).toBe(true);
    expect(fs.readFileSync(sibling, 'utf8')).toBe(fs.readFileSync(path.join(REPO, 'skills', 'xlsx', 'SKILL.md'), 'utf8'));
    expect(fs.existsSync(own)).toBe(false);
    expect(fs.readFileSync(definition, 'utf8')).toBe('original\n');
    expect(fs.existsSync(plain)).toBe(true);
  }, 60_000);
});
