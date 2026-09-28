/**
 * `webagents skills add <source>`, `skills remove` of an installed SKILL.md
 * skill, `skills list` and `doctor` for SKILL.md skills (gap-closure plan
 * item 1.4, 2026-09-26), against `python/tests/fixtures/skillmd/skillmd.json`
 * (`install`, `doctor`), which the Python suite reads too
 * (`tests/cli/test_skillmd_install.py`).
 *
 * The source is a git repository the test makes from `fixtures/skillmd/repo`
 * and reaches over a `file://` URL: nothing here touches the internet. What
 * is proved: the sources a name can be, the sample skill installed from the
 * repository with its file list shown and the lock recording the commit and
 * the digest, `--skill`, `--yes` required without a terminal, the skipped
 * skill reported and not installed, the limits, a skill the lock does not
 * know left alone, removal, and the same skill then loading into an agent
 * built from the folder with the catalog and activation the fixture pins.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseAgentMarkdown } from '../../../src/agents/index';
import { doctorReport, discoverSkills } from '../../../src/skills/skillmd/skillmd-loader';
import { SKILL_KEY, type SkillMdSkill } from '../../../src/skills/skillmd/skillmd-skill';
import {
  BINARY_EXTENSIONS,
  DOWNLOAD_LIMIT,
  EXTRACTED_LIMIT,
  FILE_LIMIT,
  LOCK_ENTRY_KEYS,
  LOCK_FILE,
  LOCK_VERSION,
  PLUGIN_MANIFESTS,
  SCRIPT_DIRS,
  SCRIPT_EXTENSIONS,
  SEARCH_DEPTH,
  SEARCH_ROOTS,
  installFromSource,
  listFiles,
  locateSkills,
  parseSource,
  readLock,
  treeDigest,
} from '../../../src/skills/skillmd/skillmd-install';
import { resolveSkillsByName } from '../../../src/skills/resolve';
import { skillmdListLines, skillsCommand } from '../../../src/cli/skills-edit';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, '../../../../python/tests/fixtures/skillmd');
const FIXTURE = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'skillmd.json'), 'utf8'));
const INSTALL = FIXTURE.install;
const MESSAGES = INSTALL.messages as Record<string, string>;
const REPO = path.join(FIXTURES, FIXTURE.sample.repo);
const tempDir = tempDirs();

const GIT_ENV = {
  GIT_AUTHOR_NAME: 'fixture',
  GIT_AUTHOR_EMAIL: 'fixture@example.invalid',
  GIT_COMMITTER_NAME: 'fixture',
  GIT_COMMITTER_EMAIL: 'fixture@example.invalid',
  GIT_AUTHOR_DATE: '2026-09-26T00:00:00Z',
  GIT_COMMITTER_DATE: '2026-09-26T00:00:00Z',
};

const fill = (template: string, values: Record<string, string | number>): string =>
  template.replace(/\{(\w+)\}/g, (_m, key: string) => String(values[key] ?? `{${key}}`));

function git(args: string[], cwd: string): string {
  const result = spawnSync('git', args, { cwd, encoding: 'utf8', env: { ...process.env, ...GIT_ENV } });
  if (result.status !== 0) throw new Error(`git ${args.join(' ')}: ${result.stderr}`);
  return result.stdout.trim();
}

/** The fixture repository as a git repository, and its file:// URL. */
function repository(): { dir: string; url: string; sha: string } {
  const dir = path.join(tempDir('wa-skillmd-src-'), 'src');
  fs.cpSync(REPO, dir, { recursive: true });
  git(['init', '-q', '-b', 'main'], dir);
  git(['add', '.'], dir);
  git(['commit', '-q', '-m', 'fixture skills'], dir);
  return { dir, url: `file://${dir}`, sha: git(['rev-parse', 'HEAD'], dir) };
}

function agentFolder(): string {
  const folder = path.join(tempDir('wa-skillmd-agent-'), 'agent');
  fs.mkdirSync(folder);
  fs.writeFileSync(path.join(folder, 'AGENT.md'), '---\nname: bot\nskills:\n  - openai\nsandbox:\n  preset: development\n---\n\nBody.\n');
  return folder;
}

async function run(
  action: 'add' | 'remove',
  names: string[],
  folder: string,
  options: { skill?: string; yes?: boolean; tty?: boolean; confirm?: (q: string) => Promise<boolean>; facts?: () => Promise<{ hasKey(v: string): boolean; signedIn: boolean }> } = {},
): Promise<{ code: number; out: string[]; err: string[] }> {
  const out: string[] = [];
  const err: string[] = [];
  const code = await skillsCommand(action, names, { folder, tty: false, ...options }, { out: (l) => out.push(l), err: (l) => err.push(l) });
  return { code, out, err };
}

describe('the fixture is the contract', () => {
  it('pins the limits, search rules and flags', () => {
    const limits = INSTALL.limits;
    expect([DOWNLOAD_LIMIT, EXTRACTED_LIMIT, FILE_LIMIT]).toEqual([limits.download_bytes, limits.extracted_bytes, limits.files]);
    expect([...SEARCH_ROOTS]).toEqual(INSTALL.search.roots);
    expect(SEARCH_DEPTH).toBe(INSTALL.search.depth);
    expect([...PLUGIN_MANIFESTS]).toEqual(INSTALL.search.plugin_manifests);
    expect([...SCRIPT_DIRS]).toEqual(INSTALL.flags.script_dirs);
    expect([...SCRIPT_EXTENSIONS]).toEqual(INSTALL.flags.script_extensions);
    expect([...BINARY_EXTENSIONS]).toEqual(INSTALL.flags.binary_extensions);
    expect(LOCK_FILE).toBe((FIXTURE.layout.lock_file as string).split('/').join(path.sep));
    expect(LOCK_VERSION).toBe(INSTALL.lock.version);
    expect([...LOCK_ENTRY_KEYS]).toEqual(INSTALL.lock.entry_keys);
  });

  it.each(INSTALL.sources as Array<{ text: string; kind: string; url?: string | null; ref?: string | null; subpath?: string | null; path?: string }>)('source $text', (c) => {
    const saved = process.cwd();
    process.chdir(tempDir('wa-skillmd-cwd-')); // no `owner/repo` folder exists here
    try {
      const source = parseSource(c.text);
      expect(source.kind).toBe(c.kind);
      if (c.kind === 'git') expect([source.url, source.ref ?? null, source.subpath ?? null]).toEqual([c.url, c.ref, c.subpath]);
      else if (c.kind === 'local') expect(source.path).toBe(c.path);
    } finally {
      process.chdir(saved);
    }
  });

  it('reads a folder that exists as a local source, and a bare word as a name', () => {
    const saved = process.cwd();
    const base = tempDir('wa-skillmd-cwd-');
    process.chdir(base);
    try {
      fs.mkdirSync(path.join(base, 'skills', 'pdf'), { recursive: true });
      expect(parseSource('skills/pdf').kind).toBe('local');
      expect(parseSource('pdf').kind).toBe('name');
      fs.mkdirSync(path.join(base, 'shell'));
      expect(parseSource('shell').kind).toBe('name');
    } finally {
      process.chdir(saved);
    }
  });

  it('computes the digest vector', () => {
    const vector = INSTALL.lock.digest_vector as { files: Record<string, string>; tree: string };
    const base = tempDir('wa-skillmd-digest-');
    for (const [relative, text] of Object.entries(vector.files)) {
      fs.mkdirSync(path.dirname(path.join(base, relative)), { recursive: true });
      fs.writeFileSync(path.join(base, relative), text);
    }
    expect(treeDigest(base)).toBe(vector.tree);
    expect(treeDigest(base, Object.keys(vector.files))).toBe(vector.tree);
    // The definition, spelled out: sha256 over "path\n<sha256 of content>\n" per sorted file.
    const digest = createHash('sha256');
    for (const relative of Object.keys(vector.files).sort()) {
      digest.update(`${relative}\n${createHash('sha256').update(vector.files[relative]).digest('hex')}\n`);
    }
    expect(vector.tree).toBe(`sha256:${digest.digest('hex')}`);
  });
});

describe('locating', () => {
  it('finds the sample repository by the skills.sh rules', () => {
    const repo = repository();
    const { located, skipped } = locateSkills(repo.dir, undefined, 'src');
    expect(located.map((c) => c.name)).toEqual(['pdf', 'xlsx']);
    expect(skipped.map((s) => [s.name, s.problem])).toEqual([['broken', 'no_description']]);
  });

  it('narrows to a subpath', () => {
    const repo = repository();
    expect(locateSkills(repo.dir, 'skills/pdf', 'src').located.map((c) => c.name)).toEqual(['pdf']);
    expect(locateSkills(repo.dir, 'skills', 'src').located.map((c) => c.name)).toEqual(['pdf', 'xlsx']);
    expect(locateSkills(repo.dir, '../outside', 'src')).toEqual({ located: [], skipped: [] });
  });

  it('finds a root skill, nested folders and a dot-agent folder, not too deep', () => {
    const root = path.join(tempDir('wa-skillmd-root-'), 'one-skill');
    const write = (relative: string, text: string): void => {
      fs.mkdirSync(path.dirname(path.join(root, relative)), { recursive: true });
      fs.writeFileSync(path.join(root, relative), text);
    };
    write('SKILL.md', '---\nname: one-skill\ndescription: d\n---\n');
    write('.cursor/skills/deep/SKILL.md', '---\ndescription: d\n---\n');
    write('skills/cat/sub/nested/SKILL.md', '---\ndescription: d\n---\n');
    write('skills/a/b/c/toodeep/SKILL.md', '---\ndescription: d\n---\n');
    expect(locateSkills(root, undefined, 'one-skill').located.map((c) => c.name)).toEqual(['one-skill', 'nested', 'deep']);
  });

  it('flags scripts and binaries', () => {
    const repo = repository();
    const pdf = path.join(repo.dir, 'skills', 'pdf');
    const { files, links } = listFiles(pdf);
    expect(links).toEqual([]);
    expect(files.map((f) => f.path).sort()).toEqual([...FIXTURE.sample.pdf.files, 'SKILL.md'].sort());
    const byPath = new Map(files.map((f) => [f.path, f]));
    expect(byPath.get('scripts/fill_form.py')).toMatchObject({ script: true, binary: false });
    expect(byPath.get('forms.md')).toMatchObject({ script: false, binary: false });
    fs.writeFileSync(path.join(pdf, 'tool.bin'), Buffer.from([0, 1]));
    fs.writeFileSync(path.join(pdf, 'run.sh'), 'echo\n');
    const again = new Map(listFiles(pdf).files.map((f) => [f.path, f]));
    expect(again.get('tool.bin')).toMatchObject({ binary: true, script: false });
    expect(again.get('run.sh')).toMatchObject({ script: true });
  });
});

describe('installing', () => {
  let profile: string | undefined;
  beforeEach(() => {
    profile = process.env.WEBAGENTS_PROFILE;
    delete process.env.WEBAGENTS_PROFILE;
  });
  afterEach(() => {
    if (profile !== undefined) process.env.WEBAGENTS_PROFILE = profile;
  });

  it('installs the sample skill and records the lock', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, out, err } = await run('add', [repo.url], folder, { yes: true, skill: 'pdf' });
    expect(code, JSON.stringify({ out, err })).toBe(0);
    expect(err).toEqual([fill(MESSAGES.skipped, { name: 'broken', reason: FIXTURE.sample.skipped[0].reason })]);
    expect(out[0]).toBe(fill(MESSAGES.found, { count: 1, source: repo.url, names: 'pdf' }));
    expect(out[1].startsWith('pdf: Work with PDF files')).toBe(true);
    const files = [...FIXTURE.sample.pdf.files, 'SKILL.md'].sort();
    const listed = out.slice(2, 2 + files.length);
    expect(listed.map((line) => line.split(' (')[0].trim())).toEqual(files);
    for (const line of listed) {
      const relative = line.split(' (')[0].trim();
      const size = fs.statSync(path.join(REPO, 'skills', 'pdf', relative)).size;
      const flag = relative.startsWith('scripts/') ? MESSAGES.script_flag : '';
      expect(line).toBe(fill(MESSAGES.file_line, { path: relative, size }) + flag);
    }
    expect(out[out.length - 1]).toBe(fill(MESSAGES.installed_git, { name: 'pdf', files: files.length, sha7: repo.sha.slice(0, 7) }));
    const installed = path.join(folder, '.agents', 'skills', 'pdf');
    expect(fs.readFileSync(path.join(installed, 'SKILL.md'))).toEqual(fs.readFileSync(path.join(REPO, 'skills', 'pdf', 'SKILL.md')));
    expect(fs.existsSync(path.join(installed, 'scripts', 'probe.py'))).toBe(true);
    expect(fs.existsSync(path.join(folder, '.agents', 'skills', 'xlsx'))).toBe(false);
    const lock = JSON.parse(fs.readFileSync(path.join(folder, LOCK_FILE), 'utf8'));
    expect(lock.version).toBe(LOCK_VERSION);
    const entry = lock.skills.pdf;
    expect(Object.keys(entry)).toEqual([...LOCK_ENTRY_KEYS].sort());
    expect([entry.source, entry.url, entry.ref]).toEqual([repo.url, repo.url, null]);
    expect(entry.subpath).toBe('skills/pdf');
    expect(entry.commit).toBe(repo.sha);
    expect(entry.files).toEqual(files);
    expect(entry.tree).toBe(treeDigest(installed));
    expect(entry.tree).toBe(treeDigest(path.join(REPO, 'skills', 'pdf')));
    expect(entry.installed_at.endsWith('Z')).toBe(true);
    // The agent file was not edited: every agent in the folder finds .agents/skills on its own.
    expect(fs.readFileSync(path.join(folder, 'AGENT.md'), 'utf8')).not.toContain('pdf');
    // Sorted keys, two-space indent, trailing newline: byte for byte what Python writes.
    const text = fs.readFileSync(path.join(folder, LOCK_FILE), 'utf8');
    expect(text.endsWith('}\n')).toBe(true);
    expect(text.startsWith('{\n  "skills": {\n    "pdf": {\n      "commit": ')).toBe(true);
  });

  it('installs every skill without --skill', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, out } = await run('add', [repo.url], folder, { yes: true });
    expect(code).toBe(0);
    expect(out[0]).toBe(fill(MESSAGES.found, { count: 2, source: repo.url, names: 'pdf, xlsx' }));
    expect(Object.keys(readLock(folder).skills).sort()).toEqual(['pdf', 'xlsx']);
  });

  it('names the skills there when --skill names none of them', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, err } = await run('add', [repo.url], folder, { yes: true, skill: 'nope' });
    expect(code).toBe(1);
    expect(err[err.length - 1]).toBe(fill(MESSAGES.no_such_skill, { name: 'nope', source: repo.url, names: 'pdf, xlsx' }));
    expect(fs.existsSync(path.join(folder, '.agents'))).toBe(false);
  });

  it('requires --yes without a terminal', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, out, err } = await run('add', [repo.url], folder, { yes: false, tty: false });
    expect(code).toBe(1);
    expect(err[err.length - 1]).toBe(MESSAGES.no_tty);
    expect(out[0].startsWith('Found ')).toBe(true);
    expect(fs.existsSync(path.join(folder, '.agents'))).toBe(false);
  });

  it('asks the person at a terminal', async () => {
    const repo = repository();
    const folder = agentFolder();
    const asked: string[] = [];
    const declined = await run('add', [repo.url], folder, {
      yes: false,
      tty: true,
      skill: 'pdf',
      confirm: async (question) => {
        asked.push(question);
        return false;
      },
    });
    expect(declined.code).toBe(0);
    expect(asked).toEqual([fill(MESSAGES.confirm, { names: 'pdf' })]);
    expect(declined.out[declined.out.length - 1]).toBe(MESSAGES.not_installed);
    expect(fs.existsSync(path.join(folder, '.agents'))).toBe(false);
    const accepted = await run('add', [repo.url], folder, { yes: false, tty: true, skill: 'pdf', confirm: async () => true });
    expect(accepted.code).toBe(0);
    expect(accepted.out[accepted.out.length - 1].startsWith('Installed pdf')).toBe(true);
  });

  it('installs from a local folder', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, out } = await run('add', [path.join(repo.dir, 'skills', 'pdf')], folder, { yes: true });
    expect(code).toBe(0);
    expect(out[out.length - 1]).toBe(fill(MESSAGES.installed_local, { name: 'pdf', files: 6 }));
    const entry = readLock(folder).skills.pdf;
    expect([entry.commit, entry.url]).toEqual([null, null]);
  });

  it('leaves a skill the lock does not know alone', async () => {
    const repo = repository();
    const folder = agentFolder();
    const mine = path.join(folder, '.agents', 'skills', 'pdf');
    fs.mkdirSync(mine, { recursive: true });
    fs.writeFileSync(path.join(mine, 'SKILL.md'), '---\ndescription: mine\n---\nMine.\n');
    const { code, err } = await run('add', [repo.url], folder, { yes: true, skill: 'pdf' });
    expect(code).toBe(1);
    expect(err[err.length - 1]).toBe(fill(MESSAGES.exists, { name: 'pdf' }));
    expect(fs.readFileSync(path.join(mine, 'SKILL.md'), 'utf8').endsWith('Mine.\n')).toBe(true);
  });

  it('replaces what the lock knows on a reinstall', async () => {
    const repo = repository();
    const folder = agentFolder();
    expect((await run('add', [repo.url], folder, { yes: true, skill: 'pdf' })).code).toBe(0);
    fs.writeFileSync(path.join(folder, '.agents', 'skills', 'pdf', 'extra.md'), 'stale');
    expect((await run('add', [repo.url], folder, { yes: true, skill: 'pdf' })).code).toBe(0);
    expect(fs.existsSync(path.join(folder, '.agents', 'skills', 'pdf', 'extra.md'))).toBe(false);
  });

  it('refuses a skill with a symbolic link', async () => {
    const repo = repository();
    const folder = agentFolder();
    fs.symlinkSync('/etc/hosts', path.join(repo.dir, 'skills', 'xlsx', 'hosts'));
    git(['add', '.'], repo.dir);
    git(['commit', '-q', '-m', 'link'], repo.dir);
    const { code, err } = await run('add', [repo.url], folder, { yes: true, skill: 'xlsx' });
    expect(code).toBe(1);
    expect(err[err.length - 1]).toBe(fill(MESSAGES.symlink, { name: 'xlsx', path: 'hosts' }));
  });

  it('refuses a source that cannot be fetched, and one that is not a folder', async () => {
    const folder = agentFolder();
    const nowhere = path.join(tempDir('wa-skillmd-nowhere-'), 'nowhere');
    const url = `file://${nowhere}`;
    const fetched = await run('add', [url], folder, { yes: true });
    expect(fetched.code).toBe(1);
    expect(fetched.err[fetched.err.length - 1].startsWith(fill(MESSAGES.fetch_failed.split('{detail}')[0], { source: url }))).toBe(true);
    const local = await run('add', [nowhere], folder, { yes: true });
    expect(local.code).toBe(1);
    expect(local.err[local.err.length - 1]).toBe(fill(MESSAGES.not_a_folder, { source: nowhere }));
  });

  it('refuses a repository without skills', async () => {
    const folder = agentFolder();
    const source = path.join(tempDir('wa-skillmd-empty-'), 'empty');
    fs.mkdirSync(source);
    fs.writeFileSync(path.join(source, 'README.md'), 'nothing');
    const { code, err } = await run('add', [source], folder, { yes: true });
    expect(code).toBe(1);
    expect(err[err.length - 1]).toBe(fill(MESSAGES.none_found, { source }));
  });

  it('takes names and sources in one command', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, out } = await run('add', ['shell', repo.url], folder, {
      yes: true,
      skill: 'xlsx',
      facts: async () => ({ hasKey: () => true, signedIn: true }),
    });
    expect(code).toBe(0);
    expect(out[0]).toBe('Added shell to AGENT.md.');
    expect(out[1]).toBe('Skills: openai, shell');
    expect(out[2]).toBe(fill(MESSAGES.found, { count: 1, source: repo.url, names: 'xlsx' }));
    expect(fs.readFileSync(path.join(folder, 'AGENT.md'), 'utf8').split('- shell').length).toBe(2);
  });
});

describe('removing', () => {
  it('removes an installed skill and its lock entry', async () => {
    const repo = repository();
    const folder = agentFolder();
    expect((await run('add', [repo.url], folder, { yes: true })).code).toBe(0);
    const { code, out, err } = await run('remove', ['pdf'], folder);
    expect(code).toBe(0);
    expect(err).toEqual([]);
    expect(out).toEqual([fill(MESSAGES.removed, { name: 'pdf' })]);
    expect(fs.existsSync(path.join(folder, '.agents', 'skills', 'pdf'))).toBe(false);
    expect(Object.keys(readLock(folder).skills)).toEqual(['xlsx']);
  });

  it('still sends a coded name to the agent file', async () => {
    const repo = repository();
    const folder = agentFolder();
    expect((await run('add', [repo.url], folder, { yes: true })).code).toBe(0);
    const { code, out } = await run('remove', ['openai'], folder);
    expect(code).toBe(0);
    expect(out).toEqual(['Removed openai from AGENT.md.', 'Skills: none']);
    expect(fs.existsSync(path.join(folder, '.agents', 'skills', 'pdf'))).toBe(true);
  });

  it('does not take a source', async () => {
    const repo = repository();
    const folder = agentFolder();
    const { code, err } = await run('remove', [repo.url], folder);
    expect(code).toBe(1);
    expect(err).toEqual([`${repo.url} is not a skill name; remove takes the names \`skills list\` shows.`]);
  });
});

describe('the installed skill runs in an agent', () => {
  it('loads into the agent built from the folder with the catalog and activation the fixture pins', async () => {
    const repo = repository();
    const folder = agentFolder();
    expect((await run('add', [repo.url], folder, { yes: true, skill: 'pdf' })).code).toBe(0);
    const parsed = parseAgentMarkdown(fs.readFileSync(path.join(folder, 'AGENT.md'), 'utf8'), path.join(folder, 'AGENT.md'));
    const resolved = await resolveSkillsByName(parsed.skillEntries, {
      agentDir: folder,
      ...(parsed.sandbox ? { sandbox: parsed.sandbox } : {}),
      ...(parsed.agentSkills ? { agentSkills: parsed.agentSkills } : {}),
    });
    expect(resolved.skillmd.skills).toEqual(['pdf']);
    const skill = resolved.byName.get(SKILL_KEY) as unknown as SkillMdSkill;
    expect([...skill.skills.keys()]).toEqual(['pdf']);
    const location = fs.realpathSync(path.join(folder, '.agents', 'skills', 'pdf', 'SKILL.md'));
    const directory = path.dirname(location);
    const catalog = skill.skillsCatalog({ auth: { authenticated: true, scope: 'owner' } } as never);
    expect(catalog.startsWith(FIXTURE.catalog.preamble)).toBe(true);
    expect(catalog).toContain('<name>pdf</name>');
    expect(catalog).toContain(`<location>${location}</location>`);
    const activate = skill.tools.find((t) => t.name === 'activate_skill')!;
    expect(await activate.handler({ name: 'pdf' }, { auth: {}, get: () => undefined } as never)).toBe(
      (FIXTURE.sample.activation as string).replace('{pdf_dir}', directory),
    );
    for (const tool of skill.tools) expect(tool.scopes).toEqual(['owner']);
  });
});

describe('listing and doctor', () => {
  it('shows the two kinds apart in skills list', async () => {
    const repo = repository();
    const folder = agentFolder();
    const before = skillmdListLines(folder);
    expect(before[0]).toBe(MESSAGES.list_header);
    expect(before).toContain(MESSAGES.list_none);
    expect(before[before.length - 1]).toBe(fill(MESSAGES.list_hint, { command: 'webagents skills add <owner/repo | git URL | folder>' }));
    expect((await run('add', [repo.url], folder, { yes: true })).code).toBe(0);
    const broken = path.join(folder, '.agents', 'skills', 'broken');
    fs.mkdirSync(broken);
    fs.writeFileSync(path.join(broken, 'SKILL.md'), '---\nname: broken\n---\n');
    const after = skillmdListLines(folder);
    const location = fs.realpathSync(path.join(folder, '.agents', 'skills', 'pdf', 'SKILL.md'));
    expect(after).toContain(fill(MESSAGES.list_line, { name: 'pdf', location }));
    expect(after).toContain(fill(MESSAGES.list_skipped, { name: 'broken', reason: FIXTURE.sample.skipped[0].reason }));
    expect(after).not.toContain(MESSAGES.list_none);
  });

  it('reports the skills and the skipped ones in doctor', async () => {
    const repo = repository();
    const folder = agentFolder();
    const doctor = FIXTURE.doctor;
    expect(doctorReport(discoverSkills(folder))).toEqual({ status: 'ok', detail: doctor.none });
    expect((await run('add', [repo.url], folder, { yes: true })).code).toBe(0);
    expect(doctorReport(discoverSkills(folder))).toEqual({ status: 'ok', detail: fill(doctor.some, { count: 2, s: 's', names: 'pdf, xlsx' }) });
    const broken = path.join(folder, '.agents', 'skills', 'broken');
    fs.mkdirSync(broken);
    fs.writeFileSync(path.join(broken, 'SKILL.md'), '---\nname: broken\n---\n');
    expect(doctorReport(discoverSkills(folder))).toEqual({
      status: 'warn',
      detail: fill(doctor.some, { count: 2, s: 's', names: 'pdf, xlsx' }) + fill(doctor.skipped, { name: 'broken', reason: FIXTURE.sample.skipped[0].reason }),
      // The fix names the folders (2026-09-26).
      fix: fill(doctor.fix, { names: 'broken' }),
    });
  });

  it('refuses an agent_skills that is not a list, with the fixture sentence', () => {
    const file = path.join(tempDir('wa-skillmd-file-'), 'AGENT.md');
    fs.writeFileSync(file, '---\nname: a\nagent_skills: ./skills\n---\n');
    expect(() => parseAgentMarkdown(fs.readFileSync(file, 'utf8'), file)).toThrow(`${file}: ${FIXTURE.load_messages.agent_skills_not_a_list}`);
  });
});
