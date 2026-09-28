/**
 * Installing SKILL.md skills from outside: `webagents skills add <source>`
 * (gap-closure plan item 1.4, 2026-09-26). The Python twin is
 * `python/webagents/agents/skills/local/skillmd/skillmd_install.py`; both run
 * `python/tests/fixtures/skillmd/skillmd.json` (`install`).
 *
 * WHAT A SOURCE IS (`parseSource`): a git URL (`https://...`, `git@host:...`,
 * `file://...`), `owner/repo` on GitHub, a `.../tree/<ref>/<path>` or
 * `.../blob/<ref>/<path>/SKILL.md` page, or a local folder (`./x`, `../x`,
 * `/x`, `~/x`). A bare word (`shell`, `pdf`) is a NAME, never a source:
 * names belong to the existing editor of the agent file's `skills:` list.
 *
 * WHY THE VETTING. ClawHavoc (341 malicious ClawHub skills) is the reason
 * nothing here installs silently: the repository is fetched at ONE commit,
 * its skills are located with the same search rules skills.sh uses, every
 * file is listed with its size, scripts and binaries are flagged, the person
 * confirms (or passes `--yes`, which is required when there is no terminal),
 * and only then is the skill copied into `<agent folder>/.agents/skills/<name>`
 * and recorded in `<agent folder>/.webagents/skills.lock` with its source,
 * the commit, a digest of its files and the file list. Limits: 10 MiB
 * fetched, 25 MiB and 1,000 files installed, no symbolic links. What is
 * installed is owner-only to the agent until `access: tools:` opens it, and
 * its scripts run only in the sandbox (`skillmd-skill.ts`).
 *
 * THE LOCK is the record of what was installed and from where: the skills
 * the CLI may replace or remove are the ones it wrote. A folder under
 * `.agents/skills` that the lock does not know is the person's, and `add`
 * and `remove` leave it alone and say so.
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

import { SKILL_FILE_NAMES, SKIPPED_DIRS, loadSkillDir, skillFileIn, type SkippedSkill } from './skillmd-loader';

/** The limits skills.sh applies, applied here too. */
export const DOWNLOAD_LIMIT = 10 * 1024 * 1024;
export const EXTRACTED_LIMIT = 25 * 1024 * 1024;
export const FILE_LIMIT = 1000;

/**
 * Where skills live in a repository (skills.sh's rules): the root itself,
 * then these folders, walked `SEARCH_DEPTH` levels deep, plus any
 * `.<agent>/skills` folder at the root and the paths a Claude Code plugin
 * manifest names. A shallower SKILL.md shadows a deeper one of the same name.
 */
export const SEARCH_ROOTS = [
  '.', 'skills', 'skills/.curated', 'skills/.experimental', 'skills/.system',
  '.agents/skills', '.claude/skills', '.codex/skills', '.opencode/skills', '.hermes/skills',
  '.cursor/skills', '.github/skills',
] as const;
export const SEARCH_DEPTH = 3;
export const PLUGIN_MANIFESTS = ['.claude-plugin/marketplace.json', '.claude-plugin/plugin.json'] as const;

/** What is flagged in the file list before the person confirms. */
export const SCRIPT_DIRS = ['scripts'] as const;
export const SCRIPT_EXTENSIONS = ['.py', '.sh', '.js', '.mjs', '.cjs', '.ts', '.rb', '.pl', '.ps1', '.bat', '.cmd'] as const;
export const BINARY_EXTENSIONS = ['.so', '.dylib', '.dll', '.exe', '.wasm', '.bin', '.pyc', '.o', '.a', '.jar', '.class', '.node'] as const;

export const LOCK_FILE = path.join('.webagents', 'skills.lock');
export const LOCK_VERSION = 1;
export const LOCK_ENTRY_KEYS = ['source', 'url', 'ref', 'subpath', 'commit', 'tree', 'files', 'installed_at'] as const;

const FORGE_HOSTS = new Set(['github.com', 'gitlab.com', 'bitbucket.org', 'codeberg.org']);
const SHA_RE = /^[0-9a-f]{40}$/;
const OWNER_REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const PAGE_RE = /^(https?:\/\/([^/]+)\/[^/]+\/[^/]+?)(?:\.git)?(?:\/(tree|blob)\/([^/]+)(?:\/(.*?))?)?\/?$/;

/** A refusal or a failure, worded for the person at the terminal. */
export class InstallError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InstallError';
  }
}

export interface Source {
  kind: 'git' | 'local' | 'name';
  text: string;
  url?: string;
  ref?: string;
  subpath?: string;
  path?: string;
}

function expandHome(p: string): string {
  return p.startsWith('~') ? path.join(os.homedir(), p.slice(1)) : p;
}

function exists(p: string): boolean {
  try {
    fs.lstatSync(p);
    return true;
  } catch {
    return false;
  }
}

function isDir(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function isLink(p: string): boolean {
  try {
    return fs.lstatSync(p).isSymbolicLink();
  } catch {
    return false;
  }
}

function realOrSelf(p: string): string {
  try {
    return fs.realpathSync(p);
  } catch {
    return path.resolve(p);
  }
}

/** What `skills add <text>` means (file comment; pinned by the fixture's `install.sources`). */
export function parseSource(text: string): Source {
  const s = text.trim();
  if (s === '.' || s === '..' || s.startsWith('./') || s.startsWith('../') || s.startsWith('/') || s.startsWith('~')) {
    return { kind: 'local', text, path: s };
  }
  if (s.includes('://')) {
    const match = PAGE_RE.exec(s);
    if (match && FORGE_HOSTS.has(match[2].toLowerCase())) {
      const [, base, , kind, ref] = match;
      let subpath = match[5];
      if (kind === 'blob' && subpath) {
        for (const name of SKILL_FILE_NAMES) {
          if (subpath.endsWith('/' + name) || subpath === name) {
            subpath = subpath.slice(0, -name.length).replace(/\/+$/, '');
            break;
          }
        }
      }
      return {
        kind: 'git',
        text,
        url: `${base}.git`,
        ...(ref ? { ref } : {}),
        ...(kind && subpath ? { subpath } : {}),
      };
    }
    return { kind: 'git', text, url: s };
  }
  if (s.startsWith('git@') && s.includes(':')) return { kind: 'git', text, url: s };
  if (OWNER_REPO_RE.test(s) && !exists(s)) return { kind: 'git', text, url: `https://github.com/${s}.git` };
  if (s.includes('/') && exists(expandHome(s))) return { kind: 'local', text, path: s };
  return { kind: 'name', text };
}

/** The repository's own name, for a skill at its root. */
export function repoName(source: Source): string {
  let base = (source.url ?? source.path ?? source.text).replace(/\/+$/, '');
  base = base.slice(base.lastIndexOf('/') + 1);
  base = base.slice(base.lastIndexOf(':') + 1);
  return base.endsWith('.git') ? base.slice(0, -4) : base;
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

function git(args: string[], cwd: string): { status: number; stdout: string; stderr: string; error?: Error } {
  const result = spawnSync('git', args, {
    cwd,
    encoding: 'utf8',
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_ASKPASS: process.env.GIT_ASKPASS ?? 'echo' },
    timeout: 600_000,
  });
  return { status: result.status ?? 1, stdout: result.stdout ?? '', stderr: result.stderr ?? '', ...(result.error ? { error: result.error } : {}) };
}

function folderBytes(root: string): number {
  let total = 0;
  const walk = (dir: string): void => {
    let names: string[];
    try {
      names = fs.readdirSync(dir);
    } catch {
      return;
    }
    for (const name of names) {
      const full = path.join(dir, name);
      let stat: fs.Stats;
      try {
        stat = fs.lstatSync(full);
      } catch {
        continue;
      }
      if (stat.isDirectory()) walk(full);
      else total += stat.size;
    }
  };
  walk(root);
  return total;
}

function detail(result: { status: number; stdout: string; stderr: string; error?: Error }): string {
  if (result.error) return result.error.message;
  const lines = `${result.stderr || result.stdout || ''}`.split('\n').map((l) => l.trim()).filter(Boolean);
  return lines.length ? lines[lines.length - 1] : `git exited with ${result.status}`;
}

/**
 * Clone `source` at one commit into `workdir`: the checkout, the commit and
 * the bytes fetched. A branch or tag name goes to `--branch`; a full commit
 * SHA is fetched by itself. Refuses when more than `DOWNLOAD_LIMIT` came down.
 */
export function fetchGit(source: Source, workdir: string): { checkout: string; commit: string; fetched: number } {
  const probe = spawnSync('git', ['--version'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) throw new InstallError(`Could not fetch ${source.text}: git is not installed`);
  const checkout = path.join(workdir, 'repo');
  const url = String(source.url);
  if (source.ref && SHA_RE.test(source.ref)) {
    fs.mkdirSync(checkout);
    for (const args of [['init', '-q'], ['remote', 'add', 'origin', url], ['fetch', '-q', '--depth', '1', 'origin', source.ref], ['checkout', '-q', 'FETCH_HEAD']]) {
      const result = git(args, checkout);
      if (result.status !== 0) throw new InstallError(`Could not fetch ${source.text}: ${detail(result)}`);
    }
  } else {
    const args = ['clone', '-q', '--depth', '1', '--single-branch'];
    if (source.ref) args.push('--branch', source.ref);
    args.push(url, checkout);
    const result = git(args, workdir);
    if (result.status !== 0) throw new InstallError(`Could not fetch ${source.text}: ${detail(result)}`);
  }
  const head = git(['rev-parse', 'HEAD'], checkout);
  if (head.status !== 0) throw new InstallError(`Could not fetch ${source.text}: ${detail(head)}`);
  const fetched = folderBytes(path.join(checkout, '.git'));
  if (fetched > DOWNLOAD_LIMIT) {
    throw new InstallError(`Refused: fetching ${source.text} took ${humanSize(fetched)}, more than the 10 MiB limit.`);
  }
  return { checkout, commit: head.stdout.trim(), fetched };
}

// ---------------------------------------------------------------------------
// Locating skills in a checkout
// ---------------------------------------------------------------------------

/** Folders holding a SKILL.md under `base`, breadth first up to `depth` levels, a folder with one not descended into. */
function skillDirsBelow(base: string, depth: number): string[] {
  const found: string[] = [];
  let level = [base];
  for (let i = 0; i < depth; i++) {
    const next: string[] = [];
    for (const directory of level) {
      let names: string[];
      try {
        names = fs.readdirSync(directory).sort();
      } catch {
        continue;
      }
      for (const name of names) {
        if (SKIPPED_DIRS.has(name) || name.startsWith('.')) continue;
        const child = path.join(directory, name);
        if (isLink(child) || !isDir(child)) continue;
        if (skillFileIn(child)) found.push(child);
        else next.push(child);
      }
    }
    level = next;
  }
  return found;
}

/** The skill folders a Claude Code plugin manifest names, relative to the root. */
function manifestPaths(root: string): string[] {
  const paths: string[] = [];
  for (const relative of PLUGIN_MANIFESTS) {
    const file = path.join(root, relative);
    let data: unknown;
    try {
      data = JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch {
      continue;
    }
    const entries: unknown[] = [];
    if (data && typeof data === 'object') {
      const record = data as Record<string, unknown>;
      if (Array.isArray(record.plugins)) {
        for (const plugin of record.plugins) {
          const skills = (plugin as Record<string, unknown> | null)?.skills;
          if (Array.isArray(skills)) entries.push(...skills);
          else if (typeof skills === 'string') entries.push(skills);
        }
      }
      if (Array.isArray(record.skills)) entries.push(...record.skills);
      else if (typeof record.skills === 'string') entries.push(record.skills);
    }
    for (const entry of entries) if (typeof entry === 'string' && entry.trim()) paths.push(entry.trim());
  }
  return paths;
}

/**
 * The skill folders in a checkout, by skills.sh's rules (`SEARCH_ROOTS`), or
 * under `subpath` when a page URL named one; and the folders whose SKILL.md
 * cannot load.
 */
export function locateSkills(rootDir: string, subpath: string | undefined, nameForRoot: string): { located: Array<{ name: string; directory: string }>; skipped: SkippedSkill[] } {
  const root = realOrSelf(rootDir);
  const candidates: Array<{ name: string; directory: string }> = [];
  const seen = new Set<string>();
  // THE INSTALLED FOLDER IS NAMED AFTER THE SKILL'S OWN VALIDATED `name:`
  // (2026-09-26, the e2e run): the folder took the repository's name (or the
  // checkout folder's), so a skill declared `greeter` in a repository
  // `greeter-skill` landed as `greeter-skill` and warned about its own name
  // on every load. A declared name that is not a skill name (upper-case, a
  // space) is not used; the folder name, or `nameForRoot` for a skill at the
  // repository root, stands in as before.
  const add = (directory: string, name?: string): void => {
    const real = realOrSelf(directory);
    if (real !== root && !real.startsWith(root + path.sep)) return;
    if (!skillFileIn(real)) return;
    const loaded = loadSkillDir(real);
    const label = ('problem' in loaded ? undefined : loaded.declaredName) ?? name ?? path.basename(real);
    if (seen.has(label)) return;
    seen.add(label);
    candidates.push({ name: label, directory: real });
  };

  if (subpath) {
    const base = realOrSelf(path.join(root, subpath));
    if (!base.startsWith(root + path.sep) || !isDir(base)) return { located: [], skipped: [] };
    if (skillFileIn(base)) add(base);
    else for (const directory of skillDirsBelow(base, SEARCH_DEPTH)) add(directory);
  } else {
    // A skill at the repository root: its validated `name:` (`add` prefers
    // it), else the repository's own name, never the checkout folder's.
    if (skillFileIn(root)) add(root, nameForRoot);
    for (const relative of SEARCH_ROOTS.slice(1)) {
      const directory = path.join(root, relative);
      if (isDir(directory)) for (const found of skillDirsBelow(directory, SEARCH_DEPTH)) add(found);
    }
    let entries: string[] = [];
    try {
      entries = fs.readdirSync(root).sort();
    } catch {
      entries = [];
    }
    for (const entry of entries) {
      if (entry.startsWith('.') && !SKIPPED_DIRS.has(entry) && isDir(path.join(root, entry, 'skills'))) {
        for (const found of skillDirsBelow(path.join(root, entry, 'skills'), SEARCH_DEPTH)) add(found);
      }
    }
    for (const relative of manifestPaths(root)) add(path.join(root, relative));
  }

  const skipped: SkippedSkill[] = [];
  const located: Array<{ name: string; directory: string }> = [];
  for (const candidate of candidates) {
    const loaded = loadSkillDir(candidate.directory);
    if ('problem' in loaded) skipped.push({ ...loaded, name: candidate.name });
    else located.push(candidate);
  }
  return { located, skipped };
}

// ---------------------------------------------------------------------------
// Files, flags, digests
// ---------------------------------------------------------------------------

export interface SkillFile {
  path: string;
  size: number;
  script: boolean;
  binary: boolean;
}

function isBinaryFile(full: string): boolean {
  try {
    const fd = fs.openSync(full, 'r');
    try {
      const head = Buffer.alloc(8192);
      const read = fs.readSync(fd, head, 0, 8192, 0);
      return head.subarray(0, read).includes(0);
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return false;
  }
}

/**
 * Every regular file under a skill folder (`.git` and the like left out),
 * sorted, with its size and flags; and the symbolic links found, which
 * refuse the install.
 */
export function listFiles(directory: string): { files: SkillFile[]; links: string[] } {
  const real = realOrSelf(directory);
  const files: SkillFile[] = [];
  const links: string[] = [];
  const walk = (dir: string): void => {
    let names: string[];
    try {
      names = fs.readdirSync(dir).sort();
    } catch {
      return;
    }
    for (const name of names) {
      const full = path.join(dir, name);
      const relative = path.relative(real, full).split(path.sep).join('/');
      if (isLink(full)) {
        links.push(relative);
        continue;
      }
      let stat: fs.Stats;
      try {
        stat = fs.statSync(full);
      } catch {
        continue;
      }
      if (stat.isDirectory()) {
        if (!SKIPPED_DIRS.has(name)) walk(full);
        continue;
      }
      if (!stat.isFile()) continue;
      const extension = path.extname(name).toLowerCase();
      const inScripts = relative.includes('/') && (SCRIPT_DIRS as readonly string[]).includes(relative.split('/')[0]);
      const executable = (stat.mode & 0o100) !== 0;
      const binary = (BINARY_EXTENSIONS as readonly string[]).includes(extension) || isBinaryFile(full);
      const script = !binary && (inScripts || (SCRIPT_EXTENSIONS as readonly string[]).includes(extension) || executable);
      files.push({ path: relative, size: stat.size, script, binary });
    }
  };
  walk(real);
  files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  return { files, links: links.sort() };
}

/**
 * `sha256:<hex>` over the sorted file list: each path, then the SHA-256 of
 * its content, one per line. The same bytes give the same digest in both
 * SDKs (fixture `install.lock.digest_vector`).
 */
export function treeDigest(directory: string, files?: readonly string[]): string {
  const real = realOrSelf(directory);
  const names = files ? [...files].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)) : listFiles(real).files.map((f) => f.path);
  const digest = createHash('sha256');
  for (const relative of names) {
    const content = createHash('sha256').update(fs.readFileSync(path.join(real, ...relative.split('/')))).digest('hex');
    digest.update(`${relative}\n${content}\n`);
  }
  return `sha256:${digest.digest('hex')}`;
}

export function humanSize(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

// ---------------------------------------------------------------------------
// The lock
// ---------------------------------------------------------------------------

export interface LockEntry {
  source: string;
  url: string | null;
  ref: string | null;
  subpath: string | null;
  commit: string | null;
  tree: string;
  files: string[];
  installed_at: string;
}

export interface Lock {
  version: number;
  skills: Record<string, LockEntry>;
}

export function lockPath(folder: string): string {
  return path.join(folder, LOCK_FILE);
}

export function readLock(folder: string): Lock {
  let data: unknown;
  try {
    data = JSON.parse(fs.readFileSync(lockPath(folder), 'utf8'));
  } catch {
    return { version: LOCK_VERSION, skills: {} };
  }
  if (!data || typeof data !== 'object' || !(data as Lock).skills || typeof (data as Lock).skills !== 'object') {
    return { version: LOCK_VERSION, skills: {} };
  }
  return { ...(data as Lock), version: LOCK_VERSION };
}

/** JSON with sorted keys and two-space indent, as Python's `json.dumps(sort_keys=True, indent=2)` writes it. */
function sortedJson(value: unknown, indent = ''): string {
  if (Array.isArray(value)) {
    if (!value.length) return '[]';
    const inner = indent + '  ';
    return `[\n${value.map((item) => inner + sortedJson(item, inner)).join(',\n')}\n${indent}]`;
  }
  if (value && typeof value === 'object') {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    if (!keys.length) return '{}';
    const inner = indent + '  ';
    return `{\n${keys.map((key) => `${inner}${JSON.stringify(key)}: ${sortedJson((value as Record<string, unknown>)[key], inner)}`).join(',\n')}\n${indent}}`;
  }
  return JSON.stringify(value);
}

export function writeLock(folder: string, data: Lock): void {
  const file = lockPath(folder);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, sortedJson(data) + '\n');
}

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

export interface InstallIO {
  out(line: string): void;
  err(line: string): void;
}

export interface InstallOptions {
  /** `--skill <name>`: the one skill to install from the source. */
  skill?: string;
  /** `-y`: install without asking. Required when there is no terminal. */
  yes?: boolean;
  /** Whether a person is at a terminal to be asked. */
  tty?: boolean;
  /** The question, answered true to proceed; asked only at a terminal without `--yes`. */
  confirm?: (question: string) => Promise<boolean>;
}

interface Candidate {
  name: string;
  directory: string;
  description: string;
  files: SkillFile[];
  links: string[];
}

function short(description: string, limit = 80): string {
  const text = description.split(/\s+/).join(' ').trim();
  return text.length <= limit ? text : `${text.slice(0, limit - 3).replace(/\s+$/, '')}...`;
}

function copyTree(from: string, to: string): void {
  fs.mkdirSync(to, { recursive: true });
  for (const name of fs.readdirSync(from)) {
    if (SKIPPED_DIRS.has(name)) continue;
    const source = path.join(from, name);
    const target = path.join(to, name);
    const stat = fs.lstatSync(source);
    if (stat.isSymbolicLink()) continue;
    if (stat.isDirectory()) copyTree(source, target);
    else if (stat.isFile()) fs.copyFileSync(source, target);
  }
}

/**
 * `skills add <source>`: fetch, locate, show, confirm, install, record.
 * Resolves to the exit code; every refusal leaves the folder untouched.
 */
export async function installFromSource(source: Source, folderPath: string, options: InstallOptions, io: InstallIO): Promise<number> {
  const folder = realOrSelf(folderPath);
  const workdir = fs.mkdtempSync(path.join(os.tmpdir(), 'webagents-skillmd-'));
  try {
    let root: string;
    let commit: string | null = null;
    try {
      if (source.kind === 'git') {
        const fetched = fetchGit(source, workdir);
        root = realOrSelf(fetched.checkout);
        commit = fetched.commit;
      } else {
        root = realOrSelf(expandHome(String(source.path)));
        if (!isDir(root)) {
          io.err(`${source.text} is not a folder.`);
          return 1;
        }
      }
    } catch (err) {
      if (err instanceof InstallError) {
        io.err(err.message);
        return 1;
      }
      throw err;
    }

    const { located, skipped } = locateSkills(root, source.subpath, repoName(source));
    for (const item of skipped) io.err(`Skipped ${item.name}: ${item.reason}`);
    if (!located.length) {
      io.err(`No SKILL.md skills found in ${source.text}.`);
      return 1;
    }
    let chosen = located;
    if (options.skill !== undefined) {
      chosen = located.filter((c) => c.name === options.skill);
      if (!chosen.length) {
        io.err(`No skill called "${options.skill}" in ${source.text}. Skills there: ${located.map((c) => c.name).join(', ')}.`);
        return 1;
      }
    }

    const candidates: Candidate[] = chosen.map((c) => {
      const loaded = loadSkillDir(c.directory);
      const { files, links } = listFiles(c.directory);
      return { name: c.name, directory: c.directory, description: 'problem' in loaded ? '' : loaded.description, files, links };
    });
    const names = candidates.map((c) => c.name).join(', ');

    // The limits, before anything is shown as installable.
    for (const candidate of candidates) {
      if (candidate.links.length) {
        io.err(`Refused: ${candidate.name} contains a symbolic link (${candidate.links[0]}), which could point outside the skill folder.`);
        return 1;
      }
    }
    const totalFiles = candidates.reduce((n, c) => n + c.files.length, 0);
    const totalSize = candidates.reduce((n, c) => n + c.files.reduce((m, f) => m + f.size, 0), 0);
    if (totalFiles > FILE_LIMIT) {
      io.err(`Refused: ${names} would install ${totalFiles} files, more than the limit of ${FILE_LIMIT}.`);
      return 1;
    }
    if (totalSize > EXTRACTED_LIMIT) {
      io.err(`Refused: ${names} would install ${humanSize(totalSize)}, more than the 25 MiB limit.`);
      return 1;
    }
    const lock = readLock(folder);
    for (const candidate of candidates) {
      const target = path.join(folder, '.agents', 'skills', candidate.name);
      if (exists(target) && !(candidate.name in lock.skills)) {
        io.err(`${candidate.name} already exists in .agents/skills and was not installed by webagents; remove it first.`);
        return 1;
      }
    }

    // The vetting: what would be installed, file by file.
    io.out(`Found ${candidates.length} in ${source.text}: ${names}`);
    for (const candidate of candidates) {
      io.out(`${candidate.name}: ${short(candidate.description)}`);
      for (const item of candidate.files) {
        const flag = item.binary ? ' [binary]' : item.script ? ' [script]' : '';
        io.out(`  ${item.path} (${item.size} B)${flag}`);
      }
    }
    if (!options.yes) {
      if (!options.tty) {
        io.err('Pass --yes to install without a prompt.');
        return 1;
      }
      const asked = options.confirm ?? askAtTerminal;
      if (!(await asked(`Install ${names} into .agents/skills? [y/N] `))) {
        io.out('Nothing installed.');
        return 0;
      }
    }

    // The install, then the record.
    for (const candidate of candidates) {
      const target = path.join(folder, '.agents', 'skills', candidate.name);
      if (exists(target)) fs.rmSync(target, { recursive: true, force: true });
      copyTree(candidate.directory, target);
      const paths = candidate.files.map((f) => f.path);
      lock.skills[candidate.name] = {
        source: source.text,
        url: source.url ?? null,
        ref: source.ref ?? null,
        subpath: source.subpath ?? path.relative(root, candidate.directory).split(path.sep).join('/'),
        commit,
        tree: treeDigest(target, paths),
        files: paths,
        installed_at: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
      };
      writeLock(folder, lock);
      if (commit) io.out(`Installed ${candidate.name} into .agents/skills/${candidate.name} (${paths.length} files) at ${commit.slice(0, 7)}.`);
      else io.out(`Installed ${candidate.name} into .agents/skills/${candidate.name} (${paths.length} files).`);
    }
    return 0;
  } finally {
    fs.rmSync(workdir, { recursive: true, force: true });
  }
}

/** The question at a terminal, answered with y or yes. */
async function askAtTerminal(question: string): Promise<boolean> {
  const { createInterface } = await import('node:readline');
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  try {
    const answer = await new Promise<string>((resolve) => rl.question(question, resolve));
    return ['y', 'yes'].includes(answer.trim().toLowerCase());
  } finally {
    rl.close();
  }
}

/** The skills the lock records, that are still there. */
export function installedSkillNames(folderPath: string): string[] {
  const folder = realOrSelf(folderPath);
  return Object.keys(readLock(folder).skills)
    .filter((name) => isDir(path.join(folder, '.agents', 'skills', name)))
    .sort();
}

/**
 * `skills remove <name>` for an installed SKILL.md skill: the folder and its
 * lock entry go. Null when `name` is not one the lock knows (the caller
 * tries the agent file's `skills:` list next); else the exit code.
 */
export function removeInstalled(name: string, folderPath: string, io: InstallIO): number | null {
  const folder = realOrSelf(folderPath);
  const lock = readLock(folder);
  if (!(name in lock.skills)) return null;
  const target = path.join(folder, '.agents', 'skills', name);
  if (exists(target)) fs.rmSync(target, { recursive: true, force: true });
  delete lock.skills[name];
  writeLock(folder, lock);
  io.out(`Removed ${name} from .agents/skills.`);
  return 0;
}
