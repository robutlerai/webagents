/**
 * Loading SKILL.md skills (the Agent Skills format, agentskills.io),
 * gap-closure plan item 1.4, 2026-09-26. The Python twin is
 * `python/webagents/agents/skills/local/skillmd/skillmd_loader.py`; both run
 * `python/tests/fixtures/skillmd/skillmd.json`.
 *
 * WHAT A SKILL.md SKILL IS. A folder holding `SKILL.md` (frontmatter with a
 * name and a description, then instructions) plus whatever files the
 * instructions refer to: `scripts/`, `references/`, `assets/`, or anything
 * else. It is the format Claude Code, Codex, OpenCode and Hermes all read, so
 * a skill written once runs under every one of them and under both of these
 * SDKs. It is what a person who does not program can write.
 *
 * WHERE THEY COME FROM (`discoverSkills`). Two places, the same in both SDKs:
 *
 *   1. `<agent folder>/.agents/skills/<name>/SKILL.md`, found on its own. This
 *      is the folder Codex, OpenCode and Hermes scan and the one `webagents
 *      skills add <source>` installs into.
 *   2. `agent_skills:` in the agent file: a list of paths, each a skill folder
 *      or a folder of skill folders, for skills kept somewhere else.
 *
 * `skills:` was NOT reused for this, because in an agent file it already
 * names CODED skills (`shell`, `openai`, `mcp`); a path in that list would be
 * refused as an unknown skill by every existing loader. A separate key keeps
 * both meanings unambiguous and lets `webagents skills list` show the two
 * kinds apart. An explicit entry wins over a discovered one of the same name.
 *
 * LENIENT LOAD, STRICT VALIDATION. The file is read as its author wrote it:
 * CRLF line ends, a byte-order mark, a closing `---` with no newline after
 * it, `skill.md` in lower case, a description with an unquoted colon (the
 * YAML is retried with the value quoted), `metadata` values written as
 * numbers (`version: 1.0` is the string "1.0" here, in both SDKs), Claude
 * Code's, Codex's and Hermes's own keys (kept in `extra`, never refused). A
 * skill is SKIPPED, and said so in `doctor` and in the loader's report, only
 * when the description is missing or the frontmatter cannot be parsed;
 * everything else is a warning. The name is the folder's name, as the
 * specification requires; a frontmatter `name` that differs is a warning,
 * not a rename.
 *
 * `allowed-tools` IS A HINT, NEVER A GRANT: it says which tools the skill's
 * author expected, and nothing here widens or narrows a tool's scope from
 * it. `!`cmd`` substitutions in a body are text and are never run.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { parse as parseYaml } from 'yaml';

/** The file names a skill folder is recognised by, in the order tried. */
export const SKILL_FILE_NAMES = ['SKILL.md', 'skill.md'] as const;

/** Where an agent folder's own SKILL.md skills live, relative to it. */
export const AUTO_SKILLS_DIR = path.join('.agents', 'skills');

/** The agent-file key naming other skill folders (a list of paths). */
export const EXPLICIT_KEY = 'agent_skills';

/** Folders never scanned for skills or listed as bundled files. */
export const SKIPPED_DIRS: ReadonlySet<string> = new Set(['.git', 'node_modules', '__pycache__']);

/** The frontmatter keys the specification defines; anything else is kept in `extra`. */
export const KNOWN_KEYS = ['name', 'description', 'license', 'compatibility', 'metadata', 'allowed-tools'] as const;

/** The specification's limits, applied as warnings. */
export const NAME_MAX = 64;
export const DESCRIPTION_MAX = 1024;
export const COMPATIBILITY_MAX = 500;
export const BODY_MAX_LINES = 500;

/** At most this many bundled files are listed on activation. */
export const LISTED_FILES_MAX = 200;

const NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** Whether `name` is a skill name: 1 to `NAME_MAX` lower-case letters, digits and single hyphens. */
export function isSkillName(name: string): boolean {
  return name.length >= 1 && name.length <= NAME_MAX && NAME_RE.test(name) && name === name.toLowerCase();
}
const KEY_LINE_RE = /^([ \t]*)([^\s#:][^:]*?):[ \t]+(.+?)[ \t]*$/;

/** A loaded SKILL.md skill. */
export interface SkillMd {
  /** The folder's name, which is the skill's name. */
  name: string;
  /**
   * The frontmatter's `name:`, when it is a valid skill name (1 to 64
   * lower-case letters, digits and single hyphens); absent otherwise. The
   * installer names the installed folder after it (2026-09-26): it named the
   * folder after the repository, so a skill declared `greeter` in a
   * repository `greeter-skill` landed as `greeter-skill` and warned about
   * its own name on every load.
   */
  declaredName?: string;
  description: string;
  /** Absolute, symlink-resolved path of the SKILL.md file: the catalog's `location`. */
  location: string;
  /** Absolute, symlink-resolved path of the skill folder. */
  directory: string;
  /** The instructions, frontmatter removed, surrounding blank lines trimmed. */
  body: string;
  allowedTools: string[];
  metadata: Record<string, string>;
  license?: string;
  compatibility?: string;
  /** Frontmatter keys the specification does not define (Claude Code's, Codex's, Hermes's), as written. */
  extra: Record<string, unknown>;
  warnings: string[];
  /** `auto` (found under .agents/skills) or `explicit` (named by `agent_skills:`). */
  source: 'auto' | 'explicit';
}

/** A skill folder that could not be loaded, and why. */
export interface SkippedSkill {
  location: string;
  name: string;
  /** `no_description`, `bad_yaml`, `not_found` or `unreadable`. */
  problem: 'no_description' | 'bad_yaml' | 'not_found' | 'unreadable';
  reason: string;
}

export interface Discovery {
  skills: SkillMd[];
  skipped: SkippedSkill[];
  warnings: string[];
}

/** Why a SKILL.md text must be skipped. */
export interface SkillProblem {
  problem: 'no_description' | 'bad_yaml';
  reason: string;
}

// ---------------------------------------------------------------------------
// The frontmatter
// ---------------------------------------------------------------------------

const isFence = (line: string): boolean => line.replace(/[ \t]+$/, '') === '---';

/**
 * `{ front, body }`, or `{ problem }`. The BOM is dropped and CRLF becomes LF
 * first, and a closing `---` on the last line with no newline after it is a
 * closing fence (the old regex required the newline).
 */
export function splitFrontmatter(text: string): { front: string; body: string } | { problem: 'no_frontmatter' | 'unclosed' } {
  if (text.startsWith('﻿')) text = text.slice(1);
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = text.split('\n');
  if (!lines.length || !isFence(lines[0])) return { problem: 'no_frontmatter' };
  const close = lines.findIndex((line, i) => i > 0 && isFence(line));
  if (close < 0) return { problem: 'unclosed' };
  return { front: lines.slice(1, close).join('\n'), body: lines.slice(close + 1).join('\n') };
}

const quote = (value: string): string => `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;

/**
 * A plain scalar this may wrap in double quotes without changing YAML's
 * reading of it: not already quoted, not a flow collection or a block scalar
 * indicator, not an anchor, alias or tag.
 */
const quotable = (value: string): boolean => !['"', "'", '[', '{', '|', '>', '&', '*', '!'].includes(value[0] ?? '');

/**
 * The values of a block-style `metadata:` mapping, quoted before parsing, so
 * `version: 1.0` reaches both SDKs as the string "1.0" rather than as a
 * number that Python prints "1.0" and JavaScript prints "1".
 */
export function quoteMetadataValues(front: string): string {
  const out: string[] = [];
  let inBlock = false;
  for (let line of front.split('\n')) {
    const stripped = line.replace(/^[ \t]+/, '');
    const indent = line.length - stripped.length;
    if (inBlock && stripped && indent === 0) inBlock = false;
    if (inBlock && stripped && !stripped.startsWith('#')) {
      const match = KEY_LINE_RE.exec(line);
      if (match && quotable(match[3])) line = `${match[1]}${match[2]}: ${quote(match[3])}`;
    } else if (indent === 0 && /^metadata[ \t]*:[ \t]*$/.test(stripped)) {
      inBlock = true;
    }
    out.push(line);
  }
  return out.join('\n');
}

/**
 * The retry after a parse error: every top-level `key: value` whose plain
 * value holds `: ` (a description such as "Use when: the file is a PDF") gets
 * its value quoted. Only the lines a colon breaks are touched.
 */
export function repairUnquotedColons(front: string): string {
  const out: string[] = [];
  for (let line of front.split('\n')) {
    const match = KEY_LINE_RE.exec(line);
    if (match && match[1] === '' && quotable(match[3]) && (match[3].includes(': ') || match[3].endsWith(':'))) {
      line = `${match[2]}: ${quote(match[3])}`;
    }
    out.push(line);
  }
  return out.join('\n');
}

/** `{ data }`, or `{ reason }` when the frontmatter is not a YAML mapping even after the colon repair. */
export function parseYamlFrontmatter(front: string): { data: Record<string, unknown> } | { reason: string } {
  const prepared = quoteMetadataValues(front);
  let data: unknown;
  try {
    data = parseYaml(prepared);
  } catch (first) {
    try {
      data = parseYaml(repairUnquotedColons(prepared));
    } catch {
      const detail = String((first as Error).message ?? first).trim().split('\n')[0];
      return { reason: `frontmatter is not valid YAML: ${detail}` };
    }
  }
  if (data === null || data === undefined) data = {};
  if (typeof data !== 'object' || Array.isArray(data)) return { reason: 'frontmatter is not a mapping of keys to values' };
  return { data: data as Record<string, unknown> };
}

/**
 * A scalar as the string both SDKs make of it; undefined for anything else.
 * An integral number prints without a `.0`, as Python prints it here.
 */
export function scalarText(value: unknown): string | undefined {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : undefined;
  if (typeof value === 'bigint') return value.toString();
  return undefined;
}

/** `allowed-tools`: space-separated by the specification, comma-separated or a YAML list as Claude Code also writes it. */
function toolList(value: unknown, warnings: string[]): string[] {
  if (value === undefined || value === null) return [];
  if (typeof value === 'string') return value.split(/[\s,]+/).filter(Boolean);
  if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    return (value as string[]).flatMap((item) => item.split(/[\s,]+/)).filter(Boolean);
  }
  warnings.push('allowed-tools is not a space-separated string; ignored');
  return [];
}

/**
 * The skill a SKILL.md's text describes, named for its folder, or the problem
 * that skips it: `no_description` or `bad_yaml`. `location` and `directory`
 * are left empty for the caller.
 */
export function parseSkillMd(text: string, dirName: string): SkillMd | SkillProblem {
  const split = splitFrontmatter(text);
  if ('problem' in split) {
    if (split.problem === 'no_frontmatter') {
      return { problem: 'no_description', reason: 'SKILL.md has no frontmatter (a --- block with name and description)' };
    }
    return { problem: 'bad_yaml', reason: 'SKILL.md opens its frontmatter with --- and never closes it' };
  }
  const parsed = parseYamlFrontmatter(split.front);
  if ('reason' in parsed) return { problem: 'bad_yaml', reason: parsed.reason };
  const data = parsed.data;

  const warnings: string[] = [];
  let description = scalarText(data.description);
  if (description === undefined || !description.trim()) {
    return { problem: 'no_description', reason: 'SKILL.md has no description in its frontmatter' };
  }
  description = description.trim();
  if (description.length > DESCRIPTION_MAX) warnings.push(`description is longer than ${DESCRIPTION_MAX} characters`);

  const declared = scalarText(data.name);
  if (declared !== undefined && declared.trim() && declared.trim() !== dirName) {
    warnings.push(`name "${declared.trim()}" does not match the folder name "${dirName}"; the folder name is used`);
  }
  if (!isSkillName(dirName)) {
    warnings.push(`name "${dirName}" is not a skill name (1 to ${NAME_MAX} lower-case letters, digits and single hyphens)`);
  }
  // The declared name the installer may use for the folder (`SkillMd.declaredName`).
  const declaredName = declared !== undefined && isSkillName(declared.trim()) ? declared.trim() : undefined;

  const metadata: Record<string, string> = {};
  const rawMetadata = data.metadata;
  if (rawMetadata && typeof rawMetadata === 'object' && !Array.isArray(rawMetadata)) {
    for (const [key, value] of Object.entries(rawMetadata as Record<string, unknown>)) {
      const textValue = scalarText(value);
      if (textValue === undefined) warnings.push(`metadata.${key} is not a string; ignored`);
      else metadata[key] = textValue;
    }
  } else if (rawMetadata !== undefined && rawMetadata !== null) {
    warnings.push('metadata is not a mapping of strings; ignored');
  }

  let license: string | undefined;
  if (data.license !== undefined && data.license !== null) {
    license = scalarText(data.license);
    if (license === undefined) warnings.push('license is not a string; ignored');
  }
  let compatibility: string | undefined;
  if (data.compatibility !== undefined && data.compatibility !== null) {
    compatibility = scalarText(data.compatibility);
    if (compatibility === undefined) warnings.push('compatibility is not a string; ignored');
    else if (compatibility.length > COMPATIBILITY_MAX) warnings.push(`compatibility is longer than ${COMPATIBILITY_MAX} characters`);
  }

  const allowedTools = toolList(data['allowed-tools'], warnings);
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) if (!(KNOWN_KEYS as readonly string[]).includes(key)) extra[key] = value;

  const body = split.body.trim();
  if (body.split('\n').length > BODY_MAX_LINES) warnings.push(`body is longer than ${BODY_MAX_LINES} lines`);

  return {
    name: dirName,
    ...(declaredName !== undefined ? { declaredName } : {}),
    description,
    location: '',
    directory: '',
    body,
    allowedTools,
    metadata,
    ...(license !== undefined ? { license } : {}),
    ...(compatibility !== undefined ? { compatibility } : {}),
    extra,
    warnings,
    source: 'auto',
  };
}

// ---------------------------------------------------------------------------
// Folders
// ---------------------------------------------------------------------------

function realOrSelf(p: string): string {
  try {
    return fs.realpathSync(p);
  } catch {
    return path.resolve(p);
  }
}

function isFile(p: string): boolean {
  try {
    return fs.statSync(p).isFile();
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

/** The SKILL.md (or skill.md) in `directory`, or undefined. */
export function skillFileIn(directory: string): string | undefined {
  for (const name of SKILL_FILE_NAMES) {
    const candidate = path.join(directory, name);
    if (isFile(candidate)) return candidate;
  }
  return undefined;
}

/** The skill in a folder, or why it was skipped. */
export function loadSkillDir(directory: string, source: 'auto' | 'explicit' = 'auto'): SkillMd | SkippedSkill {
  const real = realOrSelf(directory);
  const name = path.basename(real) || real;
  const file = skillFileIn(real);
  if (!file) return { location: real, name, problem: 'not_found', reason: 'no SKILL.md in the folder' };
  const location = realOrSelf(file);
  let text: string;
  try {
    text = fs.readFileSync(location, 'utf8');
  } catch (err) {
    return { location, name, problem: 'unreadable', reason: `SKILL.md cannot be read: ${(err as Error).message}` };
  }
  const parsed = parseSkillMd(text, name);
  if ('problem' in parsed) return { location, name, problem: parsed.problem, reason: parsed.reason };
  return { ...parsed, location, directory: real, source };
}

/** The immediate sub-folders of `directory` that hold a SKILL.md, sorted. */
function skillDirsUnder(directory: string): string[] {
  let names: string[];
  try {
    names = fs.readdirSync(directory).sort();
  } catch {
    return [];
  }
  const found: string[] = [];
  for (const name of names) {
    if (SKIPPED_DIRS.has(name) || name.startsWith('.')) continue;
    const child = path.join(directory, name);
    if (isDir(child) && skillFileIn(child)) found.push(child);
  }
  return found;
}

function expandHome(p: string): string {
  return p.startsWith('~') ? path.join(process.env.HOME ?? '', p.slice(1)) : p;
}

/**
 * Every SKILL.md skill an agent in `agentDir` has: the `agent_skills:`
 * entries (each a skill folder, or a folder of skill folders, relative to the
 * agent folder), then `.agents/skills/*`. The first of a name wins and the
 * rest are warned about; a folder that is missing or holds no SKILL.md is
 * reported as skipped, never thrown.
 */
export function discoverSkills(agentDir: string, explicit: readonly string[] = []): Discovery {
  const base = realOrSelf(agentDir);
  const result: Discovery = { skills: [], skipped: [], warnings: [] };
  const seen = new Map<string, SkillMd>();

  const take = (directory: string, source: 'auto' | 'explicit'): void => {
    const loaded = loadSkillDir(directory, source);
    if ('problem' in loaded) {
      result.skipped.push(loaded);
      return;
    }
    const earlier = seen.get(loaded.name);
    if (earlier) {
      result.warnings.push(`skill "${loaded.name}" at ${loaded.location} is shadowed by ${earlier.location}`);
      return;
    }
    seen.set(loaded.name, loaded);
    result.skills.push(loaded);
  };

  for (const entry of explicit) {
    const text = String(entry);
    const expanded = expandHome(text);
    const target = realOrSelf(path.isAbsolute(expanded) ? expanded : path.join(base, expanded));
    if (!isDir(target)) {
      result.skipped.push({ location: target, name: path.basename(target), problem: 'not_found', reason: `${EXPLICIT_KEY}: ${text} is not a folder` });
      continue;
    }
    if (skillFileIn(target)) {
      take(target, 'explicit');
      continue;
    }
    const children = skillDirsUnder(target);
    if (!children.length) {
      result.skipped.push({
        location: target,
        name: path.basename(target),
        problem: 'not_found',
        reason: `${EXPLICIT_KEY}: ${text} holds no SKILL.md and no folder with one`,
      });
      continue;
    }
    for (const child of children) take(child, 'explicit');
  }

  for (const child of skillDirsUnder(path.join(base, AUTO_SKILLS_DIR))) take(child, 'auto');

  result.skills.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return result;
}

/**
 * The files under a skill folder other than SKILL.md, as relative paths with
 * `/`, sorted; symbolic links and the skipped folders are left out. Listed,
 * never read, on activation.
 */
export function bundledFiles(directory: string): string[] {
  const real = realOrSelf(directory);
  const found: string[] = [];
  const walk = (dir: string): void => {
    let names: string[];
    try {
      names = fs.readdirSync(dir).sort();
    } catch {
      return;
    }
    for (const name of names) {
      const full = path.join(dir, name);
      if (isLink(full)) continue;
      if (isDir(full)) {
        if (!SKIPPED_DIRS.has(name)) walk(full);
        continue;
      }
      if (!isFile(full)) continue;
      if (dir === real && (SKILL_FILE_NAMES as readonly string[]).includes(name)) continue;
      found.push(path.relative(real, full).split(path.sep).join('/'));
    }
  };
  walk(real);
  return found.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

// ---------------------------------------------------------------------------
// The words the model sees (pinned by the fixture)
// ---------------------------------------------------------------------------

export const CATALOG_PREAMBLE = 'Skills available to you. Activate one with activate_skill before doing what it covers.';

export function escapeXml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** The tier-1 catalog for the system prompt, in the skills-ref `to_prompt` format; empty when there are no skills. */
export function catalogText(skills: readonly SkillMd[]): string {
  if (!skills.length) return '';
  const lines = [CATALOG_PREAMBLE, '', '<available_skills>'];
  for (const skill of [...skills].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))) {
    lines.push(
      '<skill>',
      `<name>${escapeXml(skill.name)}</name>`,
      `<description>${escapeXml(skill.description)}</description>`,
      `<location>${escapeXml(skill.location)}</location>`,
      '</skill>',
    );
  }
  lines.push('</available_skills>');
  return lines.join('\n');
}

/** The `skills` line of `webagents doctor`: status, detail and fix, the same words in both CLIs (fixture `doctor`). */
export function doctorReport(found: Discovery): { status: 'ok' | 'warn'; detail: string; fix?: string } {
  const names = found.skills.map((s) => s.name).join(', ');
  const count = found.skills.length;
  if (!found.skills.length && !found.skipped.length && !found.warnings.length) {
    return { status: 'ok', detail: 'no SKILL.md skills in this folder' };
  }
  let detail = `${count} SKILL.md skill${count === 1 ? '' : 's'}: ${names || 'none'}`;
  for (const skipped of found.skipped) detail += `; skipped: ${skipped.name} (${skipped.reason})`;
  const warnings = [...found.warnings, ...found.skills.flatMap((skill) => skill.warnings.map((w) => `${skill.name}: ${w}`))];
  for (const warning of warnings) detail += `; warning: ${warning}`;
  const problems = found.skipped.length > 0 || warnings.length > 0;
  // The fix names the folders (2026-09-26): the sentence used to stop at
  // "named". Skipped folders first, then the skills that warned; a folder
  // warning with no skill behind it is said by the detail alone.
  const named = [...new Set([...found.skipped.map((s) => s.name), ...found.skills.filter((s) => s.warnings.length).map((s) => s.name)])];
  const fix = named.length ? `Fix or remove the SKILL.md folders named: ${named.join(', ')}` : 'Fix or remove the SKILL.md folders the skills line names';
  return { status: problems ? 'warn' : 'ok', detail, ...(problems ? { fix } : {}) };
}

/** The SKILL.md part of `webagents skills list`, the same lines in both CLIs (fixture `install.messages.list_*`). */
export function listLines(found: Discovery, hintCommand: string): string[] {
  const lines = ['SKILL.md skills in this folder:', ''];
  for (const skill of found.skills) lines.push(`  ${skill.name}   ${skill.location}`);
  for (const skipped of found.skipped) lines.push(`  ${skipped.name}   skipped: ${skipped.reason}`);
  if (!found.skills.length && !found.skipped.length) lines.push('  (none)');
  lines.push('', `Add one with \`${hintCommand}\`.`);
  return lines;
}

/** The opening tag an activation puts in the conversation; its presence in an earlier tool result is what "already active" means. */
export function activationMarker(name: string): string {
  return `<skill_content name="${escapeXml(name)}">`;
}

/** The tier-2 text `activate_skill` returns: the body, then where the skill lives and what it bundles (listed, not read). */
export function activationText(skill: SkillMd): string {
  const files = bundledFiles(skill.directory);
  let listing: string;
  if (!files.length) {
    listing = 'no other files';
  } else {
    listing = files.slice(0, LISTED_FILES_MAX).join(', ');
    if (files.length > LISTED_FILES_MAX) listing += `, and ${files.length - LISTED_FILES_MAX} more`;
  }
  return [
    activationMarker(skill.name),
    skill.body,
    '',
    `Skill directory: ${skill.directory}`,
    `Files in the skill directory (read one with read_skill_file, run a script with run_skill_script): ${listing}`,
    '</skill_content>',
  ].join('\n');
}
