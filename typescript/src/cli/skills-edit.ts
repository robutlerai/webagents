/**
 * `webagents skills add` and `webagents skills remove` (2026-09-25).
 *
 * THE FILE IS THE PERSON'S. These change the `skills:` list of an agent file
 * and nothing else: every other byte (comments, key order, the body, the
 * config of a `- rest: {...}` entry, CRLF line ends) comes back as it was
 * written. So the list is edited line by line and never re-serialised: the
 * Python `AgentFile._save` learned on 2026-09-23 what a `yaml.dump` rewrite
 * costs (every comment in the front matter). A layout this cannot edit
 * safely, such as a comment at column zero inside the list or a flow list
 * over several lines, is refused with the file untouched, and every edit is
 * read back as YAML and compared with what was meant before it is written.
 *
 * ONE EDITOR IN BOTH CLIS. The Python CLI's `skills_edit.py` is this, rule for
 * rule, and `python/tests/fixtures/cli/skills_edit.json` holds the cases and
 * the words both suites run.
 *
 * WHAT A NAME IS. One `skills list` shows, or a model provider's other name
 * (`claude`, `gemini`, `grok`). `claude` and `anthropic` are one skill to the
 * loaders, so adding one where the other is listed changes nothing. A name
 * this SDK cannot load is refused with a "did you mean"; removing takes any
 * name the file lists, known here or not, since the other SDK may load it.
 *
 * WHAT A SKILL STILL NEEDS is said after an add, and only when it is missing:
 * a model provider's key, the sign-in for Robutler's models (`proxy`) or for
 * searching the platform from the chat (`discovery`).
 *
 * SKILL.md SKILLS FROM OUTSIDE (plan item 1.4, 2026-09-26). A name that is a
 * SOURCE (`owner/repo`, a git URL, a `.../tree/<ref>/<path>` page, a folder:
 * `skillmd-install.ts` `parseSource`) is not looked for in this SDK's skill
 * table: it is fetched at one commit, its skills shown file by file, and,
 * once the person confirms (`--yes` without a terminal), installed into
 * `.agents/skills/<name>` and recorded in `.webagents/skills.lock`. The agent
 * file is not edited for those: every agent in the folder finds
 * `.agents/skills` on its own. `remove <name>` takes an installed skill out
 * the same way when the name is not a coded skill's. `--skill <name>` picks
 * one skill from a source. The Python CLI does the same, and `skills list`
 * in both shows the coded names and the folder's SKILL.md skills apart.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { parse as parseYaml } from 'yaml';

import { findAgentFile, parseAgentMarkdown } from '../agents/index';
import { findProvider } from '../skills/llm/providers';
import { resolvableSkillNames } from '../skills/resolve';
import { installedSkillNames, parseSource, removeInstalled } from '../skills/skillmd/skillmd-install';
import { discoverSkills, listLines } from '../skills/skillmd/skillmd-loader';
import { cliCommand as cliCommandNow } from './config-store';
import { suggestSimilar } from './suggest';

/** One skill, however a file names it: a provider's other names are the provider (`claude` is `anthropic`). */
export function canonicalSkillName(name: string): string {
  const lower = name.trim().toLowerCase();
  return findProvider(lower)?.id ?? lower;
}

/**
 * Why an agent file may not be changed in place (2026-09-26, S-290, spec W3):
 * it is a symbolic link, lies outside its folder, or (POSIX) belongs to
 * another user; null when it is safe to write. A cloned repository can carry
 * `AGENT-x.md -> ../outside/rc`, and the editor wrote through it, prepending a
 * `name:` taken from the link's name to whatever the link pointed at. The
 * guard sits here, so `webagents skills add|remove` gets it, and the chat's
 * `/skills`, `/agent edit` and `/agent new` reuse it. `who` names the actor
 * ("this command", or "the chat"). The Python editor guards the same way
 * (`skills_edit.py`, `unsafe_target`).
 */
export function unsafeTargetReason(file: string, folder: string, who = 'this command'): string | null {
  const shown = path.basename(file);
  const refusal = `${shown} is a link or lies outside this folder, so ${who} will not change it.`;
  // A direct child of the folder, by name: the chat and the CLI only ever
  // change a file sitting in the folder they run in.
  if (path.resolve(folder, shown) !== path.resolve(file)) return refusal;
  let stat: fs.Stats | undefined;
  try {
    stat = fs.lstatSync(file);
  } catch {
    // Not there yet (a file `/agent new` is about to create): nothing to link through.
    return null;
  }
  if (stat.isSymbolicLink()) return refusal;
  // The folder itself may be reached through a link (a symlinked project
  // checkout); what must not differ is where the file's REAL path leaves the
  // real folder.
  try {
    const realFolder = fs.realpathSync(folder);
    const realFile = fs.realpathSync(file);
    if (path.dirname(realFile) !== realFolder) return refusal;
  } catch {
    return refusal;
  }
  // POSIX: another user's file is not this owner's to change.
  const getuid = (process as { getuid?: () => number }).getuid;
  if (getuid && stat.uid !== getuid()) return refusal;
  return null;
}

export type SkillsAction = 'add' | 'remove';

/** Why a list could not be edited; each has one sentence (`skillListProblemMessage`). */
export type SkillListProblem = 'unclosed' | 'not_yaml' | 'not_list' | 'layout' | 'scalar_layout';

export function skillListProblemMessage(problem: SkillListProblem, file: string, key = 'model'): string {
  switch (problem) {
    case 'unclosed':
      return `${file} opens its front matter with --- and never closes it.`;
    case 'not_yaml':
      return `The front matter of ${file} is not valid YAML. Fix it, then try again.`;
    case 'not_list':
      return `skills: in ${file} is not a list. Fix it, then try again.`;
    case 'layout':
      return `The skills: list in ${file} is laid out in a way this command cannot change safely. Edit it by hand.`;
    case 'scalar_layout':
      // The chat's `/model --save` (spec 3.5): the same sentence shape as
      // the list's, naming the key (`chat-words.ts` `modelLayout`).
      return `The ${key}: line in ${file} is laid out in a way this command cannot change safely. Edit it by hand.`;
  }
}

export class SkillListError extends Error {
  constructor(
    readonly problem: SkillListProblem,
    file: string,
    key?: string,
  ) {
    super(skillListProblemMessage(problem, file, key));
    this.name = 'SkillListError';
  }
}

/** What an edit did. `text` is the file as it is to be written; unchanged when `changed` is false. */
export interface SkillListEdit {
  text: string;
  changed: boolean;
  /** Names written into the file, as they were asked for. */
  added: string[];
  /** Asked-for names the file already lists, as the file writes them. */
  already: string[];
  /** Names taken out, as the file wrote them. */
  removed: string[];
  /** Asked-for names the file does not list, as they were asked for. */
  absent: string[];
  /** The file's skill names after the edit. */
  skills: string[];
}

/** The name a `skills:` entry uses: `shell`, or the key of `{shell: {...}}`; null for anything else. */
function entryName(entry: unknown): string | null {
  if (typeof entry === 'string') return entry;
  if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
    const keys = Object.keys(entry);
    if (keys.length === 1) return keys[0];
  }
  return null;
}

/**
 * The name an agent file with no front matter goes by in both loaders:
 * AGENT.md is `default`, AGENT-<name>.md is `<name>`. Written into the front
 * matter an add creates, because a front matter without `name:` is named
 * `assistant` by the Python loader.
 */
function nameForFile(file: string): string {
  const base = path.basename(file).replace(/\.md$/i, '');
  if (base === 'AGENT') return 'default';
  return base.startsWith('AGENT-') ? base.slice('AGENT-'.length) : base;
}

const isFence = (line: string): boolean => line.replace(/[ \t]+$/, '') === '---';
const isTrivia = (line: string): boolean => line.trim() === '' || line.trim().startsWith('#');

interface Parsed {
  bom: string;
  nl: string;
  lines: string[];
  /** Index of the closing `---`, or -1 when the file has no front matter. */
  close: number;
  data: Record<string, unknown>;
}

function parse(text: string, file: string): Parsed {
  const bom = text.startsWith('﻿') ? '﻿' : '';
  const body = bom ? text.slice(1) : text;
  const nl = body.includes('\r\n') ? '\r\n' : '\n';
  const lines = body.replace(/\r\n/g, '\n').split('\n');
  if (!isFence(lines[0])) return { bom, nl, lines, close: -1, data: {} };
  const close = lines.findIndex((line, i) => i > 0 && isFence(line));
  if (close < 0) throw new SkillListError('unclosed', file);
  let data: unknown;
  try {
    data = parseYaml(lines.slice(1, close).join('\n'));
  } catch {
    throw new SkillListError('not_yaml', file);
  }
  if (data === null || data === undefined) data = {};
  if (typeof data !== 'object' || Array.isArray(data)) throw new SkillListError('not_yaml', file);
  return { bom, nl, lines, close, data: data as Record<string, unknown> };
}

/** The `skills:` entries of parsed front matter; throws when it is not a list. */
function entriesOf(data: Record<string, unknown>, file: string): unknown[] {
  const raw = data.skills;
  if (raw === undefined || raw === null) return [];
  if (!Array.isArray(raw)) throw new SkillListError('not_list', file);
  return raw;
}

/**
 * The positions of a one-line flow list's `[` and `]` and of its top-level
 * commas, or null when the `]` is not on this line.
 */
function flowList(line: string, open: number): { close: number; commas: number[] } | null {
  let depth = 0;
  let quote: string | null = null;
  const commas: number[] = [];
  for (let i = open; i < line.length; i++) {
    const c = line[i];
    if (quote) {
      if (quote === '"' && c === '\\') i++;
      else if (c === quote) {
        if (quote === "'" && line[i + 1] === "'") i++;
        else quote = null;
      }
      continue;
    }
    if (c === '"' || c === "'") quote = c;
    else if (c === '[' || c === '{') depth++;
    else if (c === ']' || c === '}') {
      depth--;
      if (depth === 0) return { close: i, commas };
    } else if (c === ',' && depth === 1) commas.push(i);
  }
  return null;
}

/** `skills: []`, keeping a comment the key line carried. */
function emptyListLine(line: string, colon: number): string {
  const rest = line.slice(colon + 1).trim();
  return `${line.slice(0, colon + 1)} []${rest.startsWith('#') ? ` ${rest}` : ''}`;
}

/**
 * Add or remove skills in the text of the agent file `file` (file comment).
 * Throws `SkillListError` rather than write anything it cannot stand behind.
 */
export function editSkillList(
  text: string,
  action: SkillsAction,
  requested: readonly string[],
  agentFile: string,
): SkillListEdit {
  const file = path.basename(agentFile);
  const same = (a: string, b: string) => canonicalSkillName(a) === canonicalSkillName(b);
  const parsed = parse(text, file);
  const entries = entriesOf(parsed.data, file);
  const names = entries.map(entryName);

  const added: string[] = [];
  const already: string[] = [];
  const removed: string[] = [];
  const absent: string[] = [];
  const dropped = new Set<number>();

  for (const wanted of requested) {
    const hits = names.flatMap((name, i) => (name !== null && same(name, wanted) ? [i] : []));
    if (action === 'add') {
      if (hits.length) {
        const written = names[hits[0]] as string;
        if (!already.includes(written)) already.push(written);
      } else if (!added.some((name) => same(name, wanted))) {
        added.push(wanted);
      }
    } else if (!hits.length) {
      if (!absent.some((name) => same(name, wanted))) absent.push(wanted);
    } else {
      for (const i of hits) {
        if (dropped.has(i)) continue;
        dropped.add(i);
        const written = names[i] as string;
        if (!removed.includes(written)) removed.push(written);
      }
    }
  }

  const after = [...names.filter((_, i) => !dropped.has(i)), ...added];
  const result = {
    added,
    already,
    removed,
    absent,
    skills: after.filter((name): name is string => name !== null),
  };
  if (!added.length && !dropped.size) return { text, changed: false, ...result };

  const lines = [...parsed.lines];
  if (parsed.close < 0) {
    // No front matter: one is written, naming the agent as the loaders did.
    lines.unshift('---', `name: ${nameForFile(agentFile)}`, 'skills:', ...added.map((name) => `  - ${name}`), '---', '');
  } else {
    editFrontMatter(lines, parsed.close, parsed.data, entries.length, dropped, added, file);
  }
  const edited = parsed.bom + lines.join(parsed.nl);

  // Read back as YAML: the list must be exactly what was meant, or nothing is written.
  const check = parse(edited, file);
  const got = entriesOf(check.data, file).map(entryName);
  if (got.length !== after.length || got.some((name, i) => name !== after[i])) {
    throw new SkillListError('layout', file);
  }
  return { text: edited, changed: true, ...result };
}

/** What a scalar edit did. `text` is the file as it is to be written; unchanged when `changed` is false. */
export interface ScalarEdit {
  text: string;
  changed: boolean;
  /** The value the file had, as YAML read it; undefined when the key was absent. */
  previous: string | undefined;
}

/** A value written as a plain scalar when YAML reads it back as itself, else double-quoted. */
function yamlScalar(value: string): string {
  if (/^[A-Za-z0-9][A-Za-z0-9._\/:+-]*$/.test(value) && !/^(true|false|null|yes|no|on|off)$/i.test(value) && !/^[0-9.]+$/.test(value)) {
    return value;
  }
  return JSON.stringify(value);
}

/**
 * Set one top-level front-matter scalar (`model: openai/gpt-4o`) in the text
 * of the agent file `agentFile`, the way `editSkillList` edits the list
 * (2026-09-26, interactive-mode spec 3.5, `/model --save`): the same parse,
 * one line changed or added, then read back as YAML. A key line written some
 * other way (quoted, a block scalar, an anchor) is refused with the
 * `scalar_layout` sentence rather than guessed at. A file with no front
 * matter gets one, naming the agent as the loaders name it.
 */
export function editFrontMatterScalar(text: string, key: string, value: string, agentFile: string): ScalarEdit {
  const file = path.basename(agentFile);
  const parsed = parse(text, file);
  const had = parsed.data[key];
  const previous = had === undefined || had === null ? undefined : String(had);
  if (previous === value) return { text, changed: false, previous };

  const lines = [...parsed.lines];
  const written = `${key}: ${yamlScalar(value)}`;
  if (parsed.close < 0) {
    lines.unshift('---', `name: ${nameForFile(agentFile)}`, written, '---', '');
  } else {
    const keyRe = new RegExp(`^${key.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')}[ \\t]*:`);
    let at = -1;
    for (let i = 1; i < parsed.close; i++) {
      if (keyRe.test(lines[i])) {
        at = i;
        break;
      }
    }
    if (at < 0) {
      // Present under another spelling (quoted, or a complex key): not this command's to touch.
      if (key in parsed.data) throw new SkillListError('scalar_layout', file, key);
      // After `name:` when there is one, else at the end of the front matter.
      let nameAt = -1;
      for (let i = 1; i < parsed.close; i++) {
        if (/^name[ \t]*:/.test(lines[i])) {
          nameAt = i;
          break;
        }
      }
      let insertAt = parsed.close;
      if (nameAt >= 0) insertAt = nameAt + 1;
      else while (insertAt > 1 && lines[insertAt - 1].trim() === '') insertAt--;
      lines.splice(insertAt, 0, written);
    } else {
      const line = lines[at];
      const colon = line.indexOf(':');
      const rest = line.slice(colon + 1);
      // The value up to a trailing comment, which is kept. A value that
      // continues on the next line (a block scalar, a mapping) is not one line.
      const comment = /(\s+#.*)$/.exec(rest);
      const body = comment ? rest.slice(0, rest.length - comment[1].length) : rest;
      if (/^\s*[|>&*]/.test(body) || /^\s*$/.test(body)) throw new SkillListError('scalar_layout', file, key);
      const next = lines[at + 1];
      if (at + 1 < parsed.close && next !== undefined && /^\s+\S/.test(next) && !/^\s*#/.test(next)) {
        throw new SkillListError('scalar_layout', file, key);
      }
      lines[at] = `${line.slice(0, colon + 1)} ${yamlScalar(value)}${comment ? comment[1] : ''}`;
    }
  }
  const edited = parsed.bom + lines.join(parsed.nl);
  // Read back as YAML: the value must be exactly what was meant, or nothing is written.
  const check = parse(edited, file);
  if (String(check.data[key]) !== value) throw new SkillListError('scalar_layout', file, key);
  return { text: edited, changed: true, previous };
}

/** The line edit itself, in place on `lines` (front matter is lines 1 to `close - 1`). */
function editFrontMatter(
  lines: string[],
  close: number,
  data: Record<string, unknown>,
  count: number,
  dropped: Set<number>,
  added: string[],
  file: string,
): void {
  let key = -1;
  for (let i = 1; i < close; i++) {
    if (/^skills[ \t]*:/.test(lines[i])) {
      key = i;
      break;
    }
  }
  if (key < 0) {
    // Written some other way (quoted, or a complex key): not this command's to touch.
    if ('skills' in data) throw new SkillListError('layout', file);
    let at = close;
    while (at > 1 && lines[at - 1].trim() === '') at--;
    lines.splice(at, 0, 'skills:', ...added.map((name) => `  - ${name}`));
    return;
  }

  const line = lines[key];
  const colon = line.indexOf(':');
  const value = line.slice(colon + 1).trim();

  if (value.startsWith('[')) {
    const open = line.indexOf('[', colon);
    const list = flowList(line, open);
    if (!list) throw new SkillListError('layout', file);
    const tail = line.slice(list.close + 1).trim();
    if (tail && !tail.startsWith('#')) throw new SkillListError('layout', file);
    const bounds = [open, ...list.commas, list.close];
    let items = bounds.slice(1).map((end, i) => line.slice(bounds[i] + 1, end).trim());
    if (items.length && items[items.length - 1] === '') items.pop();
    if (items.length !== count || items.some((item) => item === '')) throw new SkillListError('layout', file);
    items = [...items.filter((_, i) => !dropped.has(i)), ...added];
    lines[key] = `${line.slice(0, open)}[${items.join(', ')}]${line.slice(list.close + 1)}`;
    return;
  }
  if (value !== '' && !value.startsWith('#')) throw new SkillListError('layout', file);

  // A block list: the lines after the key that are blank, indented, or items
  // at column zero, less the blanks and comments that lead into what follows.
  let end = key + 1;
  while (end < close && (lines[end].trim() === '' || /^[ \t-]/.test(lines[end]))) end++;
  while (end > key + 1 && isTrivia(lines[end - 1])) end--;
  let indent: string | null = null;
  const starts: number[] = [];
  for (let i = key + 1; i < end; i++) {
    const match = /^( *)-(?:[ \t]|$)/.exec(lines[i]);
    if (!match) continue;
    if (indent === null) indent = match[1];
    if (match[1] === indent) starts.push(i);
  }
  if (starts.length !== count) throw new SkillListError('layout', file);
  // An entry runs to the next one, less the blanks and comments before it.
  const entryEnd = (j: number): number => {
    let stop = j + 1 < starts.length ? starts[j + 1] : end;
    while (stop > starts[j] + 1 && isTrivia(lines[stop - 1])) stop--;
    return stop;
  };

  if (added.length) {
    const at = starts.length ? entryEnd(starts.length - 1) : key + 1;
    lines.splice(at, 0, ...added.map((name) => `${indent ?? '  '}- ${name}`));
  }
  for (const j of [...dropped].sort((a, b) => b - a)) {
    lines.splice(starts[j], entryEnd(j) - starts[j]);
  }
  if (dropped.size === count && !added.length) lines[key] = emptyListLine(line, colon);
}

// ============================================================================
// The command
// ============================================================================

/** What this machine has, for saying what an added skill still needs. */
export interface SkillsFacts {
  /** Whether a key variable has a value here: set in this shell, or stored with `secrets set`. */
  hasKey(envVar: string): boolean;
  /** Whether `webagents login` has signed this profile in. */
  signedIn: boolean;
}

export interface SkillsCommandOptions {
  /** `-a`: the agent in the folder to change. */
  agent?: string;
  /**
   * The agent file itself, when the caller already runs one (the chat's
   * `/skills`): `chooseAgentFile` is skipped, and the file still passes the
   * guard below (W3) before it is read.
   */
  file?: string;
  /** Who is refusing, for the guard's sentence: `this command` (the CLI) or `the chat`. */
  who?: string;
  /** Defaults to the working directory. */
  folder?: string;
  /** Defaults to this machine (`machineFacts`); asked only when an added skill has a need. */
  facts?: () => Promise<SkillsFacts>;
  /** `--skill <name>`: the one skill to install from a source. */
  skill?: string;
  /** `-y`: install a source without asking; required when there is no terminal. */
  yes?: boolean;
  /** Whether a person is at a terminal; defaults to `process.stdin.isTTY`. */
  tty?: boolean;
  /** The install question, answered true to proceed; asked only at a terminal without `--yes`. */
  confirm?: (question: string) => Promise<boolean>;
}

export interface SkillsCommandIO {
  out(line: string): void;
  err(line: string): void;
  /** What the editor changed, once, after a successful edit: the `--json` document's facts (2026-09-27). */
  edited?(facts: SkillsEdited): void;
}

/** The editor's outcome, as `--json skills add` reports it (fixture `cli/json_documents.json`, `skills_add`). */
export interface SkillsEdited {
  /** The agent file's name, as the lines show it. */
  file: string;
  added: string[];
  already: string[];
  removed: string[];
  absent: string[];
  /** The `skills:` list after the edit. */
  skills: string[];
}

/** `a`, `a and b`, `a, b and c`. */
export function spokenList(names: readonly string[]): string {
  if (names.length <= 1) return names.join('');
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/**
 * Run `skills add` or `skills remove`; resolves to the exit code. Refusals
 * (an unknown name, no agent file, a list it cannot edit) change nothing.
 * Sources among `requested` install SKILL.md skills (file comment);
 * `skill`, `yes`, `tty` and `confirm` belong to that path.
 */
export async function skillsCommand(
  action: SkillsAction,
  requested: readonly string[],
  options: SkillsCommandOptions = {},
  io: SkillsCommandIO = { out: (line) => console.log(line), err: (line) => console.error(line) },
): Promise<number> {
  const { installFromSource, parseSource, readLock, removeInstalled } = await import('../skills/skillmd/skillmd-install.js');
  const { resolvableSkillNames } = await import('../skills/resolve.js');

  const folder = options.folder ?? process.cwd();
  const known = resolvableSkillNames();
  const canonical = canonicalSkillName;

  // Sources apart from names: a source never reaches the editor, and a name
  // never reaches the installer. An installed SKILL.md skill's name, when it
  // is not a coded skill's, is removed from `.agents/skills` rather than
  // from the file.
  let names: string[] = [];
  const sources: ReturnType<typeof parseSource>[] = [];
  let exitCode = 0;
  for (const entry of requested) {
    const source = parseSource(entry);
    if (source.kind === 'name') names.push(entry);
    else if (action === 'add') sources.push(source);
    else {
      io.err(`${entry} is not a skill name; remove takes the names \`skills list\` shows.`);
      exitCode = 1;
    }
  }
  if (action === 'remove') {
    const remaining: string[] = [];
    const installed = readLock(folder).skills;
    for (const entry of names) {
      const lower = entry.trim().toLowerCase();
      if (lower && !known.includes(canonical(lower)) && lower in installed) {
        const code = removeInstalled(lower, folder, io);
        exitCode = exitCode || (code ?? 0);
      } else {
        remaining.push(entry);
      }
    }
    names = remaining;
  }

  if (names.length || options.agent) {
    const code = names.length || action === 'add' ? await editAgentFile(action, names, options, io) : 0;
    if (names.length) exitCode = exitCode || code;
    else if (code) exitCode = code;
  }
  for (const source of sources) {
    const code = await installFromSource(
      source,
      folder,
      {
        skill: options.skill,
        yes: options.yes,
        tty: options.tty ?? Boolean(process.stdin.isTTY),
        confirm: options.confirm,
      },
      io,
    );
    exitCode = exitCode || code;
  }
  return exitCode;
}

/** The editor's part of `skills add|remove`: the coded names, in the agent file. */
async function editAgentFile(
  action: SkillsAction,
  requested: readonly string[],
  options: SkillsCommandOptions,
  io: SkillsCommandIO,
): Promise<number> {
  const { cliCommand } = await import('./config-store.js');
  const { providerEnvVars } = await import('../skills/llm/providers.js');
  const { resolvableSkillNames } = await import('../skills/resolve.js');
  const { suggestSimilar } = await import('./suggest.js');

  const folder = options.folder ?? process.cwd();
  const file = await chooseAgentFile(folder, options.agent, io, cliCommand);
  if (!file) return 1;
  if (!requested.length) return 0;
  const shown = path.basename(file);

  const known = resolvableSkillNames();
  const canonical = canonicalSkillName;
  const isKnown = (name: string): boolean => known.includes(canonical(name));
  const wanted = requested.map((name) => name.trim().toLowerCase()).filter(Boolean);

  // W3 / S-290: never write through a symbolic link, a file outside the
  // folder, or another user's file. The guard is here so both CLIs get it.
  const unsafe = unsafeTargetReason(file, folder);
  if (unsafe) {
    io.err(unsafe);
    return 1;
  }

  let text: string;
  try {
    text = fs.readFileSync(file, 'utf-8');
  } catch (error) {
    io.err(`Could not read ${shown}: ${(error as Error).message}`);
    return 1;
  }

  // Every name is checked before anything is written.
  let listed: string[] = [];
  if (action === 'remove') {
    try {
      listed = editSkillList(text, 'add', [], file).skills;
    } catch (error) {
      if (!(error instanceof SkillListError)) throw error;
      io.err(error.message);
      return 1;
    }
  }
  const unknown = wanted.filter((name) => !isKnown(name) && !listed.some((have) => canonical(have) === canonical(name)));
  if (unknown.length) {
    const candidates = action === 'remove' ? [...new Set([...listed, ...known])] : known;
    for (const name of unknown) io.err(`Unknown skill '${name}'.${suggestSimilar(name, candidates)}`);
    io.err(`Run \`${cliCommand('skills list')}\` for the skills an agent file can name.`);
    return 1;
  }

  let edit: SkillListEdit;
  try {
    edit = editSkillList(text, action, wanted, file);
  } catch (error) {
    if (!(error instanceof SkillListError)) throw error;
    io.err(error.message);
    return 1;
  }
  if (edit.changed) {
    try {
      fs.writeFileSync(file, edit.text);
    } catch (error) {
      io.err(`Could not write ${shown}: ${(error as Error).message}`);
      return 1;
    }
  }

  if (edit.added.length) io.out(`Added ${spokenList(edit.added)} to ${shown}.`);
  if (edit.already.length) io.out(`${shown} already names ${spokenList(edit.already)}.`);
  if (edit.removed.length) io.out(`Removed ${spokenList(edit.removed)} from ${shown}.`);
  if (edit.absent.length) io.out(`${shown} does not name ${spokenList(edit.absent)}.`);
  io.out(`Skills: ${edit.skills.length ? edit.skills.join(', ') : 'none'}`);
  io.edited?.({ file: shown, added: [...edit.added], already: [...edit.already], removed: [...edit.removed], absent: [...edit.absent], skills: [...edit.skills] });

  // What an added skill still needs, asked of this machine only when one could need something.
  const needy = edit.added.filter((name) => {
    const provider = findProvider(canonical(name));
    return (provider && provider.credential !== 'local') || canonical(name) === 'discovery';
  });
  if (needy.length) {
    const facts = await (options.facts ?? machineFacts)();
    for (const name of needy) {
      const provider = findProvider(canonical(name));
      if (provider?.credential === 'api-key' && provider.envVar) {
        if (!providerEnvVars(provider).some((variable) => facts.hasKey(variable))) {
          io.out(`${name} needs ${provider.envVar}: add it with \`${cliCommand(`secrets set ${provider.envVar}`)}\`.`);
        }
      } else if (!facts.signedIn) {
        io.out(`${name} needs you signed in to Robutler: run \`${cliCommand('login')}\`.`);
      }
    }
  }
  return 0;
}

/** The file `-a` names, else this folder's one agent file; null (said) when there is none to change. */
async function chooseAgentFile(
  folder: string,
  agent: string | undefined,
  io: SkillsCommandIO,
  cliCommand: (rest?: string) => string,
): Promise<string | null> {
  const { agentFileFor, AgentNotFound, BUILT_IN_AGENT, folderAgents } = await import('./agent-files.js');
  if (agent) {
    let file: string | null;
    try {
      file = agentFileFor(folder, agent);
    } catch (error) {
      if (!(error instanceof AgentNotFound)) throw error;
      io.err(error.message);
      return null;
    }
    if (!file) {
      io.err(`The built-in ${BUILT_IN_AGENT} has no agent file to change. Create one with \`${cliCommand('init')}\`.`);
      return null;
    }
    return file;
  }
  const agentMd = path.join(folder, 'AGENT.md');
  if (fs.existsSync(agentMd)) return agentMd;
  let named: string[] = [];
  try {
    named = fs.readdirSync(folder).filter((name) => /^AGENT-.+\.md$/.test(name)).sort();
  } catch {
    // An unreadable folder has no agent file to offer.
  }
  if (named.length === 1) return path.join(folder, named[0]);
  if (!named.length) {
    io.err(`No agent file in this folder. Create one with \`${cliCommand('init')}\`.`);
    return null;
  }
  const names = folderAgents(folder).map((a) => a.name);
  io.err(`More than one agent in this folder: pick one with -a (${names.join(', ')}).`);
  return null;
}

/**
 * The SKILL.md part of `skills list`: what an agent in `folder` would load
 * (`.agents/skills`, plus the default agent file's `agent_skills:`), the
 * skipped ones with their reasons, and how to add one.
 */
export function skillmdListLines(folder: string): string[] {
  // Static imports: `skills list` is a listing, and these modules load nothing heavy.
  let explicit: string[] = [];
  const file = findAgentFile(folder);
  if (file) {
    try {
      explicit = parseAgentMarkdown(fs.readFileSync(file, 'utf-8'), file).agentSkills ?? [];
    } catch {
      // A broken agent file is doctor's to report.
      explicit = [];
    }
  }
  return listLines(discoverSkills(folder, explicit), cliCommandNow('skills add <owner/repo | git URL | folder>'));
}

/** This machine's keys (the shell, or `secrets set`) and sign-in. */
export async function machineFacts(): Promise<SkillsFacts> {
  const { readStoredProviderKeys } = await import('./provider-keys.js');
  const { getToken } = await import('./credentials.js');
  const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
  const signedIn = Boolean(await getToken().catch(() => null));
  return { hasKey: (variable) => Boolean(process.env[variable] || stored[variable]), signedIn };
}

// ============================================================================
// planSkills / applySkills: the chat's /skills, and the CLI's checks, in two
// halves (2026-09-26, interactive-mode spec 3.4). `planSkills` does today's
// checks and no write; `applySkills` writes. The chat shows the plan, asks,
// snapshots and applies; the fixture `chat_edits.json` and the `plans` block
// of `skills_edit.json` pin them, and the Python chat mirrors both.
// ============================================================================

export interface SkillsPlan {
  /** The agent file to edit, or null when only sources or installed skills are touched. */
  file: string | null;
  /** Refusal lines (an unknown name, a linked file, a bad layout); when set, nothing is applied. */
  errors: string[];
  /** Coded names to add to the file. */
  add: string[];
  /** Coded names to remove from the file, as the file writes them. */
  remove: string[];
  /** Asked-for names the file already lists (add), as the file writes them. */
  already: string[];
  /** Asked-for names the file does not list (remove), as they were asked for. */
  absent: string[];
  /** The file's skill names after the edit. */
  skillsAfter: string[];
  /** Installed SKILL.md skill names to take out of `.agents/skills`. */
  installedRemovals: string[];
  /** Source texts (`owner/repo`, a git URL, a folder) to install. */
  sources: string[];
  /** Whether applying the plan changes the file or removes an installed skill. */
  changed: boolean;
}

/** What `applySkills` did, for the caller to word (the chat's ✓ lines). */
export interface SkillsApplied {
  added: string[];
  removed: string[];
  skillsAfter: string[];
  /** Installed SKILL.md skills taken out, with the folder line the CLI prints. */
  installedRemoved: string[];
}

/** Resolve the agent file for `planSkills`: a passed one, else AGENT.md, else the single AGENT-<name>.md. */
function agentFileInFolder(folder: string, file?: string): string | null {
  if (file) return file;
  const agentMd = path.join(folder, 'AGENT.md');
  if (fs.existsSync(agentMd) || safeIsLink(agentMd)) return agentMd;
  let named: string[] = [];
  try {
    named = fs.readdirSync(folder).filter((name) => /^AGENT-.+\.md$/.test(name)).sort();
  } catch {
    // Nothing to offer.
  }
  return named.length === 1 ? path.join(folder, named[0]) : null;
}

function safeIsLink(file: string): boolean {
  try {
    return fs.lstatSync(file).isSymbolicLink();
  } catch {
    return false;
  }
}

/**
 * The checks for `/skills add|remove`, no write (file comment). `who` names
 * the actor in a W3 refusal. Sources are separated out for the caller to
 * install; installed SKILL.md skills are separated out for `applySkills` to
 * remove. Unknown names, a linked file and a layout it cannot edit are
 * refusals in `errors`.
 */
export function planSkills(
  action: SkillsAction,
  requested: readonly string[],
  options: { file?: string; folder: string; who?: string },
): SkillsPlan {
  const installedNames = safeInstalledNames(options.folder);
  const known = resolvableSkillNames();
  const canonical = canonicalSkillName;
  const empty: SkillsPlan = { file: null, errors: [], add: [], remove: [], already: [], absent: [], skillsAfter: [], installedRemovals: [], sources: [], changed: false };

  const names: string[] = [];
  const sources: string[] = [];
  const installedRemovals: string[] = [];
  const errors: string[] = [];
  for (const entry of requested) {
    const lower = entry.trim().toLowerCase();
    const isSource = looksLikeSource(entry);
    if (isSource) {
      if (action === 'add') sources.push(entry);
      else errors.push(`${entry} is not a skill name; remove takes the names \`skills list\` shows.`);
    } else if (action === 'remove' && lower && !known.includes(canonical(lower)) && installedNames.includes(lower)) {
      installedRemovals.push(lower);
    } else {
      names.push(entry);
    }
  }

  if (!names.length) {
    return { ...empty, sources, installedRemovals, errors, changed: installedRemovals.length > 0 };
  }

  const file = agentFileInFolder(options.folder, options.file);
  if (!file) {
    errors.push(`No agent file in this folder. Create one with \`${cliCommandNow('init')}\`.`);
    return { ...empty, sources, installedRemovals, errors, changed: installedRemovals.length > 0 };
  }
  const unsafe = unsafeTargetReason(file, options.folder, options.who);
  if (unsafe) {
    return { ...empty, file, sources, installedRemovals, errors: [...errors, unsafe], changed: false };
  }

  let text: string;
  try {
    text = fs.readFileSync(file, 'utf-8');
  } catch (error) {
    return { ...empty, file, sources, installedRemovals, errors: [...errors, `Could not read ${path.basename(file)}: ${(error as Error).message}`], changed: false };
  }

  const wanted = names.map((name) => name.trim().toLowerCase()).filter(Boolean);
  let listed: string[] = [];
  try {
    listed = editSkillList(text, 'add', [], file).skills;
  } catch (error) {
    if (!(error instanceof SkillListError)) throw error;
    return { ...empty, file, sources, installedRemovals, errors: [...errors, error.message], changed: false };
  }
  const unknown = wanted.filter((name) => !known.includes(canonical(name)) && !listed.some((have) => canonical(have) === canonical(name)));
  if (unknown.length) {
    const candidates = action === 'remove' ? [...new Set([...listed, ...known])] : known;
    for (const name of unknown) {
      errors.push(`Unknown skill '${name}'.`);
      // `suggestSkill` returns `\n(Did you mean x?)`; keep the hint on its own line.
      const suggestion = suggestSkill(name, candidates).replace(/^\n/, '');
      if (suggestion) errors.push(suggestion);
    }
    errors.push(`Run \`${cliCommandNow('skills list')}\` for the skills an agent file can name.`);
    return { ...empty, file, sources, installedRemovals, errors, changed: false };
  }

  let edit: SkillListEdit;
  try {
    edit = editSkillList(text, action, wanted, file);
  } catch (error) {
    if (!(error instanceof SkillListError)) throw error;
    return { ...empty, file, sources, installedRemovals, errors: [...errors, error.message], changed: false };
  }
  return {
    file,
    errors,
    add: edit.added,
    remove: edit.removed,
    already: edit.already,
    absent: edit.absent,
    skillsAfter: edit.skills,
    installedRemovals,
    sources,
    changed: edit.changed || installedRemovals.length > 0,
  };
}

/**
 * Apply `plan`'s file edit and installed removals (file comment); nothing is
 * printed, so the caller words the result (the chat's ✓ lines, the fixture's).
 * The file is read again and edited from its current bytes, and written by
 * renaming a temporary file so a link is never written through. Sources are
 * the caller's to install.
 */
export function applySkills(action: SkillsAction, plan: SkillsPlan, folder: string): SkillsApplied {
  const quiet: SkillsCommandIO = { out: () => {}, err: () => {} };
  const installedRemoved: string[] = [];
  for (const name of plan.installedRemovals) {
    if (removeInstalled(name, folder, quiet) === 0) installedRemoved.push(name);
  }
  if (!plan.file || (!plan.add.length && !plan.remove.length)) {
    return { added: [], removed: [], skillsAfter: plan.skillsAfter, installedRemoved };
  }
  const text = fs.readFileSync(plan.file, 'utf-8');
  const names = action === 'add' ? plan.add : plan.remove;
  const edit = editSkillList(text, action, names, plan.file);
  if (edit.changed) writeByRename(plan.file, edit.text);
  return { added: edit.added, removed: edit.removed, skillsAfter: edit.skills, installedRemoved };
}

/** Write `text` to `file` by renaming a temporary file in the same folder, so a link is never written through. */
export function writeByRename(file: string, text: string): void {
  const temp = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.tmp`);
  fs.writeFileSync(temp, text);
  fs.renameSync(temp, file);
}

/** The installed SKILL.md skill names the lock records, still present; never throws. */
function safeInstalledNames(folder: string): string[] {
  try {
    return installedSkillNames(folder);
  } catch {
    return [];
  }
}

/** Whether `entry` is a SKILL.md source (owner/repo, a git URL, a folder) rather than a skill name. */
function looksLikeSource(entry: string): boolean {
  return parseSource(entry).kind !== 'name';
}

/** `suggestSimilar` for a skill name. */
function suggestSkill(name: string, candidates: readonly string[]): string {
  return suggestSimilar(name, [...candidates]);
}

// For `mcp-import.ts`, which adds an MCP server to an agent file's `- mcp:`
// block by this module's rule (2026-09-29): the same parse, lines added and
// never re-serialised, read back as YAML before anything is written.
export type { Parsed as ParsedFrontMatter };
export { parse as parseFrontMatter, entriesOf as skillEntriesOf, yamlScalar, isTrivia as isTriviaLine };
