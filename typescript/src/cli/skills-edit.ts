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
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { parse as parseYaml } from 'yaml';

import { findProvider } from '../skills/llm/providers';

/** One skill, however a file names it: a provider's other names are the provider (`claude` is `anthropic`). */
export function canonicalSkillName(name: string): string {
  const lower = name.trim().toLowerCase();
  return findProvider(lower)?.id ?? lower;
}

export type SkillsAction = 'add' | 'remove';

/** Why a list could not be edited; each has one sentence (`skillListProblemMessage`). */
export type SkillListProblem = 'unclosed' | 'not_yaml' | 'not_list' | 'layout';

export function skillListProblemMessage(problem: SkillListProblem, file: string): string {
  switch (problem) {
    case 'unclosed':
      return `${file} opens its front matter with --- and never closes it.`;
    case 'not_yaml':
      return `The front matter of ${file} is not valid YAML. Fix it, then try again.`;
    case 'not_list':
      return `skills: in ${file} is not a list. Fix it, then try again.`;
    case 'layout':
      return `The skills: list in ${file} is laid out in a way this command cannot change safely. Edit it by hand.`;
  }
}

export class SkillListError extends Error {
  constructor(
    readonly problem: SkillListProblem,
    file: string,
  ) {
    super(skillListProblemMessage(problem, file));
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
  /** Defaults to the working directory. */
  folder?: string;
  /** Defaults to this machine (`machineFacts`); asked only when an added skill has a need. */
  facts?: () => Promise<SkillsFacts>;
}

export interface SkillsCommandIO {
  out(line: string): void;
  err(line: string): void;
}

/** `a`, `a and b`, `a, b and c`. */
export function spokenList(names: readonly string[]): string {
  if (names.length <= 1) return names.join('');
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/**
 * Run `skills add` or `skills remove`; resolves to the exit code. Refusals
 * (an unknown name, no agent file, a list it cannot edit) change nothing.
 */
export async function skillsCommand(
  action: SkillsAction,
  requested: readonly string[],
  options: SkillsCommandOptions = {},
  io: SkillsCommandIO = { out: (line) => console.log(line), err: (line) => console.error(line) },
): Promise<number> {
  const { cliCommand } = await import('./config-store.js');
  const { providerEnvVars } = await import('../skills/llm/providers.js');
  const { resolvableSkillNames } = await import('../skills/resolve.js');
  const { suggestSimilar } = await import('./suggest.js');

  const folder = options.folder ?? process.cwd();
  const file = await chooseAgentFile(folder, options.agent, io, cliCommand);
  if (!file) return 1;
  const shown = path.basename(file);

  const known = resolvableSkillNames();
  const canonical = canonicalSkillName;
  const isKnown = (name: string): boolean => known.includes(canonical(name));
  const wanted = requested.map((name) => name.trim().toLowerCase()).filter(Boolean);

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

/** This machine's keys (the shell, or `secrets set`) and sign-in. */
export async function machineFacts(): Promise<SkillsFacts> {
  const { readStoredProviderKeys } = await import('./provider-keys.js');
  const { getToken } = await import('./credentials.js');
  const stored = await readStoredProviderKeys().catch(() => ({}) as Record<string, string>);
  const signedIn = Boolean(await getToken().catch(() => null));
  return { hasKey: (variable) => Boolean(process.env[variable] || stored[variable]), signedIn };
}
