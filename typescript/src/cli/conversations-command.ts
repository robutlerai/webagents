/**
 * `webagents conversations list | delete | prune` (2026-09-29, the owner:
 * "how do we start/load/delete conversations?").
 *
 * The chat keeps every conversation under the profile
 * (`sessions/<folder>/<agent>/<id>.json`, `sessions.ts`) and, until this,
 * nothing but a hand-run `rm` removed one: they piled up forever, the owner's
 * `local` profile holding 30 across 18 folders, 16 of them test folders.
 * These commands show and remove them from outside the chat; inside it,
 * `/resume` lists and continues, and `/resume delete <number>` removes one.
 *
 *  - `list` shows this folder's conversations, every agent's, newest first,
 *    with the start of each id; `--all` shows every folder's.
 *  - `delete <id>` removes the one whose id starts so, after asking, from
 *    this folder or, with `--all`, from anywhere; `--yes` skips the question,
 *    and without a terminal the question cannot be asked, so it is required.
 *  - `prune --older-than <age>` removes the conversations last used longer ago
 *    than `30d`, `12h`, `2w` or `90m`; `--dry-run` shows them and removes
 *    nothing.
 *
 * A copy kept on Robutler (the `session: {backend: robutler}` skill) is never
 * touched, and the sentence says so. The words and the age grammar are the
 * shared fixture `python/tests/fixtures/cli/conversations.json`; the Python
 * CLI is `webagents/cli/conversations_command.py`, rule for rule.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { globalDir, profileName } from './config-store';
import { deleteSession, loadSession, sessionPreview, slugFor, whenLabel } from './sessions';

export const WORDS = {
  none: 'No conversations in {folder}.',
  noneAnywhere: 'No conversations kept under this profile.',
  row: '    {id}  {when}{count}{preview}',
  hint: '`webagents -r <id>` in its folder continues one; `{delete}` deletes one.',
  noMatch: 'No conversation {id} in {folder}; `{list}` shows them.',
  noMatchAnywhere: 'No conversation {id} under this profile.',
  ambiguous: "{id} starts {count} conversations' ids; give more of it.",
  askDelete: 'Delete the conversation last used {when} ({count} messages)? [y/N] ',
  deleted: 'Deleted the conversation last used {when}.',
  remoteStays: 'Its copy on Robutler stays; delete it there.',
  notDeleted: 'Nothing deleted.',
  needsYes: '{command} asks before it deletes, and this is not a terminal; add --yes.',
  badAge: '--older-than takes a number and a unit: 30d, 12h, 2w or 90m.',
  pruneNone: 'No conversations last used more than {age} ago.',
  wouldPrune: 'Would delete {count} conversations last used more than {age} ago:',
  askPrune: 'Delete {count} conversations last used more than {age} ago? [y/N] ',
  pruned: 'Deleted {count} conversations.',
} as const;

const AGE = /^(\d+)([mhdw])$/;
const UNIT_SECONDS: Record<string, number> = { m: 60, h: 3600, d: 86400, w: 7 * 86400 };
const ID_WIDTH = 8;

export function fill(word: keyof typeof WORDS, values: Record<string, string | number> = {}): string {
  let text: string = WORDS[word];
  for (const [key, value] of Object.entries(values)) text = text.split(`{${key}}`).join(String(value));
  return text;
}

/** `30d` as seconds; null for anything else (`badAge`). */
export function parseAge(text: string): number | null {
  const match = AGE.exec(text.trim());
  if (!match || Number(match[1]) === 0) return null;
  return Number(match[1]) * UNIT_SECONDS[match[2]];
}

/** One conversation as it is kept. */
export interface Kept {
  directory: string;
  folder: string;
  agent: string;
  id: string;
  updatedAt: string;
  messages: number;
  preview: string;
  onRobutler: boolean;
}

function when(iso: string): number {
  const at = Date.parse(iso);
  return Number.isNaN(at) ? 0 : at;
}

function dirsIn(dir: string): string[] {
  try {
    return fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();
  } catch {
    return [];
  }
}

/**
 * The conversations under one folder's directory, every agent's, newest
 * first. A file with no message from the person (a conversation never
 * started) is left out, as `/resume` leaves it out.
 */
export function keptIn(folderDir: string): Kept[] {
  const out: Kept[] = [];
  for (const agentName of dirsIn(folderDir)) {
    const agentDir = path.join(folderDir, agentName);
    let files: string[];
    try {
      files = fs.readdirSync(agentDir).filter((n) => n.endsWith('.json') && !n.startsWith('.')).sort();
    } catch {
      continue;
    }
    for (const file of files) {
      const session = loadSession(agentDir, file.slice(0, -'.json'.length));
      if (!session || !session.messages.some((m) => m && m.role === 'user')) continue;
      const metadata = session.metadata as Record<string, unknown>;
      out.push({
        directory: agentDir,
        folder: typeof metadata.folder === 'string' ? metadata.folder : path.basename(folderDir),
        agent: session.agent_name || agentName,
        id: session.session_id,
        updatedAt: session.updated_at,
        messages: session.messages.length,
        preview: sessionPreview(session.messages),
        onRobutler: typeof metadata.robutler_chat_id === 'string' && metadata.robutler_chat_id.length > 0,
      });
    }
  }
  return out.sort((a, b) => when(b.updatedAt) - when(a.updatedAt));
}

export function sessionsRoot(profile?: string): string {
  return path.join(globalDir(profileName(profile)), 'sessions');
}

function realFolder(folder: string): string {
  try {
    return fs.realpathSync(folder);
  } catch {
    return path.resolve(folder);
  }
}

export function folderDir(folder: string, profile?: string): string {
  return path.join(sessionsRoot(profile), slugFor(realFolder(folder)));
}

/** This folder's conversations, or every folder's (`--all`), newest first. */
export function gather(folder: string, everywhere: boolean, profile?: string): Kept[] {
  if (!everywhere) {
    // This folder's own path, for the files written before a conversation
    // kept its folder (their directory holds only the lossy slug).
    const here = realFolder(folder);
    return keptIn(folderDir(folder, profile)).map((k) => ({ ...k, folder: here }));
  }
  const root = sessionsRoot(profile);
  const out = dirsIn(root).flatMap((each) => keptIn(path.join(root, each)));
  return out.sort((a, b) => when(b.updatedAt) - when(a.updatedAt));
}

/** What `list` prints: a folder line, an agent line, a row per conversation, then the hint. */
export function listLines(kept: readonly Kept[], folder: string, everywhere: boolean, width: number, deleteCommand: string, now: number = Date.now()): string[] {
  if (!kept.length) return [everywhere ? WORDS.noneAnywhere : fill('none', { folder })];
  const lines: string[] = [];
  const order = new Map<string, Map<string, Kept[]>>();
  for (const k of kept) {
    if (!order.has(k.folder)) order.set(k.folder, new Map());
    const agents = order.get(k.folder)!;
    agents.set(k.agent, [...(agents.get(k.agent) ?? []), k]);
  }
  for (const [shownFolder, agents] of order) {
    lines.push(shownFolder);
    for (const [agent, rows] of agents) {
      lines.push(`  ${agent}`);
      for (const k of rows) {
        const room = Math.max(10, width - 4 - ID_WIDTH - 2 - 12 - 14);
        const text = k.preview || '(no text)';
        const preview = text.length <= room ? text : `${text.slice(0, room - 1)}…`;
        lines.push(fill('row', { id: k.id.slice(0, ID_WIDTH), when: whenLabel(k.updatedAt, now).padEnd(12), count: `${k.messages} messages`.padEnd(14), preview }));
      }
    }
  }
  lines.push('', fill('hint', { delete: deleteCommand }));
  return lines;
}

/** `delete <id>` matched none, or more than one; the message says which. */
export class NotOne extends Error {
  constructor(message: string, readonly code: 'not_found' | 'ambiguous') {
    super(message);
  }
}

export function pick(kept: readonly Kept[], prefix: string, folder: string, everywhere: boolean, listCommand: string): Kept {
  const matches = prefix ? kept.filter((k) => k.id.startsWith(prefix)) : [];
  if (!matches.length) {
    throw new NotOne(everywhere ? fill('noMatchAnywhere', { id: prefix }) : fill('noMatch', { id: prefix, folder, list: listCommand }), 'not_found');
  }
  if (matches.length > 1) throw new NotOne(fill('ambiguous', { id: prefix, count: matches.length }), 'ambiguous');
  return matches[0];
}

export function olderThan(kept: readonly Kept[], seconds: number, now: number = Date.now()): Kept[] {
  return kept.filter((k) => (now - when(k.updatedAt)) / 1000 > seconds);
}

export function remove(kept: Kept): boolean {
  return deleteSession(kept.directory, kept.id);
}

type Ask = (question: string) => Promise<string | null>;

/** A yes to `question`; null when there is no terminal to ask at. */
async function confirmAtTerminal(question: string, ask?: Ask): Promise<boolean | null> {
  let asker = ask;
  if (!asker) {
    if (!(process.stdin.isTTY && process.stdout.isTTY)) return null;
    const { promptLine } = await import('./prompt.js');
    asker = promptLine;
  }
  const answer = await asker(question);
  return /^y(es)?$/i.test((answer ?? '').trim());
}

function shown(k: Kept): Record<string, unknown> {
  return { folder: k.folder, agent: k.agent, id: k.id, updated_at: k.updatedAt, messages: k.messages, preview: k.preview, on_robutler: k.onRobutler };
}

export interface CommandIo {
  json: boolean;
  cliCommand: (rest: string) => string;
  ask?: Ask;
  width?: number;
  now?: number;
}

/** `conversations list [--all]`. */
export async function listCommand(folder: string, everywhere: boolean, io: CommandIo): Promise<number> {
  const kept = gather(folder, everywhere);
  if (io.json) {
    const { emit } = await import('./output.js');
    emit({ conversations: kept.map(shown) });
    return 0;
  }
  const width = io.width ?? (process.stdout.columns || 100);
  for (const line of listLines(kept, realFolder(folder), everywhere, width, io.cliCommand('conversations delete <id>'), io.now)) console.log(line);
  return 0;
}

/** `conversations delete <id> [--all] [--yes]`. */
export async function deleteCommand(folder: string, prefix: string, everywhere: boolean, yes: boolean, io: CommandIo): Promise<number> {
  const { emit, fail } = await import('./output.js');
  const kept = gather(folder, everywhere);
  let chosen: Kept;
  try {
    chosen = pick(kept, prefix, realFolder(folder), everywhere, io.cliCommand('conversations list'));
  } catch (error) {
    if (!(error instanceof NotOne)) throw error;
    if (io.json) fail(error.code, error.message);
    console.error(error.message);
    return 1;
  }
  const at = whenLabel(chosen.updatedAt, io.now);
  if (!yes) {
    const answer = await confirmAtTerminal(fill('askDelete', { when: at, count: chosen.messages }), io.ask);
    if (answer === null) {
      const message = fill('needsYes', { command: io.cliCommand('conversations delete') });
      if (io.json) fail('needs_yes', message);
      console.error(message);
      return 1;
    }
    if (!answer) {
      console.log(WORDS.notDeleted);
      return 0;
    }
  }
  remove(chosen);
  if (io.json) {
    emit({ deleted: chosen.id, on_robutler: chosen.onRobutler });
    return 0;
  }
  console.log(fill('deleted', { when: at }));
  if (chosen.onRobutler) console.log(WORDS.remoteStays);
  return 0;
}

/** `conversations prune --older-than <age> [--all] [--yes] [--dry-run]`. */
export async function pruneCommand(
  folder: string,
  age: string | undefined,
  everywhere: boolean,
  yes: boolean,
  dryRun: boolean,
  io: CommandIo,
): Promise<number> {
  const { emit, fail } = await import('./output.js');
  const seconds = parseAge(age ?? '');
  if (seconds === null) {
    if (io.json) fail('bad_age', WORDS.badAge, '', 2);
    console.error(WORDS.badAge);
    return 2;
  }
  const old = olderThan(gather(folder, everywhere), seconds, io.now);
  if (!old.length) {
    if (io.json) {
      emit({ [dryRun ? 'would_delete' : 'deleted']: [] });
      return 0;
    }
    console.log(fill('pruneNone', { age: age ?? '' }));
    return 0;
  }
  if (dryRun) {
    if (io.json) {
      emit({ would_delete: old.map(shown) });
      return 0;
    }
    console.log(fill('wouldPrune', { count: old.length, age: age ?? '' }));
    for (const k of old) console.log(`  ${k.id.slice(0, ID_WIDTH)}  ${whenLabel(k.updatedAt, io.now).padEnd(12)}${k.agent}  ${k.folder}`);
    return 0;
  }
  if (!yes) {
    const answer = await confirmAtTerminal(fill('askPrune', { count: old.length, age: age ?? '' }), io.ask);
    if (answer === null) {
      const message = fill('needsYes', { command: io.cliCommand('conversations prune') });
      if (io.json) fail('needs_yes', message);
      console.error(message);
      return 1;
    }
    if (!answer) {
      console.log(WORDS.notDeleted);
      return 0;
    }
  }
  const removed = old.filter((k) => remove(k));
  if (io.json) {
    emit({ deleted: removed.map((k) => k.id) });
    return 0;
  }
  console.log(fill('pruned', { count: removed.length }));
  if (removed.some((k) => k.onRobutler)) console.log(WORDS.remoteStays);
  return 0;
}
