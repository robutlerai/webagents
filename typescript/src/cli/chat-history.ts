/**
 * The chat's typed-line history on disk (2026-09-26, owner decision D1 of the
 * interactive-mode review; the S-291 fix).
 *
 * The Python chat wrote every typed line to `~/.webagents/history` with the
 * process umask (0644 in a 0755 folder) whatever `--profile` said, so another
 * account on the machine could read everything typed, pasted keys included,
 * and every profile shared one file; the TypeScript chat kept no history at
 * all. Now both keep ONE file per profile, `<profile folder>/history`
 * (`globalDir()`: `~/.webagents` or `~/.webagents-<profile>`), with the
 * folder at 0700 and the file at 0600, made that way here and put back that
 * way on every open, so a file an older version left readable is closed the
 * first time the chat runs.
 *
 * The format is prompt_toolkit's `FileHistory`, which the Python box reads
 * and writes as it is:
 *
 *     # 2026-09-26 12:00:00.000000
 *     +first line of the entry
 *     +second line
 *     <blank>
 *
 * so a line typed into one chat comes back in the other's ↑ history. The
 * shape is pinned by `python/tests/fixtures/cli/chat_commands.json` (`history`).
 *
 * EACH FOLDER ITS OWN LINES (2026-09-29). ↑ and the suggestions offered every
 * line typed under the profile, in any folder and by anything that ran the
 * chat as its owner: an e2e test agent that drove the chat under the owner's
 * profile left its prompts in the owner's ↑ (the owner: "random stuff shows up
 * in history.. prob leakage from other sessions?"). An entry now carries the
 * folder it was typed in, on a comment line prompt_toolkit's `FileHistory`
 * skips (`# folder <path>`, after the timestamp), and each chat offers only its
 * own folder's lines (`chatHistoryFolder`, the real path of where it runs). A
 * line written before folders were kept names none and is offered nowhere; it
 * stays in the file. The Python chat is `repl/session.py` `FolderHistory`;
 * pinned by the fixture's `history.folders`.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { CHAT_HISTORY } from './chat-commands';
import { globalDir } from './config-store';

/** The profile's history file. */
export function chatHistoryFile(profile?: string): string {
  return path.join(globalDir(profile), CHAT_HISTORY.file);
}

/** The folder a chat's history is kept for: the real path of where it runs. */
export function chatHistoryFolder(cwd: string = process.cwd()): string {
  try {
    return fs.realpathSync(cwd);
  } catch {
    return path.resolve(cwd);
  }
}

/** The comment line that names an entry's folder. */
const FOLDER_LINE = '# folder ';

/** Each entry of a history file's text with the folder it names (none before 2026-09-29), oldest first. */
export function parseChatHistoryEntries(text: string): Array<{ text: string; folder?: string }> {
  const entries: Array<{ text: string; folder?: string }> = [];
  let lines: string[] | null = null;
  let folder: string | undefined;
  const flush = () => {
    if (lines && lines.length) entries.push(folder === undefined ? { text: lines.join('\n') } : { text: lines.join('\n'), folder });
    lines = null;
  };
  for (const raw of text.split(/\r?\n/)) {
    if (raw.startsWith('+')) {
      if (!lines) lines = [];
      lines.push(raw.slice(1));
      continue;
    }
    flush();
    if (raw.startsWith(FOLDER_LINE)) folder = raw.slice(FOLDER_LINE.length);
    else if (raw.startsWith('#')) folder = undefined;
  }
  flush();
  return entries;
}

/** The entries of a history file's text, oldest first, as `FileHistory` reads them. */
export function parseChatHistory(text: string): string[] {
  const entries: string[] = [];
  let lines: string[] | null = null;
  const flush = () => {
    if (lines && lines.length) entries.push(lines.join('\n'));
    lines = null;
  };
  for (const raw of text.split(/\r?\n/)) {
    if (raw.startsWith('+')) {
      if (!lines) lines = [];
      lines.push(raw.slice(1));
    } else {
      flush();
    }
  }
  flush();
  return entries;
}

/**
 * One entry in the file's format: a timestamp comment, the folder it was typed
 * in when given (`# folder <path>`), and `+`-prefixed lines.
 */
export function encodeChatHistoryEntry(text: string, at: Date = new Date(), folder?: string): string {
  const stamp = at.toISOString().replace('T', ' ').replace('Z', '').replace(/(\.\d{3})$/, '$1000');
  const where = folder === undefined ? '' : `${FOLDER_LINE}${folder.replace(/[\r\n]/g, ' ')}\n`;
  return `\n# ${stamp}\n${where}${text.split('\n').map((line) => `+${line}`).join('\n')}\n`;
}

/**
 * The folder at 0700 and the file at 0600, whether they exist already or not.
 * Never throws: a home folder that cannot be written leaves the chat without
 * history rather than without a chat.
 */
export function secureChatHistory(file: string): boolean {
  try {
    const dir = path.dirname(file);
    fs.mkdirSync(dir, { recursive: true, mode: CHAT_HISTORY.dirMode });
    fs.chmodSync(dir, CHAT_HISTORY.dirMode);
    // Open for append so an existing file keeps its entries; create it owner-only.
    fs.closeSync(fs.openSync(file, 'a', CHAT_HISTORY.fileMode));
    fs.chmodSync(file, CHAT_HISTORY.fileMode);
    return true;
  } catch {
    return false;
  }
}

/**
 * The kept entries, oldest first: the last `CHAT_HISTORY.keep`, only those
 * typed in `folder` when one is given. Empty when there is no file.
 */
export function readChatHistory(file: string, folder?: string): string[] {
  try {
    const text = fs.readFileSync(file, 'utf8');
    const entries =
      folder === undefined
        ? parseChatHistory(text)
        : parseChatHistoryEntries(text)
            .filter((entry) => entry.folder === folder)
            .map((entry) => entry.text);
    return entries.slice(-CHAT_HISTORY.keep);
  } catch {
    return [];
  }
}

/** Append one typed line, with the folder it was typed in when given; quiet when the file cannot be written. */
export function appendChatHistory(file: string, text: string, folder?: string): void {
  if (!text.trim()) return;
  if (!secureChatHistory(file)) return;
  try {
    fs.appendFileSync(file, encodeChatHistoryEntry(text, new Date(), folder), { mode: CHAT_HISTORY.fileMode });
  } catch {
    // History is a convenience; the message was still sent.
  }
}
