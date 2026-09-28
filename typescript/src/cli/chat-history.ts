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
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import { CHAT_HISTORY } from './chat-commands';
import { globalDir } from './config-store';

/** The profile's history file. */
export function chatHistoryFile(profile?: string): string {
  return path.join(globalDir(profile), CHAT_HISTORY.file);
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

/** One entry in the file's format: a timestamp comment and `+`-prefixed lines, then a blank line. */
export function encodeChatHistoryEntry(text: string, at: Date = new Date()): string {
  const stamp = at.toISOString().replace('T', ' ').replace('Z', '').replace(/(\.\d{3})$/, '$1000');
  return `\n# ${stamp}\n${text.split('\n').map((line) => `+${line}`).join('\n')}\n`;
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

/** The kept entries, oldest first: the last `CHAT_HISTORY.keep`. Empty when there is no file. */
export function readChatHistory(file: string): string[] {
  try {
    return parseChatHistory(fs.readFileSync(file, 'utf8')).slice(-CHAT_HISTORY.keep);
  } catch {
    return [];
  }
}

/** Append one typed line; quiet when the file cannot be written. */
export function appendChatHistory(file: string, text: string): void {
  if (!text.trim()) return;
  if (!secureChatHistory(file)) return;
  try {
    fs.appendFileSync(file, encodeChatHistoryEntry(text), { mode: CHAT_HISTORY.fileMode });
  } catch {
    // History is a convenience; the message was still sent.
  }
}
