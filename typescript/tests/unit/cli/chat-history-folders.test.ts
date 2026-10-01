/**
 * Each folder its own history lines (2026-09-29), against the shared fixture
 * `python/tests/fixtures/cli/chat_commands.json` (`history.folders`), which the
 * Python suite reads too (`tests/cli/test_chat_history_folders.py`).
 *
 * ↑ offered every line typed under the profile, in any folder and by anything
 * that ran the chat as its owner (an e2e test agent's prompts showed up in the
 * owner's ↑). Pinned here: an entry names its folder on a comment line
 * prompt_toolkit skips, a chat reads only its own folder's lines, an older
 * entry with no folder is read nowhere, and reading without a folder still
 * gives every entry.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  appendChatHistory,
  chatHistoryFolder,
  encodeChatHistoryEntry,
  parseChatHistory,
  parseChatHistoryEntries,
  readChatHistory,
} from '../../../src/cli/chat-history';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FOLDERS = (
  JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_commands.json'), 'utf8')) as {
    history: { folders: { line: string; encoded: string; for_folder: Record<string, string[]>; all: string[] } };
  }
).history.folders;

let dir: string;
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wa-history-folders-'));
});
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

describe('history kept per folder', () => {
  it('reads each entry with the folder it names', () => {
    const entries = parseChatHistoryEntries(FOLDERS.encoded);
    expect(entries.map((e) => e.text)).toEqual(FOLDERS.all);
    expect(entries.map((e) => e.folder)).toEqual(['/work/a', '/work/b', undefined]);
  });

  it.each(Object.entries(FOLDERS.for_folder))('offers %s its own lines only', (folder, lines) => {
    const file = path.join(dir, 'history');
    fs.writeFileSync(file, FOLDERS.encoded);
    expect(readChatHistory(file, folder)).toEqual(lines);
  });

  it('still reads every entry without a folder', () => {
    expect(parseChatHistory(FOLDERS.encoded)).toEqual(FOLDERS.all);
  });

  it('encodes the folder line the fixture shows', () => {
    const first = encodeChatHistoryEntry('in a', new Date('2026-09-29T12:00:00.000Z'), '/work/a');
    expect(first).toBe(`\n# 2026-09-29 12:00:00.000000\n${FOLDERS.line}/work/a\n+in a\n`);
    expect(FOLDERS.encoded.startsWith(first)).toBe(true);
  });

  it('appends a line to its folder, and nowhere else offers it', () => {
    const file = path.join(dir, 'history');
    appendChatHistory(file, 'two\nlines', '/work/a');
    expect(readChatHistory(file, '/work/a')).toEqual(['two\nlines']);
    expect(readChatHistory(file, '/work/b')).toEqual([]);
  });

  it('keeps the history for the real path of the folder', () => {
    expect(chatHistoryFolder(dir)).toBe(fs.realpathSync(dir));
  });
});
