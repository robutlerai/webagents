/**
 * The chat's typed-line history on disk (`cli/chat-history.ts`; owner
 * decision D1, the S-291 fix, 2026-09-26): one file per profile, owner-only,
 * in prompt_toolkit's `FileHistory` format so the Python chat reads it back.
 * The shape is pinned by `python/tests/fixtures/cli/chat_commands.json`
 * (`history`), which `tests/cli/test_chat_history_interactive2.py` runs too.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CHAT_HISTORY } from '../../../src/cli/chat-commands';
import {
  appendChatHistory,
  chatHistoryFile,
  chatHistoryFolder,
  encodeChatHistoryEntry,
  parseChatHistory,
  readChatHistory,
  secureChatHistory,
} from '../../../src/cli/chat-history';
import { tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/chat_commands.json'), 'utf8'),
) as { history: { file: string; dir_mode: string; file_mode: string; keep: number; entries: string[]; encoded: string } };

const tempDir = tempDirs();
const saved: Record<string, string | undefined> = {};
const ISOLATED = ['HOME', 'WEBAGENTS_PROFILE', 'WEBAGENTS_SECRETS_BACKEND', 'ROBUTLER_API_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'WEBAGENTS_TOKEN'];
let home = '';

beforeEach(() => {
  for (const name of ISOLATED) {
    saved[name] = process.env[name];
    delete process.env[name];
  }
  home = tempDir('wa-history-home-');
  process.env.HOME = home;
  process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
  process.env.ROBUTLER_API_URL = 'http://127.0.0.1:9';
});

afterEach(() => {
  for (const name of ISOLATED) {
    if (saved[name] === undefined) delete process.env[name];
    else process.env[name] = saved[name];
  }
});

const posix = process.platform !== 'win32';

describe('the format is prompt_toolkit FileHistory', () => {
  it('the fixture shape is the one both chats keep', () => {
    expect(FIXTURE.history.file).toBe(CHAT_HISTORY.file);
    expect(parseInt(FIXTURE.history.dir_mode, 8)).toBe(CHAT_HISTORY.dirMode);
    expect(parseInt(FIXTURE.history.file_mode, 8)).toBe(CHAT_HISTORY.fileMode);
    expect(FIXTURE.history.keep).toBe(CHAT_HISTORY.keep);
  });

  it('reads the Python-written sample back as its entries, oldest first', () => {
    expect(parseChatHistory(FIXTURE.history.encoded)).toEqual(FIXTURE.history.entries);
  });

  it('encodes an entry the way prompt_toolkit writes one, multi-line included', () => {
    const at = new Date('2026-09-26T12:00:02.000Z');
    expect(encodeChatHistoryEntry('two\nlines', at)).toBe('\n# 2026-09-26 12:00:02.000000\n+two\n+lines\n');
    const text = FIXTURE.history.entries.map((e, i) => encodeChatHistoryEntry(e, new Date(Date.UTC(2026, 8, 26, 12, 0, i)))).join('');
    expect(text).toBe(FIXTURE.history.encoded);
  });
});

describe('the file on disk', () => {
  it('lives in the profile folder, and is made owner-only in an owner-only folder', () => {
    process.env.WEBAGENTS_PROFILE = 'work';
    const file = chatHistoryFile();
    expect(file).toBe(path.join(home, '.webagents-work', 'history'));
    appendChatHistory(file, 'hello');
    appendChatHistory(file, '/status');
    expect(readChatHistory(file)).toEqual(['hello', '/status']);
    if (posix) {
      expect(fs.statSync(path.dirname(file)).mode & 0o777).toBe(0o700);
      expect(fs.statSync(file).mode & 0o777).toBe(0o600);
    }
  });

  it('closes a file an older version left readable (S-291)', () => {
    const file = chatHistoryFile();
    fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o755 });
    fs.writeFileSync(file, FIXTURE.history.encoded, { mode: 0o644 });
    expect(secureChatHistory(file)).toBe(true);
    if (posix) {
      expect(fs.statSync(path.dirname(file)).mode & 0o777).toBe(0o700);
      expect(fs.statSync(file).mode & 0o777).toBe(0o600);
    }
    // And keeps what was there.
    expect(readChatHistory(file)).toEqual(FIXTURE.history.entries);
  });

  it('keeps the last entries only, and skips blank lines', () => {
    const file = chatHistoryFile();
    for (let i = 0; i < CHAT_HISTORY.keep + 5; i++) appendChatHistory(file, `line ${i}`);
    appendChatHistory(file, '   ');
    const got = readChatHistory(file);
    expect(got.length).toBe(CHAT_HISTORY.keep);
    expect(got[got.length - 1]).toBe(`line ${CHAT_HISTORY.keep + 4}`);
  });

  it('the chat loads earlier lines, newest first, at construction', async () => {
    const file = chatHistoryFile();
    // Its own folder's lines (2026-09-29, `chat-history-folders.test.ts`): a
    // line typed elsewhere, or written before folders were kept, is not offered.
    appendChatHistory(file, 'first', chatHistoryFolder());
    appendChatHistory(file, 'elsewhere', '/some/other/folder');
    appendChatHistory(file, 'second', chatHistoryFolder());
    appendChatHistory(file, 'before folders');
    const { InteractiveREPL } = await import('../../../src/cli/app');
    const repl = new InteractiveREPL({ interactive: true }) as unknown as { inputHistory: string[] };
    expect(repl.inputHistory).toEqual(['second', 'first']);
  });
});
