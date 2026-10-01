/**
 * `webagents conversations list | delete | prune` (`src/cli/conversations-command.ts`,
 * 2026-09-29), against the shared fixture
 * `python/tests/fixtures/cli/conversations.json`, which the Python suite reads
 * too (`tests/cli/test_conversations_command.py`).
 *
 * Until this, a kept conversation could only be removed by hand. Pinned here:
 * the words; the `--older-than` grammar; what `list` prints, this folder's or
 * every folder's; which conversation an id prefix picks; which are older than
 * an age. Then the commands in a throwaway HOME: `delete` asks, and without a
 * terminal it needs `--yes`; `prune --dry-run` removes nothing; a copy on
 * Robutler is said to stay; and the real CLI answers `--json` with one document.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  NotOne,
  WORDS,
  deleteCommand,
  fill,
  listLines,
  olderThan,
  parseAge,
  pick,
  pruneCommand,
  type Kept,
} from '../../../src/cli/conversations-command';
import { saveSession, sessionsDir } from '../../../src/cli/sessions';
import { CLI_ARGS, TSX_PROBLEM, tempDirs } from '../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
type RawKept = { folder: string; agent: string; id: string; updated_at: string; messages: number; preview: string; on_robutler: boolean };
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/cli/conversations.json'), 'utf8')) as {
  words: Record<string, string>;
  ages: Array<[string, number | null]>;
  now: string;
  list: { width: number; delete_command: string; cases: Array<{ about: string; everywhere: boolean; folder: string; kept: RawKept[]; lines: string[] }> };
  pick: { folder: string; list_command: string; kept: RawKept[]; cases: Array<{ prefix: string; everywhere: boolean; picks?: number; refused?: string; code?: string }> };
  older_than: { seconds: number; kept: RawKept[]; ids: string[] };
};
const NOW = Date.parse(FIXTURE.now);
const kept = (raw: RawKept): Kept => ({
  directory: '/nowhere',
  folder: raw.folder,
  agent: raw.agent,
  id: raw.id,
  updatedAt: raw.updated_at,
  messages: raw.messages,
  preview: raw.preview,
  onRobutler: raw.on_robutler,
});
const cliCommand = (rest: string): string => `webagents ${rest}`;
const tempDir = tempDirs();

describe('the fixture', () => {
  it('has the words', () => {
    expect(WORDS).toEqual(FIXTURE.words);
  });

  it.each(FIXTURE.ages.map(([text, seconds]) => [JSON.stringify(text), text, seconds] as const))('reads the age %s', (_label, text, seconds) => {
    expect(parseAge(text)).toBe(seconds);
  });

  it.each(FIXTURE.list.cases.map((c) => [c.about, c] as const))('list: %s', (_about, c) => {
    expect(listLines(c.kept.map(kept), c.folder, c.everywhere, FIXTURE.list.width, FIXTURE.list.delete_command, NOW)).toEqual(c.lines);
  });

  it.each(FIXTURE.pick.cases.map((c) => [`${JSON.stringify(c.prefix)} everywhere=${c.everywhere}`, c] as const))('pick %s', (_label, c) => {
    const all = FIXTURE.pick.kept.map(kept);
    if (c.refused !== undefined) {
      let caught: unknown;
      try {
        pick(all, c.prefix, FIXTURE.pick.folder, c.everywhere, FIXTURE.pick.list_command);
      } catch (error) {
        caught = error;
      }
      expect(caught).toBeInstanceOf(NotOne);
      expect((caught as NotOne).message).toBe(c.refused);
      expect((caught as NotOne).code).toBe(c.code);
      return;
    }
    expect(all.indexOf(pick(all, c.prefix, FIXTURE.pick.folder, c.everywhere, FIXTURE.pick.list_command))).toBe(c.picks);
  });

  it('finds the ones older than an age', () => {
    expect(olderThan(FIXTURE.older_than.kept.map(kept), FIXTURE.older_than.seconds, NOW).map((k) => k.id)).toEqual(FIXTURE.older_than.ids);
  });
});

describe('the commands', () => {
  const ISOLATED = ['HOME', 'WEBAGENTS_SECRETS_BACKEND', 'WEBAGENTS_PROFILE'];
  const saved: Record<string, string | undefined> = {};
  const cwd = process.cwd();
  let here = '';
  let elsewhere = '';
  let printed: string[] = [];

  beforeEach(() => {
    for (const name of ISOLATED) {
      saved[name] = process.env[name];
      delete process.env[name];
    }
    const root = tempDir('wa-conversations-');
    process.env.HOME = path.join(root, 'home');
    process.env.WEBAGENTS_SECRETS_BACKEND = 'file';
    here = path.join(root, 'here');
    elsewhere = path.join(root, 'elsewhere');
    fs.mkdirSync(here);
    fs.mkdirSync(elsewhere);
    process.chdir(here);
    const blank = { created_at: '', updated_at: '', input_tokens: 0, output_tokens: 0 };
    for (const [folder, agent, id, text, chatId] of [
      [here, 'helper', 'aaaa1111-0000-4000-8000-000000000001', 'plan the launch', undefined],
      [here, 'writer', 'bbbb2222-0000-4000-8000-000000000002', 'draft the post', 'chat-9'],
      [elsewhere, 'helper', 'cccc3333-0000-4000-8000-000000000003', 'somewhere else', undefined],
    ] as const) {
      saveSession(sessionsDir(folder, agent), {
        ...blank,
        session_id: id,
        agent_name: agent,
        messages: [{ role: 'user', content: text }, { role: 'assistant', content: 'ok' }],
        metadata: { folder: fs.realpathSync(folder), ...(chatId ? { robutler_chat_id: chatId } : {}) },
      });
    }
    // A conversation never started (no message from the person) is not one.
    saveSession(sessionsDir(here, 'helper'), { ...blank, session_id: 'dddd4444-0000-4000-8000-000000000004', agent_name: 'helper', messages: [], metadata: {} });
    printed = [];
    for (const method of ['log', 'error'] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        printed.push(args.map(String).join(' '));
      });
    }
  });

  afterEach(() => {
    process.chdir(cwd);
    for (const name of ISOLATED) {
      if (saved[name] === undefined) delete process.env[name];
      else process.env[name] = saved[name];
    }
    vi.restoreAllMocks();
  });

  const file = (folder: string, agent: string, id: string) => path.join(sessionsDir(folder, agent), `${id}.json`);

  it('delete asks at a terminal, and says a copy on Robutler stays', async () => {
    expect(await deleteCommand(here, 'aaaa', false, false, { json: false, cliCommand, ask: async () => 'n' })).toBe(0);
    expect(fs.existsSync(file(here, 'helper', 'aaaa1111-0000-4000-8000-000000000001'))).toBe(true);
    const questions: string[] = [];
    const ask = async (q: string) => {
      questions.push(q);
      return 'y';
    };
    expect(await deleteCommand(here, 'aaaa', false, false, { json: false, cliCommand, ask })).toBe(0);
    expect(questions).toEqual([fill('askDelete', { when: 'just now', count: 2 })]);
    expect(fs.existsSync(file(here, 'helper', 'aaaa1111-0000-4000-8000-000000000001'))).toBe(false);
    printed = [];
    expect(await deleteCommand(here, 'bbbb', false, true, { json: false, cliCommand })).toBe(0);
    expect(printed).toEqual(['Deleted the conversation last used just now.', WORDS.remoteStays]);
    expect(await deleteCommand(here, 'cccc', false, true, { json: false, cliCommand })).toBe(1);
    expect(await deleteCommand(here, 'cccc', true, true, { json: false, cliCommand })).toBe(0);
  });

  it('prune reads the age, removes only what is older, and --dry-run removes nothing', async () => {
    expect(await pruneCommand(here, 'soon', false, true, false, { json: false, cliCommand })).toBe(2);
    expect(printed).toContain(WORDS.badAge);
    expect(await pruneCommand(here, undefined, false, true, false, { json: false, cliCommand })).toBe(2);
    printed = [];
    expect(await pruneCommand(here, '1d', false, true, false, { json: false, cliCommand })).toBe(0);
    expect(printed).toEqual(['No conversations last used more than 1d ago.']);
    const later = Date.parse('2099-01-01T00:00:00Z');
    expect(await pruneCommand(here, '1d', true, false, true, { json: false, cliCommand, now: later })).toBe(0);
    expect(fs.existsSync(file(elsewhere, 'helper', 'cccc3333-0000-4000-8000-000000000003'))).toBe(true);
    expect(await pruneCommand(here, '1d', false, true, false, { json: false, cliCommand, now: later })).toBe(0);
    expect(fs.existsSync(file(here, 'helper', 'aaaa1111-0000-4000-8000-000000000001'))).toBe(false);
    expect(fs.existsSync(file(elsewhere, 'helper', 'cccc3333-0000-4000-8000-000000000003'))).toBe(true);
  });

  function runCli(args: string[]): Promise<{ code: number | null; out: string; err: string }> {
    return new Promise((resolve) => {
      const child = spawn(process.execPath, [...CLI_ARGS, ...args], { cwd: here, env: { ...process.env, WEBAGENTS_PROFILE: '' }, stdio: ['ignore', 'pipe', 'pipe'] });
      let out = '';
      let err = '';
      child.stdout.on('data', (d) => (out += d));
      child.stderr.on('data', (d) => (err += d));
      child.on('close', (code) => resolve({ code, out, err }));
    });
  }

  it.skipIf(TSX_PROBLEM !== null)('the real CLI lists, answers --json, and needs --yes without a terminal', async () => {
    const listed = await runCli(['conversations', 'list']);
    expect(listed.code, listed.err).toBe(0);
    expect(listed.out).toContain(fs.realpathSync(here));
    expect(listed.out).toContain('plan the launch');
    expect(listed.out).not.toContain('somewhere else');
    expect(listed.out).not.toContain('dddd4444');
    const everywhere = await runCli(['conversations', 'list', '--all']);
    expect(everywhere.out).toContain('somewhere else');
    const document = JSON.parse((await runCli(['--json', 'conversations', 'list'])).out) as { ok: boolean; data: { conversations: Array<{ id: string }> } };
    expect(document.ok).toBe(true);
    expect(document.data.conversations.map((d) => d.id.slice(0, 4)).sort()).toEqual(['aaaa', 'bbbb']);
    const refused = await runCli(['conversations', 'delete', 'bbbb']);
    expect(refused.code).toBe(1);
    expect(refused.err).toContain(fill('needsYes', { command: 'webagents conversations delete' }));
    expect(fs.existsSync(file(here, 'writer', 'bbbb2222-0000-4000-8000-000000000002'))).toBe(true);
  }, 60_000);
});
