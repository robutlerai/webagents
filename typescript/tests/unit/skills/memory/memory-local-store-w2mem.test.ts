/**
 * The local memory tier (`src/skills/memory/local-store.ts`): the files are
 * the truth, the index is rebuilt from them, the log records this machine's
 * changes, and the sync with the portal moves changes both ways, merged by
 * entry id with the last writer winning.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import path from 'node:path';

import { LocalMemoryStore, parseEntryFile, renderEntryFile } from '../../../../src/skills/memory/local-store';
import { MemorySkill } from '../../../../src/skills/memory/skill';
import { fakePortal } from './fake-portal-w2mem';
import { tempDirs } from '../../../helpers/cli';

const tempDir = tempDirs();
const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ALICE = { authenticated: true, provider: 'platform', scopes: [], user_id: 'alice' };
const ctx = (auth: unknown, metadata: Record<string, unknown> = {}) => ({ auth, metadata, get: () => undefined, set: () => undefined }) as never;

describe('the local store', () => {
  it('uses node:sqlite FTS5 on this Node, and the plain index when asked', async () => {
    const sqlite = new LocalMemoryStore({ root: path.join(tempDir('mem-store-'), 'memory'), store: 'helper' });
    await sqlite.open();
    expect(sqlite.indexKind).toBe('sqlite');
    expect(fs.existsSync(path.join(sqlite.root, 'index.db'))).toBe(true);
    sqlite.close();
    const plain = new LocalMemoryStore({ root: path.join(tempDir('mem-store-plain-'), 'memory'), store: 'helper', plainIndex: true });
    await plain.open();
    expect(plain.indexKind).toBe('plain');
  });

  for (const plainIndex of [false, true]) {
    it(`finds a note a person wrote by hand, and one edited by hand (${plainIndex ? 'plain' : 'sqlite'} index)`, async () => {
      const root = path.join(tempDir('mem-store-hand-'), 'memory');
      const store = new LocalMemoryStore({ root, store: 'helper', plainIndex });
      await store.open();
      await store.put('owner', 'preferences', 'Short answers.', 'tool');
      store.close();

      // Written by hand, with front matter (a copied note) and without (a plain Markdown file).
      fs.writeFileSync(path.join(root, 'owner', 'venue.md'), renderEntryFile({ id: 'x', namespace: 'owner', key: 'venue', content: 'The Forum, on the river.', source: 'owner', createdAt: '2026-01-01T00:00:00.000Z', updatedAt: '2026-01-01T00:00:00.000Z' }));
      fs.writeFileSync(path.join(root, 'owner', 'preferences.md'), fs.readFileSync(path.join(root, 'owner', 'preferences.md'), 'utf8').replace('Short answers.', 'Long answers.'));

      const again = new LocalMemoryStore({ root, store: 'helper', plainIndex });
      await again.open();
      expect((await again.search('river', null, 10)).map((e) => e.key)).toEqual(['venue']);
      expect((await again.search('answers', ['owner'], 10)).map((e) => e.content)).toEqual(['Long answers.']);
      expect((await again.search('answers', ['shared'], 10))).toEqual([]);
      expect((await again.get('owner', 'venue'))?.id).toMatch(/^[0-9a-f-]{36}$/);
      again.close();
    });
  }

  it('a file without front matter is a hand-written note under a known namespace, and a caller file without a caller is left alone', () => {
    // 2026-09-27, `chat-fixes-hand-written-note.test.ts` has the shared cases.
    expect(parseEntryFile('just text', { namespace: 'owner', key: 'a', store: 's' })).toMatchObject({ namespace: 'owner', key: 'a', content: 'just text', source: 'owner' });
    expect(parseEntryFile('just text', { namespace: '', key: 'a', store: 's' })).toBeNull();
    expect(parseEntryFile('---\nkey: a\n---\nbody\n', { namespace: '', key: 'a', store: 's' })).toBeNull();
    const parsed = parseEntryFile('---\nkey: a\nnamespace: caller:user:bob\nsource: compaction\n---\nbody\n', { namespace: '', key: 'a', store: 's' });
    expect(parsed).toMatchObject({ key: 'a', namespace: 'caller:user:bob', source: 'compaction', content: 'body' });
  });

  it('logs every local change in order, and applies the other tier\'s without logging them', async () => {
    const store = new LocalMemoryStore({ root: path.join(tempDir('mem-store-log-'), 'memory'), store: 'helper', plainIndex: true, now: () => new Date('2026-09-26T10:00:00.000Z') });
    await store.open();
    await store.put('owner', 'a', 'one', 'tool');
    await store.put('owner', 'a', 'two', 'tool');
    await store.forget('owner', 'a');
    expect(store.logSince(0).map((l) => [l.seq, l.op, l.key, l.content])).toEqual([[1, 'put', 'a', 'one'], [2, 'put', 'a', 'two'], [3, 'delete', 'a', undefined]]);
    expect(store.logSince(2).map((l) => l.seq)).toEqual([3]);

    const applied = await store.apply([
      { seq: 9, op: 'put', id: 'x', namespace: 'shared', key: 'hours', content: '9 to 5', source: 'tool', at: '2026-09-26T11:00:00.000Z' },
      { seq: 10, op: 'put', id: 'x', namespace: 'shared', key: 'hours', content: 'older', source: 'tool', at: '2026-09-26T09:00:00.000Z' },
      { seq: 11, op: 'delete', id: 'y', namespace: 'shared', key: 'missing', at: '2026-09-26T11:00:00.000Z' },
    ]);
    expect(applied).toBe(1);
    expect((await store.get('shared', 'hours'))?.content).toBe('9 to 5');
    expect((await store.get('shared', 'hours'))?.source).toBe('sync');
    expect(store.logSince(0)).toHaveLength(3);
    store.close();
  });
});

describe('local and portal together', () => {
  it('pushes every local change to the platform and pulls what is newer there, merged by entry id', async () => {
    const portal = fakePortal();
    const dir = tempDir('mem-sync-');
    const skill = new MemorySkill({ local: true, portal: true, agentDir: dir, agentId: portal.agentId, portalUrl: 'https://portal.test', apiKey: 'agent-key', fetchImpl: portal.fetch, plainIndex: true });
    await skill.initialize();

    await skill.memoryWrite({ key: 'plan', content: 'October' }, ctx(OWNER));
    await skill.memoryWrite({ key: 'name', content: 'Ada' }, ctx(ALICE));
    expect([...portal.rows.values()].map((r) => `${r.namespace}/${r.key}=${r.content}`).sort()).toEqual(['caller:user:alice/name=Ada', 'owner/plan=October']);
    const state = JSON.parse(fs.readFileSync(path.join(dir, '.webagents', 'memory', 'sync-state.json'), 'utf8'));
    expect(state.pushedSeq).toBe(2);

    // Something changed on the platform (the owner edited the plan there, and shared a note).
    portal.put('owner', 'plan', 'November', 'owner', '2999-01-01T00:00:00.000Z');
    portal.put('shared', 'hours', '9 to 5', 'owner', '2999-01-01T00:00:00.000Z');
    await skill.pullForCaller({}, ctx(OWNER));
    expect(((await skill.memorySearch({ query: 'November' }, ctx(OWNER))) as { entries: Array<{ content: string }> }).entries.map((e) => e.content)).toEqual(['November']);
    expect(fs.readFileSync(path.join(dir, '.webagents', 'memory', 'shared', 'hours.md'), 'utf8')).toContain('9 to 5');
    // What was pulled is not pushed back: the log holds this machine's two changes only.
    expect(portal.seen.filter((r) => r.method === 'POST')).toHaveLength(2);
    // Search answers from the local index first, then the platform's semantic leg for what it missed.
    expect(((await skill.memorySearch({ query: 'Ada' }, ctx(ALICE))) as { entries: Array<{ key: string }> }).entries.map((e) => e.key)).toEqual(['name']);
  });
});
