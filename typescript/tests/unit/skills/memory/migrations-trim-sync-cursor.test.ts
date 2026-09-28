/**
 * S-307 (logged 2026-09-27, fixed by the migrations-trim lane): the portal
 * answered a push with `Number.MAX_SAFE_INTEGER` as the cursor, this skill
 * kept it with `Math.max`, and every later pull asked for "after the end":
 * the owner's edits and forgets on Robutler never reached the agent's local
 * notes. The suites missed it because the fake portal minted its own
 * cursor. Pinned here against the fake portal that now follows the
 * platform's contract, with the SHARED fixture the portal and the Python
 * suite read too (`python/tests/fixtures/memory_tool/migrations_trim_sync.json`):
 *
 *   - a second pull after a push sees the owner's later edit and forget;
 *   - a push sends no cursor back and none is kept; a pull keeps the
 *     platform's string as it came, and sends it back as `since`;
 *   - a numeric cursor an older state file holds (S-307's stuck one) is
 *     dropped, and that namespace pulls from the start;
 *   - a pull takes every page (`more`) in one go.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import path from 'node:path';

import { MemorySkill } from '../../../../src/skills/memory/skill';
import { PortalMemoryStore } from '../../../../src/skills/memory/portal-store';
import { fakePortal } from './fake-portal-w2mem';
import { tempDirs } from '../../../helpers/cli';

const tempDir = tempDirs();
const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ctx = (auth: unknown, metadata: Record<string, unknown> = {}) => ({ auth, metadata, get: () => undefined, set: () => undefined }) as never;

const FIXTURE = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../../../../python/tests/fixtures/memory_tool/migrations_trim_sync.json'), 'utf8'),
) as {
  wire: { push: { answer_keys: string[] }; pull: { answer_keys: string[] }; max_pages_per_pull: number };
  cursor: { pattern: string; retired_state_file: Record<string, unknown>; retired_state_file_keeps: Record<string, string> };
  scenario: {
    namespace: string;
    agent_writes: Array<{ key: string; content: string }>;
    owner_edit: { key: string; content: string };
    owner_forget: { key: string };
    local_after: Record<string, string>;
    local_gone: string[];
  };
};
const CURSOR = new RegExp(FIXTURE.cursor.pattern);

function skillFor(portal: ReturnType<typeof fakePortal>, dir: string, now = '2026-09-27T09:00:00.000Z') {
  return new MemorySkill({
    local: true,
    portal: true,
    agentDir: dir,
    agentId: portal.agentId,
    portalUrl: 'https://portal.test',
    apiKey: 'agent-key',
    fetchImpl: portal.fetch,
    plainIndex: true,
    now: () => new Date(now),
  });
}

const stateOf = (dir: string) => JSON.parse(fs.readFileSync(path.join(dir, '.webagents', 'memory', 'sync-state.json'), 'utf8')) as { pushedSeq: number; cursors: Record<string, unknown> };
const syncGets = (portal: ReturnType<typeof fakePortal>) => portal.seen.filter((r) => r.method === 'GET' && r.url.searchParams.get('action') === 'sync');

describe('S-307: the scenario of the shared fixture', () => {
  it('a second pull after a push sees the owner\'s later edit and forget', async () => {
    const S = FIXTURE.scenario;
    const portal = fakePortal();
    const dir = tempDir('mt-sync-scenario-');
    const skill = skillFor(portal, dir);
    await skill.initialize();
    for (const w of S.agent_writes) await skill.memoryWrite({ key: w.key, content: w.content }, ctx(OWNER));
    expect([...portal.rows.values()].map((r) => `${r.key}=${r.content}`).sort()).toEqual(S.agent_writes.map((w) => `${w.key}=${w.content}`).sort());
    // The pushes answered no cursor, and none was kept.
    for (const post of portal.seen.filter((r) => r.method === 'POST')) expect(post.url.searchParams.get('action')).toBe('sync');
    expect(stateOf(dir).cursors).toEqual({});

    await skill.pull([S.namespace]);
    const first = stateOf(dir).cursors[S.namespace];
    expect(first).toMatch(CURSOR);

    // On Robutler, after the agent wrote: the owner edits one note and forgets the other.
    portal.put(S.namespace, S.owner_edit.key, S.owner_edit.content, 'owner', '2026-09-27T10:00:00.000Z');
    portal.del(S.namespace, S.owner_forget.key, '2026-09-27T10:00:01.000Z');

    await skill.pull([S.namespace]);
    const gets = syncGets(portal);
    expect(gets.at(-1)!.url.searchParams.get('since')).toBe(first);
    for (const [key, content] of Object.entries(S.local_after)) {
      const entries = (await skill.memoryList({ namespace: S.namespace }, ctx(OWNER))) as { entries: Array<{ key: string }> };
      expect(entries.entries.map((e) => e.key)).toContain(key);
      expect(fs.readFileSync(path.join(dir, '.webagents', 'memory', S.namespace, `${key}.md`), 'utf8')).toContain(content);
    }
    for (const key of S.local_gone) expect(fs.existsSync(path.join(dir, '.webagents', 'memory', S.namespace, `${key}.md`))).toBe(false);
    const second = stateOf(dir).cursors[S.namespace];
    expect(second).toMatch(CURSOR);
    expect(second > (first as string)).toBe(true);
    // What was pulled is not pushed back.
    expect(portal.seen.filter((r) => r.method === 'POST')).toHaveLength(S.agent_writes.length);
  });
});

describe('the cursor', () => {
  it('the store sends `since` only when it has one, reads the page, and a push answers no cursor', async () => {
    const portal = fakePortal();
    const store = new PortalMemoryStore({ portalUrl: 'https://portal.test', token: () => 'agent-key', agentId: portal.agentId, fetchImpl: portal.fetch });
    const empty = await store.pull('owner', null);
    expect(empty).toEqual({ lines: [], cursor: null, more: false });
    expect(syncGets(portal)[0].url.searchParams.has('since')).toBe(false);
    const pushed = await store.push('owner', [{ seq: 1, op: 'put', id: 'x', namespace: 'owner', key: 'plan', content: 'October', source: 'tool', at: '2026-09-27T09:00:00.000Z' }]);
    expect(Object.keys(pushed)).toEqual(FIXTURE.wire.push.answer_keys);
    const page = await store.pull('owner', null);
    expect(Object.keys(page).sort()).toEqual([...FIXTURE.wire.pull.answer_keys].sort());
    expect(page.lines.map((l) => l.key)).toEqual(['plan']);
    expect(page.cursor).toMatch(CURSOR);
    await store.pull('owner', page.cursor);
    expect(syncGets(portal).at(-1)!.url.searchParams.get('since')).toBe(page.cursor);
  });

  it('drops the numeric cursors an older state file holds (S-307\'s stuck one included), so those namespaces pull from the start', async () => {
    const portal = fakePortal();
    const dir = tempDir('mt-sync-retired-');
    fs.mkdirSync(path.join(dir, '.webagents', 'memory'), { recursive: true });
    fs.writeFileSync(path.join(dir, '.webagents', 'memory', 'sync-state.json'), JSON.stringify(FIXTURE.cursor.retired_state_file));
    portal.put('owner', 'plan', 'November', 'owner', '2026-09-27T10:00:00.000Z');
    const skill = skillFor(portal, dir);
    await skill.initialize();
    await skill.pull(['owner', 'shared']);
    for (const get of syncGets(portal)) expect(get.url.searchParams.has('since')).toBe(false);
    expect(fs.readFileSync(path.join(dir, '.webagents', 'memory', 'owner', 'plan.md'), 'utf8')).toContain('November');
    const cursors = stateOf(dir).cursors;
    expect(cursors.owner).toMatch(CURSOR);
    expect('shared' in cursors).toBe(false); // nothing served there, nothing kept
    expect(Object.keys(FIXTURE.cursor.retired_state_file_keeps)).toEqual([]);
  });

  it('a pull takes every page in one go, and the pages join without a gap', async () => {
    const portal = fakePortal(undefined, { pageSize: 2 });
    for (let i = 0; i < 5; i += 1) portal.put('shared', `k${i}`, `note ${i}`, 'owner', '2026-09-27T10:00:00.000Z');
    const dir = tempDir('mt-sync-pages-');
    const skill = skillFor(portal, dir);
    await skill.initialize();
    await skill.pull(['shared']);
    expect(syncGets(portal)).toHaveLength(3);
    for (let i = 0; i < 5; i += 1) expect(fs.existsSync(path.join(dir, '.webagents', 'memory', 'shared', `k${i}.md`))).toBe(true);
    expect(FIXTURE.wire.max_pages_per_pull).toBe(20);
  });
});
