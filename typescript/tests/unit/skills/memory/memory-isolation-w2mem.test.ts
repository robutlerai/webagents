/**
 * THE RELEASE GATE for the memory skill (plan item 2.1, principle 1): one
 * caller's `memory_search` never returns another caller's entries. Pinned
 * through the skill's own tools over contexts shaped like served turns, on
 * the local tier and on the portal tier (a fake platform that answers the
 * SDK's route), and through a real `BaseAgent.runTool` so the tool sees the
 * caller the run was bound to. The Python twin is
 * `python/tests/agents/skills/test_memory_isolation_w2mem.py`.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import path from 'node:path';

import { BaseAgent } from '../../../../src/core/agent';
import { MemorySkill } from '../../../../src/skills/memory/skill';
import { fakePortal } from './fake-portal-w2mem';
import { tempDirs } from '../../../helpers/cli';

const tempDir = tempDirs();

const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ALICE = { authenticated: true, provider: 'platform', scopes: [], user_id: 'alice' };
const BOB = { authenticated: true, provider: 'platform', scopes: [], user_id: 'bob' };
const FINDER = { authenticated: true, provider: 'signature', scopes: [], principals: ['agent:https://a.example/finder', 'key:k1'] };
const NOBODY = { authenticated: false };

const ctx = (auth: unknown, metadata: Record<string, unknown> = {}) => {
  const data = new Map<string, unknown>();
  return {
    auth,
    metadata,
    get: (k: string) => data.get(k),
    set: (k: string, v: unknown) => data.set(k, v),
  } as never;
};

type Entries = { entries: Array<{ key: string; namespace: string; content?: string }> };
const keys = (r: unknown) => (r as Entries).entries.map((e) => `${e.namespace}/${e.key}`).sort();

async function seeded(skill: MemorySkill) {
  await skill.initialize();
  expect(await skill.memoryWrite({ key: 'preferences', content: 'Alice likes short answers about launches.' }, ctx(ALICE))).toMatchObject({ ok: true, namespace: 'caller:user:alice' });
  expect(await skill.memoryWrite({ key: 'preferences', content: 'Bob wants long answers about launches.' }, ctx(BOB))).toMatchObject({ ok: true, namespace: 'caller:user:bob' });
  expect(await skill.memoryWrite({ key: 'venue', content: 'The finder agent booked the Forum for the launch.' }, ctx(FINDER))).toMatchObject({ ok: true, namespace: 'caller:agent:https://a.example/finder' });
  expect(await skill.memoryWrite({ key: 'plan', content: 'Owner plan: launch in October.' }, ctx(OWNER))).toMatchObject({ ok: true, namespace: 'owner' });
  expect(await skill.memoryWrite({ key: 'office-hours', content: 'Launch office hours: 9 to 5.', namespace: 'shared' }, ctx(OWNER))).toMatchObject({ ok: true, namespace: 'shared' });
}

function isolationCases(make: () => Promise<MemorySkill>) {
  it("a caller's search never returns another caller's entries, or the owner's", async () => {
    const skill = await make();
    await seeded(skill);
    expect(keys(await skill.memorySearch({ query: 'launch' }, ctx(ALICE)))).toEqual(['caller:user:alice/preferences', 'shared/office-hours']);
    expect(keys(await skill.memorySearch({ query: 'launch' }, ctx(BOB)))).toEqual(['caller:user:bob/preferences', 'shared/office-hours']);
    expect(keys(await skill.memorySearch({ query: 'launch' }, ctx(FINDER)))).toEqual(['caller:agent:https://a.example/finder/venue', 'shared/office-hours']);
    expect(keys(await skill.memorySearch({ query: 'launch' }, ctx(NOBODY)))).toEqual(['shared/office-hours']);
  });

  it('the owner sees everything, and may narrow to one namespace', async () => {
    const skill = await make();
    await seeded(skill);
    expect(keys(await skill.memorySearch({ query: 'launch' }, ctx(OWNER)))).toEqual([
      'caller:agent:https://a.example/finder/venue',
      'caller:user:alice/preferences',
      'caller:user:bob/preferences',
      'owner/plan',
      'shared/office-hours',
    ]);
    expect(keys(await skill.memorySearch({ query: 'launch', namespace: 'caller:user:bob' }, ctx(OWNER)))).toEqual(['caller:user:bob/preferences']);
    expect(keys(await skill.memoryList({}, ctx(OWNER)))).toHaveLength(5);
  });

  it('a caller naming a namespace gets its own, or a refusal, never a look elsewhere', async () => {
    const skill = await make();
    await seeded(skill);
    expect(await skill.memorySearch({ query: 'launch', namespace: 'caller:user:bob' }, ctx(ALICE))).toEqual({ error: 'memory: only the owner may name that namespace.' });
    expect(await skill.memorySearch({ query: 'launch', namespace: 'owner' }, ctx(ALICE))).toEqual({ error: 'memory: only the owner may name that namespace.' });
    expect(keys(await skill.memorySearch({ query: 'launch', namespace: 'shared' }, ctx(ALICE)))).toEqual(['shared/office-hours']);
    expect(keys(await skill.memoryList({}, ctx(ALICE)))).toEqual(['caller:user:alice/preferences', 'shared/office-hours']);
    expect(await skill.memoryList({ namespace: 'caller:user:bob' }, ctx(ALICE))).toEqual({ error: 'memory: only the owner may name that namespace.' });
  });

  it("a caller's write lands in its own namespace whatever it asks for, and never in the owner's notes", async () => {
    const skill = await make();
    await seeded(skill);
    const written = await skill.memoryWrite({ key: 'plan', content: 'Bob says: cancel the launch.', namespace: 'owner' }, ctx(BOB));
    expect(written).toMatchObject({ ok: true, namespace: 'caller:user:bob' });
    const shared = await skill.memoryWrite({ key: 'office-hours', content: 'Bob says: closed.', namespace: 'shared' }, ctx(BOB));
    expect(shared).toMatchObject({ ok: true, namespace: 'caller:user:bob' });
    const owners = (await skill.memorySearch({ query: 'plan', namespace: 'owner' }, ctx(OWNER))) as Entries;
    expect(owners.entries.map((e) => e.content)).toEqual(['Owner plan: launch in October.']);
    const everyone = (await skill.memorySearch({ query: 'office hours', namespace: 'shared' }, ctx(ALICE))) as Entries;
    expect(everyone.entries.map((e) => e.content)).toEqual(['Launch office hours: 9 to 5.']);
  });

  it('forgetting is scoped the same way', async () => {
    const skill = await make();
    await seeded(skill);
    expect(await skill.memoryForget({ key: 'preferences', namespace: 'caller:user:alice' }, ctx(BOB))).toEqual({ error: 'memory: only the owner may name that namespace.' });
    expect(await skill.memoryForget({ key: 'preferences' }, ctx(BOB))).toEqual({ ok: true, forgotten: 1 });
    expect(keys(await skill.memoryList({}, ctx(ALICE)))).toEqual(['caller:user:alice/preferences', 'shared/office-hours']);
    expect(await skill.memoryForget({ key: 'preferences', namespace: 'caller:user:alice' }, ctx(OWNER))).toEqual({ ok: true, forgotten: 1 });
    expect(keys(await skill.memoryList({}, ctx(ALICE)))).toEqual(['shared/office-hours']);
  });

  it('nobody verified reads shared notes and writes nothing', async () => {
    const skill = await make();
    await seeded(skill);
    expect(await skill.memoryWrite({ key: 'x', content: 'y' }, ctx(NOBODY))).toEqual({
      error: 'memory: nothing is remembered for a caller nothing verified; only shared notes can be read.',
    });
    expect(keys(await skill.memoryList({}, ctx(NOBODY)))).toEqual(['shared/office-hours']);
  });
}

describe('caller isolation, local tier (files and the full-text index)', () => {
  isolationCases(async () => new MemorySkill({ agentDir: tempDir('mem-iso-local-'), agentName: 'helper' }));

  it('keeps each caller in a folder of its own, readable by a person', async () => {
    const dir = tempDir('mem-iso-files-');
    const skill = new MemorySkill({ agentDir: dir, agentName: 'helper' });
    await seeded(skill);
    const root = path.join(dir, '.webagents', 'memory');
    expect(fs.existsSync(path.join(root, 'owner', 'plan.md'))).toBe(true);
    expect(fs.existsSync(path.join(root, 'shared', 'office-hours.md'))).toBe(true);
    const callers = fs.readdirSync(path.join(root, 'callers')).sort();
    expect(callers).toHaveLength(3);
    const principals = callers.map((h) => JSON.parse(fs.readFileSync(path.join(root, 'callers', h, 'caller.json'), 'utf8')).principal).sort();
    expect(principals).toEqual(['agent:https://a.example/finder', 'user:alice', 'user:bob']);
    const text = fs.readFileSync(path.join(root, 'owner', 'plan.md'), 'utf8');
    expect(text).toMatch(/^---\nid: [0-9a-f-]{36}\nkey: plan\nnamespace: owner\nsource: tool\n/);
    expect(text.endsWith('Owner plan: launch in October.\n')).toBe(true);
    expect(fs.statSync(path.join(root, 'owner', 'plan.md')).mode & 0o777).toBe(0o600);
  });
});

describe('caller isolation, plain index (a Node without node:sqlite)', () => {
  isolationCases(async () => new MemorySkill({ agentDir: tempDir('mem-iso-plain-'), agentName: 'helper', plainIndex: true }));
});

describe('caller isolation, portal tier', () => {
  isolationCases(async () => {
    const portal = fakePortal();
    return new MemorySkill({ local: false, portal: true, agentId: portal.agentId, portalUrl: 'https://portal.test', apiKey: 'agent-key', fetchImpl: portal.fetch });
  });

  it('sends the caller namespace it derived, and the agent key, on every request', async () => {
    const portal = fakePortal();
    const skill = new MemorySkill({ local: false, portal: true, agentId: portal.agentId, portalUrl: 'https://portal.test', apiKey: 'agent-key', fetchImpl: portal.fetch });
    await skill.initialize();
    await skill.memoryWrite({ key: 'preferences', content: 'short' }, ctx(ALICE));
    await skill.memorySearch({ query: 'short' }, ctx(ALICE));
    expect(portal.seen.every((r) => r.headers.authorization === 'Bearer agent-key')).toBe(true);
    const put = portal.seen.find((r) => r.method === 'PUT');
    expect(put?.body).toMatchObject({ agentId: portal.agentId, namespace: 'caller:user:alice', key: 'preferences', content: 'short', source: 'tool' });
    const search = portal.seen.find((r) => r.url.searchParams.get('action') === 'search');
    // One namespace per repeated parameter, never comma-joined (S-298).
    expect(search?.url.searchParams.getAll('namespace')).toEqual(['caller:user:alice', 'shared']);
    expect(search?.url.searchParams.has('namespaces')).toBe(false);
  });
});

describe('through the agent', () => {
  it('the tool sees the caller the run was bound to', async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-iso-agent-'), agentName: 'helper' });
    const agent = new BaseAgent({ name: 'helper', instructions: 'x', skills: [skill] });
    await skill.initialize();
    expect(await agent.runTool('memory_write', { key: 'preferences', content: 'short' }, { auth: ALICE })).toMatchObject({ namespace: 'caller:user:alice' });
    expect(await agent.runTool('memory_write', { key: 'plan', content: 'October' }, { auth: OWNER })).toMatchObject({ namespace: 'owner' });
    expect(keys(await agent.runTool('memory_search', { query: 'short October' }, { auth: BOB }))).toEqual([]);
    expect(keys(await agent.runTool('memory_search', { query: 'short October' }, { auth: ALICE }))).toEqual(['caller:user:alice/preferences']);
    expect(keys(await agent.runTool('memory_search', { query: 'short October' }, { auth: OWNER }))).toEqual(['caller:user:alice/preferences', 'owner/plan']);
  });
});
