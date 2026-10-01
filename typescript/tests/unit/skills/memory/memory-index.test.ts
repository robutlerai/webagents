/**
 * The memory index (2026-09-29, the owner: "is it index based like in
 * claude?"), the same in the Python skill
 * (`python/tests/agents/skills/test_memory_index.py`).
 *
 * The prompt's notes are now an index, as Claude Code's `MEMORY.md` is: one
 * line per note, its key and a one-line description, and `memory_read` gives
 * a note in full. Pinned here: `memory_write` keeps a description (in the note
 * file's front matter, one line); `memory_list`, `memory_search` and
 * `memory_read` give it back; search finds a note by it; it survives the store
 * reopening and the sync; the index shows it, or the note's first line without
 * one; and `memory_read` reads only what the caller may.
 */

import { describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { LocalMemoryStore } from '../../../../src/skills/memory/local-store';
import { NOTES_GUIDE, NOTES_HEADING } from '../../../../src/skills/memory/notes';
import { MemorySkill } from '../../../../src/skills/memory/skill';
import { tempDirs } from '../../../helpers/cli';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(HERE, '../../../../../python/tests/fixtures/memory_tool/definition.json'), 'utf8')) as {
  notes: { heading: string; guide: string };
};
const tempDir = tempDirs();
const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ALICE = { authenticated: true, provider: 'platform', scopes: [], user_id: 'alice' };
const BOB = { authenticated: true, provider: 'platform', scopes: [], user_id: 'bob' };
const NOT_YOURS = 'memory: only the owner may name that namespace.';

const ctx = (auth: unknown, metadata: Record<string, unknown> = {}) => {
  const data = new Map<string, unknown>();
  return { auth, metadata, get: (k: string) => data.get(k), set: (k: string, v: unknown) => data.set(k, v) } as never;
};

type Any = Record<string, unknown> & { entries?: Array<Record<string, unknown>> };

async function skillIn(dir: string): Promise<MemorySkill> {
  const skill = new MemorySkill({ agentDir: dir, agentName: 'helper', now: () => new Date('2026-09-29T10:00:00.000Z') });
  await skill.initialize();
  return skill;
}

describe('the memory index', () => {
  it('has the fixture words', () => {
    expect(NOTES_HEADING).toBe(FIXTURE.notes.heading);
    expect(NOTES_GUIDE).toBe(FIXTURE.notes.guide);
  });

  it('keeps a description, gives it back and finds by it', async () => {
    const dir = tempDir('mem-index-');
    const skill = await skillIn(dir);
    const wrote = (await skill.memoryWrite(
      { key: 'launch', content: 'The launch is on 1 October.\nVenue: the Forum.', description: 'When and where\nthe launch is' },
      ctx(OWNER),
    )) as Any;
    expect(wrote.ok).toBe(true);
    const text = fs.readFileSync(path.join(dir, '.webagents', 'memory', 'owner', 'launch.md'), 'utf8');
    expect(text).toContain('\ndescription: When and where the launch is\n');
    expect(((await skill.memoryList({}, ctx(OWNER))) as Any).entries?.[0].description).toBe('When and where the launch is');
    const read = (await skill.memoryRead({ key: 'launch' }, ctx(OWNER))) as Any;
    expect([read.key, read.namespace, read.description]).toEqual(['launch', 'owner', 'When and where the launch is']);
    expect(read.content).toBe('The launch is on 1 October.\nVenue: the Forum.');
    expect(((await skill.memorySearch({ query: 'where' }, ctx(OWNER))) as Any).entries?.map((e) => e.key)).toEqual(['launch']);
    expect(await skill.memoryRead({ key: 'nope' }, ctx(OWNER))).toEqual({ error: 'memory: no note called nope.' });
  });

  it('shows descriptions, and first lines where there is none', async () => {
    const skill = await skillIn(tempDir('mem-index-'));
    await skill.memoryWrite({ key: 'launch', content: 'The launch is on 1 October.', description: 'When the launch is' }, ctx(OWNER));
    await skill.memoryWrite({ key: 'tone', content: '# Tone\nFormal, no emoji.' }, ctx(OWNER));
    const notes = await skill.frozenNotes(ctx(OWNER, { session_id: 's1' }));
    expect(notes.startsWith(`${NOTES_HEADING}\n${NOTES_GUIDE}\nYour notes (owner):\n`)).toBe(true);
    expect(notes).toContain('- launch: When the launch is');
    expect(notes).toContain('- tone: Tone');
    expect(notes).not.toContain('The launch is on 1 October.');
  });

  it('lets a caller read only its own and the shared notes', async () => {
    const skill = await skillIn(tempDir('mem-index-'));
    await skill.memoryWrite({ key: 'office-hours', content: '9 to 5.', namespace: 'shared' }, ctx(OWNER));
    await skill.memoryWrite({ key: 'name', content: 'Ada.', description: 'What to call her' }, ctx(ALICE));
    expect(((await skill.memoryRead({ key: 'office-hours' }, ctx(ALICE))) as Any).namespace).toBe('shared');
    expect(((await skill.memoryRead({ key: 'name' }, ctx(ALICE))) as Any).description).toBe('What to call her');
    expect(await skill.memoryRead({ key: 'name' }, ctx(BOB))).toEqual({ error: 'memory: no note called name.' });
    expect(await skill.memoryRead({ key: 'name', namespace: 'caller:user:alice' }, ctx(BOB))).toEqual({ error: NOT_YOURS });
    expect(((await skill.memoryRead({ key: 'name', namespace: 'caller:user:alice' }, ctx(OWNER))) as Any).content).toBe('Ada.');
  });

  it('keeps the description across reopening and the sync', async () => {
    const root = tempDir('mem-index-store-');
    const store = new LocalMemoryStore({ root: path.join(root, 'memory'), store: 'helper' });
    await store.open();
    await store.put('owner', 'launch', 'On 1 October.', 'tool', undefined, 'When the launch is');
    const line = store.logSince(0).at(-1) as { description?: string };
    store.close();
    const again = new LocalMemoryStore({ root: path.join(root, 'memory'), store: 'helper' });
    await again.open();
    expect((await again.get('owner', 'launch'))?.description).toBe('When the launch is');
    expect((await again.search('launch is', ['owner'])).map((e) => e.key)).toEqual(['launch']);
    expect(line.description).toBe('When the launch is');
    const other = new LocalMemoryStore({ root: path.join(root, 'other'), store: 'helper' });
    await other.open();
    await other.apply([
      { seq: 1, op: 'put', id: 'x', namespace: 'owner', key: 'venue', content: 'The Forum.', description: 'Where it is', source: 'tool', at: '2026-09-29T10:00:00.000Z' },
    ]);
    expect((await other.get('owner', 'venue'))?.description).toBe('Where it is');
  });
});
