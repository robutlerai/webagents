/**
 * Compaction through the skill's own hook (plan items 2.1 and 2.4): past
 * the threshold the working conversation is rewritten in place, the summary
 * becomes an episode in the CALLER's namespace, a summarizing run of the
 * agent's own model is left alone, and the frozen notes stay frozen for the
 * session. Deterministic under a stub model; the Python twin is
 * `python/tests/agents/skills/test_memory_compaction_w2mem.py`.
 */

import { describe, expect, it } from 'vitest';

import { MemorySkill } from '../../../../src/skills/memory/skill';
import { COMPACTION_PREFIX } from '../../../../src/skills/memory/compaction';
import { tempDirs } from '../../../helpers/cli';

const tempDir = tempDirs();

const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ALICE = { authenticated: true, provider: 'platform', scopes: [], user_id: 'alice' };

const ctx = (auth: unknown, metadata: Record<string, unknown> = {}) => {
  const data = new Map<string, unknown>();
  return {
    auth,
    metadata,
    get: (k: string) => data.get(k),
    set: (k: string, v: unknown) => data.set(k, v),
  } as never;
};

const stub = async (transcript: string) => `[stub summary of ${transcript.split('\n').length} lines]`;

function longConversation() {
  return [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Plan the launch.' },
    { role: 'assistant', content: 'Which week?' },
    { role: 'user', content: 'The first of October.' },
    { role: 'assistant', content: 'Noted.' },
    { role: 'user', content: 'Book a venue.' },
    { role: 'assistant', content: 'Done.' },
  ];
}

describe('compaction through the hook', () => {
  it('rewrites the working conversation in place and keeps the summary as the caller\'s episode', async () => {
    let calls = 0;
    const skill = new MemorySkill({
      agentDir: tempDir('mem-compact-'),
      agentName: 'helper',
      compaction: { threshold: 20, keep: 2 },
      summarize: async (t, i) => {
        calls += 1;
        expect(i).toMatch(/^Summarize the conversation below/);
        return stub(t);
      },
      now: () => new Date('2026-09-26T10:00:00.000Z'),
    });
    await skill.initialize();
    const conversation = longConversation();
    const context = ctx(ALICE);
    (context as { set: (k: string, v: unknown) => void }).set('_agentic_messages', conversation);

    await skill.compactBeforeCall({}, context);

    expect(calls).toBe(1);
    expect(conversation).toEqual([
      { role: 'system', content: 'You are helpful.' },
      { role: 'system', content: `${COMPACTION_PREFIX}[stub summary of 4 lines]` },
      { role: 'user', content: 'Book a venue.' },
      { role: 'assistant', content: 'Done.' },
    ]);
    expect(skill.compactions).toBe(1);

    const alice = (await skill.memoryList({}, ctx(ALICE))) as { entries: Array<{ key: string; namespace: string }> };
    expect(alice.entries).toEqual([{ key: 'episode-2026-09-26T10-00-00-000Z', namespace: 'caller:user:alice', updated_at: '2026-09-26T10:00:00.000Z' }]);
    const found = (await skill.memorySearch({ query: 'stub summary' }, ctx(ALICE))) as { entries: Array<{ content: string }> };
    expect(found.entries.map((e) => e.content)).toEqual(['[stub summary of 4 lines]']);
    // The owner's memory did not gain Alice's episode, and Alice's notes prompt does not carry episodes.
    expect(((await skill.memoryList({ namespace: 'owner' }, ctx(OWNER))) as { entries: unknown[] }).entries).toEqual([]);
    skill.unfreezeNotes();
    expect(await skill.frozenNotes(ctx(ALICE))).toBe('');
  });

  it('does nothing under the threshold, and stays out of its own summarizing run', async () => {
    let calls = 0;
    const skill = new MemorySkill({
      agentDir: tempDir('mem-compact-quiet-'),
      agentName: 'helper',
      compaction: { threshold: 20, keep: 2 },
      summarize: async (t) => {
        calls += 1;
        return stub(t);
      },
    });
    await skill.initialize();
    const short = [{ role: 'system', content: 'You are helpful.' }, { role: 'user', content: 'Hi' }];
    const quiet = ctx(ALICE);
    (quiet as { set: (k: string, v: unknown) => void }).set('_agentic_messages', short);
    await skill.compactBeforeCall({}, quiet);
    expect(short).toHaveLength(2);

    const nested = ctx(ALICE, { memory_compaction: true });
    const long = longConversation();
    (nested as { set: (k: string, v: unknown) => void }).set('_agentic_messages', long);
    await skill.compactBeforeCall({}, nested);
    expect(long).toHaveLength(7);
    expect(await skill.frozenNotes(nested)).toBe('');
    expect(calls).toBe(0);
  });

  it('a summary that comes back empty leaves the conversation as it was', async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-compact-empty-'), agentName: 'helper', compaction: { threshold: 20, keep: 2 }, summarize: async () => '' });
    await skill.initialize();
    const conversation = longConversation();
    const context = ctx(ALICE);
    (context as { set: (k: string, v: unknown) => void }).set('_agentic_messages', conversation);
    await skill.compactBeforeCall({}, context);
    expect(conversation).toHaveLength(7);
    expect(skill.compactions).toBe(0);
  });
});

describe('the frozen notes', () => {
  it('are computed once per session and never updated mid-session', async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-notes-'), agentName: 'helper', notes_budget: 400 });
    await skill.initialize();
    await skill.memoryWrite({ key: 'office-hours', content: '9 to 5.', namespace: 'shared' }, ctx(OWNER));
    await skill.memoryWrite({ key: 'name', content: 'Ada.' }, ctx(ALICE));

    const session = { session_id: 'sess-1' };
    const first = await skill.frozenNotes(ctx(ALICE, session));
    expect(first).toBe('## Memory\nShared notes:\n- office-hours: 9 to 5.\nNotes about this caller:\n- name: Ada.');

    await skill.memoryWrite({ key: 'tone', content: 'Formal.' }, ctx(ALICE));
    expect(await skill.frozenNotes(ctx(ALICE, session))).toBe(first);
    // A new session sees the new note; the owner sees the owner's sections; Bob sees nothing of Alice's.
    expect(await skill.frozenNotes(ctx(ALICE, { session_id: 'sess-2' }))).toContain('- tone: Formal.');
    expect(await skill.frozenNotes(ctx(OWNER, session))).toBe('## Memory\nShared notes:\n- office-hours: 9 to 5.');
    const bob = { authenticated: true, provider: 'platform', scopes: [], user_id: 'bob' };
    expect(await skill.frozenNotes(ctx(bob, session))).toBe('## Memory\nShared notes:\n- office-hours: 9 to 5.');
  });
});
