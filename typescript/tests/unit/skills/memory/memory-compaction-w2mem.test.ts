/**
 * The memory skill's part in compaction (2026-09-29): compaction is the
 * agent's (`core/context-compaction.ts`), and the skill keeps each summary as
 * an episode in the CALLER's namespace, once (`onCompaction`); the owner's
 * between turns in the chat, where there is no run; nothing for a compaction
 * that made no summary; and a summary run is left alone. Through the real
 * agent: one compaction, one episode, and nothing more on the next turn. Also
 * here: the frozen notes stay frozen for the session. The Python twin is
 * `python/tests/agents/skills/test_memory_compaction_w2mem.py`.
 */

import { describe, expect, it } from 'vitest';

import { BaseAgent } from '../../../../src/core/agent';
import { COMPACTION_RUN, DEFAULT_POLICY, WORDS, type CompactMessage } from '../../../../src/core/context-compaction';
import { MemorySkill } from '../../../../src/skills/memory/skill';
import { tempDirs } from '../../../helpers/cli';

const tempDir = tempDirs();

const OWNER = { authenticated: true, provider: 'platform', scopes: ['owner'], user_id: 'owner-1' };
const ALICE = { authenticated: true, provider: 'platform', scopes: [], user_id: 'alice' };
const NOW = () => new Date('2026-09-26T10:00:00.000Z');

const ctx = (auth: unknown, metadata: Record<string, unknown> = {}) => {
  const data = new Map<string, unknown>();
  return {
    auth,
    metadata,
    get: (k: string) => data.get(k),
    set: (k: string, v: unknown) => data.set(k, v),
  } as never;
};

type Listing = { entries: Array<{ key: string; namespace: string; content?: string }> };

describe("the memory skill's part in compaction", () => {
  it("keeps the summary as the caller's episode", async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-compact-'), agentName: 'helper', now: NOW });
    await skill.initialize();
    await skill.onCompaction({ summary: '[stub summary of 4 lines]' }, ctx(ALICE));
    expect(skill.compactions).toBe(1);
    const alice = (await skill.memoryList({}, ctx(ALICE))) as Listing;
    expect(alice.entries.map((e) => [e.key, e.namespace])).toEqual([['episode-2026-09-26T10-00-00-000Z', 'caller:user:alice']]);
    const found = (await skill.memorySearch({ query: 'stub summary' }, ctx(ALICE))) as Listing;
    expect(found.entries.map((e) => e.content)).toEqual(['[stub summary of 4 lines]']);
    expect(((await skill.memoryList({ namespace: 'owner' }, ctx(OWNER))) as Listing).entries).toEqual([]);
    skill.unfreezeNotes();
    expect(await skill.frozenNotes(ctx(ALICE))).toBe('');
  });

  it("between turns in the chat, the episode is the owner's", async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-compact-'), agentName: 'helper', now: NOW });
    await skill.initialize();
    await skill.onCompaction({ summary: "the chat's summary" }, undefined);
    expect(((await skill.memoryList({ namespace: 'owner' }, ctx(OWNER))) as Listing).entries.map((e) => e.key)).toEqual(['episode-2026-09-26T10-00-00-000Z']);
  });

  it('keeps nothing for a compaction that made no summary', async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-compact-'), agentName: 'helper' });
    await skill.initialize();
    await skill.onCompaction({}, ctx(ALICE));
    expect(((await skill.memoryList({}, ctx(ALICE))) as Listing).entries).toEqual([]);
    expect(skill.compactions).toBe(0);
  });

  it('leaves a summary run alone', async () => {
    const skill = new MemorySkill({ agentDir: tempDir('mem-compact-'), agentName: 'helper' });
    await skill.initialize();
    await skill.memoryWrite({ key: 'office-hours', content: '9 to 5.', namespace: 'shared' }, ctx(OWNER));
    expect(await skill.frozenNotes(ctx(ALICE, { [COMPACTION_RUN]: true }))).toBe('');
  });

  it('through the agent: one compaction keeps one episode, and the next turn adds none', async () => {
    const memory = new MemorySkill({ agentDir: tempDir('mem-compact-'), agentName: 'helper', now: NOW });
    const agent = new BaseAgent({ name: 'helper', instructions: 'x', skills: [memory] });
    await agent.initialize();
    const seen: Array<{ messages: Array<{ content: string }>; options: Record<string, unknown> }> = [];
    (agent as unknown as { run: unknown }).run = async (messages: Array<{ content: string }>, options: Record<string, unknown>) => {
      seen.push({ messages, options });
      return { content: '  the summary  ' };
    };
    agent.compactionPolicy = { ...DEFAULT_POLICY, at: 60, keep: 10, hard: 90 };
    const conversation: CompactMessage[] = [
      { role: 'user', content: 'Plan the launch for the first of October, with a venue.' },
      { role: 'assistant', content: 'Which city, and how many people are coming to it?' },
      { role: 'user', content: 'Paris, about two hundred people.' },
      { role: 'assistant', content: 'Noted: Paris, two hundred.' },
      { role: 'user', content: 'Book a venue.' },
      { role: 'assistant', content: 'Done.' },
    ];
    const outcome = await agent.compactIfNeeded(conversation);
    expect(outcome.stage).toBe('summarized');
    expect(outcome.summary).toBe('the summary');
    expect(outcome.messages[0].content).toBe(`${WORDS.summaryPrefix}the summary`);
    expect(agent.lastCompaction).toBe(outcome);
    expect(memory.compactions).toBe(1);
    expect(seen[0].options).toMatchObject({ metadata: { [COMPACTION_RUN]: true } });
    expect(seen[0].messages[0].content).toContain('user: Plan the launch');
    expect((await agent.compactIfNeeded(outcome.messages)).stage).toBe('none');
    expect(memory.compactions).toBe(1);
    const forced = await agent.compact(conversation, { focus: 'the venue' });
    expect(forced.stage).toBe('summarized');
    expect(seen.at(-1)?.messages[0].content).toContain('Pay particular attention to: the venue');
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
    expect(first).toBe('## Memory\nOne line per note you keep, newest first. memory_read gives a note in full; memory_write keeps one, with a one-line description.\nShared notes:\n- office-hours: 9 to 5.\nNotes about this caller:\n- name: Ada.');

    await skill.memoryWrite({ key: 'tone', content: 'Formal.' }, ctx(ALICE));
    expect(await skill.frozenNotes(ctx(ALICE, session))).toBe(first);
    // A new session sees the new note; the owner sees the owner's sections; Bob sees nothing of Alice's.
    expect(await skill.frozenNotes(ctx(ALICE, { session_id: 'sess-2' }))).toContain('- tone: Formal.');
    expect(await skill.frozenNotes(ctx(OWNER, session))).toBe('## Memory\nOne line per note you keep, newest first. memory_read gives a note in full; memory_write keeps one, with a one-line description.\nShared notes:\n- office-hours: 9 to 5.');
    const bob = { authenticated: true, provider: 'platform', scopes: [], user_id: 'bob' };
    expect(await skill.frozenNotes(ctx(bob, session))).toBe('## Memory\nOne line per note you keep, newest first. memory_read gives a note in full; memory_write keeps one, with a one-line description.\nShared notes:\n- office-hours: 9 to 5.');
  });
});
