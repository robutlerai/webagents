/**
 * Sessions, the same in both SDKs (2026-09-25). The cases are
 * `python/tests/fixtures/sessions/sessions.json`, which the Python suite runs
 * too (`tests/cli/test_session_fixture.py`): where a served agent keeps a
 * caller's conversation, which session a request names, what is kept of a
 * turn, what the chat records on Robutler, how `/resume` merges this
 * machine's list with Robutler's, and the sentences both say.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { callerKey } from '../../../src/cli/sessions';
import {
  failureSentence,
  mergeConversations,
  unavailableReason,
  wordsSince,
  type PlatformConversation,
} from '../../../src/cli/robutler-sessions';
import {
  conversationOwner,
  conversationToKeep,
  requestSessionId,
  sessionBackendOf,
} from '../../../src/skills/session/skill';
import type { AuthInfo } from '../../../src/core/types';

const HERE = path.dirname(fileURLToPath(import.meta.url));

interface OwnerCase {
  case: string;
  tier: 'owner' | 'admin' | null;
  principals: string[] | null;
  user_id: string | null;
  whose: string | null;
}
interface MergeRow {
  id: string | null;
  chat_id: string | null;
  session_id: string | null;
  updated_at: string;
  local_count: number;
  platform_count: number;
  preview: string;
  only_on_robutler: boolean;
}

const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, '../../../../python/tests/fixtures/sessions/sessions.json'), 'utf8'),
) as {
  caller_keys: { principal: string; key: string }[];
  owners: OwnerCase[];
  session_ids: { metadata: unknown; id: string | null }[];
  keep: { case: string; request: { role: string; content: unknown }[]; reply: unknown; kept: unknown[] }[];
  words_since: { case: string; messages: { role: string; content: unknown }[]; from: number; words: unknown[] }[];
  merge: {
    case: string;
    local: { id: string; updated_at: string; message_count: number; preview: string; chat_id: string | null }[];
    platform: PlatformConversation[];
    rows: MergeRow[];
  }[];
  sentences: Record<string, string>;
};

const command = (rest = '') => `webagents ${rest}`.trim();

/** The caller in this SDK's terms: no `principals` when no access block ran. */
function auth(c: OwnerCase): Partial<AuthInfo> {
  const anonymous = !c.tier && !c.principals && !c.user_id;
  return {
    authenticated: !anonymous,
    provider: 'platform',
    scopes: c.tier ? [c.tier] : [],
    ...(c.user_id ? { user_id: c.user_id } : {}),
    ...(c.principals !== null ? { principals: c.principals } : {}),
  } as Partial<AuthInfo>;
}

describe('sessions (shared fixture)', () => {
  it.each(FIXTURE.caller_keys)('the directory for $principal', ({ principal, key }) => {
    expect(callerKey(principal)).toBe(key);
  });

  it.each(FIXTURE.owners)('whose conversation: $case', (c) => {
    expect(conversationOwner(auth(c))).toBe(c.whose);
  });

  it('no auth at all is not kept', () => {
    expect(conversationOwner(undefined)).toBeNull();
  });

  it.each(FIXTURE.session_ids)('the session a request names: %j', ({ metadata, id }) => {
    expect(requestSessionId(metadata) ?? null).toBe(id);
  });

  it.each(FIXTURE.keep)('what is kept of a turn: $case', ({ request, reply, kept }) => {
    expect(conversationToKeep(request, reply)).toEqual(kept);
  });

  it.each(FIXTURE.words_since)('what a turn records on Robutler: $case', ({ messages, from, words }) => {
    expect(wordsSince(messages, from)).toEqual(words);
  });

  it.each(FIXTURE.merge)('resume merges: $case', ({ local, platform, rows }) => {
    const merged = mergeConversations(
      local.map((l) => ({
        id: l.id,
        updatedAt: l.updated_at,
        messageCount: l.message_count,
        preview: l.preview,
        ...(l.chat_id ? { chatId: l.chat_id } : {}),
      })),
      platform,
    );
    expect(
      merged.map((r) => ({
        id: r.id ?? null,
        chat_id: r.chatId ?? null,
        session_id: r.sessionId ?? null,
        updated_at: r.updatedAt,
        local_count: r.localCount,
        platform_count: r.platformCount,
        preview: r.preview,
        only_on_robutler: r.onlyOnRobutler,
      })),
    ).toEqual(rows);
  });

  it('the sentences', () => {
    const said = FIXTURE.sentences;
    expect(unavailableReason('signed_out', command)).toBe(said.signed_out);
    expect(unavailableReason('not_published', command)).toBe(said.not_published);
    for (const status of [401, 404, 0, 502]) expect(failureSentence(status, command)).toBe(said[`failure_${status}`]);
    expect(() => sessionBackendOf('cloud')).toThrow(said.backend);
    expect(sessionBackendOf(undefined)).toBe('local');
    expect(sessionBackendOf('robutler')).toBe('robutler');
  });
});
