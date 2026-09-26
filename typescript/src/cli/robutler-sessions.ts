/**
 * The chat's conversations on Robutler (2026-09-25): the `robutler` backend of
 * the session skill, as the chat uses it.
 *
 * ON ROBUTLER A CONVERSATION IS A CHAT. So a person's conversation with their
 * own agent in this terminal is recorded into their chat with that agent on
 * the platform (`/api/agents/{id}/conversations`, the portal's
 * `lib/messaging/recorded-conversations.ts`): their words as them, the
 * agent's replies as the agent, without waking the agent there or notifying
 * anyone. It shows in their chat list, and `/resume` on another machine, or
 * after a chat on the web, continues it.
 *
 * WHAT IT NEEDS: the person signed in (`webagents login`, the token the
 * request carries; the platform answers only the agent's owner) and the agent
 * on Robutler (`webagents publish`, whose folder link names its platform id).
 * Without either, conversations stay on this machine and the chat says why,
 * once. The Python chat's twin is `python/webagents/cli/robutler_sessions.py`;
 * `python/tests/fixtures/sessions/sessions.json` holds the words both say.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

/** One of the owner's conversations with the agent, as the platform lists it. */
export interface PlatformConversation {
  chatId: string;
  /** The session id this chat was recorded from; null for a chat started on the web. */
  sessionId: string | null;
  updatedAt: string;
  messageCount: number;
  preview: string;
}

/** Where and as whom: the platform, the person's token, the agent's platform id. */
export interface ConversationsTarget {
  base: string;
  token: string;
  agentId: string;
}

/** Why conversations cannot be kept on Robutler here, as the chat says it (`unavailableReason`). */
export type Unavailable = 'signed_out' | 'not_published';

/** The sentence the chat shows, once, when the `robutler` backend cannot work here. */
export function unavailableReason(why: Unavailable, command: (rest: string) => string): string {
  return why === 'signed_out'
    ? `Conversations stay on this machine: sign in with \`${command('login')}\` to keep them on Robutler too.`
    : `Conversations stay on this machine: publish this agent with \`${command('publish')}\` to keep them on Robutler too.`;
}

/** A refusal or failure from the platform, with the sentence to show. */
export class PlatformConversationsError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'PlatformConversationsError';
  }
}

/** The sentence for a platform answer that is not a success (both SDKs say the same). */
export function failureSentence(status: number, command: (rest: string) => string): string {
  if (status === 401) return `your sign-in has expired: run \`${command('login')}\``;
  if (status === 404) return 'Robutler does not know this agent as yours';
  if (status === 0) return 'Robutler could not be reached';
  return `Robutler answered ${status}`;
}

/**
 * The folder's link to its platform agent (`webagents publish` / `link`), when
 * it names THIS agent: `link.agentId` and `link.agentName` in the folder's own
 * `.webagents/config.json`.
 */
export function linkedPlatformAgent(folder: string, agentName: string): string | undefined {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(folder, '.webagents', 'config.json'), 'utf-8')) as Record<string, unknown>;
    const id = typeof data['link.agentId'] === 'string' ? (data['link.agentId'] as string) : undefined;
    const name = typeof data['link.agentName'] === 'string' ? (data['link.agentName'] as string) : undefined;
    if (!id) return undefined;
    // `alice.helper` is the agent `helper`; a link naming another agent in
    // this folder is not this one's.
    if (name && name !== agentName && !name.endsWith(`.${agentName}`)) return undefined;
    return id;
  } catch {
    return undefined;
  }
}

async function call<T>(target: ConversationsTarget, route: string, command: (rest: string) => string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${target.base}/api/agents/${encodeURIComponent(target.agentId)}/conversations${route}`, {
      ...init,
      headers: {
        Authorization: `Bearer ${target.token}`,
        ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      },
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    throw new PlatformConversationsError(failureSentence(0, command), 0);
  }
  if (!res.ok) throw new PlatformConversationsError(failureSentence(res.status, command), res.status);
  return (await res.json()) as T;
}

/** The owner's conversations with the agent, newest first. */
export async function listPlatformConversations(
  target: ConversationsTarget,
  command: (rest: string) => string,
): Promise<PlatformConversation[]> {
  const body = await call<{ conversations?: PlatformConversation[] }>(target, '?limit=20', command);
  return Array.isArray(body.conversations) ? body.conversations : [];
}

/** One conversation's words, oldest first. */
export async function readPlatformConversation(
  target: ConversationsTarget,
  chatId: string,
  command: (rest: string) => string,
): Promise<{ role: 'user' | 'assistant'; content: string }[]> {
  const body = await call<{ messages?: { role: 'user' | 'assistant'; content: string }[] }>(
    target,
    `/${encodeURIComponent(chatId)}`,
    command,
  );
  return Array.isArray(body.messages) ? body.messages : [];
}

/** Record messages into the owner's chat with the agent; answers the chat's id. */
export async function recordPlatformTurn(
  target: ConversationsTarget,
  turn: { sessionId: string; chatId?: string; messages: { role: 'user' | 'assistant'; content: string }[] },
  command: (rest: string) => string,
): Promise<string> {
  const body = await call<{ chatId?: string }>(target, '', command, { method: 'POST', body: JSON.stringify(turn) });
  if (typeof body.chatId !== 'string') throw new PlatformConversationsError(failureSentence(500, command), 500);
  return body.chatId;
}

/** Messages one record call carries, and characters one message keeps: the platform's own limits. */
export const RECORD_BATCH = 50;
export const RECORD_CHARS = 100_000;

/**
 * What a turn adds to the record: the person's and the agent's words since
 * `from`, text only (the platform takes nothing else), in order, each cut to
 * `RECORD_CHARS`.
 */
export function wordsSince(
  messages: readonly { role: string; content?: unknown }[],
  from: number,
): { role: 'user' | 'assistant'; content: string }[] {
  return messages
    .slice(from)
    .filter(
      (m): m is { role: 'user' | 'assistant'; content: string } =>
        (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string' && m.content.trim() !== '',
    )
    .map((m) => ({ role: m.role, content: m.content.slice(0, RECORD_CHARS) }));
}

/**
 * Record `words` into the conversation, `RECORD_BATCH` at a time, the first
 * batch finding or making the chat; answers the chat's id.
 */
export async function recordWords(
  target: ConversationsTarget,
  sessionId: string,
  chatId: string | undefined,
  words: { role: 'user' | 'assistant'; content: string }[],
  command: (rest: string) => string,
): Promise<string | undefined> {
  let chat = chatId;
  for (let i = 0; i < words.length; i += RECORD_BATCH) {
    chat = await recordPlatformTurn(
      target,
      { sessionId, ...(chat ? { chatId: chat } : {}), messages: words.slice(i, i + RECORD_BATCH) },
      command,
    );
  }
  return chat;
}

/** One row of `/resume`: a conversation on this machine, on Robutler, or both. */
export interface ConversationEntry {
  /** Its id on this machine, when it is here. */
  id?: string;
  /** Its platform chat, when it is on Robutler. */
  chatId?: string;
  /** The session it was recorded from, for a conversation only on Robutler. */
  sessionId?: string;
  updatedAt: string;
  /** Messages here, and on Robutler (0 where it is not). */
  localCount: number;
  platformCount: number;
  preview: string;
  /** Only on Robutler: started on the web, or on another machine. */
  onlyOnRobutler: boolean;
}

const when = (iso: string): number => {
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? 0 : ms;
};

/**
 * This machine's conversations and Robutler's, as one list, newest first: a
 * conversation recorded from here is one row (matched by its chat, or by the
 * session it was recorded from), the rest are rows of their own.
 */
export function mergeConversations(
  local: readonly { id: string; updatedAt: string; messageCount: number; preview: string; chatId?: string }[],
  platform: readonly PlatformConversation[],
): ConversationEntry[] {
  const used = new Set<string>();
  const rows: ConversationEntry[] = local.map((l) => {
    const p = platform.find((c) => !used.has(c.chatId) && (c.chatId === l.chatId || (c.sessionId !== null && c.sessionId === l.id)));
    if (p) used.add(p.chatId);
    return {
      id: l.id,
      ...(p || l.chatId ? { chatId: p?.chatId ?? l.chatId } : {}),
      updatedAt: p && when(p.updatedAt) > when(l.updatedAt) ? p.updatedAt : l.updatedAt,
      localCount: l.messageCount,
      platformCount: p?.messageCount ?? 0,
      preview: l.preview || p?.preview || '',
      onlyOnRobutler: false,
    };
  });
  for (const p of platform) {
    if (used.has(p.chatId)) continue;
    rows.push({
      chatId: p.chatId,
      ...(p.sessionId ? { sessionId: p.sessionId } : {}),
      updatedAt: p.updatedAt,
      localCount: 0,
      platformCount: p.messageCount,
      preview: p.preview,
      onlyOnRobutler: true,
    });
  }
  return rows.sort((a, b) => when(b.updatedAt) - when(a.updatedAt));
}
