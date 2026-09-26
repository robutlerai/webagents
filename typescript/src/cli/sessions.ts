/**
 * The conversations the chat keeps (2026-09-24), in the same files the Python
 * chat writes, so either CLI can pick up the other's.
 *
 * WHERE: `~/.webagents[-profile]/sessions/<folder>/<agent>/<id>.json`, with a
 * `.latest` pointer beside them. Under the profile's own directory, never in
 * the project: a chat started in any folder must not leave files there. The
 * Python chat used to write `.webagents/sessions` next to the agent file,
 * which for the built-in agent was inside the installed package, one pile
 * shared by every folder. `<folder>` is the folder's absolute path with every
 * character other than a letter, a digit, `.`, `_` or `-` turned into `-`
 * (`/Users/me/x` -> `-Users-me-x`), the same in both SDKs.
 *
 * WHAT: `session_id`, `agent_name`, `created_at`, `updated_at`, `messages`
 * (OpenAI shape), `metadata`, `input_tokens`, `output_tokens`, the same in
 * both SDKs. Owner-only files, written whole and then renamed into place, so
 * a crash never leaves half a conversation: they hold the conversation.
 *
 * WHOSE (2026-09-25). The person at the terminal is the agent's owner, and
 * their conversations are the folder's own directory. A SERVED agent's
 * session skill (`skills/session/skill.ts`) keeps every other verified
 * caller's under `callers/<hash of the caller>/` beside them, one namespace
 * per caller, so no caller's conversations are another's.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { globalDir, profileName } from './config-store';

export interface SessionMessage {
  role: string;
  content: string | null;
  [key: string]: unknown;
}

export interface StoredSession {
  session_id: string;
  agent_name: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
  metadata: Record<string, unknown>;
  input_tokens: number;
  output_tokens: number;
}

export interface SessionSummary {
  id: string;
  updatedAt: string;
  messageCount: number;
  /** The first thing the person said, on one line. */
  preview: string;
  /** The platform chat it is recorded into, when it is (`robutler-sessions.ts`). */
  chatId?: string;
}

/** A path or a name as one directory name. */
export function slugFor(text: string): string {
  return text.replace(/[^A-Za-z0-9._-]/g, '-');
}

/**
 * Where this folder's conversations with `agentName` are kept. The folder's
 * REAL path, symlinks resolved, as Python's `Path.resolve()` gives it, so a
 * folder reached through a link maps to the same place in both SDKs.
 */
export function sessionsDir(folder: string, agentName: string, profile?: string): string {
  let real: string;
  try {
    real = fs.realpathSync(folder);
  } catch {
    real = path.resolve(folder);
  }
  return path.join(globalDir(profileName(profile)), 'sessions', slugFor(real), slugFor(agentName));
}

/** One caller's directory name: the first 32 hex digits of the SHA-256 of their principal (`user:...`). */
export function callerKey(principal: string): string {
  return createHash('sha256').update(principal, 'utf8').digest('hex').slice(0, 32);
}

/** Where a served agent keeps one verified caller's conversations (file comment, WHOSE). */
export function callerSessionsDir(folder: string, agentName: string, principal: string, profile?: string): string {
  return path.join(sessionsDir(folder, agentName, profile), 'callers', callerKey(principal));
}

export function newSessionId(): string {
  return randomUUID();
}

/** Write `text` to `file`, owner-only, whole or not at all. */
function writePrivate(file: string, text: string): void {
  const temp = `${file}.${process.pid}.${randomUUID().slice(0, 8)}.tmp`;
  fs.writeFileSync(temp, text, { mode: 0o600 });
  fs.renameSync(temp, file);
}

/** Write the session and point `.latest` at it. */
export function saveSession(dir: string, session: StoredSession): void {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const now = new Date().toISOString();
  const data: StoredSession = { ...session, updated_at: now, created_at: session.created_at || now };
  writePrivate(path.join(dir, `${slugFor(session.session_id)}.json`), `${JSON.stringify(data, null, 2)}\n`);
  writePrivate(path.join(dir, '.latest'), session.session_id);
}

/**
 * Note on a saved conversation that its first `count` messages are recorded
 * into the platform chat `chatId`, leaving everything else as it was (its
 * `updated_at` and `.latest` included). For a turn recorded after the person
 * already moved on to another conversation.
 */
export function markRecorded(dir: string, id: string, chatId: string, count: number): void {
  const session = loadSession(dir, id);
  if (!session) return;
  const data = { ...session, metadata: { ...session.metadata, robutler_chat_id: chatId, robutler_recorded: count } };
  writePrivate(path.join(dir, `${slugFor(id)}.json`), `${JSON.stringify(data, null, 2)}\n`);
}

export function loadSession(dir: string, id: string): StoredSession | null {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(dir, `${slugFor(id)}.json`), 'utf-8')) as Partial<StoredSession>;
    if (!Array.isArray(data.messages)) return null;
    return {
      session_id: data.session_id ?? id,
      agent_name: data.agent_name ?? '',
      created_at: data.created_at ?? '',
      updated_at: data.updated_at ?? '',
      messages: data.messages,
      metadata: data.metadata ?? {},
      input_tokens: data.input_tokens ?? 0,
      output_tokens: data.output_tokens ?? 0,
    };
  } catch {
    return null;
  }
}

/** The first thing the person said, as one line. */
export function sessionPreview(messages: readonly SessionMessage[]): string {
  const first = messages.find((m) => m.role === 'user' && typeof m.content === 'string' && m.content.trim());
  return first ? String(first.content).replace(/\s+/g, ' ').trim() : '';
}

/** This folder's conversations with the agent, newest first. */
export function listSessions(dir: string): SessionSummary[] {
  let names: string[];
  try {
    names = fs.readdirSync(dir).filter((name) => name.endsWith('.json'));
  } catch {
    return [];
  }
  const out: SessionSummary[] = [];
  for (const name of names) {
    const session = loadSession(dir, name.slice(0, -'.json'.length));
    if (!session || !session.messages.some((m) => m.role === 'user')) continue;
    const chatId = session.metadata.robutler_chat_id;
    out.push({
      id: session.session_id,
      updatedAt: session.updated_at,
      messageCount: session.messages.length,
      preview: sessionPreview(session.messages),
      ...(typeof chatId === 'string' && chatId ? { chatId } : {}),
    });
  }
  return out.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
}

/** "just now", "5 min ago", "3 h ago", "2 days ago", or the date. */
export function whenLabel(iso: string, now: number = Date.now()): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const seconds = Math.max(0, (now - then) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 7 * 86400) {
    const days = Math.floor(seconds / 86400);
    return days === 1 ? 'yesterday' : `${days} days ago`;
  }
  return iso.slice(0, 10);
}
