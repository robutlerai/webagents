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
 * WHAT: the Python session skill's format (`Session.to_dict` in
 * `python/webagents/agents/skills/local/session/skill.py`): `session_id`,
 * `agent_name`, `created_at`, `updated_at`, `messages` (OpenAI shape),
 * `metadata`, `input_tokens`, `output_tokens`. Owner-only files: they hold the
 * conversation.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { randomUUID } from 'node:crypto';
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

export function newSessionId(): string {
  return randomUUID();
}

/** Write the session and point `.latest` at it. */
export function saveSession(dir: string, session: StoredSession): void {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const now = new Date().toISOString();
  const data: StoredSession = { ...session, updated_at: now, created_at: session.created_at || now };
  const file = path.join(dir, `${slugFor(session.session_id)}.json`);
  fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`, { mode: 0o600 });
  fs.writeFileSync(path.join(dir, '.latest'), session.session_id, { mode: 0o600 });
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
    out.push({
      id: session.session_id,
      updatedAt: session.updated_at,
      messageCount: session.messages.length,
      preview: sessionPreview(session.messages),
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
