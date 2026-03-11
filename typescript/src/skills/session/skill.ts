/**
 * Session Skill
 *
 * Manages conversational state across turns. Provides persistent
 * memory, scratchpad, and context management for agents that need
 * to maintain state between messages.
 *
 * State can be stored in-memory (ephemeral) or via a backend
 * (Redis, portal API, filesystem) for persistence across restarts.
 */

import { Skill } from '../../core/skill.js';
import { tool, hook } from '../../core/decorators.js';
import type { Context, HookData } from '../../core/types.js';

export interface SessionConfig {
  name?: string;
  enabled?: boolean;
  /** Storage backend: 'memory' (default), 'file', 'portal' */
  backend?: 'memory' | 'file' | 'portal';
  /** File path for 'file' backend */
  storagePath?: string;
  /** Portal API URL for 'portal' backend */
  portalUrl?: string;
  /** API key for portal backend */
  apiKey?: string;
  /** Max entries per session (default 1000) */
  maxEntries?: number;
  /** Session TTL in ms (default: 1 hour) */
  sessionTtl?: number;
}

interface SessionEntry {
  key: string;
  value: unknown;
  createdAt: number;
  updatedAt: number;
}

interface SessionData {
  id: string;
  entries: Map<string, SessionEntry>;
  createdAt: number;
  lastAccessedAt: number;
  metadata: Record<string, unknown>;
}

export class SessionSkill extends Skill {
  private sessions = new Map<string, SessionData>();
  private maxEntries: number;
  /** TTL in ms — reserved for future eviction logic */
  public sessionTtl: number;

  constructor(config: SessionConfig = {}) {
    super({ ...config, name: config.name || 'session' });
    this.maxEntries = config.maxEntries ?? 1000;
    this.sessionTtl = config.sessionTtl ?? 3_600_000;
  }

  private getOrCreateSession(sessionId: string): SessionData {
    let session = this.sessions.get(sessionId);
    if (!session) {
      session = {
        id: sessionId,
        entries: new Map(),
        createdAt: Date.now(),
        lastAccessedAt: Date.now(),
        metadata: {},
      };
      this.sessions.set(sessionId, session);
    }
    session.lastAccessedAt = Date.now();
    return session;
  }

  private getSessionId(context: Context): string {
    return (context.metadata?.sessionId as string)
      ?? (context.metadata?.chatId as string)
      ?? 'default';
  }

  @hook({ lifecycle: 'before_run', priority: 10 })
  async injectSessionContext(_data: HookData, context: Context): Promise<void> {
    const sessionId = this.getSessionId(context);
    const session = this.getOrCreateSession(sessionId);
    context.metadata = {
      ...context.metadata,
      sessionId: session.id,
      sessionEntryCount: session.entries.size,
    };
  }

  @tool({
    name: 'session_get',
    description: 'Get a value from the current session state.',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'State key' },
      },
      required: ['key'],
    },
  })
  async sessionGet(params: { key: string }, context: Context): Promise<unknown> {
    const session = this.getOrCreateSession(this.getSessionId(context));
    const entry = session.entries.get(params.key);
    return entry?.value ?? null;
  }

  @tool({
    name: 'session_set',
    description: 'Store a value in the current session state.',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'State key' },
        value: { description: 'Value to store' },
      },
      required: ['key', 'value'],
    },
  })
  async sessionSet(
    params: { key: string; value: unknown },
    context: Context,
  ): Promise<string> {
    const session = this.getOrCreateSession(this.getSessionId(context));

    if (session.entries.size >= this.maxEntries && !session.entries.has(params.key)) {
      const oldest = [...session.entries.entries()].sort(
        (a, b) => a[1].updatedAt - b[1].updatedAt,
      )[0];
      if (oldest) session.entries.delete(oldest[0]);
    }

    const now = Date.now();
    session.entries.set(params.key, {
      key: params.key,
      value: params.value,
      createdAt: session.entries.get(params.key)?.createdAt ?? now,
      updatedAt: now,
    });
    return 'OK';
  }

  @tool({
    name: 'session_delete',
    description: 'Delete a key from session state.',
    parameters: {
      type: 'object',
      properties: {
        key: { type: 'string', description: 'State key to delete' },
      },
      required: ['key'],
    },
  })
  async sessionDelete(params: { key: string }, context: Context): Promise<string> {
    const session = this.getOrCreateSession(this.getSessionId(context));
    session.entries.delete(params.key);
    return 'OK';
  }

  @tool({
    name: 'session_list',
    description: 'List all keys in the current session state.',
    parameters: { type: 'object', properties: {} },
  })
  async sessionList(_params: Record<string, unknown>, context: Context): Promise<string[]> {
    const session = this.getOrCreateSession(this.getSessionId(context));
    return [...session.entries.keys()];
  }

  @tool({
    name: 'session_clear',
    description: 'Clear all state for the current session.',
    parameters: { type: 'object', properties: {} },
  })
  async sessionClear(_params: Record<string, unknown>, context: Context): Promise<string> {
    const session = this.getOrCreateSession(this.getSessionId(context));
    session.entries.clear();
    return 'OK';
  }

  @tool({
    name: 'session_get_all',
    description: 'Get all key-value pairs from the current session state.',
    parameters: { type: 'object', properties: {} },
  })
  async sessionGetAll(
    _params: Record<string, unknown>,
    context: Context,
  ): Promise<Record<string, unknown>> {
    const session = this.getOrCreateSession(this.getSessionId(context));
    const result: Record<string, unknown> = {};
    for (const [key, entry] of session.entries) {
      result[key] = entry.value;
    }
    return result;
  }

  override async cleanup(): Promise<void> {
    this.sessions.clear();
  }
}
