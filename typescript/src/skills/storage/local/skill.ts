/**
 * LocalMemorySkill - SQLite-backed memory for standalone agents.
 *
 * Same tool interface as RobutlerMemorySkill but stores data locally
 * in a SQLite database instead of calling the portal API.
 *
 * Requires `better-sqlite3` as an optional peer dependency.
 * Falls back to `sql.js` (WASM) if available.
 */

import { Skill } from '../../../core/skill.js';
import type { Context, Tool } from '../../../core/types.js';

export interface LocalMemoryConfig {
  name?: string;
  enabled?: boolean;
  storagePath?: string;
  agentId: string;
  contextStores?: Array<{ storeId: string; label: string }>;
  defaultTtl?: number;
}

type SqliteDb = {
  prepare(sql: string): any;
  exec(sql: string): void;
  close(): void;
};

export class LocalMemorySkill extends Skill {
  private agentId: string;
  private storagePath: string;
  private contextStores: Array<{ storeId: string; label: string }>;
  private defaultTtl: number;
  private db: SqliteDb | null = null;
  private dbReady: Promise<void>;

  constructor(config: LocalMemoryConfig) {
    super({ ...config, name: config.name || 'local-memory' });
    this.agentId = config.agentId;
    this.storagePath = config.storagePath ?? './.webagents/memory.db';
    this.contextStores = config.contextStores ?? [];
    this.defaultTtl = config.defaultTtl ?? 0;
    this.dbReady = this._initDb();
    this._registerMemoryTool();
  }

  private async _initDb(): Promise<void> {
    try {
      const BetterSqlite3 = (await import('better-sqlite3')).default;
      const { mkdirSync } = await import('fs');
      const { dirname } = await import('path');
      mkdirSync(dirname(this.storagePath), { recursive: true });
      this.db = new BetterSqlite3(this.storagePath) as unknown as SqliteDb;
    } catch {
      try {
        const initSqlJs = (await import('sql.js')).default;
        const SQL = await initSqlJs();
        this.db = new SQL.Database() as unknown as SqliteDb;
      } catch {
        throw new Error(
          'LocalMemorySkill requires either better-sqlite3 or sql.js. ' +
          'Install one: npm install better-sqlite3 or npm install sql.js',
        );
      }
    }
    this._createTables();
  }

  private _createTables(): void {
    if (!this.db) return;
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memory (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        store_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        namespace TEXT NOT NULL DEFAULT '',
        key TEXT NOT NULL,
        value TEXT,
        in_context INTEGER NOT NULL DEFAULT 1,
        encrypted INTEGER NOT NULL DEFAULT 0,
        ttl INTEGER DEFAULT 0,
        expires_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(store_id, owner_id, namespace, key)
      );
      CREATE INDEX IF NOT EXISTS memory_store_idx ON memory(store_id);
      CREATE INDEX IF NOT EXISTS memory_expires_idx ON memory(expires_at) WHERE expires_at IS NOT NULL;

      CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        key, value, content=memory, content_rowid=rowid
      );

      CREATE TABLE IF NOT EXISTS memory_grants (
        store_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        level TEXT NOT NULL DEFAULT 'read',
        granted_by TEXT NOT NULL,
        expires_at TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY(store_id, agent_id)
      );
      CREATE INDEX IF NOT EXISTS memory_grants_agent_idx ON memory_grants(agent_id);
    `);
  }

  private _registerMemoryTool(): void {
    const storeLines: string[] = [];
    storeLines.push(`- ${this.agentId} (self): Your persistent memory`);
    for (const s of this.contextStores) {
      storeLines.push(`- ${s.storeId} (${s.label}): ${s.label} memory`);
    }

    const description =
      `Persistent memory. Store and retrieve information across conversations.\n` +
      `\nAvailable stores:\n${storeLines.join('\n')}\n` +
      `\nActions:\n` +
      `- get(store, key): retrieve a stored value\n` +
      `- set(store, key, value, ttl?): store a value\n` +
      `- delete(store, key): remove a key (own entries only)\n` +
      `- list(store, prefix?): list keys in a store\n` +
      `- search(query, store?): full-text search\n` +
      `- share(store, agent, level?): grant another agent access\n` +
      `- unshare(store, agent): revoke a grant\n` +
      `- stores(): list all stores you can access`;

    this.registerTool({
      name: 'memory',
      description,
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['get', 'set', 'delete', 'list', 'search', 'share', 'unshare', 'stores'],
            description: 'Operation to perform',
          },
          store: { type: 'string', description: `Store UUID. Default: ${this.agentId}` },
          key: { type: 'string', description: 'Storage key' },
          value: { description: 'Value to store (any JSON-serializable)' },
          ttl: { type: 'number', description: 'Time-to-live in seconds' },
          prefix: { type: 'string', description: 'Key prefix filter (for list)' },
          query: { type: 'string', description: 'Search query (for search)' },
          agent: { type: 'string', description: 'Agent UUID (for share/unshare)' },
          level: { type: 'string', enum: ['search', 'read', 'readwrite'], description: 'Access level (for share)' },
        },
        required: ['action'],
      },
      scopes: ['all'],
      enabled: true,
      handler: this._handleMemory.bind(this),
    } as Tool);
  }

  private _canAccess(agentId: string, storeId: string): { allowed: boolean; level: string } {
    if (storeId === agentId) return { allowed: true, level: 'readwrite' };
    if (!this.db) return { allowed: false, level: 'search' };

    const now = new Date().toISOString();
    const grant = this.db.prepare(
      `SELECT level FROM memory_grants WHERE store_id = ? AND agent_id = ? AND (expires_at IS NULL OR expires_at > ?)`,
    ).get(storeId, agentId, now) as { level: string } | undefined;

    if (grant) return { allowed: true, level: grant.level };
    return { allowed: false, level: 'search' };
  }

  private _hasLevel(actual: string, required: string): boolean {
    const rank: Record<string, number> = { search: 0, read: 1, readwrite: 2 };
    return (rank[actual] ?? -1) >= (rank[required] ?? 99);
  }

  private async _handleMemory(
    params: {
      action: string;
      store?: string;
      key?: string;
      value?: unknown;
      ttl?: number;
      prefix?: string;
      query?: string;
      agent?: string;
      level?: string;
    },
    _context: Context,
  ): Promise<unknown> {
    await this.dbReady;
    if (!this.db) return { error: 'Database not initialized' };

    const agentId = this.agentId;
    const store = params.store ?? agentId;
    const now = new Date().toISOString();

    switch (params.action) {
      case 'get': {
        if (!params.key) return { error: 'key is required' };
        const access = this._canAccess(agentId, store);
        if (!access.allowed || !this._hasLevel(access.level, 'read')) return { error: 'Access denied' };
        const row = this.db.prepare(
          `SELECT key, value, namespace, owner_id, in_context, created_at, updated_at FROM memory
           WHERE store_id = ? AND key = ? AND (expires_at IS NULL OR expires_at > ?)`,
        ).get(store, params.key, now) as any;
        if (!row) return null;
        return { ...row, value: row.value ? JSON.parse(row.value) : null };
      }

      case 'set': {
        if (!params.key) return { error: 'key is required' };
        if (params.value === undefined) return { error: 'value is required' };
        const access = this._canAccess(agentId, store);
        if (!access.allowed || !this._hasLevel(access.level, 'readwrite')) return { error: 'Access denied' };
        const ttl = params.ttl ?? this.defaultTtl;
        const expiresAt = ttl > 0 ? new Date(Date.now() + ttl * 1000).toISOString() : null;
        const valueStr = JSON.stringify(params.value);
        this.db.prepare(`
          INSERT INTO memory (store_id, owner_id, namespace, key, value, in_context, ttl, expires_at)
          VALUES (?, ?, '', ?, ?, 1, ?, ?)
          ON CONFLICT(store_id, owner_id, namespace, key) DO UPDATE SET
            value = excluded.value, ttl = excluded.ttl, expires_at = excluded.expires_at,
            updated_at = datetime('now')
        `).run(store, agentId, params.key, valueStr, ttl, expiresAt);
        return 'OK';
      }

      case 'delete': {
        if (!params.key) return { error: 'key is required' };
        const access = this._canAccess(agentId, store);
        if (!access.allowed || !this._hasLevel(access.level, 'readwrite')) return { error: 'Access denied' };
        this.db.prepare(
          `DELETE FROM memory WHERE store_id = ? AND owner_id = ? AND key = ?`,
        ).run(store, agentId, params.key);
        return 'OK';
      }

      case 'list': {
        const access = this._canAccess(agentId, store);
        if (!access.allowed || !this._hasLevel(access.level, 'read')) return { error: 'Access denied' };
        let query = `SELECT key, value, namespace, owner_id, in_context FROM memory
                     WHERE store_id = ? AND in_context = 1 AND (expires_at IS NULL OR expires_at > ?)`;
        const args: any[] = [store, now];
        if (params.prefix) {
          query += ` AND key LIKE ?`;
          args.push(`${params.prefix}%`);
        }
        query += ` LIMIT 1000`;
        const rows = this.db.prepare(query).all(...args) as any[];
        return { entries: rows.map((r: any) => ({ ...r, value: r.value ? JSON.parse(r.value) : null })) };
      }

      case 'search': {
        if (!params.query) return { error: 'query is required' };
        const rows = this.db.prepare(
          `SELECT m.store_id, m.key, m.value, m.namespace, m.owner_id, m.in_context
           FROM memory_fts f JOIN memory m ON f.rowid = m.rowid
           WHERE memory_fts MATCH ? AND m.encrypted = 0 AND (m.expires_at IS NULL OR m.expires_at > ?)
           LIMIT 50`,
        ).all(params.query, now) as any[];
        const accessible = rows.filter((r: any) => this._canAccess(agentId, r.store_id).allowed);
        return { entries: accessible.map((r: any) => ({ ...r, value: r.value ? JSON.parse(r.value) : null })) };
      }

      case 'share': {
        if (!params.agent) return { error: 'agent is required' };
        const access = this._canAccess(agentId, store);
        if (!access.allowed || !this._hasLevel(access.level, 'readwrite')) return { error: 'Access denied' };
        const level = params.level ?? 'read';
        this.db.prepare(`
          INSERT INTO memory_grants (store_id, agent_id, level, granted_by)
          VALUES (?, ?, ?, ?)
          ON CONFLICT(store_id, agent_id) DO UPDATE SET level = excluded.level, granted_by = excluded.granted_by
        `).run(store, params.agent, level, agentId);
        return 'OK';
      }

      case 'unshare': {
        if (!params.agent) return { error: 'agent is required' };
        const access = this._canAccess(agentId, store);
        if (!access.allowed || !this._hasLevel(access.level, 'readwrite')) return { error: 'Access denied' };
        this.db.prepare(`DELETE FROM memory_grants WHERE store_id = ? AND agent_id = ?`).run(store, params.agent);
        return 'OK';
      }

      case 'stores': {
        const stores: Array<{ storeId: string; level: string; source: string }> = [];
        stores.push({ storeId: agentId, level: 'readwrite', source: 'self' });
        const grants = this.db.prepare(
          `SELECT store_id, level, granted_by FROM memory_grants WHERE agent_id = ? AND (expires_at IS NULL OR expires_at > ?)`,
        ).all(agentId, now) as any[];
        for (const g of grants) {
          stores.push({ storeId: g.store_id, level: g.level, source: 'grant' });
        }
        for (const s of this.contextStores) {
          stores.push({ storeId: s.storeId, level: 'readwrite', source: s.label });
        }
        return { stores };
      }

      default:
        return { error: `Unknown action: ${params.action}` };
    }
  }

  async getInternal(store: string, key: string): Promise<unknown> {
    await this.dbReady;
    if (!this.db) return null;
    const row = this.db.prepare(
      `SELECT value FROM memory WHERE store_id = ? AND key = ? AND owner_id = ? AND (expires_at IS NULL OR expires_at > ?)`,
    ).get(store, key, this.agentId, new Date().toISOString()) as any;
    if (!row) return null;
    return row.value ? JSON.parse(row.value) : null;
  }

  async setInternal(store: string, key: string, value: unknown, opts?: {
    encrypted?: boolean;
    ttl?: number;
  }): Promise<boolean> {
    await this.dbReady;
    if (!this.db) return false;
    const ttl = opts?.ttl ?? 0;
    const expiresAt = ttl > 0 ? new Date(Date.now() + ttl * 1000).toISOString() : null;
    this.db.prepare(`
      INSERT INTO memory (store_id, owner_id, namespace, key, value, in_context, encrypted, ttl, expires_at)
      VALUES (?, ?, '', ?, ?, 0, ?, ?, ?)
      ON CONFLICT(store_id, owner_id, namespace, key) DO UPDATE SET
        value = excluded.value, encrypted = excluded.encrypted, ttl = excluded.ttl,
        expires_at = excluded.expires_at, updated_at = datetime('now')
    `).run(store, this.agentId, key, JSON.stringify(value), opts?.encrypted ? 1 : 0, ttl, expiresAt);
    return true;
  }

  override async cleanup(): Promise<void> {
    this.db?.close();
  }
}
