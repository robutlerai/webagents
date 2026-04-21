/**
 * Storage Skills
 *
 * Platform storage capabilities via portal API:
 * - Memory: Store-based key-value storage with grants, search, sharing
 * - Files: Consolidated file storage with action parameter
 * - JSON: Structured JSON document storage (opt-in, not default)
 */

import { Skill } from '../../core/skill';
import { tool } from '../../core/decorators';
import type { Context, Tool } from '../../core/types';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface StorageConfig {
  name?: string;
  enabled?: boolean;
  /** Portal API base URL */
  portalUrl?: string;
  /** API key for portal auth */
  apiKey?: string;
  /** Agent ID for namespace scoping */
  agentId?: string;
  /** Default TTL for KV entries (seconds, 0 = no expiry) */
  defaultTtl?: number;
  /** Max file size in bytes (default 50MB) */
  maxFileSize?: number;
}

export interface MemoryConfig extends StorageConfig {
  /** Contextual stores to expose: { storeId, label } pairs */
  contextStores?: Array<{ storeId: string; label: string }>;
  /** Chat ID for chat-scoped store */
  chatId?: string;
  /** User ID for user-scoped store */
  userId?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function portalFetch(
  baseUrl: string,
  path: string,
  apiKey: string | undefined,
  init?: RequestInit,
): Promise<Response> {
  const url = `${baseUrl.replace(/\/$/, '')}${path}`;
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string> ?? {}),
  };
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }
  return fetch(url, { ...init, headers });
}

// ---------------------------------------------------------------------------
// Memory Skill (store-based, replaces KV)
// ---------------------------------------------------------------------------

export class RobutlerMemorySkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;
  private defaultTtl: number;
  private contextStores: Array<{ storeId: string; label: string }>;
  private chatId?: string;
  private userId?: string;

  constructor(config: MemoryConfig = {}) {
    super({ ...config, name: config.name || 'robutler-memory' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
    this.defaultTtl = config.defaultTtl ?? 0;
    this.contextStores = config.contextStores ?? [];
    this.chatId = config.chatId;
    this.userId = config.userId;

    this._registerMemoryTool();
  }

  private _registerMemoryTool(): void {
    const storeLines: string[] = [];
    if (this.agentId) {
      storeLines.push(`- /memories/ (self): Your persistent memory`);
    }
    for (const s of this.contextStores) {
      storeLines.push(`- /memories/shared/${s.storeId}/ (${s.label}): ${s.label} memory`);
    }

    const storesSection = storeLines.length > 0
      ? `\nAvailable memory paths:\n${storeLines.join('\n')}\nUse 'stores' command to discover additional shared stores.\n`
      : '';

    // share/unshare are intentionally NOT exposed to the LLM right now (the
    // grantee field requires an agent UUID the LLM rarely has, and unmediated
    // agent-to-agent sharing is a prompt-injection footgun). The underlying
    // _handleMemory switch cases for 'share' / 'unshare' are kept so future
    // settings-UI-driven sharing — or programmatic callers — keep working
    // without a server-side change. To re-enable as a first-class LLM
    // capability, uncomment the enum entries and the `agent` / `level`
    // property blocks below, and re-add the corresponding lines in
    // `description`. Mirror the change in `lib/llm/platform-tools.ts`.
    const description =
      `Persistent memory across conversations. Address a bucket with \`scope\` (NOT a path):\n` +
      `- scope: "agent" → your own (self) bucket; SHARED across all callers.\n` +
      `- scope: "user"  → per-CALLER private bucket; isolated across callers.\n` +
      `- scope: "chat"  → per-conversation bucket, shared with chat participants.\n` +
      `Entries are key→value (NOT files). Pass keys as plain names (e.g. "user-profile") — no .md/.json suffixes.\n` +
      `Granted custom stores: pass \`store: "<storeId>"\` instead of \`scope\`. Use \`stores\` to discover.\n` +
      `Legacy: \`path: "/memories/<key>"\` is accepted (treated as scope="agent" + key=<key>) for back-compat.` +
      storesSection +
      `\nCommands:\n` +
      `- view(scope, key?): read an entry, or list keys in the scope when key is omitted\n` +
      `- create(scope, key, content): create a new memory entry\n` +
      `- edit(scope, key, old_str, new_str): edit an existing memory entry via str_replace\n` +
      `- delete(scope, key): remove a memory entry\n` +
      `- rename(scope, key, new_str): rename a memory entry within the same scope\n` +
      `- search(query, scope?): full-text + semantic search; omit scope to search all enabled scopes\n` +
      // `- share(scope|store, agent, level?): grant another agent access\n` +
      // `- unshare(scope|store, agent): revoke a previously granted access\n` +
      `- stores(): list all memory stores you can access`;

    this.registerTool({
      name: 'memory',
      description,
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties: {
          command: {
            type: 'string',
            // share/unshare temporarily removed — see comment block above.
            enum: ['view', 'create', 'edit', 'delete', 'rename', 'search', /* 'share', 'unshare', */ 'stores'],
            description:
              'view: read an entry (scope+key) or list keys in a scope (scope only, no key). ' +
              'create: write a new entry (requires scope+key+content). ' +
              'edit: str_replace inside an existing entry (requires scope+key+old_str+new_str; old_str must be unique). ' +
              'delete: remove an entry (requires scope+key). ' +
              'rename: move an entry to a new key (new_str holds the destination key). ' +
              'search: full-text + semantic search across one scope (when scope is set) or all enabled scopes (when omitted). Requires query. ' +
              'stores: list every memory bucket the caller can reach.',
          },
          scope: {
            type: 'string',
            enum: ['agent', 'user', 'chat'],
            description:
              'Which built-in bucket to act on. Mutually exclusive with `store`. ' +
              'For `search`, omit scope to search across all enabled scopes.',
          },
          store: {
            type: 'string',
            maxLength: 64,
            description:
              'Granted custom storeId for buckets you have an external grant to. Use `stores` to discover. ' +
              'Mutually exclusive with `scope`.',
          },
          key: {
            type: 'string',
            maxLength: 200,
            description: 'Entry name within the chosen scope/store. Plain names — no file extensions. Omit to list.',
          },
          path: {
            type: 'string',
            pattern: '^/memories(/.*)?$',
            maxLength: 256,
            description:
              'DEPRECATED — pass `scope` + `key` instead. Kept for back-compat: ' +
              '`/memories/<key>` is treated as scope="agent" + key=<key>; ' +
              '`/memories/shared/<storeId>/<key>` is treated as store=<storeId> + key=<key>.',
          },
          content: { type: 'string', description: 'Content for create' },
          old_str: { type: 'string', description: 'Text to find (for edit). Must appear exactly once in the entry.' },
          new_str: { type: 'string', description: 'Replacement text (for edit) or new key (for rename). Used literally — no $-substitution.' },
          query: { type: 'string', description: 'Search query text (for search)' },
          // share/unshare-only fields — re-enable alongside the enum entries
          // above when sharing comes back as a first-class LLM capability.
          // agent: { type: 'string', description: 'Agent UUID (for share)' },
          // level: {
          //   type: 'string',
          //   enum: ['search', 'read', 'readwrite'],
          //   description: 'Access level (for share). search = vector/FTS only; read = view + search; readwrite = full access. Default: read',
          // },
        },
        required: ['command'],
        oneOf: [
          { properties: { command: { const: 'view' } } },
          { properties: { command: { const: 'stores' } } },
          { properties: { command: { const: 'create' } }, required: ['command', 'content'] },
          { properties: { command: { const: 'edit' } }, required: ['command', 'old_str', 'new_str'] },
          { properties: { command: { const: 'delete' } } },
          { properties: { command: { const: 'rename' } }, required: ['command', 'new_str'] },
          { properties: { command: { const: 'search' } }, required: ['command', 'query'] },
        ],
      },
      scopes: ['all'],
      enabled: true,
      handler: (params: Record<string, unknown>, context: Context) =>
        this._handleMemory(this._normalizeScopeKey(params as any) as any, context),
    } as Tool);
  }

  /**
   * Translate the new (scope|store, key) addressing back to the legacy
   * `/memories/...` path that _handleMemory + the portal HTTP API still
   * speak. Keeps the SDK schema aligned with the portal-side MEMORY_TOOL
   * without rewriting the handler / route layer here. If `path` is already
   * present (legacy callers), leave it alone.
   */
  private _normalizeScopeKey(params: Record<string, unknown>): Record<string, unknown> {
    if (params.path) return params;
    const scope = typeof params.scope === 'string' ? params.scope : undefined;
    const store = typeof params.store === 'string' ? params.store : undefined;
    const key = typeof params.key === 'string' ? params.key : undefined;

    if (store) {
      return { ...params, path: `/memories/shared/${store}${key ? `/${key}` : '/'}` };
    }
    if (scope === 'agent' || !scope) {
      // Built-in agent scope is the SDK's existing "self" bucket — same as
      // the legacy bare /memories/<key>. user / chat are routed by the portal
      // handler; the SDK skill (out-of-process) just forwards path + the
      // userId/chatId context fields it already sets on every request.
      return { ...params, path: key ? `/memories/${key}` : `/memories/` };
    }
    // For user / chat scopes, the SDK relies on the portal handler to
    // resolve the storeId from the (userId, chatId) it ships with every
    // request. Encode the scope as a synthetic /memories/<scope>/<key>
    // path; the portal handler treats unknown leading segments as bare keys
    // today, so this falls back to scope=agent if the portal hasn't been
    // upgraded — same behavior as before.
    const synth = key ? `/memories/${key}` : `/memories/`;
    return { ...params, path: synth, scope };
  }

  private _parseMemoryPath(memPath: string): { store: string; key: string } {
    const cleaned = memPath.replace(/^\/memories\/?/, '').replace(/\.md$/, '');
    if (cleaned.startsWith('shared/')) {
      const parts = cleaned.replace('shared/', '').split('/');
      const store = parts[0] || this.agentId || 'default';
      const key = parts.slice(1).join('/') || '';
      return { store, key };
    }
    return { store: this.agentId || 'default', key: cleaned };
  }

  private _extractStoreFromPath(memPath: string): string {
    const cleaned = memPath.replace(/^\/memories\/?/, '').replace(/\/$/, '');
    if (cleaned.startsWith('shared/')) {
      return cleaned.replace('shared/', '').split('/')[0] || this.agentId || 'default';
    }
    return this.agentId || 'default';
  }

  private async _handleMemory(
    params: {
      command: string;
      path?: string;
      content?: string;
      old_str?: string;
      new_str?: string;
      query?: string;
      agent?: string;
      level?: string;
    },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;

    switch (params.command) {
      case 'view': {
        const memPath = params.path || '/memories/';
        const isDir = memPath.endsWith('/');
        if (isDir) {
          const store = this._extractStoreFromPath(memPath);
          const qs = new URLSearchParams({ agentId: agentId!, store, inContext: 'true' });
          if (this.chatId) qs.set('chatId', this.chatId);
          if (this.userId) qs.set('userId', this.userId);
          const res = await portalFetch(this.portalUrl, `/api/storage/memory?${qs}`, this.apiKey);
          if (!res.ok) {
            if (res.status === 403) return { error: 'Access denied to this store' };
            return { error: `Memory list failed: ${res.status}` };
          }
          return res.json();
        } else {
          const { store, key } = this._parseMemoryPath(memPath);
          if (!key) return { error: 'path is required for view' };
          const qs = new URLSearchParams({ agentId: agentId!, store });
          if (this.chatId) qs.set('chatId', this.chatId);
          if (this.userId) qs.set('userId', this.userId);
          const res = await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(key)}?${qs}`, this.apiKey);
          if (!res.ok) {
            if (res.status === 404) return null;
            if (res.status === 403) return { error: 'Access denied to this store' };
            return { error: `Memory get failed: ${res.status}` };
          }
          return res.json();
        }
      }

      case 'create': {
        if (!params.path) return { error: 'path is required for create' };
        if (params.content === undefined) return { error: 'content is required for create' };
        const { store, key } = this._parseMemoryPath(params.path);
        if (!key) return { error: 'path must include a filename for create' };
        const res = await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(key)}`, this.apiKey, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: params.content, ttl: this.defaultTtl, agentId, store, chatId: this.chatId, userId: this.userId }),
        });
        if (!res.ok) {
          if (res.status === 403) return { error: 'Access denied to this store' };
          if (res.status === 413) return { error: 'Value too large' };
          if (res.status === 429) return { error: 'Key limit reached for this store' };
          return { error: `Memory create failed: ${res.status}` };
        }
        return `Created ${params.path}`;
      }

      case 'edit': {
        if (!params.path) return { error: 'path is required for edit' };
        if (params.old_str === undefined || params.new_str === undefined) {
          return { error: 'old_str and new_str are required for edit' };
        }
        const { store, key } = this._parseMemoryPath(params.path);
        const qs = new URLSearchParams({ agentId: agentId!, store });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const getRes = await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(key)}?${qs}`, this.apiKey);
        if (!getRes.ok) return { error: `Memory not found: ${params.path}` };
        const getData = await getRes.json() as { value?: unknown };
        if (typeof getData.value !== 'string') {
          return {
            error:
              `Cannot edit ${params.path}: value is ${getData.value === null ? 'null' : typeof getData.value}, not text. ` +
              `Use create to overwrite, or fetch + delete + create to convert.`,
          };
        }
        const current = getData.value;
        const occurrences = current.split(params.old_str).length - 1;
        if (occurrences === 0) return { error: `old_str not found in ${params.path}` };
        if (occurrences > 1) {
          return {
            error:
              `old_str matches ${occurrences} times in ${params.path}. ` +
              `Add more surrounding context so the match is unique.`,
          };
        }
        // Callback form keeps `$&`, `$1`, `$$` etc. literal in `new_str`.
        const updated = current.replace(params.old_str, () => params.new_str!);
        const setRes = await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(key)}`, this.apiKey, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: updated, agentId, store, chatId: this.chatId, userId: this.userId }),
        });
        if (!setRes.ok) return { error: `Memory edit failed: ${setRes.status}` };
        const idx = current.indexOf(params.old_str);
        const ctx = 40;
        const before = current.slice(Math.max(0, idx - ctx), idx);
        const after = current.slice(idx + params.old_str.length, idx + params.old_str.length + ctx);
        return `Edited ${params.path}\n  - …${before}[${params.old_str}]${after}…\n  + …${before}[${params.new_str}]${after}…`;
      }

      case 'delete': {
        if (!params.path) return { error: 'path is required for delete' };
        const { store, key } = this._parseMemoryPath(params.path);
        if (!key) return { error: 'path must include a filename for delete' };
        const qs = new URLSearchParams({ agentId: agentId!, store });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(key)}?${qs}`, this.apiKey, { method: 'DELETE' });
        if (!res.ok) {
          if (res.status === 403) return { error: 'Access denied to this store' };
          return { error: `Memory delete failed: ${res.status}` };
        }
        return `Deleted ${params.path}`;
      }

      case 'rename': {
        if (!params.path || !params.new_str) return { error: 'path and new_str (new path) are required for rename' };
        const src = this._parseMemoryPath(params.path);
        const dst = this._parseMemoryPath(params.new_str);
        const srcQs = new URLSearchParams({ agentId: agentId!, store: src.store });
        if (this.chatId) srcQs.set('chatId', this.chatId);
        if (this.userId) srcQs.set('userId', this.userId);
        const getRes = await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(src.key)}?${srcQs}`, this.apiKey);
        if (!getRes.ok) return { error: `Memory not found: ${params.path}` };
        const getData = await getRes.json() as { value?: unknown };
        await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(dst.key)}`, this.apiKey, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ value: getData.value, agentId, store: dst.store, chatId: this.chatId, userId: this.userId }),
        });
        await portalFetch(this.portalUrl, `/api/storage/memory/${encodeURIComponent(src.key)}?${srcQs}`, this.apiKey, { method: 'DELETE' });
        return `Renamed ${params.path} → ${params.new_str}`;
      }

      case 'search': {
        if (!params.query) return { error: 'query is required for search' };
        const qs = new URLSearchParams({ action: 'search', agentId: agentId!, q: params.query });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(this.portalUrl, `/api/storage/memory?${qs}`, this.apiKey);
        if (!res.ok) return { error: `Memory search failed: ${res.status}` };
        return res.json();
      }

      case 'share': {
        if (!params.agent) return { error: 'agent is required for share' };
        const store = params.path ? this._extractStoreFromPath(params.path) : agentId;
        const res = await portalFetch(this.portalUrl, `/api/storage/memory?action=share`, this.apiKey, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agentId, store, grantee: params.agent, level: params.level ?? 'read', chatId: this.chatId, userId: this.userId }),
        });
        if (!res.ok) {
          if (res.status === 403) return { error: 'You need write access to share this store' };
          return { error: `Memory share failed: ${res.status}` };
        }
        return 'OK';
      }

      case 'unshare': {
        if (!params.agent) return { error: 'agent is required for unshare' };
        const store = params.path ? this._extractStoreFromPath(params.path) : agentId;
        const qs = new URLSearchParams({ action: 'share', store: store!, grantee: params.agent });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(this.portalUrl, `/api/storage/memory?${qs}`, this.apiKey, {
          method: 'DELETE',
        });
        if (!res.ok) {
          if (res.status === 403) return { error: 'You need write access to unshare this store' };
          return { error: `Memory unshare failed: ${res.status}` };
        }
        return 'OK';
      }

      case 'stores': {
        const qs = new URLSearchParams({ action: 'stores', agentId: agentId! });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(this.portalUrl, `/api/storage/memory?${qs}`, this.apiKey);
        if (!res.ok) return { error: `Memory stores failed: ${res.status}` };
        return res.json();
      }

      default:
        return { error: `Unknown command: ${params.command}. Use view, create, edit, delete, rename, search, share, unshare, or stores.` };
    }
  }

  /** Internal API: get a value without exposing to LLM (inContext=false entries too) */
  async getInternal(store: string, key: string, namespace?: string): Promise<unknown> {
    const qs = new URLSearchParams({ agentId: this.agentId!, store });
    if (namespace) qs.set('namespace', namespace);
    const res = await portalFetch(
      this.portalUrl,
      `/api/storage/memory/${encodeURIComponent(key)}?${qs}`,
      this.apiKey,
    );
    if (!res.ok) return null;
    return res.json();
  }

  /** Internal API: set a value not exposed to LLM */
  async setInternal(store: string, key: string, value: unknown, opts?: {
    namespace?: string;
    encrypted?: boolean;
    ttl?: number;
  }): Promise<boolean> {
    const res = await portalFetch(
      this.portalUrl,
      `/api/storage/memory/${encodeURIComponent(key)}`,
      this.apiKey,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          value,
          agentId: this.agentId,
          store,
          namespace: opts?.namespace ?? '',
          inContext: false,
          encrypted: opts?.encrypted ?? false,
          ttl: opts?.ttl ?? 0,
        }),
      },
    );
    return res.ok;
  }
}

/** @deprecated Use RobutlerMemorySkill instead */
export class RobutlerKVSkill extends RobutlerMemorySkill {
  constructor(config: StorageConfig = {}) {
    super(config);
  }
}

// ---------------------------------------------------------------------------
// JSON Document Storage Skill (opt-in, not in default set)
// ---------------------------------------------------------------------------

export class RobutlerJSONSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;

  constructor(config: StorageConfig = {}) {
    super({ ...config, name: config.name || 'robutler-json' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
  }

  @tool({
    name: 'json_get',
    description: 'Get a JSON document by ID.',
    parameters: {
      type: 'object',
      properties: {
        collection: { type: 'string', description: 'Collection name' },
        id: { type: 'string', description: 'Document ID' },
      },
      required: ['collection', 'id'],
    },
  })
  async jsonGet(
    params: { collection: string; id: string },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(
      this.portalUrl,
      `/api/storage/json/${encodeURIComponent(params.collection)}/${encodeURIComponent(params.id)}?agentId=${agentId}`,
      this.apiKey,
    );
    if (!res.ok) {
      if (res.status === 404) return null;
      return { error: `JSON get failed: ${res.status}` };
    }
    return res.json();
  }

  @tool({
    name: 'json_put',
    description: 'Create or update a JSON document.',
    parameters: {
      type: 'object',
      properties: {
        collection: { type: 'string', description: 'Collection name' },
        id: { type: 'string', description: 'Document ID (auto-generated if omitted)' },
        data: { type: 'object', description: 'JSON document to store' },
      },
      required: ['collection', 'data'],
    },
  })
  async jsonPut(
    params: { collection: string; id?: string; data: Record<string, unknown> },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(
      this.portalUrl,
      `/api/storage/json/${encodeURIComponent(params.collection)}`,
      this.apiKey,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: params.id, data: params.data, agentId }),
      },
    );
    if (!res.ok) return { error: `JSON put failed: ${res.status}` };
    return res.json();
  }

  @tool({
    name: 'json_delete',
    description: 'Delete a JSON document.',
    parameters: {
      type: 'object',
      properties: {
        collection: { type: 'string', description: 'Collection name' },
        id: { type: 'string', description: 'Document ID' },
      },
      required: ['collection', 'id'],
    },
  })
  async jsonDelete(
    params: { collection: string; id: string },
    context: Context,
  ): Promise<string> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(
      this.portalUrl,
      `/api/storage/json/${encodeURIComponent(params.collection)}/${encodeURIComponent(params.id)}?agentId=${agentId}`,
      this.apiKey,
      { method: 'DELETE' },
    );
    if (!res.ok) return `JSON delete failed: ${res.status}`;
    return 'OK';
  }

  @tool({
    name: 'json_query',
    description: 'Query JSON documents in a collection with optional filters.',
    parameters: {
      type: 'object',
      properties: {
        collection: { type: 'string', description: 'Collection name' },
        filter: { type: 'object', description: 'Filter conditions (field: value pairs)' },
        sort: { type: 'string', description: 'Sort field (prefix with - for descending)' },
        limit: { type: 'number', description: 'Max documents to return (default 50)' },
        offset: { type: 'number', description: 'Skip first N results' },
      },
      required: ['collection'],
    },
  })
  async jsonQuery(
    params: {
      collection: string;
      filter?: Record<string, unknown>;
      sort?: string;
      limit?: number;
      offset?: number;
    },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const res = await portalFetch(
      this.portalUrl,
      `/api/storage/json/${encodeURIComponent(params.collection)}/query`,
      this.apiKey,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filter: params.filter,
          sort: params.sort,
          limit: params.limit ?? 50,
          offset: params.offset ?? 0,
          agentId,
        }),
      },
    );
    if (!res.ok) return { error: `JSON query failed: ${res.status}` };
    return res.json();
  }
}

// ---------------------------------------------------------------------------
// File Storage Skill (consolidated as `files`)
// ---------------------------------------------------------------------------

export class RobutlerFilesSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;
  private maxFileSize: number;

  constructor(config: StorageConfig = {}) {
    super({ ...config, name: config.name || 'robutler-files' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
    this.maxFileSize = config.maxFileSize ?? 50 * 1024 * 1024;
  }

  @tool({
    name: 'files',
    description:
      'File storage. Upload, download, list, and manage files. Use this for ' +
      'images, documents, or any binary content that needs to be stored or shared.\n\n' +
      'Actions:\n' +
      '- upload: store a file (provide base64 content and MIME type)\n' +
      '- download: retrieve a file\'s content (returned as base64)\n' +
      '- list: list stored files (optionally filtered by directory prefix)\n' +
      '- delete: remove a file\n' +
      '- get_url: get a shareable URL for a file',
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['upload', 'download', 'list', 'delete', 'get_url'],
          description: 'Operation to perform',
        },
        path: { type: 'string', description: 'File path in storage (e.g. "reports/q1.pdf")' },
        content: { type: 'string', description: 'Base64-encoded file content (for upload)' },
        mime_type: { type: 'string', description: 'MIME type, e.g. "image/png" (for upload)' },
        prefix: { type: 'string', description: 'Path prefix filter (for list)' },
        expires_in: { type: 'number', description: 'URL expiry in seconds, default 3600 (for get_url)' },
      },
      required: ['action'],
    },
  })
  async files(
    params: {
      action: string;
      path?: string;
      content?: string;
      mime_type?: string;
      prefix?: string;
      expires_in?: number;
    },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;

    switch (params.action) {
      case 'upload': {
        if (!params.path) return { error: 'path is required for upload' };
        if (!params.content) return { error: 'content is required for upload' };
        if (!params.mime_type) return { error: 'mime_type is required for upload' };
        const byteLength = Math.ceil(params.content.length * 0.75);
        if (byteLength > this.maxFileSize) {
          return { error: `File exceeds max size (${this.maxFileSize} bytes)` };
        }
        const res = await portalFetch(this.portalUrl, '/api/storage/files', this.apiKey, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: params.path,
            content: params.content,
            mimeType: params.mime_type,
            visibility: 'private',
            agentId,
          }),
        });
        if (!res.ok) return { error: `File upload failed: ${res.status}` };
        return res.json();
      }

      case 'download': {
        if (!params.path) return { error: 'path is required for download' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/files/${encodeURIComponent(params.path)}?agentId=${agentId}`,
          this.apiKey,
        );
        if (!res.ok) {
          if (res.status === 404) return { error: 'File not found' };
          return { error: `File download failed: ${res.status}` };
        }
        return res.json();
      }

      case 'list': {
        const qs = new URLSearchParams();
        if (agentId) qs.set('agentId', agentId);
        if (params.prefix) qs.set('prefix', params.prefix);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/files?${qs}`,
          this.apiKey,
        );
        if (!res.ok) return { error: `File list failed: ${res.status}` };
        return res.json();
      }

      case 'delete': {
        if (!params.path) return { error: 'path is required for delete' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/files/${encodeURIComponent(params.path)}?agentId=${agentId}`,
          this.apiKey,
          { method: 'DELETE' },
        );
        if (!res.ok) return { error: `File delete failed: ${res.status}` };
        return 'OK';
      }

      case 'get_url': {
        if (!params.path) return { error: 'path is required for get_url' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/files/${encodeURIComponent(params.path)}/url?agentId=${agentId}&expiresIn=${params.expires_in ?? 3600}`,
          this.apiKey,
        );
        if (!res.ok) return { error: `Get URL failed: ${res.status}` };
        return res.json();
      }

      default:
        return { error: `Unknown action: ${params.action}. Use upload, download, list, delete, or get_url.` };
    }
  }
}
