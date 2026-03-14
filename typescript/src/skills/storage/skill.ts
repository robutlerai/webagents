/**
 * Storage Skills
 *
 * Platform storage capabilities via portal API:
 * - Memory: Store-based key-value storage with grants, search, sharing
 * - Files: Consolidated file storage with action parameter
 * - JSON: Structured JSON document storage (opt-in, not default)
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context, Tool } from '../../core/types.js';

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
      storeLines.push(`- ${this.agentId} (self): Your persistent memory`);
    }
    for (const s of this.contextStores) {
      storeLines.push(`- ${s.storeId} (${s.label}): ${s.label} memory`);
    }

    const storesSection = storeLines.length > 0
      ? `\nAvailable stores:\n${storeLines.join('\n')}\nUse 'stores' action to discover additional stores shared with you.\n`
      : '';

    const description =
      `Persistent memory. Store and retrieve information across conversations.` +
      storesSection +
      `\nActions:\n` +
      `- get(store, key): retrieve a stored value\n` +
      `- set(store, key, value, ttl?): store a value\n` +
      `- delete(store, key): remove a key (own entries only)\n` +
      `- list(store, prefix?): list keys in a store\n` +
      `- search(query, store?): full-text search (omit store to search all accessible stores)\n` +
      `- share(store, agent, level?): grant another agent access (search, read, or readwrite)\n` +
      `- unshare(store, agent): revoke a grant\n` +
      `- stores(): list all stores you can access`;

    const defaultStore = this.agentId ?? 'self';

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
          store: {
            type: 'string',
            format: 'uuid',
            description: `UUID of the store. Default: ${defaultStore} (your own memory).`,
          },
          key: { type: 'string', description: 'Storage key (required for get/set/delete)' },
          value: { description: 'Value to store - any JSON-serializable value (required for set)' },
          ttl: { type: 'number', description: 'Time-to-live in seconds, 0 = no expiry (for set)' },
          prefix: { type: 'string', description: 'Filter keys by prefix (for list)' },
          query: { type: 'string', description: 'Search query text (for search)' },
          agent: { type: 'string', description: 'Agent UUID (for share/unshare)' },
          level: {
            type: 'string',
            enum: ['search', 'read', 'readwrite'],
            description: 'Access level (for share). Default: read',
          },
        },
        required: ['action'],
      },
      scopes: ['all'],
      enabled: true,
      handler: (params: Record<string, unknown>, context: Context) =>
        this._handleMemory(params as any, context),
    } as Tool);
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
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;
    const store = params.store ?? agentId;

    switch (params.action) {
      case 'get': {
        if (!params.key) return { error: 'key is required for get' };
        const qs = new URLSearchParams({ agentId: agentId!, store: store! });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory/${encodeURIComponent(params.key)}?${qs}`,
          this.apiKey,
        );
        if (!res.ok) {
          if (res.status === 404) return null;
          if (res.status === 403) return { error: 'Access denied to this store' };
          return { error: `Memory get failed: ${res.status}` };
        }
        return res.json();
      }

      case 'set': {
        if (!params.key) return { error: 'key is required for set' };
        if (params.value === undefined) return { error: 'value is required for set' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory/${encodeURIComponent(params.key)}`,
          this.apiKey,
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              value: params.value,
              ttl: params.ttl ?? this.defaultTtl,
              agentId,
              store,
              chatId: this.chatId,
              userId: this.userId,
            }),
          },
        );
        if (!res.ok) {
          if (res.status === 403) return { error: 'Access denied to this store' };
          if (res.status === 413) return { error: 'Value too large' };
          if (res.status === 429) return { error: 'Key limit reached for this store' };
          return { error: `Memory set failed: ${res.status}` };
        }
        return 'OK';
      }

      case 'delete': {
        if (!params.key) return { error: 'key is required for delete' };
        const qs = new URLSearchParams({ agentId: agentId!, store: store! });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory/${encodeURIComponent(params.key)}?${qs}`,
          this.apiKey,
          { method: 'DELETE' },
        );
        if (!res.ok) {
          if (res.status === 403) return { error: 'Access denied to this store' };
          return { error: `Memory delete failed: ${res.status}` };
        }
        return 'OK';
      }

      case 'list': {
        const qs = new URLSearchParams({ agentId: agentId!, store: store!, inContext: 'true' });
        if (params.prefix) qs.set('prefix', params.prefix);
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory?${qs}`,
          this.apiKey,
        );
        if (!res.ok) {
          if (res.status === 403) return { error: 'Access denied to this store' };
          return { error: `Memory list failed: ${res.status}` };
        }
        return res.json();
      }

      case 'search': {
        if (!params.query) return { error: 'query is required for search' };
        const qs = new URLSearchParams({ action: 'search', agentId: agentId!, q: params.query });
        if (store && store !== agentId) qs.set('store', store);
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory?${qs}`,
          this.apiKey,
        );
        if (!res.ok) return { error: `Memory search failed: ${res.status}` };
        return res.json();
      }

      case 'share': {
        if (!params.agent) return { error: 'agent is required for share' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory?action=share`,
          this.apiKey,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              agentId,
              store,
              grantee: params.agent,
              level: params.level ?? 'read',
              chatId: this.chatId,
              userId: this.userId,
            }),
          },
        );
        if (!res.ok) {
          if (res.status === 403) return { error: 'You need readwrite access to share this store' };
          return { error: `Memory share failed: ${res.status}` };
        }
        return 'OK';
      }

      case 'unshare': {
        if (!params.agent) return { error: 'agent is required for unshare' };
        const qs = new URLSearchParams({
          action: 'share',
          agentId: agentId!,
          store: store!,
          grantee: params.agent,
        });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory?${qs}`,
          this.apiKey,
          { method: 'DELETE' },
        );
        if (!res.ok) return { error: `Memory unshare failed: ${res.status}` };
        return 'OK';
      }

      case 'stores': {
        const qs = new URLSearchParams({ action: 'stores', agentId: agentId! });
        if (this.chatId) qs.set('chatId', this.chatId);
        if (this.userId) qs.set('userId', this.userId);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/memory?${qs}`,
          this.apiKey,
        );
        if (!res.ok) return { error: `Memory stores failed: ${res.status}` };
        return res.json();
      }

      default:
        return { error: `Unknown action: ${params.action}. Use get, set, delete, list, search, share, unshare, or stores.` };
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
