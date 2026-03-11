/**
 * Storage Skills
 *
 * Platform storage capabilities via portal API:
 * - Memory (KV): Consolidated key-value storage with action parameter
 * - Files: Consolidated file storage with action parameter
 * - JSON: Structured JSON document storage (opt-in, not default)
 *
 * All storage is scoped to the agent's namespace and authenticated
 * via the portal API.
 */

import { Skill } from '../../core/skill.js';
import { tool } from '../../core/decorators.js';
import type { Context } from '../../core/types.js';

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
// KV Storage Skill (consolidated as `memory`)
// ---------------------------------------------------------------------------

export class RobutlerKVSkill extends Skill {
  private portalUrl: string;
  private apiKey?: string;
  private agentId?: string;
  private defaultTtl: number;

  constructor(config: StorageConfig = {}) {
    super({ ...config, name: config.name || 'robutler-kv' });
    this.portalUrl = config.portalUrl ?? process.env.PORTAL_URL ?? 'https://robutler.ai';
    this.apiKey = config.apiKey ?? process.env.PLATFORM_SERVICE_KEY;
    this.agentId = config.agentId;
    this.defaultTtl = config.defaultTtl ?? 0;
  }

  @tool({
    name: 'memory',
    description:
      'Persistent key-value memory. Use this to remember things across conversations -- ' +
      'user preferences, task results, notes, or any data you want to recall later.\n\n' +
      'Actions:\n' +
      '- get: retrieve a stored value by key\n' +
      '- set: store a value (string, number, object, or array)\n' +
      '- delete: remove a stored key\n' +
      '- list: list all stored keys (optionally filtered by prefix)',
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['get', 'set', 'delete', 'list'],
          description: 'Operation to perform',
        },
        key: { type: 'string', description: 'Storage key (required for get/set/delete)' },
        value: { description: 'Value to store - any JSON-serializable value (required for set)' },
        ttl: { type: 'number', description: 'Time-to-live in seconds, 0 = no expiry (for set)' },
        prefix: { type: 'string', description: 'Filter keys by prefix (for list)' },
      },
      required: ['action'],
    },
  })
  async memory(
    params: { action: string; key?: string; value?: unknown; ttl?: number; prefix?: string },
    context: Context,
  ): Promise<unknown> {
    const agentId = this.agentId ?? context.auth?.agentId;

    switch (params.action) {
      case 'get': {
        if (!params.key) return { error: 'key is required for get' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/kv/${encodeURIComponent(params.key)}?agentId=${agentId}`,
          this.apiKey,
        );
        if (!res.ok) {
          if (res.status === 404) return null;
          return { error: `Memory get failed: ${res.status}` };
        }
        return res.json();
      }

      case 'set': {
        if (!params.key) return { error: 'key is required for set' };
        if (params.value === undefined) return { error: 'value is required for set' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/kv/${encodeURIComponent(params.key)}`,
          this.apiKey,
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              value: params.value,
              ttl: params.ttl ?? this.defaultTtl,
              agentId,
            }),
          },
        );
        if (!res.ok) return { error: `Memory set failed: ${res.status}` };
        return 'OK';
      }

      case 'delete': {
        if (!params.key) return { error: 'key is required for delete' };
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/kv/${encodeURIComponent(params.key)}?agentId=${agentId}`,
          this.apiKey,
          { method: 'DELETE' },
        );
        if (!res.ok) return { error: `Memory delete failed: ${res.status}` };
        return 'OK';
      }

      case 'list': {
        const qs = new URLSearchParams();
        if (agentId) qs.set('agentId', agentId);
        if (params.prefix) qs.set('prefix', params.prefix);
        const res = await portalFetch(
          this.portalUrl,
          `/api/storage/kv?${qs}`,
          this.apiKey,
        );
        if (!res.ok) return { error: `Memory list failed: ${res.status}` };
        return res.json();
      }

      default:
        return { error: `Unknown action: ${params.action}. Use get, set, delete, or list.` };
    }
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
        const res = await portalFetch(this.portalUrl, `/api/storage/files?${qs}`, this.apiKey);
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
          `/api/storage/files/${encodeURIComponent(params.path)}/url`,
          this.apiKey,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              expiresIn: params.expires_in ?? 3600,
              agentId,
            }),
          },
        );
        if (!res.ok) return { error: `Get URL failed: ${res.status}` };
        return res.json();
      }

      default:
        return { error: `Unknown action: ${params.action}. Use upload, download, list, delete, or get_url.` };
    }
  }
}
