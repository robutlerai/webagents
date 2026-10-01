/**
 * The portal memory tier, as the SDK reaches it (gap-closure plan item 2.1,
 * principle 5, 2026-09-26): the platform's memory store with per-caller
 * namespaces, semantic search on its embedding and Milvus stack, and the
 * sync. One route, `/api/storage/memory-scoped` (portal
 * `lib/storage/memory-scoped-service.ts`), authenticated as the AGENT with its
 * own key (`WEBAGENTS_AGENT_TOKEN`, or the key `publish` stored), never as a
 * caller: the platform trusts the agent about which of its callers an entry
 * belongs to, because only the agent verified them, and never trusts one
 * agent about another agent's store (S-257).
 *
 * Every method throws an `Error` whose message starts with `memory:` when the
 * platform cannot be reached or refuses; the skill turns that into a tool
 * result and keeps the local tier working.
 */

import type { EntrySource, ListOptions, MemoryEntry, MemoryLogLine } from './local-store';

/** One page of a pull (portal `syncLinesSince`). */
export interface SyncPage {
  lines: MemoryLogLine[];
  /** Where the next pull starts; null when nothing was ever served. */
  cursor: string | null;
  /** Another page waits: pull again from `cursor`. */
  more: boolean;
}

export interface PortalMemoryStoreOptions {
  portalUrl: string;
  /** The agent's platform key, resolved when first needed. */
  token: () => Promise<string | undefined> | string | undefined;
  /** The agent's platform id, when the key is not bound to one. */
  agentId?: string;
  fetchImpl?: typeof fetch;
}

interface WireEntry {
  id: string;
  namespace: string;
  key: string;
  content: string;
  description?: string;
  source?: string;
  created_at: string;
  updated_at: string;
}

/** Only entries of the namespaces asked for (S-298): what the platform answers is checked, not trusted. */
function onlyWithin(entries: MemoryEntry[], namespaces: readonly string[] | null): MemoryEntry[] {
  if (namespaces === null) return entries;
  return entries.filter((e) => namespaces.includes(e.namespace));
}

function fromWire(e: WireEntry): MemoryEntry {
  return {
    id: e.id,
    namespace: e.namespace,
    key: e.key,
    content: e.content,
    source: (e.source ?? 'tool') as EntrySource,
    createdAt: e.created_at,
    updatedAt: e.updated_at,
    ...(e.description ? { description: e.description } : {}),
  };
}

export class PortalMemoryStore {
  readonly portalUrl: string;
  private readonly token: PortalMemoryStoreOptions['token'];
  private readonly agentId?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: PortalMemoryStoreOptions) {
    this.portalUrl = options.portalUrl.replace(/\/+$/, '');
    this.token = options.token;
    this.agentId = options.agentId;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  /**
   * A list value is sent as a REPEATED parameter, one value each, never
   * comma-joined (S-298, 2026-09-26): a caller principal may carry a comma,
   * and a joined list would be split on the portal into a namespace the
   * caller was never given.
   */
  private async request<T>(method: string, query: Record<string, string | readonly string[] | undefined>, body?: unknown): Promise<T> {
    const token = await this.token();
    if (!token) throw new Error('memory: no platform credential for this agent (set WEBAGENTS_AGENT_TOKEN or publish the agent).');
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === '') continue;
      if (typeof v === 'string') qs.set(k, v);
      else for (const item of v) qs.append(k, item);
    }
    if (this.agentId) qs.set('agentId', this.agentId);
    const url = `${this.portalUrl}/api/storage/memory-scoped${qs.size ? `?${qs}` : ''}`;
    let res: Response;
    try {
      res = await this.fetchImpl(url, {
        method,
        headers: { Authorization: `Bearer ${token}`, ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
        body: body !== undefined ? JSON.stringify({ ...(this.agentId ? { agentId: this.agentId } : {}), ...(body as Record<string, unknown>) }) : undefined,
      });
    } catch (err) {
      throw new Error(`memory: the platform could not be reached (${(err as Error).message}).`);
    }
    if (!res.ok) {
      let detail = '';
      try {
        detail = String(((await res.json()) as { error?: unknown }).error ?? '');
      } catch {
        // no JSON body
      }
      throw new Error(`memory: the platform answered ${res.status}${detail ? ` (${detail})` : ''}.`);
    }
    return (await res.json()) as T;
  }

  async get(namespace: string, key: string): Promise<MemoryEntry | null> {
    const data = await this.request<{ entry: WireEntry | null }>('GET', { action: 'get', namespace, key });
    return data.entry ? fromWire(data.entry) : null;
  }

  async list(namespaces: readonly string[] | null, options: ListOptions = {}): Promise<MemoryEntry[]> {
    const data = await this.request<{ entries: WireEntry[] }>('GET', {
      action: 'list',
      namespace: namespaces ?? undefined,
      prefix: options.prefix,
      limit: options.limit ? String(options.limit) : undefined,
      excludeSource: options.excludeSources?.length ? options.excludeSources : undefined,
    });
    return onlyWithin((data.entries ?? []).map(fromWire), namespaces);
  }

  async search(query: string, namespaces: readonly string[] | null, limit = 10): Promise<MemoryEntry[]> {
    const data = await this.request<{ entries: WireEntry[] }>('GET', {
      action: 'search',
      q: query,
      namespace: namespaces ?? undefined,
      limit: String(limit),
    });
    return onlyWithin((data.entries ?? []).map(fromWire), namespaces);
  }

  async put(namespace: string, key: string, content: string, source: EntrySource = 'tool', at?: string, description = ''): Promise<MemoryEntry> {
    const data = await this.request<{ entry: WireEntry }>('PUT', {}, { namespace, key, content, source, at, ...(description ? { description } : {}) });
    return fromWire(data.entry);
  }

  async forget(namespace: string, key: string): Promise<boolean> {
    const data = await this.request<{ forgotten: number }>('DELETE', { namespace, key });
    return (data.forgotten ?? 0) > 0;
  }

  /**
   * One page of what changed on the platform in a namespace after `since`
   * (null: from the start), the cursor to ask from next, and whether another
   * page waits. The cursor is the platform's own string, kept and sent back
   * as it came (S-307, 2026-09-27): it is never compared or computed here.
   */
  async pull(namespace: string, since: string | null): Promise<SyncPage> {
    const data = await this.request<{ lines?: unknown; cursor?: unknown; more?: unknown }>('GET', {
      action: 'sync',
      namespace,
      since: since ?? undefined,
    });
    return {
      lines: Array.isArray(data.lines) ? (data.lines as MemoryLogLine[]) : [],
      cursor: typeof data.cursor === 'string' && data.cursor ? data.cursor : null,
      more: data.more === true,
    };
  }

  /**
   * This machine's changes for a namespace, merged on the platform by entry
   * id. The answer carries no cursor: a push never moves a pull cursor (a
   * push answering one was S-307, which stopped every later pull).
   */
  async push(namespace: string, lines: readonly MemoryLogLine[]): Promise<{ applied: number }> {
    const data = await this.request<{ applied?: number }>('POST', { action: 'sync' }, {
      namespace,
      lines: lines.map((l) => ({
        op: l.op,
        id: l.id,
        namespace: l.namespace,
        key: l.key,
        content: l.content,
        ...(l.description ? { description: l.description } : {}),
        source: l.source,
        at: l.at,
      })),
    });
    return { applied: data.applied ?? 0 };
  }
}
