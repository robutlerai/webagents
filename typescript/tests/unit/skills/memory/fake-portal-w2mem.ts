/**
 * A stand-in for the platform's `/api/storage/memory-scoped` route
 * (portal `lib/storage/memory-scoped-service.ts`), for the memory tests: the
 * same actions, an in-memory store per (namespace, key), word-match search,
 * and the sync as the platform serves it since 2026-09-27 (S-307; the shared
 * fixture `python/tests/fixtures/memory_tool/migrations_trim_sync.json`):
 *
 *   - every change stamps the row with the SERVER's clock (microseconds,
 *     only growing), apart from the change's own `at` that the merge compares;
 *   - a forget leaves a tombstone (no content) that the pull serves as a delete;
 *   - a pull answers the rows stamped after `since`, oldest first, a page at
 *     a time, with the last row's stamp as an opaque string cursor; anything
 *     that is not such a cursor pulls from the start;
 *   - a push answers `{applied}` and no cursor.
 *
 * It used to mint its own cursor (the log length) on a push, which is why
 * the SDK suites never saw the platform's S-307 cursor. Every request is
 * recorded in `seen`; `rows` holds the live entries only.
 */

import { entryIdFor } from '../../../../src/skills/memory/namespace';

interface Row {
  id: string;
  namespace: string;
  key: string;
  content: string;
  source: string;
  created_at: string;
  /** The change's own time, as the writer stamped it (what the merge compares). */
  updated_at: string;
}

interface Stored {
  row: Row;
  deleted: boolean;
  /** The server's stamp, the cursor's unit. */
  stampUs: number;
}

interface LogLine {
  seq?: number;
  op: 'put' | 'delete';
  id: string;
  namespace: string;
  key: string;
  content?: string;
  source?: string;
  at: string;
}

export interface SeenRequest {
  method: string;
  url: URL;
  headers: Record<string, string>;
  body?: Record<string, unknown>;
}

const CURSOR_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$/;

function stampText(us: number): string {
  const seconds = Math.floor(us / 1_000_000);
  return `${new Date(seconds * 1000).toISOString().slice(0, 19)}.${String(us - seconds * 1_000_000).padStart(6, '0')}Z`;
}

function stampOf(text: string): number {
  const m = /^(.{19})(?:\.(\d{1,6}))?Z$/.exec(text)!;
  return Date.parse(`${m[1]}Z`) * 1000 + Number((m[2] ?? '').padEnd(6, '0'));
}

export function fakePortal(agentId = '7f3c2a10-5b6e-4c8d-9e1f-0a2b3c4d5e6f', options: { pageSize?: number } = {}) {
  const store = new Map<string, Stored>();
  const rows = new Map<string, Row>();
  const seen: SeenRequest[] = [];
  const pageSize = options.pageSize ?? 500;
  let clock = 0;
  let serverUs = Date.parse('2023-11-14T22:13:20Z') * 1000;

  const now = () => new Date(1_700_000_000_000 + (clock += 1000)).toISOString();
  const nextStamp = () => (serverUs += 1_000);
  const rowKey = (ns: string, key: string) => `${ns}\n${key}`;

  /** A change made on the platform (the owner's panel, another machine): it wins over what the agent has when its `at` is later. */
  function put(ns: string, key: string, content: string, source: string, at?: string): Row {
    const existing = store.get(rowKey(ns, key));
    const when = at ?? now();
    const row: Row = {
      id: entryIdFor(agentId, ns, key),
      namespace: ns,
      key,
      content,
      source,
      created_at: existing && !existing.deleted ? existing.row.created_at : when,
      updated_at: when,
    };
    store.set(rowKey(ns, key), { row, deleted: false, stampUs: nextStamp() });
    rows.set(rowKey(ns, key), row);
    return row;
  }

  function del(ns: string, key: string, at?: string): number {
    const existing = store.get(rowKey(ns, key));
    if (!existing || existing.deleted) return 0;
    const when = at ?? now();
    store.set(rowKey(ns, key), { row: { ...existing.row, content: '', updated_at: when }, deleted: true, stampUs: nextStamp() });
    rows.delete(rowKey(ns, key));
    return 1;
  }

  const json = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url);
    const method = (init?.method ?? 'GET').toUpperCase();
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries((init?.headers as Record<string, string>) ?? {})) headers[k.toLowerCase()] = v;
    const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : undefined;
    seen.push({ method, url, headers, body });
    if (headers.authorization !== 'Bearer agent-key') return json(401, { error: 'Unauthorized' });
    const q = url.searchParams;
    // Lists are repeated parameters (S-298); the retired comma-joined form is refused as the route refuses it.
    if (q.has('namespaces') || q.has('excludeSources')) return json(400, { error: 'namespaces is not accepted; repeat namespace=<value> once per value' });
    const namespaces = q.getAll('namespace').length ? q.getAll('namespace') : null;
    const within = (r: Row) => !namespaces || namespaces.includes(r.namespace);

    if (method === 'GET' && q.get('action') === 'search') {
      const words = (q.get('q') ?? '').toLowerCase().match(/[a-z0-9]+/g) ?? [];
      const hits = [...rows.values()]
        .filter(within)
        .filter((r) => words.some((w) => `${r.key} ${r.content}`.toLowerCase().includes(w)))
        .slice(0, Number(q.get('limit') ?? 10));
      return json(200, { entries: hits });
    }
    if (method === 'GET' && q.get('action') === 'list') {
      const excluded = q.getAll('excludeSource');
      const prefix = q.get('prefix') ?? '';
      const entries = [...rows.values()]
        .filter(within)
        .filter((r) => r.key.startsWith(prefix) && !excluded.includes(r.source))
        .sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1))
        .slice(0, Number(q.get('limit') ?? 50));
      return json(200, { entries });
    }
    if (method === 'GET' && q.get('action') === 'get') {
      return json(200, { entry: rows.get(rowKey(q.get('namespace')!, q.get('key')!)) ?? null });
    }
    if (method === 'GET' && q.get('action') === 'sync') {
      const raw = q.get('since');
      const since = raw && CURSOR_RE.test(raw) && stampOf(raw) <= serverUs ? raw : null;
      const changed = [...store.values()]
        .filter((s) => s.row.namespace === q.get('namespace') && (since === null || s.stampUs > stampOf(since)))
        .sort((a, b) => a.stampUs - b.stampUs);
      const page = changed.slice(0, pageSize);
      const lines: LogLine[] = page.map((s) =>
        s.deleted
          ? { op: 'delete', id: s.row.id, namespace: s.row.namespace, key: s.row.key, at: s.row.updated_at }
          : { op: 'put', id: s.row.id, namespace: s.row.namespace, key: s.row.key, content: s.row.content, source: s.row.source, at: s.row.updated_at },
      );
      return json(200, { lines, cursor: page.length ? stampText(page[page.length - 1].stampUs) : since, more: changed.length > pageSize });
    }
    if (method === 'PUT') {
      const b = body as { namespace: string; key: string; content: string; source?: string; at?: string };
      return json(200, { entry: put(b.namespace, b.key, b.content, b.source ?? 'tool', b.at) });
    }
    if (method === 'DELETE') {
      return json(200, { forgotten: del(q.get('namespace')!, q.get('key')!) });
    }
    if (method === 'POST' && q.get('action') === 'sync') {
      const b = body as { namespace: string; lines: LogLine[] };
      let applied = 0;
      for (const line of b.lines) {
        const existing = store.get(rowKey(line.namespace, line.key));
        if (existing && existing.row.updated_at >= line.at) continue;
        if (line.op === 'put') {
          put(line.namespace, line.key, line.content ?? '', line.source === 'sync' ? 'tool' : line.source ?? 'tool', line.at);
          applied += 1;
        } else {
          applied += del(line.namespace, line.key, line.at);
        }
      }
      return json(200, { applied });
    }
    return json(400, { error: 'Unknown action' });
  }) as unknown as typeof fetch;

  return { agentId, rows, store, seen, fetch: fetchImpl, put, del };
}
