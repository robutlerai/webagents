/**
 * Stub `ctx` shape exposed to webapp templates.
 *
 * The templates target the `function-runtime` ctx surface, which the
 * function-executor materialises at run-time. For test purposes we
 * only need a faithful enough shape to (a) exercise each template's
 * happy path and (b) catch a regression where a template starts
 * relying on a field that's not actually exposed.
 *
 * NOT a full Context impl — just the slice the templates touch.
 */

export interface StubKv {
  get<T = unknown>(args: string | { user_id: string; key: string }): Promise<T | undefined>;
  put<T = unknown>(
    args: string | { user_id: string; key: string },
    value: T,
    opts?: { ttlSeconds?: number },
  ): Promise<void>;
  delete(args: string | { user_id: string; key: string }): Promise<void>;
  list(args: { user_id: string; prefix?: string }): Promise<Array<{ key: string; value: unknown }>>;
}

export interface StubSecrets {
  get(name: string): Promise<string | undefined>;
}

export interface StubPortal {
  signRequest(
    url: string,
    init?: { method?: string },
  ): Promise<{ url: string; headers: Record<string, string> }>;
}

export interface StubRequest {
  method: string;
  path: string;
  query?: Record<string, string>;
  headers: Record<string, string>;
  body?: unknown;
}

export interface StubAuth {
  authenticated: boolean;
  /** Canonical visitor user id (camelCase). */
  userId?: string | null;
  /** Canonical agent id (camelCase). */
  agentId?: string | null;
  profile?: { displayName?: string; avatarUrl?: string; email?: string };
}

export interface StubMetadata {
  agentId: string;
  agentSlug: string;
}

export interface StubCtx {
  request: StubRequest;
  metadata: StubMetadata;
  auth?: StubAuth;
  kv: StubKv;
  secrets: StubSecrets;
  portal: StubPortal;
  fetch: (url: string, init?: RequestInit) => Promise<Response>;
}

/** Default in-memory ctx with sensible defaults. Tests override fields per case. */
export function makeStubCtx(overrides: Partial<StubCtx> = {}): StubCtx {
  const store = new Map<string, unknown>();
  const kv: StubKv = {
    async get(args) {
      const key = typeof args === 'string' ? args : keyOf(args);
      return store.get(key) as never;
    },
    async put(args, value) {
      store.set(typeof args === 'string' ? args : keyOf(args), value);
    },
    async delete(args) {
      store.delete(typeof args === 'string' ? args : keyOf(args));
    },
    async list(args) {
      const prefix = `${args.user_id}::${args.prefix ?? ''}`;
      const out: Array<{ key: string; value: unknown }> = [];
      for (const [k, v] of store) {
        if (k.startsWith(prefix)) out.push({ key: k.slice(args.user_id.length + 2), value: v });
      }
      return out;
    },
  };
  const secrets: StubSecrets = { async get() { return undefined; } };
  const portal: StubPortal = {
    async signRequest(url) {
      return { url, headers: { 'x-portal-token': 'stub-token' } };
    },
  };
  return {
    request: { method: 'GET', path: '/page', headers: { host: 'robutler.ai' } },
    metadata: { agentId: 'agent-uuid-aaaa', agentSlug: 'myagent' },
    kv,
    secrets,
    portal,
    fetch: async () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    ...overrides,
  };
}

function keyOf(a: { user_id: string; key: string }): string {
  return `${a.user_id}::${a.key}`;
}
