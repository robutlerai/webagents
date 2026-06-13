/**
 * `CustomHttpSkill`
 *
 * Reads `agent_configs.skills.custom_http.endpoints[]` and registers each
 * entry as an HTTP endpoint on the agent. The handler resolves the
 * referenced function via `FunctionRuntimeSkill.invoke(use, ctx)` and
 * shapes the executor response into a `Response`.
 *
 * Auth modes (`public` / `signature` / `session` / `portal_token`) are
 * pass-through to the dispatcher — this skill does NOT verify auth itself.
 * The dispatcher's `auth` mode tells it how to populate `ctx.auth`.
 */

import { Skill } from '../../core/skill';
import { prompt, tool } from '../../core/decorators';
import type {
  Context,
  HttpAuthMode,
  HttpEndpoint,
  HttpHandler,
  HttpMethod,
  SkillConfig,
} from '../../core/types';
import type { FunctionRuntimeSkill } from '../functions/skill';
import type { SerializableContext } from '../functions/executor-client';
import type { FunctionLimitsResolved } from '../functions/context';
import {
  WEBAPP_TEMPLATES,
  WEBAPP_TEMPLATE_NAMES,
  type WebappTemplate,
  type WebappTemplateName,
} from './templates';

/** Single endpoint declaration. */
export interface CustomHttpEndpointEntry {
  id: string;
  /** URL path; supports `:name` template params. */
  path: string;
  method: HttpMethod;
  auth: HttpAuthMode;
  /** Function name to invoke (required for custom_http — no host fallback). */
  use: string;
  /** Per-endpoint timeout (defaults to function manifest limits.wallMs). */
  timeoutMs?: number;
  /** Maximum inbound body bytes (post Content-Length check). */
  bodyLimitBytes?: number;
  enabled?: boolean;
  description?: string;
  /**
   * Phase 10 — flag the endpoint as a home-screen widget. The
   * dispatcher relaxes the default `X-Frame-Options: DENY` /
   * `frame-ancestors 'none'` so the platform widget runner can
   * embed the rasterised HTML in same-origin iframes for snapshot.
   */
  widget?: WidgetManifest;
}

/** Widget metadata declared on a `custom_http` endpoint. */
export interface WidgetManifest {
  /** Short human-readable label shown by the widget gallery. */
  title?: string;
  /** Default rasterisation viewport (CSS px). */
  defaultSize?: { w: number; h: number };
  /** Optional override for narrow mobile widgets. */
  mobileSize?: { w: number; h: number };
  /** When true, the platform may snapshot at multiple sizes. */
  supportsResize?: boolean;
}

export interface CustomHttpSkillConfig extends SkillConfig {
  endpoints?: CustomHttpEndpointEntry[];
  /** Optional explicit runtime skill. When undefined, the runtime is looked up from the agent registry at first invocation. */
  runtime?: FunctionRuntimeSkill;
  /**
   * Lookup function for the FunctionRuntimeSkill. Called lazily so the
   * agent's normal mount flow (which auto-mounts function-runtime via
   * `dependencies`) can satisfy the lookup without explicit wiring.
   */
  resolveRuntime?: () => FunctionRuntimeSkill | undefined;
}

const DEFAULT_LIMITS: FunctionLimitsResolved = {
  wallMs: 30_000,
  cpuMs: 5_000,
  memoryMb: 128,
  ingressBytes: 1_000_000,
  egressBytes: 1_000_000,
};

/**
 * Build a default lazy-resolver: walks the agent's skills array (set via
 * `setAgent`) for an instance whose `name === 'function-runtime'`. The
 * dep-validation pass guarantees one is mounted whenever this skill is
 * present, so the lookup is safe at first invoke.
 */
function defaultRuntimeResolver(skill: CustomHttpSkill): () => FunctionRuntimeSkill | undefined {
  return () => {
    // Cast through `unknown` to read the private `_agent` field — the
    // skill's `hasFunctionRuntime` method below also reads this field
    // directly, which is what keeps noUnusedLocals satisfied.
    const agent = (skill as unknown as { _agent?: { skills?: Array<{ name?: string }> } })._agent;
    if (!agent?.skills) return undefined;
    return agent.skills.find((s) => s?.name === 'function-runtime') as FunctionRuntimeSkill | undefined;
  };
}

export class CustomHttpSkill extends Skill {
  readonly name = 'custom_http';
  readonly dependencies = ['function-runtime'] as const;

  private readonly endpoints: CustomHttpEndpointEntry[];
  private readonly resolveRuntime: () => FunctionRuntimeSkill | undefined;
  private explicitRuntime?: FunctionRuntimeSkill;
  /**
   * Set by `BaseAgent.addSkill` when it sees a `setAgent` method.
   * Read by `hasFunctionRuntime` directly (which is what keeps
   * noUnusedLocals happy) and by `defaultRuntimeResolver` via cast.
   */
  private _agent?: { skills?: Array<{ name?: string }> };

  constructor(config: CustomHttpSkillConfig = {}) {
    super(config);
    this.endpoints = (config.endpoints ?? []).filter((e) => e.enabled !== false);
    this.explicitRuntime = config.runtime;
    this.resolveRuntime =
      config.resolveRuntime ?? defaultRuntimeResolver(this);
    this.registerAll();
  }

  /** Called by `BaseAgent.addSkill` (it checks for `setAgent`). */
  setAgent(agent: unknown): void {
    this._agent = agent as { skills?: Array<{ name?: string }> };
  }

  list(): readonly CustomHttpEndpointEntry[] {
    return this.endpoints;
  }

  /** Validate endpoints — used by the save route. */
  validate(declaredFunctions: ReadonlySet<string>): Array<{ id: string; error: string }> {
    const errors: Array<{ id: string; error: string }> = [];
    const seenRoutes = new Set<string>();
    for (const e of this.endpoints) {
      if (!e.id) errors.push({ id: '', error: 'endpoint.id is required' });
      if (!e.path?.startsWith('/')) errors.push({ id: e.id, error: 'path must start with /' });
      if (!e.method) errors.push({ id: e.id, error: 'method is required' });
      if (!e.use) errors.push({ id: e.id, error: 'custom_http endpoints require "use"' });
      if (e.use && !declaredFunctions.has(e.use)) {
        errors.push({ id: e.id, error: `function "${e.use}" not declared in agent_configs.functions` });
      }
      const key = `${e.method} ${e.path}`;
      if (seenRoutes.has(key)) {
        errors.push({ id: e.id, error: `route conflict: ${key} declared twice` });
      } else {
        seenRoutes.add(key);
      }
    }
    return errors;
  }

  private registerAll(): void {
    for (const ep of this.endpoints) {
      const handler = this.makeHandler(ep);
      // Project the function manifest's `permissions.visitor_profile`
      // onto the endpoint metadata so the dispatcher can decide which
      // profile fields to load WITHOUT having to re-invoke the runtime.
      // Resolved lazily because the runtime may not be mounted yet at
      // skill construction time — fall back to undefined when the
      // function isn't resolvable.
      let visitorProfile: HttpEndpoint['visitorProfile'] | undefined;
      try {
        const r = this.explicitRuntime ?? this.resolveRuntime();
        const decl = r?.get(ep.use);
        const perms = decl?.manifest.permissions as
          | { visitor_profile?: readonly ('name' | 'avatar' | 'email')[] }
          | undefined;
        if (perms?.visitor_profile && Array.isArray(perms.visitor_profile)) {
          visitorProfile = perms.visitor_profile;
        }
      } catch {
        // Runtime not yet mounted — dispatcher will fall back to no
        // profile, which is the safe default.
      }
      const endpoint: HttpEndpoint = {
        path: ep.path,
        method: ep.method,
        auth: ep.auth,
        enabled: true,
        visitorProfile,
        widget: ep.widget,
        handler,
      };
      this.registerHttpEndpoint(endpoint);
    }
  }

  private resolveRuntimeOrThrow(): FunctionRuntimeSkill {
    const r = this.explicitRuntime ?? this.resolveRuntime();
    if (!r) {
      throw new Error(
        'CustomHttpSkill requires FunctionRuntimeSkill to be mounted on the agent ' +
          '(usually auto-mounted via Skill.dependencies). ',
      );
    }
    return r;
  }

  private makeHandler(ep: CustomHttpEndpointEntry): HttpHandler {
    return async (request, ctx) => {
      const runtime = this.resolveRuntimeOrThrow();
      const url = new URL(request.url);
      const params = matchRouteTemplate(ep.path, url.pathname);
      const headers: Record<string, string> = {};
      request.headers.forEach((v, k) => {
        const lower = k.toLowerCase();
        // Defensive mirror of the dispatcher's Phase 1 sanitisation —
        // the dispatcher is the primary enforcement point, but a
        // misconfigured caller (test harness, future code path that
        // bypasses the dispatcher) could re-leak cookies if we trusted
        // the request blindly. Strip the same set of headers and apply
        // the same per-agent cookie namespace filter here too.
        if (
          lower === 'authorization' ||
          lower === 'set-cookie' ||
          lower === 'x-forwarded-host' ||
          lower === 'x-forwarded-proto' ||
          lower.startsWith('x-robutler-')
        ) {
          return;
        }
        if (lower === 'cookie') {
          const filtered = filterAgentCookies(v, extractAgentId(ctx));
          if (filtered) headers[lower] = filtered;
          return;
        }
        headers[lower] = v;
      });

      const includeRawBody = runtime.get(ep.use)?.manifest.permissions?.rawBody === true;
      let body: unknown = null;
      let rawBody: Uint8Array | undefined;
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        try {
          if (includeRawBody) {
            const buf = new Uint8Array(await request.arrayBuffer());
            rawBody = buf;
            body = decodeBody(buf, headers['content-type'] ?? '');
          } else {
            body = await request.json().catch(() => null);
            if (body === null) {
              const text = await request.clone().text().catch(() => '');
              if (text) body = text;
            }
          }
        } catch {
          // ignore — empty body
        }
      }

      const query: Record<string, string> = {};
      url.searchParams.forEach((v, k) => {
        query[k] = v;
      });

      const serializable: SerializableContext = {
        source: {
          skill: 'custom_http',
          consumerId: ep.id,
          invocationId: cryptoRandomId(),
        },
        request: {
          method: request.method,
          path: url.pathname,
          params,
          query,
          headers,
          body,
          rawBody,
        },
        auth: extractAuth(ctx),
        limits: { ...DEFAULT_LIMITS, wallMs: ep.timeoutMs ?? DEFAULT_LIMITS.wallMs },
      };

      const result = await runtime.invoke<unknown>(ep.use, serializable);

      if (!result.ok) {
        return new Response(
          JSON.stringify({
            error: { code: result.errorCode ?? 'FUNCTION_ERROR', message: result.errorMessage ?? 'function failed' },
          }),
          {
            status: errorCodeToStatus(result.errorCode),
            headers: { 'content-type': 'application/json' },
          },
        );
      }

      return shapeResponse(result.result);
    };
  }

  // ---------------------------------------------------------------------
  // Webapp template tools — surface the 14 reference recipes through
  // discrete LLM-callable tools so the agent doesn't have to memorise
  // the catalogue. The companion `webappPatterns` @prompt advertises
  // the names + a one-liner; `get_webapp_template` returns the full
  // metadata + source so the LLM can adapt it.
  // ---------------------------------------------------------------------

  // ---------------------------------------------------------------------
  // DEPRECATED — webapp template tools.
  //
  // These two tools are kept as thin shims so existing prompts that
  // still call `list_webapp_templates` / `get_webapp_template` keep
  // working through the deprecation window. New code should use the
  // unified factory knowledge surface:
  //
  //   - `get_doc("templates/<name>")`  → full template body
  //   - the `factory-knowledge-index` prompt advertises every slug
  //
  // These shims will be removed once `FactoryKnowledgeSkill` is mounted
  // on every surface that consumes templates.
  // ---------------------------------------------------------------------

  @tool({
    name: 'list_webapp_templates',
    description:
      '[DEPRECATED — prefer `get_doc("templates/<name>")` via FactoryKnowledgeSkill] List the available custom_http webapp templates (name + 1-line description). Call `get_webapp_template` to fetch the source for one of them.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
  })
  async list_webapp_templates(): Promise<{
    deprecated: true;
    replacement: string;
    templates: Array<{ name: WebappTemplateName; description: string; slug: string }>;
  }> {
    return {
      deprecated: true,
      replacement: 'get_doc("templates/<name>") via FactoryKnowledgeSkill',
      templates: WEBAPP_TEMPLATE_NAMES.map((name) => ({
        name,
        description: WEBAPP_TEMPLATES[name].description,
        slug: `templates/${name}`,
      })),
    };
  }

  @tool({
    name: 'get_webapp_template',
    description:
      '[DEPRECATED — prefer `get_doc("templates/<name>")` via FactoryKnowledgeSkill] Fetch the full metadata + reference source for a named custom_http webapp template.',
    parameters: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          enum: WEBAPP_TEMPLATE_NAMES as unknown as string[],
          description: 'The template name; use `list_webapp_templates` to see options.',
        },
      },
      required: ['pattern'],
      additionalProperties: false,
    },
  })
  async get_webapp_template(args: { pattern: WebappTemplateName }): Promise<
    WebappTemplate & { deprecated: true; replacement: string }
  > {
    const t = WEBAPP_TEMPLATES[args.pattern];
    if (!t) {
      throw new Error(
        `Unknown webapp template "${args.pattern}". Call list_webapp_templates for the catalogue, or use get_doc("templates/<name>").`,
      );
    }
    return {
      ...t,
      deprecated: true,
      replacement: `get_doc("templates/${args.pattern}") via FactoryKnowledgeSkill`,
    };
  }

  /**
   * Webapp template catalogue prompt.
   *
   * Gated on `function-runtime` being mounted: the templates all rely
   * on `ctx.kv` / `ctx.secrets` / `ctx.fetch`, so they're useless
   * without the runtime. We probe the live agent's skills array
   * (populated by `setAgent`) at render time — `Skill.dependencies` is
   * static and can't tell us what's actually mounted on this agent.
   *
   * Stays under 1500 chars to leave room for `customHttpRuntime`
   * (priority 71, ~500 chars) and the common system preamble.
   */
  @prompt({ priority: 70, name: 'webappPatterns', scope: 'all' })
  webappPatterns(_ctx: Context): string {
    if (!this.hasFunctionRuntime()) return '';
    const lines = [
      '## Webapp templates',
      'Reference recipes for serving HTML/JSON pages from `custom_http`. Call `get_webapp_template` with the name to read the source + `required_permissions`.',
      '',
    ];
    for (const name of WEBAPP_TEMPLATE_NAMES) {
      lines.push(`- \`${name}\` — ${WEBAPP_TEMPLATES[name].description}`);
    }
    lines.push(
      '',
      '### Cookies + Set-Cookie (HARD-ENFORCED)',
      "- Cookie name MUST start with `agt_<ctx.auth.agentId>_` — the canonical agent UUID, not the username.",
      '- NEVER set `Domain=` on a Set-Cookie. The dispatcher rejects with 502.',
      '- NEVER use reserved names: `session`, `logged_in`, anything starting `_robutler_`. Hard-rejected with 502.',
      '- Inbound cookies are filtered to the same `agt_<id>_` prefix — your function never sees the platform session cookie.',
      '',
      '### Identity model',
      "- For 'who is this visitor?' use `auth: 'visitor_session'` + `permissions.visitor_profile: ['name','avatar']`. Do NOT roll Google OAuth for identity — use it ONLY when you need Google API access (Calendar, Gmail, Drive).",
    );
    return lines.join('\n');
  }

  /** True when a `function-runtime` skill is mounted on the live agent. */
  private hasFunctionRuntime(): boolean {
    const agent = this._agent;
    if (!agent?.skills) return !!this.explicitRuntime;
    return agent.skills.some((s) => s?.name === 'function-runtime');
  }

  @prompt({ priority: 71, name: 'customHttpRuntime', scope: 'all' })
  customHttpRuntime(_ctx: Context): string {
    const lines: string[] = ['## Custom HTTP endpoints'];

    if (this.endpoints.length === 0) {
      lines.push(
        'No custom HTTP endpoints are configured. To expose one, declare a function (`declare_function`) and attach it via `add_to_skill skill="custom_http"`.',
      );
    } else {
      lines.push('Endpoints exposed by this agent (handled by the function executor — NOT by you in chat):');
      for (const e of this.endpoints) {
        lines.push(
          `- ${e.method} ${e.path} (auth: ${e.auth}) -> ${e.use}${e.description ? ` — ${e.description}` : ''}`,
        );
      }
    }

    lines.push(
      '',
      '### Behavior',
      '- Endpoints are served from the function executor, not from this chat. The canonical browser-facing URL is `https://robutler.ai/agents/<id>/<path>` (the legacy `/api/agents/<id>/<path>` form keeps working). DO NOT try to call them via `url_fetch` from inside the chat as a substitute for invoking the function directly.',
      '- **Auth modes** affect what `ctx.auth` looks like inside the function:',
      '  - `public`: anyone can call. `ctx.auth = { authenticated: false }`.',
      '  - `signature`: HMAC-signed by the caller (third-party webhooks).',
      '  - `session`: OWNER-ONLY. Requires the agent OWNER to be signed in to robutler.ai; rejects anyone else with 401. Use ONLY for owner admin pages.',
      '  - `visitor_session`: ANY signed-in Robutler visitor. `ctx.auth.user_id` set without ownership check. Pair with `permissions.visitor_profile: ["name","avatar"]` for `ctx.auth.profile.{displayName,avatarUrl}` (Robutler-as-IdP — no Google OAuth needed for "who is this visitor"). NO permissive CORS — same-origin XHR only.',
      '  - `portal_token`: scoped portal JWT required (inter-agent RPC). `ctx.auth.{user_id,agent_id,scopes}` set.',
      '- **Default response headers (text/html only)**: CSP (`default-src \'self\'`, `script/style-src \'self\' \'unsafe-inline\'`, `connect-src \'self\'`, `frame-ancestors \'none\'`), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, COOP, CORP. Override per-key by setting the same header in your response. Widgets (`widget: {...}` on the endpoint) relax to `frame-ancestors \'self\'` + `XFO SAMEORIGIN`.',
      '- **2 MB response cap** is the default. Configurable per-endpoint via `responseSizeLimitBytes`; overflow returns 502 + owner warning.',
      '- **Cookies**: incoming cookies are filtered to the `agt_<ctx.auth.agentId>_*` namespace (your function never sees the platform session cookie or other agents\' cookies). Outgoing `Set-Cookie` MUST start with `agt_<ctx.auth.agentId>_` — the dispatcher hard-rejects (502) anything with `Domain=`, reserved names (`session`, `logged_in`, `_robutler_*`), or wrong prefix.',
      '- **Agent-app sessions**: store your session record in `ctx.kv` keyed under `ctx.auth.userId` (visitor) or `ctx.auth.agentId` (self). For OAuth flows, see `oauth_google_login` template. Browser-storage caveat: `localStorage` is shared across all agent webapps on the same `robutler.ai` origin — never store secrets there; use HttpOnly cookies + `ctx.kv`.',
      '- **Custom domains**: `visitor_session` only works on `robutler.ai`-origin requests because the platform cookie is host-scoped. Custom-domain agents (`https://myagent.com/...`) get anonymous `ctx.auth = { authenticated: false }`; either roll your own auth (see `oauth_google_login` template) or redirect users to `/login` with the `signin_with_robutler` template.',
      '- **Idempotency**: external callers will retry on 5xx. Make the function idempotent (key state writes by request id / external event id), or document a sensible at-least-once contract back to the user.',
      '- **Body limits**: default ingress is 1 MB; override per-endpoint via `bodyLimitBytes`. Larger bodies fail before the function runs.',
      '- **Error shape**: `{ "error": { "code": "<errorCode>", "message": "..." } }` with HTTP status mapped from the error code (FN_NOT_FOUND→404, *QUOTA*→429, WALL_TIMEOUT→504, RESPONSE_SIZE_CAP_EXCEEDED→502, others→500). The executor emits `WALL_TIMEOUT` when the function exceeds `limits.wallMs`; functions either `throw new Error(...)` for unexpected failures or return a structured `{ status, headers, body }` for explicit non-200 responses.',
      '- Routes match `method + path`. `:name` parameters in the path become `ctx.request.params.name`. Route conflicts (same method+path twice) fail validation at save time.',
    );

    return lines.join('\n');
  }
}

function matchRouteTemplate(template: string, actual: string): Record<string, string> {
  const tParts = template.split('/').filter(Boolean);
  const aParts = actual.split('/').filter(Boolean);
  if (tParts.length !== aParts.length) return {};
  const out: Record<string, string> = {};
  for (let i = 0; i < tParts.length; i++) {
    if (tParts[i].startsWith(':')) {
      out[tParts[i].slice(1)] = decodeURIComponent(aParts[i]);
    } else if (tParts[i] !== aParts[i]) {
      return {};
    }
  }
  return out;
}

function extractAuth(ctx: Context): SerializableContext['auth'] {
  const auth = ctx.auth as
    | (Record<string, unknown> & {
        authenticated?: boolean;
        user_id?: string | null;
        userId?: string | null;
        agent_id?: string | null;
        agentId?: string | null;
        scopes?: readonly string[];
        claims?: Record<string, unknown>;
        profile?: { displayName?: string; avatarUrl?: string; email?: string };
      })
    | undefined;
  return {
    userId: auth?.user_id ?? auth?.userId ?? null,
    agentId: auth?.agent_id ?? auth?.agentId ?? null,
    scopes: auth?.scopes ?? [],
    claims: auth?.claims,
    authenticated: auth?.authenticated ?? !!(auth?.user_id ?? auth?.userId),
    profile: auth?.profile,
  };
}

/**
 * Read the canonical agent UUID from the dispatcher-populated
 * `ctx.metadata.agentId` (host-side dispatcher field, NOT the
 * sandbox-visible `ctx.auth.agentId` user code reads). Falls back to an empty string if the field
 * is missing — the cookie filter will then drop ALL cookies, which is
 * the safe default.
 */
function extractAgentId(ctx: Context): string {
  const meta = ctx?.metadata;
  if (!meta) return '';
  const id = (meta as { agentId?: unknown }).agentId;
  return typeof id === 'string' ? id : '';
}

/**
 * Defensive mirror of the dispatcher's per-agent cookie filter: only
 * cookies whose name starts with `agt_<canonicalAgentId>_` are
 * forwarded to the function. Returns the rebuilt Cookie header string
 * (or `''` if nothing matches — caller should treat that as "drop the
 * header"). Uses a strict RFC 6265 cookie-list parser instead of regex.
 */
function filterAgentCookies(rawHeader: string, canonicalAgentId: string): string {
  if (!rawHeader || !canonicalAgentId) return '';
  const prefix = `agt_${canonicalAgentId}_`;
  const out: string[] = [];
  // Cookie pairs are separated by `; ` (RFC 6265 §5.4). Names are
  // tokens and never contain `=`; we split on the first `=` only.
  for (const pair of rawHeader.split(/;\s*/)) {
    if (!pair) continue;
    const eq = pair.indexOf('=');
    if (eq < 0) continue;
    const name = pair.slice(0, eq).trim();
    const value = pair.slice(eq + 1);
    if (!name) continue;
    if (name === 'session' || name === 'logged_in' || name.startsWith('_robutler_')) continue;
    if (!name.startsWith(prefix)) continue;
    out.push(`${name}=${value}`);
  }
  return out.join('; ');
}

function decodeBody(buf: Uint8Array, contentType: string): unknown {
  const ct = contentType.toLowerCase();
  if (ct.includes('application/json')) {
    try {
      return JSON.parse(new TextDecoder().decode(buf));
    } catch {
      return null;
    }
  }
  if (ct.startsWith('text/') || ct.includes('application/x-www-form-urlencoded')) {
    return new TextDecoder().decode(buf);
  }
  return buf;
}

function errorCodeToStatus(code?: string): number {
  switch (code) {
    case 'FN_NOT_FOUND':
      return 404;
    case 'FN_CHAIN_TOO_DEEP':
    case 'FN_CYCLE_DETECTED':
    case 'FN_QUOTA_EXHAUSTED':
    case 'QUOTA_EXCEEDED':
    // Executor admission rejections are transient back-pressure — the
    // caller should retry, not treat it as a server fault. Mapping
    // these to 500 buried a load-shedding signal inside generic error
    // noise (widget heartbeats logged hard failures for what was
    // "try again in a second").
    case 'CPU_PRESSURE':
    case 'POOL_SATURATED':
    case 'CONCURRENCY_EXCEEDED':
      return 429;
    // Executor emits `WALL_TIMEOUT` (mapErrorCode in fn-runner). The
    // legacy `TIMEOUT` alias is kept for any fixture/test that still
    // sends the old code; both map to 504 Gateway Timeout.
    case 'WALL_TIMEOUT':
    case 'TIMEOUT':
      return 504;
    case 'WS_NOT_YET_SUPPORTED':
      return 501;
    default:
      return 500;
  }
}

function shapeResponse(result: unknown): Response {
  // Functions can return a Response object directly, a string body, or
  // structured `{ status, headers, body }` shapes. Normalise here.
  if (result instanceof Response) return result;
  if (typeof result === 'string') {
    return new Response(result, { status: 200, headers: { 'content-type': 'text/plain' } });
  }
  if (result && typeof result === 'object' && 'status' in (result as object)) {
    const r = result as { status: number; headers?: Record<string, string>; body?: unknown };
    const body = typeof r.body === 'string' ? r.body : JSON.stringify(r.body ?? null);
    return new Response(body, {
      status: r.status,
      headers: r.headers ?? { 'content-type': 'application/json' },
    });
  }
  return new Response(JSON.stringify(result ?? null), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

function cryptoRandomId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `inv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}
