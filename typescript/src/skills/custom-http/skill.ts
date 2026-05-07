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
import { prompt } from '../../core/decorators';
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
  /** Set by `BaseAgent.addSkill` when it sees a `setAgent` method. Read by the default resolver. */
  // @ts-expect-error read indirectly through `defaultRuntimeResolver`
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
      const endpoint: HttpEndpoint = {
        path: ep.path,
        method: ep.method,
        auth: ep.auth,
        enabled: true,
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
        headers[k.toLowerCase()] = v;
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

  @prompt({ priority: 71, name: 'customHttpRuntime', scope: 'all' })
  customHttpRuntime(_ctx: Context): string {
    if (this.endpoints.length === 0) return '';
    const lines = this.endpoints.map(
      (e) => `- ${e.method} ${e.path} (auth: ${e.auth}) -> ${e.use}`,
    );
    return `Custom HTTP endpoints exposed by this agent:\n${lines.join('\n')}`;
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
  return {
    userId: ctx.auth?.user_id ?? null,
    agentId: ctx.auth?.agent_id ?? ctx.auth?.agentId ?? null,
    scopes: ctx.auth?.scopes ?? [],
    claims: ctx.auth?.claims,
  };
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
      return 429;
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
