/**
 * `FunctionRuntimeSkill` — substrate skill consumed by `cron`, `custom_http`,
 * `custom_tools`, `manual` invocations, and `host-self-edit`.
 *
 * Reads `agent_configs.functions` (passed in via `config.functions`) and
 * exposes `invoke(name, ctx)` to the consuming skills. Resolves codeRefs,
 * threads the invocation chain (for `ctx.fn`), and proxies to the
 * `ExecutorClient` over mTLS.
 *
 * This skill is `auto-mounted` by the consumer skills via
 * `Skill.dependencies = ['function-runtime']` — agent owners enable
 * consumer skills only.
 */

import { Skill } from '../../core/skill';
import { prompt } from '../../core/decorators';
import type { Context, SkillConfig } from '../../core/types';
import {
  type FunctionContext,
  type FunctionFnApi,
  type FunctionInvocationResult,
  type FunctionLimitsResolved,
  type FunctionSource,
  type InvocationChain,
  DEFAULT_MAX_FN_CHAIN_DEPTH,
  FN_BUDGET_BUFFER_MS,
  FN_ERR_CHAIN_TOO_DEEP,
  FN_ERR_CYCLE_DETECTED,
  FN_ERR_QUOTA_EXHAUSTED,
} from './context';
import {
  StubExecutorClient,
  type ExecutorClient,
  type HostBridge,
  type InvocationEnvelope,
  type SerializableContext,
} from './executor-client';
import type { FunctionManifest } from './manifest';

/** User-action issue surfaced from the runtime skill (e.g. "open the function drawer to fix X"). */
export interface FunctionRequiresUserAction {
  kind: string;
  functionName: string;
  reason: string;
}

/** Async minter for a host-bridge ticket — called once per invoke. */
export type HostBridgeMinter = (args: {
  agentId: string;
  functionName: string;
  invocationId: string;
  /**
   * Caller attribution — usually the parent invocation's `source.consumerId`.
   * Required by `ctx.portal.payment.{lock,settle,release}`; optional otherwise.
   */
  consumerId?: string;
  /**
   * Surface-owner billing principal (`ctx.source.billedTo`), threaded into
   * the token so nested `ctx.fn.invoke` children bill the SAME principal
   * as their parent (fn-quota-enforcement-plan.md D7).
   */
  billedTo?: string;
  /**
   * Verified visitor id — when the dispatcher resolved a Robutler session
   * for this invocation (e.g. visitor_session / session). Stamped into the
   * fn-host token so `ctx.kv.*` can authorize visitor-scoped reads/writes.
   */
  verifiedVisitorId?: string;
  /**
   * Widget scope partitions (ADR-0023) — passed through from
   * `ctx.source.widget` so the minter can stamp verified partition claims
   * into the fn-host token for widget-scoped KV.
   */
  widget?: import('./context').WidgetInvocationScope;
}) => Promise<HostBridge>;

/**
 * Per-function declaration as it appears in `agent_configs.functions[name]`.
 * The `manifest` is the parsed + validated frontmatter / `manifest.json`;
 * `codeRef` may either be an explicit override or implicit (the function's
 * content row IS the bytes to execute).
 */
export interface DeclaredFunction {
  /** Stable agent-local name (`agent_configs.functions` key). */
  name: string;
  manifest: FunctionManifest;
  /** CodeRef the executor will resolve; usually `{ kind: 'content', contentId, sha256 }`. */
  codeRef: import('./manifest').CodeRef;
  /** Pre-pinned bundle hash for cache-key stability. */
  bundleSha256: string;
}

/** Configuration for `FunctionRuntimeSkill`. */
export interface FunctionRuntimeSkillConfig extends SkillConfig {
  /** Functions declared on this agent. */
  functions?: Record<string, DeclaredFunction>;
  /** Owning agent id (used in invocation envelopes / sandbox keys). */
  agentId?: string;
  /** Owner user id (for billing attribution + payment helpers). */
  ownerId?: string;
  /** Default per-invocation limits — overridden by manifest hint / agent override / plan ceiling. */
  defaultLimits?: FunctionLimitsResolved;
  /** Maximum nested-call depth (plan-tier configurable). */
  maxChainDepth?: number;
  /** Executor client (defaults to a no-op stub). */
  executor?: ExecutorClient;
  /**
   * Host-bridge ticket minter. Stamped onto each `InvocationEnvelope` so
   * the executor can call back into the portal for stateful APIs. When
   * undefined, `ctx.{secrets,kv,content,folders,fn,portal}` are
   * unavailable in the sandbox.
   */
  hostBridge?: HostBridgeMinter;
  /**
   * Issues surfaced to the owner UI (e.g. "missing secret binding" — the
   * factory passes them through; the skill stores them for retrieval).
   */
  requiresUserAction?: FunctionRequiresUserAction[];
  /**
   * Quota gate — called once per invocation (all entry paths, including
   * nested `ctx.fn.invoke`), before the host-bridge mint and executor
   * dispatch. The host injects the implementation (portal: account-bucket
   * check-and-increment); absent means ungated. A hook that THROWS is
   * treated as ok (fail open — quota infrastructure must never take the
   * function surface down).
   */
  gate?: FunctionGateHook;
}

/** Gate decision: proceed (with optional release/settle) or block. */
export type FunctionGateDecision =
  | {
      ok: true;
      /** Frees the concurrency slot — invoked in `finally` around the executor. */
      release?: () => Promise<void>;
      /** Posts actual cpu/ingress/egress over the estimate; fire-and-forget. */
      settle?: (actual: { cpuMs: number; ingressBytes: number; egressBytes: number }) => Promise<void>;
    }
  | {
      ok: false;
      errorCode: string;
      errorMessage: string;
      retryAfterSec?: number;
    };

/** Host-injected quota gate (see `FunctionRuntimeSkillConfig.gate`). */
export type FunctionGateHook = (args: {
  functionName: string;
  manifest: FunctionManifest;
  source: FunctionSource;
  /**
   * Dispatcher-resolved caller identity (`ctx.auth`) — server-derived,
   * never from request bodies. The host uses it for principal fallback
   * on ownerless (system) agents.
   */
  auth?: { userId?: string | null; agentId?: string };
  validateOnly?: boolean;
}) => Promise<FunctionGateDecision>;

const DEFAULT_LIMITS: FunctionLimitsResolved = {
  wallMs: 30_000,
  cpuMs: 5_000,
  memoryMb: 128,
  ingressBytes: 1_000_000,
  egressBytes: 1_000_000,
};

/**
 * Substrate skill — provides `invoke(name, ctx)` to consumer skills via a
 * shared instance reference. Consumer skills (CronSkill / CustomHttpSkill /
 * CustomToolsSkill) declare `dependencies = ['function-runtime']` so the
 * registry auto-mounts a default instance when no explicit one was passed.
 */
export class FunctionRuntimeSkill extends Skill {
  readonly name = 'function-runtime';
  // Memory is intentionally NOT a runtime dependency. Functions access
  // KV via the executor (which talks to the host's memory REST surface
  // over mTLS) and secrets via the host-side resolver — neither path
  // calls into a `RobutlerMemorySkill` instance. Mounting memory is an
  // orthogonal, opt-in choice the agent author makes through their
  // skill list / config UI.
  readonly dependencies = [] as const;

  private readonly functions: Map<string, DeclaredFunction>;
  private readonly executor: ExecutorClient;
  private readonly agentId: string;
  private readonly ownerId: string;
  private readonly defaultLimits: FunctionLimitsResolved;
  private readonly maxChainDepth: number;
  private readonly hostBridgeMinter?: HostBridgeMinter;
  private readonly userActions: FunctionRequiresUserAction[];
  private readonly gate?: FunctionGateHook;

  constructor(config: FunctionRuntimeSkillConfig = {}) {
    super(config);
    this.functions = new Map(Object.entries(config.functions ?? {}));
    this.executor = config.executor ?? new StubExecutorClient();
    this.agentId = config.agentId ?? '';
    this.ownerId = config.ownerId ?? '';
    this.defaultLimits = config.defaultLimits ?? DEFAULT_LIMITS;
    this.maxChainDepth = config.maxChainDepth ?? DEFAULT_MAX_FN_CHAIN_DEPTH;
    this.hostBridgeMinter = config.hostBridge;
    this.userActions = config.requiresUserAction ?? [];
    this.gate = config.gate;
  }

  /** Surface user-action issues for the drawer UI. */
  getUserActions(): readonly FunctionRequiresUserAction[] {
    return this.userActions;
  }

  /** Owning user id (for billing attribution / payment helpers). */
  getOwnerId(): string {
    return this.ownerId;
  }

  /** Names of declared functions (for `ctx.fn.list()` / capability summary). */
  list(): string[] {
    return Array.from(this.functions.keys()).sort();
  }

  /** Lookup; returns undefined if the name is not declared. */
  get(name: string): DeclaredFunction | undefined {
    return this.functions.get(name);
  }

  /**
   * Invoke a function. The consumer skills (cron / custom_http /
   * custom_tools) call this with the appropriate `source.skill` value and
   * payload (`request` / `schedule` / `toolCall`).
   *
   * Recursion + cycle + quota checks happen here so every entry path —
   * direct invocation, manual replay, `ctx.fn.invoke` — is gated.
   */
  async invoke<T = unknown>(
    name: string,
    ctx: SerializableContext,
    opts: {
      chain?: InvocationChain;
      idempotencyKey?: string;
      validateOnly?: boolean;
    } = {},
  ): Promise<FunctionInvocationResult<T>> {
    const fn = this.functions.get(name);
    if (!fn) {
      return failure<T>('FN_NOT_FOUND', `Function "${name}" is not declared on this agent`);
    }

    const chain = opts.chain;
    if (chain) {
      if (chain.depth >= this.maxChainDepth) {
        return failure<T>(
          FN_ERR_CHAIN_TOO_DEEP,
          `Function chain depth ${chain.depth} >= max ${this.maxChainDepth}`,
        );
      }
      const allowSelf = fn.manifest.permissions?.selfRecursion === true;
      if (chain.path.includes(name) && !allowSelf) {
        return failure<T>(
          FN_ERR_CYCLE_DETECTED,
          `Function "${name}" is already in the chain path: ${chain.path.join(' -> ')}`,
        );
      }
      if (
        chain.budgetRemaining.wallMs <= FN_BUDGET_BUFFER_MS ||
        chain.budgetRemaining.cpuMs <= 0
      ) {
        return failure<T>(
          FN_ERR_QUOTA_EXHAUSTED,
          'Cumulative chain budget exhausted',
        );
      }
    }

    // Host-injected quota gate — every entry path funnels through here
    // (validate-only runs use the separate per-minute validations bucket).
    let gateOk: Extract<FunctionGateDecision, { ok: true }> | undefined;
    if (this.gate && !opts.validateOnly) {
      let decision: FunctionGateDecision;
      try {
        decision = await this.gate({
          functionName: name,
          manifest: fn.manifest,
          source: ctx.source,
          auth: ctx.auth
            ? { userId: ctx.auth.userId, agentId: ctx.auth.agentId ?? undefined }
            : undefined,
          validateOnly: opts.validateOnly,
        });
      } catch {
        decision = { ok: true }; // fail open — see config.gate contract
      }
      if (!decision.ok) {
        return failure<T>(decision.errorCode, decision.errorMessage, decision.retryAfterSec);
      }
      gateOk = decision;
    }

    // Everything past the gate runs under try/finally so the reserved
    // concurrency slot frees on EVERY exit — mint failure, executor throw,
    // or normal return.
    let result: FunctionInvocationResult<T>;
    try {
      let hostBridge: HostBridge | undefined;
      if (this.hostBridgeMinter && !opts.validateOnly) {
        try {
          // Visitor identity flows from the dispatcher via `ctx.auth.userId`
          // ONLY when the dispatcher resolved a server-side session for the
          // invocation. We never trust client-controlled `user_id` values
          // here — `ctx.auth` is built from the session cookie / portal
          // token in the dispatcher, not from request bodies/headers.
          const verifiedVisitorId =
            ctx.auth?.userId && ctx.auth.userId !== this.agentId ? ctx.auth.userId : undefined;
          hostBridge = await this.hostBridgeMinter({
            agentId: this.agentId,
            functionName: name,
            invocationId: ctx.source.invocationId,
            // Consumer attribution: a trusted billing override (the widget fn
            // route stamps the mounted widget's OWNER) wins over the default
            // consuming-skill entry id — ADR-0023 Phase 2 (owner-billing).
            consumerId: ctx.source.billedTo ?? ctx.source.consumerId,
            billedTo: ctx.source.billedTo,
            verifiedVisitorId,
            widget: ctx.source.widget,
          });
        } catch (e) {
          return failure<T>(
            'HOST_BRIDGE_MINT_FAILED',
            `Failed to mint host bridge token: ${(e as Error).message}`,
          );
        }
      }

      const envelope: InvocationEnvelope = {
        functionName: name,
        agentId: this.agentId,
        bundleSha256: fn.bundleSha256,
        manifest: fn.manifest,
        codeRef: fn.codeRef,
        context: ctx,
        chain,
        idempotencyKey: opts.idempotencyKey,
        validateOnly: opts.validateOnly,
        hostBridge,
      };

      result = await this.executor.invoke<T>(envelope);
    } finally {
      await gateOk?.release?.().catch(() => {});
    }
    if (gateOk?.settle) {
      // Fire-and-forget: settle failure never fails the response.
      void gateOk
        .settle({
          cpuMs: result.cpuMs ?? 0,
          ingressBytes: result.ingressBytes ?? 0,
          egressBytes: result.egressBytes ?? 0,
        })
        .catch(() => {});
    }
    return result;
  }

  /**
   * Build a `ctx.fn` API bound to the parent invocation chain. Consumer
   * skills call this when assembling the executor's host-API bridge.
   */
  buildFnApi(parentChain: InvocationChain | undefined, baseSource: FunctionSource): FunctionFnApi {
    const skill = this;
    return {
      list: () => skill.list(),
      invoke: async <T = unknown>(
        name: string,
        args: unknown,
        opts?: { timeoutMs?: number; idempotencyKey?: string },
      ): Promise<T> => {
        const chain: InvocationChain = parentChain
          ? {
              rootInvocationId: parentChain.rootInvocationId,
              depth: parentChain.depth + 1,
              path: [...parentChain.path, name],
              budgetRemaining: subtractBuffer(parentChain.budgetRemaining),
              traceparent: parentChain.traceparent,
            }
          : {
              rootInvocationId: baseSource.invocationId,
              depth: 1,
              path: [name],
              budgetRemaining: skill.defaultLimits,
            };

        const childCtx: SerializableContext = {
          // `billedTo` rides the chain: nested work of a usage-initiated
          // invocation bills the SAME surface owner as its parent (without
          // this, nested calls silently shift to the author's bucket).
          source: {
            skill: 'function',
            consumerId: baseSource.invocationId,
            invocationId: cryptoRandomId(),
            ...(baseSource.billedTo ? { billedTo: baseSource.billedTo } : {}),
          },
          auth: { userId: null, agentId: skill.agentId, scopes: ['function:nested'] },
          limits: chain.budgetRemaining,
          toolCall: { name, params: args, callId: cryptoRandomId() },
        };

        const r = await skill.invoke<T>(name, childCtx, {
          chain,
          idempotencyKey: opts?.idempotencyKey,
        });
        if (!r.ok) {
          const err = new Error(r.errorMessage ?? 'function invocation failed');
          (err as Error & { code?: string }).code = r.errorCode;
          throw err;
        }
        return r.result as T;
      },
    };
  }

  /**
   * `@prompt` contribution that lists installed functions on this agent.
   * Renders only when functions exist so the empty case stays zero-token.
   */
  @prompt({ priority: 60, name: 'functionsRuntime', scope: 'all' })
  functionsRuntime(_ctx: Context): string {
    if (this.functions.size === 0) return '';
    const names = this.list().join(', ');
    return `Installed functions on this agent: ${names}. Use them via the available skill tools/endpoints. Functions are sandboxed and metered.`;
  }

  /**
   * Static runtime constraints + error playbook + templates + iteration
   * loop guidance for the function executor (js-v1, the only enabled
   * runtime). Always rendered when this skill is mounted — the rules are
   * universal regardless of how many functions are declared.
   *
   * This block is the single source of truth for functions runtime
   * guidance. Other surfaces (`portal-side functions-awareness`,
   * `host-self-edit selfEditGuide`) are being slimmed in lockstep so we
   * don't carry three drifting copies.
   */
  @prompt({ priority: 61, name: 'functionsRuntimeStatic', scope: 'all' })
  functionsRuntimeStatic(_ctx: Context): string {
    return [
      '## Functions runtime (js-v1)',
      '',
      'js-v1 is the only enabled runtime. Setting `runtime` to anything else fails validation with `RUNTIME_DISABLED` (ADR-0008).',
      '',
      '### Sandbox',
      '- Bare V8 isolate (isolated-vm). NO Node globals: `process`, `Buffer`, `require`, `fs`, `eval`, `Function` are all blocked.',
      '- NO npm packages or Node-only modules — bundling is deferred to v2. If you need a library, inline the small subset you actually use.',
      '- Entrypoint: an async handler. Prefer `export default async function handler(ctx) { ... }`. `export async function handler(ctx)` and `module.exports = async (ctx) => ...` also work.',
      '- Globals available: `URL`, `URLSearchParams`, `atob`, `btoa`, `JSON`, `Math`, `Date`, `Promise`, `Map`, `Set`, `RegExp`, `Symbol`, `Proxy`, `Reflect`, `Intl`, `console`, `TextEncoder`/`TextDecoder`, `structuredClone`, `crypto.{randomUUID,getRandomValues,subtle}`.',
      '- Host APIs (only via `ctx`, all permission-gated by `manifest.permissions`): `ctx.fetch` (URL allowlist), `ctx.secrets`, `ctx.kv` (`none`/`ro`/`rw` or object form `{ self?, visitor?, agent_scope? }`), `ctx.content` (`{ read?, write? }`), `ctx.folders`, `ctx.fn` (sibling functions, chain-budgeted), `ctx.portal.{verifyToken, verifyHmac, lookupAgent, callTool, getOwner, notifyOwner, signContentUrl, payment.lock, payment.settle, payment.release}`, `ctx.log`, `ctx.emit`. Full reference: `get_doc("reference/host-apis")`, `get_doc("reference/portal-helpers")`.',
      '- Inline source cap: 16 KB UTF-8 (`inline`) or 64 KB base64 (`inlineB64`); bigger source must move to a content row.',
      '- Defaults: `wallMs=30s` (fallback `10s` for some entry paths), `cpuMs=5s`, `memoryMb=128` (max 512), `ingressBytes=egressBytes=1MB`. Long-running work MUST finish within `wallMs` — there is a host-side watchdog.',
      '',
      '### Error playbook (errorCode → what to do)',
      'When `invocation.ok === false`, map the `errorCode` to the user-facing fix:',
      '- `FN_NOT_FOUND` — function name is wrong or not declared. Use `list_runtime_catalog` / declared-functions list to confirm the canonical name; do NOT auto-create a same-named function.',
      '- `FN_CHAIN_TOO_DEEP` — `ctx.fn.invoke` recursed past the per-plan depth cap. Restructure to a flat fan-out instead of a chain; if recursion is intentional, set `manifest.permissions.selfRecursion: true`.',
      '- `FN_CYCLE_DETECTED` — same function appeared twice in the chain path. Same fix as above; cycles are blocked even when `selfRecursion` is on.',
      '- `FN_QUOTA_EXHAUSTED` / `QUOTA_EXCEEDED` — cumulative wall/cpu/network budget for the chain ran out. Move heavy work to a single top-level invocation, or cache via `ctx.kv` and short-circuit on a hit.',
      '- `TIMEOUT` — single invocation exceeded `wallMs`. Profile the slow step (most often a synchronous JSON.parse on a large body or a slow `ctx.fetch`). Split the work, lower the body size, or raise `wallMs` in the manifest (capped at 30s).',
      '- `RUNTIME_DISABLED` — manifest pinned `python-pyodide-v1`/`wasm-v1`. Switch to `js-v1`; the others are reserved.',
      '- `HOST_BRIDGE_MINT_FAILED` — host token couldn\'t be minted (auth/quota issue on the portal). Surface to the user; do NOT retry with the same envelope, the failure is on the host side.',
      '- `EXECUTOR_THREW` / `FUNCTION_ERROR` — function code threw. The `errorMessage` carries the user-thrown message — surface it (sanitised) to the user; if you authored the function, fix the throw.',
      '- `HOST_QUOTA_EXCEEDED` — agent owner ran out of plan budget for a host API (KV writes, fetch egress, content storage). Tell the user the owner needs to upgrade their plan; do NOT retry.',
      '',
      '### Iteration loop for declaring NEW functions',
      'Always: `declare_function` → manual invoke as a smoke test → read the invocation result (success or `errorCode`) → iterate. NEVER attach a freshly-declared function to a skill (`add_to_skill cron|custom_http|custom_tools`) before the smoke test passes — a broken function attached to cron will fail silently every minute, and a broken `custom_http` endpoint surfaces 500s to whoever calls it.',
      '',
      '### Common templates (start from these instead of guessing)',
      '',
      '1. **Fetch + parse JSON** (with `manifest.permissions.fetch: ["https://api.example.com"]`):',
      '   ```',
      '   export default async function handler(ctx) {',
      '     const r = await ctx.fetch("https://api.example.com/v1/items");',
      '     if (!r.ok) throw new Error(`upstream ${r.status}`);',
      '     return await r.json();',
      '   }',
      '   ```',
      '',
      '2. **KV read/write** — KV API is `get`/`put`/`delete`/`list` (NEVER `kv.set`). Permissions: bare string "none"|"ro"|"rw" (= self only) or object form `{ self?, visitor?, agent_scope? }`. Per-visitor calls require `permissions.kv.visitor` AND `ctx.auth.userId`. Cross-function reads use `scope: "agent"` (requires `agent_scope: true`).',
      '   ```',
      '   export default async function handler(ctx) {',
      '     const seen = await ctx.kv.get(`seen:${ctx.toolCall.params.id}`);',
      '     if (seen) return { cached: true, value: seen };',
      '     const fresh = computeSomething(ctx.toolCall.params);',
      '     await ctx.kv.put(`seen:${ctx.toolCall.params.id}`, fresh, { ttlSeconds: 3600 });',
      '     // Per-visitor / cross-function: object form',
      '     // await ctx.kv.put({ user_id: ctx.auth.userId, key: "pref", value: { theme: "dark" } });',
      '     // await ctx.kv.get({ scope: "agent", key: "shared_state" });',
      '     return { cached: false, value: fresh };',
      '   }',
      '   ```',
      '',
      '3. **Chain to another function** (with declared `targetFn` in `agent_configs.functions`):',
      '   ```',
      '   export default async function handler(ctx) {',
      '     const inner = await ctx.fn.invoke("targetFn", { foo: 1 });',
      '     return { wrapped: inner };',
      '   }',
      '   ```',
      '',
      '4. **Throw a structured error** (so the caller sees `errorMessage`):',
      '   ```',
      '   export default async function handler(ctx) {',
      '     if (!ctx.toolCall.params.email) throw new Error("missing email");',
      '     return { ok: true };',
      '   }',
      '   ```',
      '',
      '5. **Emit a non-result side-channel event** (e.g. progress for a long task):',
      '   ```',
      '   export default async function handler(ctx) {',
      '     ctx.emit({ type: "progress", percent: 42 });',
      '     // ... continue work',
      '     return { done: true };',
      '   }',
      '   ```',
      '',
      '### HTML / browser-renderable returns (custom_http endpoints)',
      'Functions wired to a `custom_http` endpoint can return HTML directly — that\'s how agents serve webpages.',
      '',
      '- **Return shape**: `{ status, headers: { "content-type": "text/html; charset=utf-8" }, body: "<html>...</html>" }`. The dispatcher inspects `content-type` to decide which response wrapper applies.',
      '- **Default security headers (auto-injected for `text/html`)**: `Content-Security-Policy` (default-src \'self\', script-src \'self\' \'unsafe-inline\', style-src \'self\' \'unsafe-inline\', img-src \'self\' data: https:, connect-src \'self\', frame-ancestors \'none\', base-uri \'none\', form-action \'self\'), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, plus COOP/CORP. Override per-key by setting the same header in your response — the agent always wins.',
      '- **2 MB response cap** is enforced; oversized responses become a 502 with `RESPONSE_SIZE_CAP_EXCEEDED` and an owner warning.',
      '- **Same-origin fetch from your HTML**: served from `https://robutler.ai/agents/<id>/<...>` (canonical URL), so the page can `fetch("./api")` to call other endpoints on the same agent without CORS gymnastics.',
      '- **Visitor identity (Robutler-as-IdP)**: declare `auth: "visitor_session"` on the endpoint to get `ctx.auth.user_id` for logged-in visitors. Add `permissions.visitor_profile: ["name", "avatar"]` to receive `ctx.auth.profile.{displayName,avatarUrl}` (and optionally `email` — PII, opt-in only). Prefer this over rolling Google OAuth for identity.',
      '- **Agent-scoped sessions** (your own login): set cookies that MUST start with `agt_${ctx.auth.agentId}_`. The dispatcher hard-rejects (502) any Set-Cookie with `Domain=`, reserved names (`session`, `logged_in`, `_robutler_*`), or wrong prefix.',
      '- **Reference recipes**: `CustomHttpSkill` exposes a `get_webapp_template` tool with 14 named patterns (e.g. `minimal_html_page`, `visitor_session_personalized`, `kv_visitor_state`, `csrf_protected_form`, `oauth_google_login`, `widget_personalized_card`). Call `list_webapp_templates` first, then fetch the source for the one you want.',
    ].join('\n');
  }
}

function failure<T>(code: string, message: string, retryAfterSec?: number): FunctionInvocationResult<T> {
  return {
    ok: false,
    errorCode: code,
    errorMessage: message,
    ...(retryAfterSec !== undefined ? { retryAfterSec } : {}),
    durationMs: 0,
    cpuMs: 0,
    ingressBytes: 0,
    egressBytes: 0,
  };
}

function subtractBuffer(b: FunctionLimitsResolved): FunctionLimitsResolved {
  return {
    wallMs: Math.max(0, b.wallMs - FN_BUDGET_BUFFER_MS),
    cpuMs: Math.max(0, b.cpuMs - FN_BUDGET_BUFFER_MS),
    memoryMb: b.memoryMb,
    ingressBytes: b.ingressBytes,
    egressBytes: b.egressBytes,
  };
}

function cryptoRandomId(): string {
  // Lightweight uuid-ish — uses globalThis.crypto when available.
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `inv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Helper to build the human-bridged `FunctionContext` from a serialisable
 * context plus host-API implementations. Used by tests and by the
 * localhost executor.
 */
export function bridgeFunctionContext(args: {
  serializable: SerializableContext;
  hostApis: Pick<FunctionContext, 'fetch' | 'secrets' | 'kv' | 'content' | 'folders' | 'log' | 'portal'>;
  fn: FunctionFnApi;
  emit?: FunctionContext['emit'];
}): FunctionContext {
  return {
    ...args.serializable,
    ...args.hostApis,
    fn: args.fn,
    emit: args.emit ?? (() => undefined),
  };
}
