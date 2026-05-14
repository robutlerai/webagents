/**
 * `FunctionContext` — the unified shape passed to every function invocation.
 *
 * Same context across cron / custom_http / custom_tools / manual / function-
 * to-function calls; the `source.skill` discriminator tells the function
 * where the invocation originated. Per-skill payload (request / schedule /
 * toolCall) is populated as appropriate.
 *
 * Host APIs (`fetch`, `secrets`, `kv`, `content`, `folders`, `log`,
 * `portal`, `fn`) are wired to the executor coordinator over mTLS — user
 * code never touches portal credentials directly.
 */

import type { CodeRef, FetchAllowlist, FunctionLimits } from './manifest';

/** Discriminator for the consuming skill. */
export type FunctionSourceSkill =
  | 'cron'
  | 'custom_http'
  | 'custom_tools'
  | 'manual'
  | 'function';

/** Where the invocation came from. */
export interface FunctionSource {
  /** Consuming skill that triggered this invocation. */
  skill: FunctionSourceSkill;
  /** Id of the consuming-skill entry (cron schedule id, http endpoint id, …). */
  consumerId: string;
  /** Unique per invocation; correlates logs/metrics/audit. */
  invocationId: string;
}

/** Verified caller (or internal-context-derived for cron / tool / manual). */
export interface FunctionAuth {
  userId: string | null;
  agentId: string | null;
  scopes: readonly string[];
  payment?: { tokenSub: string; balance: bigint } | null;
  claims?: Record<string, unknown>;
  /**
   * `true` iff the request carried a valid Robutler session (visitor or owner).
   * Always present so user code can branch with `if (!ctx.auth.authenticated)`
   * without dealing with `undefined`.
   */
  authenticated?: boolean;
  /**
   * Visitor profile fields the endpoint manifest opted into via
   * `permissions.visitor_profile` (subset of `name | avatar | email`).
   * Only populated for `visitor_session`-authed endpoints.
   */
  profile?: {
    displayName?: string;
    avatarUrl?: string;
    email?: string;
  };
}

/** HTTP-shaped invocation payload (only set when `source.skill === 'custom_http'`). */
export interface FunctionRequest {
  method: string;
  /** Post route-template substitution. */
  path: string;
  params: Record<string, string>;
  query: Record<string, string>;
  /**
   * Raw passthrough — no portal-side stripping; lower-cased keys (Node
   * convention). Includes signature/auth headers so `signature` mode
   * handlers can verify themselves via `ctx.portal.verifyHmac`.
   */
  headers: Record<string, string>;
  /** Pre-parsed for json/form, raw for octet-stream. */
  body: unknown;
  /** Populated when the manifest sets `permissions.rawBody = true`. */
  rawBody?: Uint8Array;
}

/** Schedule payload (only set when `source.skill === 'cron'`). */
export interface FunctionScheduleInfo {
  plannedAt: string;
  firedAt: string;
}

/** Tool-call payload (only set when `source.skill === 'custom_tools'`). */
export interface FunctionToolCall {
  name: string;
  params: unknown;
  callId: string;
}

/** `ctx.fetch` request init — mirrors the Web Fetch API. */
export interface FunctionFetchInit {
  method?: string;
  headers?: Record<string, string>;
  body?: string | Uint8Array | null;
  signal?: AbortSignal;
  /** Maximum response bytes; default per-runtime. */
  maxBytes?: number;
  /** Override timeout for this call (ms). */
  timeoutMs?: number;
}

/** `ctx.secrets` API. Reads gated by manifest `permissions.secrets[]`. */
export interface FunctionSecrets {
  get(name: string): Promise<string | undefined>;
  /** Self-write. Requires `'write'` in `permissions.secrets`. */
  put(name: string, value: string): Promise<void>;
  /** Names only, never values. */
  list(): Promise<string[]>;
}

/**
 * Scope axis for `ctx.kv` (Phase 5).
 *
 *  - `'function'` (default): namespace is per-function. Two functions on
 *    the same agent CANNOT see each other's keys at this scope.
 *  - `'agent'`: namespace is shared across every function on the agent.
 *    Lets multi-function webapps share session/user records. Requires
 *    `permissions.kv.agent_scope: true` on every participating function.
 */
export type KvScope = 'function' | 'agent';

/**
 * Object-form parameters for `ctx.kv.*` (Phase 5).
 *
 * Validation rules (enforced in fn-host before each call):
 *   - `user_id` MUST equal `ctx.auth.agentId` (agent's own data) OR
 *     `ctx.auth.userId` when authenticated (visitor's data). Anything else
 *     → PERMISSION_DENIED.
 *   - `scope: 'agent'` requires `permissions.kv.agent_scope === true`.
 *   - `user_id !== ctx.auth.agentId` requires `permissions.kv.visitor`.
 */
export interface KvCallArgs {
  /** Owner of the data — agent's own UUID OR a verified visitor's UUID. */
  user_id: string;
  /** KV key. */
  key: string;
  /** TTL in seconds. Optional. */
  ttlSeconds?: number;
  /** Scope axis. Defaults to `'function'`. */
  scope?: KvScope;
}

/**
 * `ctx.kv` API.
 *
 * Backwards-compatible: the legacy single-string-key form
 * (`get('foo')`, `put('foo', val)`) is treated as
 * `{ user_id: ctx.auth.agentId, key: 'foo', scope: 'function' }`
 * and continues to read/write the legacy `fn:<functionName>` namespace
 * with no data migration. The new object-form lets agents address
 * per-visitor data (Phase 4 webapp pattern) and cross-function data
 * (Option B: split webapp across login/page/api functions).
 */
export interface FunctionKv {
  get<T = unknown>(key: string): Promise<T | undefined>;
  get<T = unknown>(args: Omit<KvCallArgs, 'ttlSeconds'>): Promise<T | undefined>;

  put<T = unknown>(key: string, value: T, opts?: { ttlMs?: number }): Promise<void>;
  put<T = unknown>(args: KvCallArgs & { value: T }): Promise<void>;

  delete(key: string): Promise<void>;
  delete(args: Omit<KvCallArgs, 'ttlSeconds'>): Promise<void>;

  list(
    prefix?: string,
    opts?: { limit?: number; cursor?: string },
  ): Promise<{ keys: string[]; cursor?: string }>;
  list(args: {
    user_id: string;
    prefix?: string;
    scope?: KvScope;
    limit?: number;
    cursor?: string;
  }): Promise<{ keys: string[]; cursor?: string }>;
}

/** Content access API — mediated by content ACL. */
export interface FunctionContentApi {
  get(id: string): Promise<Blob>;
  put?(item: { kind: string; data: Blob | string; meta?: Record<string, unknown> }): Promise<{ id: string }>;
}

/** Folder binding API — listing, reading, writing within a binding scope. */
export interface FunctionFolderApi {
  list(opts?: { prefix?: string; limit?: number; cursor?: string }): Promise<{
    items: Array<{ name: string; size: number; modified: string }>;
    cursor?: string;
  }>;
  read(name: string): Promise<Blob>;
  write?(name: string, data: Blob | Uint8Array | string): Promise<void>;
}

/** Structured logger — emitted to executor logs with traceparent context. */
export interface FunctionLog {
  debug(...args: unknown[]): void;
  info(...args: unknown[]): void;
  warn(...args: unknown[]): void;
  error(...args: unknown[]): void;
}

/**
 * Portal helper API — typed gateway from inside a function back into the
 * portal. All calls routed over mTLS through the executor coordinator,
 * scoped to the calling agent's permissions.
 */
export interface PortalHelpers {
  verifyToken(
    token: string,
    opts?: {
      /** Defaults to `PLATFORM_ISS`. */
      expectAudience?: string;
      /** When true, calls `/api/payments/verify` (DB-checked); else local JWKS. */
      expectBalance?: boolean;
    },
  ): Promise<{ valid: boolean; claims?: Record<string, unknown>; balance?: bigint }>;

  verifyHmac(opts: {
    algo: 'sha256' | 'sha1';
    /** Names a `permissions.secrets` entry — secret values never re-enter user code. */
    secretBinding: string;
    payload: string | Uint8Array;
    expected: string;
  }): Promise<boolean>;

  lookupAgent(
    idOrUsername: string,
  ): Promise<{ id: string; username: string; ownerId: string } | null>;

  callTool(
    agentRef: string,
    toolName: string,
    params: unknown,
    opts?: { timeoutMs?: number; paymentToken?: string },
  ): Promise<unknown>;

  getOwner(): Promise<{ id: string; email: string; planName: string }>;

  notifyOwner(opts: {
    title: string;
    body: string;
    severity?: 'info' | 'warn' | 'error';
    deepLink?: string;
  }): Promise<void>;

  signContentUrl(
    contentId: string,
    opts?: { expiresInSeconds?: number },
  ): Promise<string>;

  payment: {
    /**
     * Reserve `amountNanocents` against a caller-supplied payment token.
     * The token is the credential the paying party gave the agent (e.g.
     * via `Authorization: Bearer <jwt>`); the agent's id is the holder.
     * Returns the lock id used for `settle` / `release`.
     */
    lock(
      paymentToken: string,
      amountNanocents: bigint,
      reason: string,
    ): Promise<{ lockId: string; expiresAt: string }>;
    /**
     * Charge `amountNanocents` against the lock and credit `recipientId`
     * (defaults to the agent's owner). The amount must be ≤ the locked
     * amount; the remainder is auto-released.
     */
    settle(
      lockId: string,
      amountNanocents: bigint,
      recipientId?: string,
    ): Promise<{ ok: true } | { ok: false; reason: string }>;
    /** Release a lock without charging. */
    release(lockId: string): Promise<void>;
  };
}

/**
 * `ctx.fn` — function-to-function calls. Recursion is bounded:
 *   - max chain depth 5 (plan-tier configurable) → `FN_CHAIN_TOO_DEEP`
 *   - cycle detection on the path → `FN_CYCLE_DETECTED`
 *   - cumulative quota inheritance (50ms buffer) → `FN_QUOTA_EXHAUSTED`
 *   - serialised admission (no concurrent fan-out)
 */
export interface FunctionFnApi {
  invoke<T = unknown>(
    name: string,
    args: unknown,
    opts?: { timeoutMs?: number; idempotencyKey?: string },
  ): Promise<T>;
  list(): string[];
}

/**
 * Per-invocation quota envelope passed alongside `ctx`. The executor
 * applies these as `MIN(planCeiling, agentOverride, manifestHint)` minus
 * any budget already consumed earlier in the chain.
 */
export interface FunctionLimitsResolved {
  wallMs: number;
  cpuMs: number;
  memoryMb: number;
  ingressBytes: number;
  egressBytes: number;
}

/**
 * The single shape user code reads from. Most functions only touch
 * `ctx.auth.userId`, `ctx.request.body`, `ctx.kv`, and `ctx.fetch` — the
 * rest exists for advanced use (HMAC, sibling tool calls, signed URLs).
 */
export interface FunctionContext {
  source: FunctionSource;
  request?: FunctionRequest;
  schedule?: FunctionScheduleInfo;
  toolCall?: FunctionToolCall;
  auth: FunctionAuth;

  // Host APIs
  fetch: (url: string, init?: FunctionFetchInit) => Promise<Response>;
  secrets: FunctionSecrets;
  kv: FunctionKv;
  content: FunctionContentApi;
  folders: Record<string, FunctionFolderApi>;
  fn: FunctionFnApi;
  log: FunctionLog;
  portal: PortalHelpers;

  // Limits & metering
  limits: FunctionLimitsResolved;
  /**
   * Emit additional metering during execution. The executor merges these
   * into the final `_metering` envelope automatically; functions only
   * call this if they have out-of-band measurements (rare).
   */
  emit: (m: Partial<{ cpu_ms: number; ingress_bytes: number; egress_bytes: number }>) => void;
}

/**
 * Invocation chain envelope — propagated by the coordinator across nested
 * `ctx.fn.invoke` calls.
 */
export interface InvocationChain {
  rootInvocationId: string;
  /** 0 for the root; +1 per nested call. */
  depth: number;
  /** Function names visited so far — used for cycle detection. */
  path: readonly string[];
  budgetRemaining: FunctionLimitsResolved;
  /** W3C traceparent shared across the chain. */
  traceparent?: string;
}

/** Maximum nested depth for `ctx.fn.invoke` (plan-tier configurable). */
export const DEFAULT_MAX_FN_CHAIN_DEPTH = 5;

/** Buffer subtracted from inherited budget when descending. */
export const FN_BUDGET_BUFFER_MS = 50;

/** Error codes thrown when chain limits trip. */
export const FN_ERR_CHAIN_TOO_DEEP = 'FN_CHAIN_TOO_DEEP';
export const FN_ERR_CYCLE_DETECTED = 'FN_CYCLE_DETECTED';
export const FN_ERR_QUOTA_EXHAUSTED = 'FN_QUOTA_EXHAUSTED';

/**
 * Public summary of a function invocation, returned to the calling skill
 * so it can shape its response (HTTP status code from `result`, tool body
 * from `result`, etc).
 */
export interface FunctionInvocationResult<T = unknown> {
  ok: boolean;
  result?: T;
  /** Error code, e.g. `FN_CHAIN_TOO_DEEP`, `FN_CYCLE_DETECTED`, `QUOTA_EXCEEDED`. */
  errorCode?: string;
  errorMessage?: string;
  durationMs: number;
  /** Active-wall-time CPU meter (v1). */
  cpuMs: number;
  ingressBytes: number;
  egressBytes: number;
  /** Total settled cost in nanocents. */
  costNanocents?: bigint;
  /** Pass-through metering for the payment skill. */
  _metering?: {
    cpu_ms: number;
    ingress_bytes: number;
    egress_bytes: number;
  };
}

/** Re-export commonly used companion types so consumers don't need both modules. */
export type { CodeRef, FetchAllowlist, FunctionLimits };
