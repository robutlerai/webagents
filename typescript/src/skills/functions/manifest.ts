/**
 * Function manifest types
 *
 * A "function" is a content item (single file with frontmatter, or folder
 * with `manifest.json`) executed in the sandboxed function-executor service.
 * Functions are declared once at the top of `agent_configs.functions` and
 * consumed by name from skills (`cron`, `custom_http`, `custom_tools`).
 *
 * This module defines the shapes of:
 *  - `CodeRef` — discriminated union for code source
 *  - `FunctionManifest` — frontmatter / manifest.json contents
 *  - `FunctionPermissions` — declared egress / KV / secret / portal helpers
 *  - `FunctionLimits` — per-function self-imposed caps (manifest hint)
 *  - `FunctionRuntimeId` — runtime allowlist (`js-v1`, `python-pyodide-v1`,
 *    `wasm-v1` reserved-but-disabled)
 *
 * The validator in `./validator.ts` parses + checks both shapes against
 * these types and produces structured errors keyed for the owner UX
 * inline-diagnostics rendering.
 */

/**
 * Runtime identifier.
 *
 * - `js-v1` — V8 isolate, the only enabled runtime in v1.
 * - `python-pyodide-v1` — slot reserved but disabled (see ADR-0008).
 *   Persisted manifests pinning this id surface as `RUNTIME_DISABLED`.
 * - `wasm-v1` — slot reserved but disabled.
 */
export type FunctionRuntimeId = 'js-v1' | 'python-pyodide-v1' | 'wasm-v1';

/** Subset of runtimes actually executable in v1. */
export const SUPPORTED_RUNTIMES: readonly FunctionRuntimeId[] = ['js-v1'] as const;

/**
 * `CodeRef` — discriminated union for "where does the code live".
 *
 * - `content`    — content row in the portal (single-file or folder).
 * - `https`      — fetched at install time, hash-pinned.
 * - `file`       — localhost-only path under the daemon's `--code-root`.
 * - `inline`     — ≤ 16 KB UTF-8 source.
 * - `inlineB64`  — ≤ 64 KB base64-packed `esbuild` output.
 */
export type CodeRef =
  // Content refs point at a portal content row that the runtime resolves
  // by id; the bytes are loaded from disk and the sha256 is computed
  // post-resolution. The field is optional here so programmatic code-ref
  // construction (the portal's runtime factory) doesn't need a phantom
  // hash; user-supplied frontmatter may pin one for documentation.
  | { kind: 'content'; contentId: string; sha256?: string }
  | { kind: 'https'; url: string; sha256: string }
  | { kind: 'file'; path: string; sha256?: string }
  | { kind: 'inline'; source: string }
  | { kind: 'inlineB64'; source: string; sha256?: string };

/**
 * `permissions.fetch` semantics — declared egress contract:
 *
 * - omitted / `[]` — no external fetch allowed (safest default).
 * - `["*"]` — unrestricted (after the SSRF guard floor).
 * - explicit list — `https://` URLs / hostname patterns; wildcards match
 *   a single label (`*.example.com` matches `a.example.com` but not
 *   `a.b.example.com`).
 */
export type FetchAllowlist = readonly string[];

/** KV access mode for `ctx.kv`. */
export type KvMode = 'none' | 'ro' | 'rw';

/**
 * Object-form KV permission block (Phase 5 of custom_http hardening).
 *
 * Backwards-compatible: `permissions.kv` may still be the bare string
 * (`'none' | 'ro' | 'rw'`) — that maps to `{ self: <string-value> }`
 * via `normalizeKvPermissions()`.
 *
 * The richer form lets agents express:
 *   - `self`      — read/write the agent's OWN data (default scope='function').
 *                   The legacy form maps here.
 *   - `visitor`   — read/write data keyed on `ctx.auth.user_id` (i.e. data the
 *                   visitor owns). Required for per-visitor preferences,
 *                   sessions, etc.
 *   - `agent_scope` — opt into the cross-function `scope: 'agent'` axis.
 *                     Lets multiple functions on the same agent share
 *                     KV records (e.g. `session:<sid>`).
 */
export interface KvPermissionsObject {
  /** Mode for the agent's own data. Defaults to `'none'`. */
  self?: KvMode;
  /** Mode for visitor data (keyed on `ctx.auth.user_id`). Defaults to `'none'`. */
  visitor?: KvMode;
  /** Allow `scope: 'agent'` on `ctx.kv.*` calls. Defaults to false. */
  agent_scope?: boolean;
  /**
   * WIDGET scopes (ADR-0023) — only meaningful when the function is invoked
   * through a widget instance (the portal fn route resolves and VERIFIES the
   * widget/app/project identity server-side; without that context every
   * widget-scope call is denied regardless of these modes).
   *
   *   - `app`           — cross-instance store shared by every deployment of
   *                       this widget/app bundle (leaderboards, marketplaces,
   *                       counters). PII-free by policy.
   *   - `instance`      — one widget item's store (per-deployment state).
   *   - `instance_owner`— projection into the INSTANCE OWNER's store
   *                       (owner-gated reads; may hold viewer PII).
   *   - `project`       — shared store of the widget's project folder.
   *   - `project_owner` — projection into the PROJECT OWNER's store
   *                       (owner-gated reads; analytics rollups live here).
   */
  app?: KvMode;
  instance?: KvMode;
  instance_owner?: KvMode;
  project?: KvMode;
  project_owner?: KvMode;
}

/** Either the legacy string or the richer object form. */
export type KvPermissions = KvMode | KvPermissionsObject;

/** Normalised internal shape produced by `normalizeKvPermissions`. */
export interface NormalizedKvPermissions {
  self: KvMode;
  visitor: KvMode;
  agent_scope: boolean;
  app: KvMode;
  instance: KvMode;
  instance_owner: KvMode;
  project: KvMode;
  project_owner: KvMode;
}

/**
 * Convert the public `KvPermissions` union into the canonical object
 * form. Bare-string callers get `{ self: <value>, visitor: 'none', agent_scope: false }`.
 * This helper is the single source of truth — every consumer (fn-host,
 * factory, validator) MUST route through it.
 */
export function normalizeKvPermissions(
  raw: KvPermissions | undefined | null,
): NormalizedKvPermissions {
  const none: NormalizedKvPermissions = {
    self: 'none', visitor: 'none', agent_scope: false,
    app: 'none', instance: 'none', instance_owner: 'none', project: 'none', project_owner: 'none',
  };
  if (!raw) return none;
  if (typeof raw === 'string') {
    return { ...none, self: raw };
  }
  return {
    ...none,
    self: raw.self ?? 'none',
    visitor: raw.visitor ?? 'none',
    agent_scope: !!raw.agent_scope,
    app: raw.app ?? 'none',
    instance: raw.instance ?? 'none',
    instance_owner: raw.instance_owner ?? 'none',
    project: raw.project ?? 'none',
    project_owner: raw.project_owner ?? 'none',
  };
}

/**
 * Visitor-profile fields the manifest can opt into. The `'email'` field
 * is gated separately because it's PII; agents only request it when
 * they actually need a verified email address.
 *
 * Used by `permissions.visitor_profile` on `visitor_session` endpoints —
 * the dispatcher loads the requested fields from the `users` table and
 * surfaces them on `ctx.auth.profile`.
 */
export type VisitorProfileField = 'name' | 'avatar' | 'email';

/** Content access mode for `ctx.content`. */
export interface ContentPermission {
  read?: boolean;
  write?: boolean;
}

/**
 * Folder binding — exposes a content folder (cloud) or a local directory
 * (localhost only) under `ctx.folders.<binding>`.
 */
export type FolderBinding =
  | { binding: string; kind: 'content'; folderId: string; mode: 'ro' | 'rw' }
  | { binding: string; kind: 'local'; path: string; mode: 'ro' | 'rw' };

/** Portal helper allowlist — names of `ctx.portal.*` methods the function may call. */
export type PortalHelperName =
  | 'verifyToken'
  | 'verifyHmac'
  | 'lookupAgent'
  | 'callTool'
  | 'getOwner'
  | 'notifyOwner'
  | 'signContentUrl'
  | 'payment.lock'
  | 'payment.settle'
  | 'payment.release';

/**
 * Full permission block — every external capability a function asks for.
 * Anything not declared here is rejected at runtime.
 */
export interface FunctionPermissions {
  fetch?: FetchAllowlist;
  /**
   * Secret bindings — names from `fn-secret:<functionName>` that the
   * function may read. Use `'write'` as a sentinel inside the array to
   * also grant `ctx.secrets.put` self-write access (e.g. OAuth callback).
   */
  secrets?: readonly string[];
  /**
   * KV access. Backwards-compatible: the legacy bare string
   * (`'none' | 'ro' | 'rw'`) maps to `{ self: <value> }`. The richer
   * object form lets the function request `visitor` access (data keyed
   * on `ctx.auth.user_id`) and the cross-function `scope: 'agent'` axis
   * — see `KvPermissionsObject`.
   */
  kv?: KvPermissions;
  content?: ContentPermission;
  folders?: readonly FolderBinding[];
  portal?: readonly PortalHelperName[];
  /**
   * Allow the function to call itself via `ctx.fn.invoke('self', ...)`.
   * Defaults to false — plain recursion is rejected as a cycle.
   */
  selfRecursion?: boolean;
  /**
   * For HTTP-mode endpoints that need access to the raw bytes (HMAC over
   * raw body). When true, the executor passes `ctx.request.rawBody` as a
   * `Uint8Array`.
   */
  rawBody?: boolean;
  /**
   * Visitor-profile fields to surface on `ctx.auth.profile` for
   * `visitor_session` endpoints. Robutler-as-IdP — the dispatcher
   * loads `displayName`/`avatarUrl`/`email` from the users table when
   * the visitor is logged in. `'email'` is gated separately as PII —
   * include it only when actually needed.
   */
  visitor_profile?: readonly VisitorProfileField[];
}

/**
 * Per-function self-imposed cap. Applied as `MIN(planCeiling, agentOverride,
 * manifestHint)` at quota check time.
 */
export interface FunctionLimits {
  wallMs?: number;
  cpuMs?: number;
  memoryMb?: number;
  ingressBytes?: number;
  egressBytes?: number;
  /** Maximum body size accepted on inbound HTTP (post Content-Length check). */
  bodyBytesMax?: number;
}

/**
 * Function manifest — frontmatter (single-file) or `manifest.json` (folder).
 */
export interface FunctionManifest {
  /**
   * Function display name. The agent-local identifier (the key in
   * `agent_configs.functions`) is the SOURCE OF TRUTH for stable
   * identity; this name is shown in the UI / catalog.
   */
  name?: string;
  description?: string;
  runtime: FunctionRuntimeId;
  /** Exported function in the bundle (default `default` for js-v1). */
  entrypoint?: string;
  /**
   * Optional explicit codeRef when the manifest lives in a content folder
   * but the executable bytes are stored elsewhere (e.g. an `https` link to
   * a CDN). When omitted, the runtime resolves the entrypoint relative to
   * the content row.
   */
  code?: CodeRef;
  permissions?: FunctionPermissions;
  limits?: FunctionLimits;
  /** Pinned bundle hash. Set by the validator at save time; immutable thereafter. */
  bundleSha256?: string;
  /** JSON Schema for `custom_tools` parameter validation. Optional. */
  parameters?: Record<string, unknown>;
}

// NOTE: there is intentionally no `type` field on the manifest.
// "How is this function exposed?" (HTTP / cron / LLM tool / WebSocket)
// is a binding-level concern, not a function-level one — it lives on
// `agent_configs.skills.{custom_http,cron,custom_tools}.<entry>` and is
// authored via `add_to_skill`. A single function can fan out to several
// bindings; pinning a `type` on the manifest forecloses that.

/**
 * Manifest validation result — structured errors so the owner UX can
 * render inline diagnostics on the offending lines.
 */
export interface ManifestValidationError {
  code: ManifestErrorCode;
  message: string;
  path?: string;
  /** Optional line/column for editor squiggles. */
  loc?: { line: number; column?: number };
}

export type ManifestErrorCode =
  | 'MISSING_RUNTIME'
  | 'RUNTIME_UNKNOWN'
  | 'RUNTIME_DISABLED'
  | 'WS_NOT_YET_SUPPORTED'
  | 'INVALID_CODE_REF'
  | 'INVALID_PERMISSIONS'
  | 'INVALID_LIMITS'
  | 'INVALID_FETCH_PATTERN'
  | 'INVALID_SECRETS'
  | 'INVALID_PARAMETERS_SCHEMA'
  | 'INVALID_NAME'
  | 'CODE_TOO_LARGE'
  | 'BUNDLE_HASH_MISMATCH'
  | 'FILE_REF_REJECTED_IN_CLOUD'
  | 'INTERNAL';

/** Maximum source bytes for `inline` codeRef (16 KB). */
export const MAX_INLINE_BYTES = 16 * 1024;
/** Maximum source bytes for `inlineB64` codeRef (64 KB). */
export const MAX_INLINE_B64_BYTES = 64 * 1024;

/**
 * Build a CodeRef carrying the bytes inline.
 *
 * - `<= MAX_INLINE_BYTES` → `{ kind: 'inline', source }` (UTF-8 source).
 * - `<= MAX_INLINE_B64_BYTES` → `{ kind: 'inlineB64', source: <base64>, sha256 }`.
 * - Larger → throws `CodeTooLargeError` (caller surfaces as `CODE_TOO_LARGE`).
 *
 * Used by the portal factory at envelope-build time to ship bytes the
 * executor can resolve without hitting the host-bridge content path.
 *
 * Async because computing SHA-256 portably across Node and browsers
 * requires `crypto.subtle.digest`. The factory awaits this once per
 * function at agent-build time.
 */
export class CodeTooLargeError extends Error {
  readonly code = 'CODE_TOO_LARGE' as const;
  constructor(actualBytes: number, maxBytes: number) {
    super(
      `Function source is ${actualBytes} bytes; exceeds inline cap ${maxBytes}. ` +
        'Move to a content row reference (when content-fetch host-bridge lands) ' +
        'or shrink the function.',
    );
  }
}

export async function encodeInlineCodeRef(source: string): Promise<CodeRef> {
  const utf8 = new TextEncoder().encode(source);
  if (utf8.byteLength <= MAX_INLINE_BYTES) {
    return { kind: 'inline', source };
  }
  if (utf8.byteLength <= MAX_INLINE_B64_BYTES) {
    const sha256 = await sha256Hex(utf8);
    const b64 = utf8ToBase64(utf8);
    return { kind: 'inlineB64', source: b64, sha256 };
  }
  throw new CodeTooLargeError(utf8.byteLength, MAX_INLINE_B64_BYTES);
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const subtle = (globalThis as { crypto?: { subtle?: SubtleCrypto } }).crypto?.subtle;
  if (!subtle) throw new Error('crypto.subtle.digest unavailable');
  // Pass an ArrayBuffer view, not a Uint8Array (jose+jsdom-quirk safe).
  const hashBuf = await subtle.digest('SHA-256', bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  const view = new Uint8Array(hashBuf);
  let out = '';
  for (let i = 0; i < view.length; i++) out += view[i].toString(16).padStart(2, '0');
  return out;
}

function utf8ToBase64(bytes: Uint8Array): string {
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  const g = globalThis as { btoa?: (s: string) => string };
  if (typeof g.btoa !== 'function') throw new Error('btoa unavailable');
  return g.btoa(bin);
}
