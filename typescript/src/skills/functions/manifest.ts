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

/** Runtime identifier. `wasm-v1` slot reserved but ships disabled in v1. */
export type FunctionRuntimeId = 'js-v1' | 'python-pyodide-v1' | 'wasm-v1';

/** Subset of runtimes actually executable in v1. */
export const SUPPORTED_RUNTIMES: readonly FunctionRuntimeId[] = [
  'js-v1',
  'python-pyodide-v1',
] as const;

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
  kv?: KvMode;
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
  /**
   * `websocket` is reserved in the type union for v2 but the validator
   * rejects it with `WS_NOT_YET_SUPPORTED` in v1.
   */
  type?: 'http' | 'websocket' | 'tool' | 'cron' | 'function';
}

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
