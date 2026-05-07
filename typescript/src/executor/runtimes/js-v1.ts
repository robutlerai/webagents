/**
 * `js-v1` runtime — V8 isolate via `isolated-vm`.
 *
 * Lifecycle of a single invocation:
 *
 *   1. Worker resolves `codeRef` → UTF-8 source string.
 *   2. `transformSource` rewrites ESM/CJS exports to a global
 *      `__handler` (see js-v1-transform.ts).
 *   3. The isolate is created with `memoryLimit` from the manifest.
 *   4. Web Platform globals are installed:
 *      - V8-native (URL, URLSearchParams, atob, btoa, console, etc.)
 *        as direct setters.
 *      - Bridged (crypto.subtle, AbortController, Request/Response,
 *        structuredClone, URLPattern) as Reference<Function> callbacks
 *        backed by Node 20 host primitives. URLPattern is missing
 *        natively in Node 20 → polyfilled via `urlpattern-polyfill`.
 *   5. `ctx` synchronous fields are shipped via `ExternalCopy`; host
 *      APIs are installed as Reference callbacks that either POST to
 *      `/api/internal/fn-host` (state) or run directly (`ctx.fetch`).
 *   6. The transformed source compiles + runs once (sets `__handler`).
 *   7. `evalClosure('return __handler(ctx);', { result: { promise: true }})`
 *      runs the user handler (no `await` inside the snippet — ivm's
 *      closure is not async).
 *   8. A wall-clock watchdog (`setTimeout(wallMs)`) calls
 *      `isolate.dispose()` if the handler exceeds its budget — this
 *      covers async waits that `evalClosure({timeout})` cannot.
 *   9. The marshalled result is returned; errors are mapped to
 *      stable error codes the portal logs and bills against.
 */

import { performance } from 'perf_hooks';
import { URLPattern as URLPatternPolyfill } from 'urlpattern-polyfill';
import type {
  ExecutorRuntime,
  RuntimeSandbox,
  ExecutorRuntimeId,
} from '../types';
import type {
  HostBridge,
  InvocationEnvelope,
  ExecutorResponse,
  ExecutorValidationResult,
} from '../../skills/functions/executor-client';
import type { FunctionManifest } from '../../skills/functions/manifest';
import { transformSource, TransformError } from './js-v1-transform';
import { ISOLATE_BOOTSTRAP, CTX_BUILDER_SOURCE } from './js-v1-bootstrap';

// ---------------------------------------------------------------------------
// URLPattern polyfill install
//
// `URLPattern` is exposed to user code via the bootstrap bridge, which calls
// the host's `globalThis.URLPattern`. Node 20 (our pinned base image — see
// the executor Dockerfile and ADR-0010) does not ship URLPattern natively;
// install the WICG polyfill once at module load if the global is missing.
// The polyfill is a pure-JS class implementing the same spec surface, so
// user code calling `new URLPattern(...)` inside the isolate sees identical
// behaviour whether the host is Node 20 (polyfill) or Node 22+ (native).
// On Node 22+, the native URLPattern wins and the polyfill is a no-op
// at runtime — only the import cost (a few KB) is paid.
// ---------------------------------------------------------------------------
if (typeof (globalThis as { URLPattern?: unknown }).URLPattern === 'undefined') {
  (globalThis as { URLPattern?: unknown }).URLPattern = URLPatternPolyfill;
}

// ---------------------------------------------------------------------------
// `isolated-vm` is a native module — load lazily so the executor can boot
// in environments where it's not installed (unit tests, type-only builds).
// ---------------------------------------------------------------------------

type IvmReference<T = unknown> = {
  apply(thisArg: unknown, args: unknown[], opts?: unknown): Promise<unknown>;
  applySync(thisArg: unknown, args: unknown[], opts?: unknown): unknown;
  applyIgnored(thisArg: unknown, args: unknown[]): void;
  copySync?(): T;
};
type IvmExternalCopy<T = unknown> = {
  copyInto(opts?: { release?: boolean }): T;
  release(): void;
};
type IvmContext = {
  global: { setSync(name: string, value: unknown, opts?: { reference?: boolean }): void };
  release(): void;
  evalClosure(
    src: string,
    args: unknown[],
    opts?: { timeout?: number; arguments?: { copy?: boolean }; result?: { copy?: boolean; promise?: boolean } },
  ): Promise<unknown>;
};
type IvmIsolate = {
  createContext(): Promise<IvmContext>;
  compileScript(src: string): Promise<{ run(ctx: IvmContext, opts?: { timeout?: number }): Promise<unknown>; release(): void }>;
  cpuTime: bigint;
  isDisposed: boolean;
  dispose(): void;
};
type IvmModule = {
  Isolate: new (opts: { memoryLimit: number }) => IvmIsolate;
  Reference: new <T>(value: T, opts?: { unsafeInherit?: boolean }) => IvmReference<T>;
  ExternalCopy: new <T>(value: T, opts?: { transferList?: ArrayBuffer[] }) => IvmExternalCopy<T>;
};

let ivmModuleCache: IvmModule | null | undefined;

const runtimeDynamicImport = new Function(
  'specifier',
  'return import(specifier)',
) as (specifier: string) => Promise<unknown>;

/**
 * `isolated-vm` ships as CommonJS. `await import('isolated-vm')` returns a
 * module namespace where `Isolate` may live on `default` (the CJS
 * `module.exports` object) rather than as a top-level named export —
 * depending on Node version and how the package is pre-bundled into
 * `webagents/dist`. Normalise so we always get the object that actually
 * carries `Isolate`, `Reference`, and `ExternalCopy` constructors.
 */
function coerceIvmModule(raw: unknown): IvmModule | null {
  if (!raw || typeof raw !== 'object') return null;
  const ns = raw as Record<string, unknown>;
  const fromNs = ns as unknown as IvmModule;
  if (
    typeof fromNs.Isolate === 'function' &&
    typeof fromNs.Reference === 'function' &&
    typeof fromNs.ExternalCopy === 'function'
  ) {
    return fromNs;
  }
  const dflt = ns.default;
  if (dflt && typeof dflt === 'object') {
    const fromDefault = dflt as unknown as IvmModule;
    if (
      typeof fromDefault.Isolate === 'function' &&
      typeof fromDefault.Reference === 'function' &&
      typeof fromDefault.ExternalCopy === 'function'
    ) {
      return fromDefault;
    }
  }
  return null;
}

async function loadIvm(): Promise<IvmModule | null> {
  if (ivmModuleCache !== undefined) return ivmModuleCache;
  try {
    const raw = await runtimeDynamicImport('isolated-vm');
    const mod = coerceIvmModule(raw);
    ivmModuleCache = mod;
    return mod;
  } catch {
    ivmModuleCache = null;
    return null;
  }
}

// ---------------------------------------------------------------------------
// Per-invocation metering buffer. Lives on the host side; user code
// touches it only through Reference callbacks.
// ---------------------------------------------------------------------------

interface MeterBuffer {
  ingressBytes: number;
  egressBytes: number;
  hostCalls: number;
  logs: Array<{ level: 'debug' | 'info' | 'warn' | 'error'; message: string; data?: unknown; ts: number }>;
}

function newMeter(): MeterBuffer {
  return { ingressBytes: 0, egressBytes: 0, hostCalls: 0, logs: [] };
}

// ---------------------------------------------------------------------------
// Host-bridge HTTP client (worker-thread side)
// ---------------------------------------------------------------------------

async function callHostBridge(
  bridge: HostBridge,
  invocationId: string,
  callSeq: number,
  method: string,
  args: unknown,
): Promise<unknown> {
  const r = await fetch(`${bridge.baseUrl}/api/internal/fn-host`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${bridge.token}`,
      'idempotency-key': `${invocationId}:${callSeq}`,
    },
    body: JSON.stringify({ method, args }),
  });
  const text = await r.text();
  let parsed: { ok: boolean; result?: unknown; error?: { code: string; message: string } };
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(`HOST_BRIDGE_NON_JSON: ${r.status} ${text.slice(0, 200)}`);
  }
  if (!parsed.ok) {
    const code = parsed.error?.code ?? 'HOST_BRIDGE_ERROR';
    const msg = parsed.error?.message ?? 'host bridge call failed';
    const e = new Error(`${code}: ${msg}`);
    (e as Error & { code?: string }).code = code;
    throw e;
  }
  return parsed.result;
}

// ---------------------------------------------------------------------------
// Web Platform global bridges
//
// Strategy: V8 ships URL, URLSearchParams, atob, btoa, JSON, Math, Date,
// Promise, Map, Set, WeakMap, WeakSet, RegExp, Symbol, Proxy, Reflect,
// Error/TypeError/etc., Intl natively. We expose those by simply NOT
// blocking them.
//
// For the rest (TextEncoder, TextDecoder, crypto.{subtle,randomUUID,
// getRandomValues}, console, Request, Response, Headers, AbortController,
// AbortSignal, EventTarget, structuredClone, URLPattern), we install
// Reference<Function> callbacks that delegate to the host's Node 20
// runtime (URLPattern is polyfilled — see top-of-file install). The
// user-visible surface mirrors the WHATWG specs; data
// crosses the isolate boundary as plain JS values copied via
// ExternalCopy.
// ---------------------------------------------------------------------------

function installCoreGlobals(
  ivm: IvmModule,
  ctx: IvmContext,
  meter: MeterBuffer,
): void {
  // console — rewrite to push to host logs.
  const consoleProxy = (level: 'debug' | 'info' | 'warn' | 'error') =>
    new ivm.Reference((...args: unknown[]) => {
      const message = args
        .map((a) =>
          typeof a === 'string'
            ? a
            : (() => {
                try {
                  return JSON.stringify(a);
                } catch {
                  return String(a);
                }
              })(),
        )
        .join(' ');
      meter.logs.push({ level, message, ts: Date.now() });
    });
  ctx.global.setSync('__hostConsoleDebug', consoleProxy('debug'), { reference: true });
  ctx.global.setSync('__hostConsoleInfo', consoleProxy('info'), { reference: true });
  ctx.global.setSync('__hostConsoleWarn', consoleProxy('warn'), { reference: true });
  ctx.global.setSync('__hostConsoleError', consoleProxy('error'), { reference: true });

  // crypto.randomUUID + crypto.getRandomValues — native v8 doesn't ship
  // them; bridge to Node webcrypto.
  ctx.global.setSync(
    '__hostRandomUUID',
    new ivm.Reference(() => crypto.randomUUID()),
    { reference: true },
  );
  ctx.global.setSync(
    '__hostGetRandomValues',
    new ivm.Reference((nBytes: number): ArrayBuffer => {
      const u = new Uint8Array(nBytes);
      crypto.getRandomValues(u);
      // Return a fresh ArrayBuffer — the worker copies into the isolate.
      return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength);
    }),
    { reference: true },
  );

  // crypto.subtle — bridge digest/sign/verify/encrypt/decrypt/etc.
  // We expose a small dispatcher rather than 12 separate references.
  ctx.global.setSync(
    '__hostSubtle',
    new ivm.Reference(async (op: string, payload: Record<string, unknown>) => {
      // Re-hydrate Uint8Arrays / ArrayBuffers from the isolate-copied
      // form (they arrive as plain ArrayBuffers).
      const subtle = (globalThis as { crypto: Crypto }).crypto.subtle;
      switch (op) {
        case 'digest':
          return await subtle.digest(payload.algorithm as AlgorithmIdentifier, payload.data as ArrayBuffer);
        case 'importKey': {
          // `importKey` has two overloads: a binary one (`raw`/`pkcs8`/`spki`,
          // BufferSource) and a JWK one (`jwk`, JsonWebKey). Branch so each
          // call site picks the matching signature.
          const fmt = payload.format as 'raw' | 'pkcs8' | 'spki' | 'jwk';
          if (fmt === 'jwk') {
            return await subtle.importKey(
              fmt,
              payload.keyData as JsonWebKey,
              payload.algorithm as AlgorithmIdentifier,
              payload.extractable as boolean,
              payload.keyUsages as KeyUsage[],
            );
          }
          return await subtle.importKey(
            fmt,
            payload.keyData as ArrayBuffer,
            payload.algorithm as AlgorithmIdentifier,
            payload.extractable as boolean,
            payload.keyUsages as KeyUsage[],
          );
        }
        case 'sign':
          return await subtle.sign(
            payload.algorithm as AlgorithmIdentifier,
            payload.key as CryptoKey,
            payload.data as ArrayBuffer,
          );
        case 'verify':
          return await subtle.verify(
            payload.algorithm as AlgorithmIdentifier,
            payload.key as CryptoKey,
            payload.signature as ArrayBuffer,
            payload.data as ArrayBuffer,
          );
        case 'encrypt':
          return await subtle.encrypt(
            payload.algorithm as AlgorithmIdentifier,
            payload.key as CryptoKey,
            payload.data as ArrayBuffer,
          );
        case 'decrypt':
          return await subtle.decrypt(
            payload.algorithm as AlgorithmIdentifier,
            payload.key as CryptoKey,
            payload.data as ArrayBuffer,
          );
        default:
          throw new Error(`UNSUPPORTED_SUBTLE_OP: ${op}`);
      }
    }),
    { reference: true },
  );

  // structuredClone — bridge to host's native structuredClone (Node 17+).
  ctx.global.setSync(
    '__hostStructuredClone',
    new ivm.Reference((value: unknown) => structuredClone(value)),
    { reference: true },
  );

  // URLPattern — provided by the host (native on Node 22+, polyfill on
  // Node 20 via `urlpattern-polyfill` — see top-of-file import).
  // Bridge creation / test / exec via a per-isolate handle store so we
  // don't have to keep a Reference per pattern instance (which would
  // leak memory across invocations).
  type HostURLPatternCtor = new (input: unknown, baseURL?: string) => {
    test(input: unknown, baseURL?: string): boolean;
    exec(input: unknown, baseURL?: string): unknown;
  };
  const HostURLPattern = (globalThis as { URLPattern?: HostURLPatternCtor }).URLPattern;
  const patterns = new Map<number, InstanceType<HostURLPatternCtor>>();
  let patternSeq = 0;
  ctx.global.setSync(
    '__hostUrlPatternCreate',
    new ivm.Reference((input: unknown, baseURL?: string): { id: number } | { error: string } => {
      if (!HostURLPattern) {
        // Should be unreachable — the top-of-file polyfill install
        // guarantees `globalThis.URLPattern` exists before this bridge
        // is registered. If we ever see this, the polyfill failed to
        // load (missing dep?) and that's worth surfacing verbatim.
        return { error: 'URLPattern is not available on the host (urlpattern-polyfill missing?)' };
      }
      try {
        const p = new HostURLPattern(input, baseURL);
        const id = ++patternSeq;
        patterns.set(id, p);
        return { id };
      } catch (e) {
        return { error: (e as Error).message ?? 'URLPattern construction failed' };
      }
    }),
    { reference: true },
  );
  ctx.global.setSync(
    '__hostUrlPatternTest',
    new ivm.Reference((id: number, input: unknown, baseURL?: string): boolean => {
      const p = patterns.get(id);
      if (!p) return false;
      try {
        return p.test(input, baseURL);
      } catch {
        return false;
      }
    }),
    { reference: true },
  );
  ctx.global.setSync(
    '__hostUrlPatternExec',
    new ivm.Reference((id: number, input: unknown, baseURL?: string): unknown => {
      const p = patterns.get(id);
      if (!p) return null;
      try {
        return p.exec(input, baseURL);
      } catch {
        return null;
      }
    }),
    { reference: true },
  );
}

// `ISOLATE_BOOTSTRAP` and `CTX_BUILDER_SOURCE` live in
// `js-v1-bootstrap.ts` — kept separate so they're readable and
// unit-testable as plain strings.

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

class JsV1Sandbox implements RuntimeSandbox {
  constructor(
    public readonly id: string,
    private readonly ivm: IvmModule,
    private isolate: IvmIsolate | null,
    private context: IvmContext | null,
    private readonly bootstrap: { run(ctx: IvmContext, opts?: { timeout?: number }): Promise<unknown>; release(): void },
    private readonly userScript: { run(ctx: IvmContext, opts?: { timeout?: number }): Promise<unknown>; release(): void },
    private readonly resolvedEntrypoint: string,
  ) {}

  async invoke(env: InvocationEnvelope): Promise<ExecutorResponse> {
    if (!this.isolate || !this.context) {
      return failure('SANDBOX_DISPOSED', 'sandbox disposed', 0, 0);
    }
    const isolate = this.isolate;
    const context = this.context;
    const ivm = this.ivm;

    const t0 = performance.now();
    const cpu0 = isolate.cpuTime;
    const wallMs = env.context?.limits?.wallMs ?? env.manifest.limits?.wallMs ?? 30_000;
    const meter = newMeter();

    // Per-invocation state shared with host callbacks.
    const invocationId = env.context?.source?.invocationId ?? `inv_${Date.now().toString(36)}`;
    let callSeq = 0;

    // ---------------------------------------------------------------
    // Wall-clock watchdog. evalClosure({timeout}) only accounts for
    // sync v8 time — async work (host-bridge HTTP, fetch waits) doesn't
    // count. We arm a timer that disposes the isolate; that's the
    // hardest stop available short of killing the worker.
    // ---------------------------------------------------------------
    let timedOut = false;
    const watchdog = setTimeout(() => {
      timedOut = true;
      try {
        if (!isolate.isDisposed) isolate.dispose();
      } catch {
        // dispose can throw if already disposed — ignore.
      }
    }, wallMs + 50);

    try {
      // ---- Install ctx ---------------------------------------------------
      // Sync fields go via ExternalCopy. Host-side APIs are Reference callbacks.
      const sync = {
        source: env.context.source,
        request: env.context.request
          ? {
              ...env.context.request,
              // rawBody is special — Uint8Array transferred as ArrayBuffer.
              rawBody: undefined,
            }
          : undefined,
        schedule: env.context.schedule,
        toolCall: env.context.toolCall,
        auth: env.context.auth,
        limits: env.context.limits,
      };
      const syncCopy = new ivm.ExternalCopy(sync);
      context.global.setSync('__ctxSync', syncCopy.copyInto({ release: true }));

      // rawBody — only when manifest declares permissions.rawBody and the
      // request actually carried one. The Node side already has it as a
      // Uint8Array; we copy as ArrayBuffer.
      const rawBody =
        env.manifest.permissions?.rawBody && env.context.request && (env.context.request as { rawBody?: Uint8Array }).rawBody
          ? new ivm.ExternalCopy((env.context.request as { rawBody: Uint8Array }).rawBody.buffer.slice(0))
          : null;
      if (rawBody) context.global.setSync('__ctxRawBody', rawBody.copyInto({ release: true }));

      // log.* + emit
      context.global.setSync(
        '__ctxLog',
        new ivm.Reference((level: 'debug' | 'info' | 'warn' | 'error', message: string, data?: unknown) => {
          meter.logs.push({ level, message, data, ts: Date.now() });
        }),
        { reference: true },
      );
      context.global.setSync(
        '__ctxEmit',
        new ivm.Reference((event: string, payload?: unknown) => {
          meter.logs.push({ level: 'info', message: `event:${event}`, data: payload, ts: Date.now() });
        }),
        { reference: true },
      );

      // fetch — runs in worker thread directly, no host-bridge.
      context.global.setSync(
        '__ctxFetch',
        new ivm.Reference(async (url: string, init?: { method?: string; headers?: Record<string, string>; body?: string | ArrayBuffer; timeoutMs?: number }) => {
          const allow = env.manifest.permissions?.fetch ?? [];
          if (allow.length === 0) {
            throw new Error('FETCH_FORBIDDEN: permissions.fetch is empty');
          }
          if (!isFetchAllowed(allow as readonly string[], url)) {
            throw new Error(`FETCH_FORBIDDEN: ${url} not in allowlist`);
          }
          const ac = new AbortController();
          const timer = init?.timeoutMs ? setTimeout(() => ac.abort(), init.timeoutMs) : null;
          try {
            const r = await fetch(url, {
              method: init?.method ?? 'GET',
              headers: init?.headers,
              body: init?.body as BodyInit | undefined,
              signal: ac.signal,
            });
            const buf = await r.arrayBuffer();
            meter.ingressBytes += buf.byteLength;
            const reqBytes =
              typeof init?.body === 'string'
                ? Buffer.byteLength(init.body, 'utf8')
                : init?.body instanceof ArrayBuffer
                  ? init.body.byteLength
                  : 0;
            meter.egressBytes += reqBytes;
            const headers: Record<string, string> = {};
            r.headers.forEach((v, k) => {
              headers[k] = v;
            });
            return {
              status: r.status,
              statusText: r.statusText,
              ok: r.ok,
              headers,
              bodyBytes: buf,
            };
          } finally {
            if (timer) clearTimeout(timer);
          }
        }),
        { reference: true },
      );

      // Host-bridge backed APIs (state). When envelope omits `hostBridge`,
      // these throw HOST_BRIDGE_DISABLED on first call. The shim surfaces
      // them as plain ctx APIs.
      const bridge = env.hostBridge;
      const requireBridge = () => {
        if (!bridge) throw new Error('HOST_BRIDGE_DISABLED: stateful APIs require a host bridge');
        return bridge;
      };
      const callBridge = async (method: string, args: unknown) => {
        const b = requireBridge();
        const seq = ++callSeq;
        meter.hostCalls++;
        return await callHostBridge(b, invocationId, seq, method, args);
      };
      context.global.setSync(
        '__ctxHost',
        new ivm.Reference(async (method: string, args: unknown) => {
          return await callBridge(method, args);
        }),
        { reference: true },
      );

      // ---- Build user-visible ctx inside the isolate ---------------------
      // Wrap __ctxSync, the host references, and the meter into a single
      // `globalThis.__ctx` that the handler invocation script consumes.
      await context.evalClosure(
        CTX_BUILDER_SOURCE,
        [],
        { timeout: 1_000 },
      );

      // ---- Run the user handler ------------------------------------------
      // evalClosure({timeout}) counts only synchronous V8 time — the
      // wall-clock watchdog above is what actually bounds async work.
      // Do NOT write `await` inside this snippet — ivm compiles the
      // closure as a non-async function; `await` is a syntax error. When
      // the handler returns a Promise, `result: { promise: true }`
      // unwraps it on the host bridge.
      const result = (await context.evalClosure(
        `return globalThis.__handler(globalThis.__ctx);`,
        [],
        {
          timeout: wallMs,
          result: { copy: true, promise: true },
        },
      )) as unknown;

      clearTimeout(watchdog);

      const cpuMs = Number((isolate.cpuTime - cpu0) / 1_000_000n);
      const durationMs = performance.now() - t0;
      return {
        ok: true,
        result,
        durationMs,
        cpuMs,
        ingressBytes: meter.ingressBytes,
        egressBytes: meter.egressBytes,
        logs: meter.logs,
      };
    } catch (e) {
      clearTimeout(watchdog);
      const durationMs = performance.now() - t0;
      const cpuMs = isolate && !isolate.isDisposed ? Number((isolate.cpuTime - cpu0) / 1_000_000n) : 0;
      const msg = (e as Error).message ?? String(e);
      const errorCode = mapErrorCode(msg, isolate?.isDisposed === true, timedOut);
      return {
        ok: false,
        errorCode,
        errorMessage: msg,
        durationMs,
        cpuMs,
        ingressBytes: meter.ingressBytes,
        egressBytes: meter.egressBytes,
        logs: meter.logs,
      };
    }
  }

  async dispose(): Promise<void> {
    try {
      this.bootstrap.release();
    } catch {
      // already released
    }
    try {
      this.userScript.release();
    } catch {
      // already released
    }
    try {
      this.context?.release();
    } catch {
      // already released
    }
    try {
      if (this.isolate && !this.isolate.isDisposed) this.isolate.dispose();
    } catch {
      // already disposed
    }
    this.isolate = null;
    this.context = null;
    void this.resolvedEntrypoint; // keep field referenced
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function failure(code: string, message: string, ingress: number, egress: number): ExecutorResponse {
  return {
    ok: false,
    errorCode: code,
    errorMessage: message,
    durationMs: 0,
    cpuMs: 0,
    ingressBytes: ingress,
    egressBytes: egress,
  };
}

function mapErrorCode(msg: string, isDisposed: boolean, timedOut: boolean): string {
  if (timedOut || (isDisposed && /isolate is disposed/i.test(msg))) return 'WALL_TIMEOUT';
  if (/script execution timed out/i.test(msg)) return 'WALL_TIMEOUT';
  if (/array buffer allocation failed|isolate was disposed during execution/i.test(msg))
    return 'MEMORY_LIMIT_EXCEEDED';
  if (/JS_NO_HANDLER/i.test(msg)) return 'JS_NO_HANDLER';
  if (/EVAL_DENIED/i.test(msg)) return 'EVAL_DENIED';
  if (/FUNCTION_DENIED/i.test(msg)) return 'FUNCTION_DENIED';
  if (/FETCH_FORBIDDEN/i.test(msg)) return 'FETCH_FORBIDDEN';
  if (/HOST_QUOTA_EXCEEDED/i.test(msg)) return 'HOST_QUOTA_EXCEEDED';
  if (/PERMISSION_DENIED/i.test(msg)) return 'PERMISSION_DENIED';
  if (/HOST_BRIDGE_/i.test(msg)) return 'HOST_BRIDGE_ERROR';
  if (/Failed to (transfer|copy) the value into the isolate/i.test(msg))
    return 'JS_RESULT_NOT_SERIALIZABLE';
  return 'JS_RUNTIME_ERROR';
}

/** Check whether `url` matches one of the manifest's fetch allowlist patterns. */
export function isFetchAllowed(allow: readonly string[], url: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') return false;
  if (allow.includes('*')) return true;
  for (const pat of allow) {
    if (matchPattern(pat, parsed)) return true;
  }
  return false;
}

function matchPattern(pat: string, url: URL): boolean {
  // Exact URL or host pattern. `*.example.com` matches one label.
  if (pat.startsWith('https://') || pat.startsWith('http://')) {
    return url.toString().startsWith(pat);
  }
  if (pat.startsWith('*.')) {
    const tail = pat.slice(1); // ".example.com"
    return url.hostname.endsWith(tail) && url.hostname !== tail.slice(1);
  }
  return url.hostname === pat;
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

export class JsV1Runtime implements ExecutorRuntime {
  readonly id: ExecutorRuntimeId = 'js-v1';
  get enabled() {
    // `loadIvm` is async but `enabled` is sync — return true unless we
    // know the dynamic load definitively failed.
    return ivmModuleCache !== null;
  }

  async prepare(source: string, manifest: FunctionManifest): Promise<RuntimeSandbox> {
    const ivm = await loadIvm();
    if (!ivm) throw new Error('isolated-vm not installed');

    let transformed: { source: string; resolvedEntrypoint: string };
    try {
      transformed = transformSource(source, manifest.entrypoint);
    } catch (e) {
      if (e instanceof TransformError) {
        const err = new Error(e.message);
        (err as Error & { code?: string }).code = e.code;
        throw err;
      }
      throw e;
    }

    const memoryMb = manifest.limits?.memoryMb ?? 64;
    const isolate = new ivm.Isolate({ memoryLimit: memoryMb });
    const context = await isolate.createContext();

    // Install host-side core globals (same set on every invoke; per-
    // invocation state — meter buffer — is keyed in `invoke()` via
    // separate references for ctx.log/emit.).
    const installMeter = newMeter();
    installCoreGlobals(ivm, context, installMeter);

    // Compile + run the bootstrap once at prepare time. User source
    // compiles at prepare-time but the *script run* happens once per
    // sandbox to land __handler. This is fine because each sandbox is
    // dedicated to one (agentId, fn, sha256) cache key.
    const bootstrapScript = await isolate.compileScript(ISOLATE_BOOTSTRAP);
    await bootstrapScript.run(context, { timeout: 5_000 });
    const userScript = await isolate.compileScript(transformed.source);
    await userScript.run(context, { timeout: 5_000 });

    const id = `js-v1:${Date.now()}:${Math.random().toString(36).slice(2)}`;
    return new JsV1Sandbox(
      id,
      ivm,
      isolate,
      context,
      bootstrapScript,
      userScript,
      transformed.resolvedEntrypoint,
    );
  }

  async validate(source: string, manifest: FunctionManifest): Promise<ExecutorValidationResult> {
    if (!source || source.length === 0) {
      return { ok: false, warnings: [], errors: [{ code: 'SOURCE_EMPTY', message: 'no source provided' }] };
    }
    if (source.length > 256 * 1024) {
      return { ok: false, warnings: [], errors: [{ code: 'SOURCE_TOO_LARGE', message: 'source exceeds 256KB' }] };
    }
    try {
      transformSource(source, manifest.entrypoint);
    } catch (e) {
      if (e instanceof TransformError) {
        return { ok: false, warnings: [], errors: [{ code: e.code, message: e.message }] };
      }
      return {
        ok: false,
        warnings: [],
        errors: [{ code: 'JS_TRANSFORM_ERROR', message: (e as Error).message }],
      };
    }
    try {
      const sandbox = await this.prepare(source, manifest);
      await sandbox.dispose();
      return { ok: true, warnings: [], errors: [] };
    } catch (e) {
      return {
        ok: false,
        warnings: [],
        errors: [{ code: 'JS_COMPILE_ERROR', message: (e as Error).message }],
      };
    }
  }
}
