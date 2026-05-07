/**
 * `ExecutorClient` — typed mTLS-aware client for the function-executor service.
 *
 * In v1 we ship two transports:
 *   - HTTPS+mTLS to the cloud `function-executor` workload.
 *   - Unix socket to the `webagents-executor` daemon on localhost.
 *
 * The client is transport-agnostic at the type level so tests can swap in
 * an in-process stub. Concrete transport selection happens in the
 * `PortalFunctionRuntimeFactory` (cloud) or the local daemon bootstrap.
 */

import type { FunctionContext, FunctionInvocationResult, InvocationChain } from './context';
import type { CodeRef, FunctionManifest } from './manifest';

/** Envelope sent to the executor for a single invocation. */
export interface InvocationEnvelope {
  /** Stable agent-local function name (the `agent_configs.functions` key). */
  functionName: string;
  /** Agent id — sandbox key includes this, never reused across tenants. */
  agentId: string;
  /** Pinned bundle hash; mismatch fails fast. */
  bundleSha256: string;
  /** Manifest as resolved at agent build time. */
  manifest: FunctionManifest;
  /** CodeRef pointing at the bytes the executor will load + bundle. */
  codeRef: CodeRef;
  /** Per-invocation context (without host-bridged APIs — those live in the executor). */
  context: SerializableContext;
  /** Recursion + budget state for `ctx.fn` nested calls. */
  chain?: InvocationChain;
  /**
   * Optional Idempotency-Key — when set the coordinator dedupes against
   * Redis (24h TTL); duplicate envelopes return the original result.
   */
  idempotencyKey?: string;
  /** Validation-only mode (no network, no secrets, no folders). */
  validateOnly?: boolean;
}

/**
 * Subset of `FunctionContext` that travels over the wire — host APIs are
 * injected on the executor side and bridged back through the coordinator.
 */
export interface SerializableContext {
  source: FunctionContext['source'];
  request?: FunctionContext['request'];
  schedule?: FunctionContext['schedule'];
  toolCall?: FunctionContext['toolCall'];
  auth: FunctionContext['auth'];
  limits: FunctionContext['limits'];
}

/** Response shape from the executor. */
export interface ExecutorResponse<T = unknown> extends FunctionInvocationResult<T> {
  /** Optional structured logs streamed back for the function drawer Activity tab. */
  logs?: Array<{
    level: 'debug' | 'info' | 'warn' | 'error';
    message: string;
    data?: unknown;
    ts: number;
  }>;
}

/** Validation result returned by the executor's `validate` envelope. */
export interface ExecutorValidationResult {
  ok: boolean;
  bundleSha256?: string;
  /** Lint diagnostics (non-fatal). */
  warnings: Array<{ line?: number; column?: number; code: string; message: string }>;
  /** Structured errors (fatal). */
  errors: Array<{ line?: number; column?: number; code: string; message: string }>;
  /** Smoke run result envelope (when validation reached the smoke stage). */
  smokeResult?: ExecutorResponse;
}

/**
 * Source payload accepted by `ExecutorClient.validate`. The portal's
 * /validate route uses this to pre-flight un-saved editor content
 * before the source ever lands in a content row, so the wire format is
 * source-bearing rather than codeRef-pointing. (Saved code goes through
 * `invoke` with `validateOnly: true` instead, which carries a CodeRef.)
 */
export interface ExecutorValidateArgs {
  runtime: string;
  source: string;
  manifest: FunctionManifest;
}

/** Hook for the runtime skill — implemented by transport layers. */
export interface ExecutorClient {
  /** Run a single function invocation. */
  invoke<T = unknown>(envelope: InvocationEnvelope): Promise<ExecutorResponse<T>>;
  /** Validate a function (parse + lint + bundle + smoke). */
  validate(args: ExecutorValidateArgs): Promise<ExecutorValidationResult>;
  /** Shut down (close mTLS pool / unix socket). */
  close?(): Promise<void>;
}

/**
 * Stub executor — returns a fixed response. Used by tests and as the
 * default before the real client is wired (so unit tests of the
 * consumer skills don't need to spin up a sandbox).
 */
export class StubExecutorClient implements ExecutorClient {
  constructor(
    private readonly handler: <T = unknown>(env: InvocationEnvelope) => Promise<ExecutorResponse<T>> =
      async () => ({
        ok: true,
        result: undefined,
        durationMs: 0,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      }) as never,
  ) {}

  async invoke<T = unknown>(envelope: InvocationEnvelope): Promise<ExecutorResponse<T>> {
    return this.handler<T>(envelope);
  }

  async validate(_args: ExecutorValidateArgs): Promise<ExecutorValidationResult> {
    return { ok: true, warnings: [], errors: [] };
  }
}
