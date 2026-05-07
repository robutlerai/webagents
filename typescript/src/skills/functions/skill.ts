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
  type InvocationEnvelope,
  type SerializableContext,
} from './executor-client';
import type { FunctionManifest } from './manifest';

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
  /** Default per-invocation limits — overridden by manifest hint / agent override / plan ceiling. */
  defaultLimits?: FunctionLimitsResolved;
  /** Maximum nested-call depth (plan-tier configurable). */
  maxChainDepth?: number;
  /** Executor client (defaults to a no-op stub). */
  executor?: ExecutorClient;
}

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
  private readonly defaultLimits: FunctionLimitsResolved;
  private readonly maxChainDepth: number;

  constructor(config: FunctionRuntimeSkillConfig = {}) {
    super(config);
    this.functions = new Map(Object.entries(config.functions ?? {}));
    this.executor = config.executor ?? new StubExecutorClient();
    this.agentId = config.agentId ?? '';
    this.defaultLimits = config.defaultLimits ?? DEFAULT_LIMITS;
    this.maxChainDepth = config.maxChainDepth ?? DEFAULT_MAX_FN_CHAIN_DEPTH;
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
    };

    const result = await this.executor.invoke<T>(envelope);
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
          source: { skill: 'function', consumerId: baseSource.invocationId, invocationId: cryptoRandomId() },
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
   * `@prompt` contribution that lists installed functions for the runtime
   * agent's LLM. Returns "" when no functions are declared so token cost
   * stays zero in the empty case.
   */
  @prompt({ priority: 60, name: 'functionsRuntime', scope: 'all' })
  functionsRuntime(_ctx: Context): string {
    if (this.functions.size === 0) return '';
    const names = this.list().join(', ');
    return `Installed functions on this agent: ${names}. Use them via the available skill tools/endpoints. Functions are sandboxed and metered.`;
  }
}

function failure<T>(code: string, message: string): FunctionInvocationResult<T> {
  return {
    ok: false,
    errorCode: code,
    errorMessage: message,
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
