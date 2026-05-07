/**
 * Executor service public types.
 */

import type { InvocationEnvelope, ExecutorResponse, ExecutorValidationResult } from '../skills/functions/executor-client';
import type { FunctionManifest } from '../skills/functions/manifest';

export type ExecutorRuntimeId = 'js-v1' | 'python-pyodide-v1' | 'wasm-v1';

export interface ExecutorRuntime {
  readonly id: ExecutorRuntimeId;
  readonly enabled: boolean;
  /** Cold-start a sandbox for the given source; returns a handle the pool reuses. */
  prepare(source: string, manifest: FunctionManifest): Promise<RuntimeSandbox>;
  /** One-shot validation pass. Stricter limits than `invoke`. */
  validate(source: string, manifest: FunctionManifest): Promise<ExecutorValidationResult>;
}

export interface RuntimeSandbox {
  /** Stable id (e.g. content-sha) — used as the pool cache key. */
  readonly id: string;
  /** Single function invocation. */
  invoke(env: InvocationEnvelope): Promise<ExecutorResponse>;
  /** Tear down the sandbox; pool calls this when the worker is reaped. */
  dispose(): Promise<void>;
}

export interface ExecutorPoolMetrics {
  workersTotal: number;
  workersBusy: number;
  cpuLoadPct: number;
  invocationsInFlight: number;
  invocationsPerSecond: number;
  admissionRejections: number;
  // Per-runtime sandbox cache hits / misses.
  cacheHits: Record<ExecutorRuntimeId, number>;
  cacheMisses: Record<ExecutorRuntimeId, number>;
}

export type AdmissionDecision =
  | { ok: true }
  | { ok: false; reason: 'CPU_PRESSURE' | 'POOL_SATURATED' | 'CONCURRENCY_EXCEEDED' };
