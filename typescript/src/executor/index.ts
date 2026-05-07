/**
 * Function executor service — public entrypoint.
 *
 * This module exposes the in-process API the executor binary boots:
 *
 *   - `startExecutorServer({ port, mtls? })` — start an HTTPS/HTTP server
 *     that accepts `/invoke` and `/validate` envelopes over mTLS.
 *   - `WorkerPool` — oversubscribed worker-thread pool that runs one
 *     invocation per worker for fairness; admission gate watches CPU
 *     pressure.
 *   - `RuntimeRegistry` — runtime dispatch (js-v1 → V8 isolate via
 *     isolated-vm; python-pyodide-v1 deferred per ADR-0008; wasm-v1
 *     reserved, disabled).
 *
 * The same package can run inside the cloud-deployed executor pod or
 * locally as `webagents-executor` (both wrap `startExecutorServer`).
 */

export { startExecutorServer } from './server';
export { WorkerPool } from './worker-pool';
export { RuntimeRegistry, registerRuntime } from './runtime-registry';
export { JsV1Runtime } from './runtimes/js-v1';
export { WasmV1Runtime } from './runtimes/wasm-v1';
export type {
  ExecutorRuntime,
  ExecutorRuntimeId,
  ExecutorPoolMetrics,
  AdmissionDecision,
} from './types';

// Local-runner surface re-exported from `./local` so the standalone
// `webagents-executor` npm package and any embedded host can boot the
// executor with a single import: `import { cli, runLocalExecutor, … }
// from 'webagents/executor'`. Keeping these here (rather than under
// `webagents/executor/local`) keeps the SDK exports map flat.
export {
  runLocalExecutor,
  cli,
  probeLocalExecutor,
  DEFAULT_LOCAL_SOCKET,
} from './local';
export type { LocalExecutorOptions } from './local';
