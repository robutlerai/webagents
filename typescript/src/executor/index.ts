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
 *     isolated-vm; python-pyodide-v1 → warm Pyodide instance; wasm-v1
 *     reserved, disabled).
 *
 * The same package can run inside the cloud-deployed executor pod or
 * locally as `webagents-executor` (both wrap `startExecutorServer`).
 */

export { startExecutorServer } from './server';
export { WorkerPool } from './worker-pool';
export { RuntimeRegistry, registerRuntime } from './runtime-registry';
export { JsV1Runtime } from './runtimes/js-v1';
export { PythonPyodideV1Runtime } from './runtimes/python-pyodide-v1';
export { WasmV1Runtime } from './runtimes/wasm-v1';
export type {
  ExecutorRuntime,
  ExecutorRuntimeId,
  ExecutorPoolMetrics,
  AdmissionDecision,
} from './types';
