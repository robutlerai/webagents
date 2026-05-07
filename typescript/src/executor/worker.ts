/**
 * Worker thread bootstrap.
 *
 * One thread per worker pool slot. Receives `InvocationEnvelope` over
 * postMessage, looks up (or cold-starts) the matching sandbox in this
 * thread's LRU cache, runs the invocation, and posts the
 * `ExecutorResponse` back.
 *
 * Each worker keeps an LRU cache of `RuntimeSandbox`es keyed by
 * `(agentId, functionName, sha256)`. The cache size is bounded by RSS
 * — when the pool admission gate sees memory pressure it asks workers
 * to evict cold sandboxes.
 */

import { parentPort } from 'worker_threads';
import { RuntimeRegistry, registerRuntime } from './runtime-registry';
import { JsV1Runtime } from './runtimes/js-v1';
import { PythonPyodideV1Runtime } from './runtimes/python-pyodide-v1';
import { WasmV1Runtime } from './runtimes/wasm-v1';
import type { InvocationEnvelope, ExecutorResponse } from '../skills/functions/executor-client';
import type { ExecutorRuntimeId, RuntimeSandbox } from './types';

registerRuntime(new JsV1Runtime());
registerRuntime(new PythonPyodideV1Runtime());
registerRuntime(new WasmV1Runtime());

const SANDBOX_CACHE = new Map<string, RuntimeSandbox>();
const MAX_SANDBOXES = 32;

if (!parentPort) throw new Error('worker.ts must run inside a Worker thread');

/** Resolve a codeRef inside the worker. Cloud mode rejects file://. */
async function resolveSource(env: InvocationEnvelope): Promise<string> {
  const ref = env.codeRef;
  if (ref.kind === 'inline') return ref.source;
  if (ref.kind === 'inlineB64') return Buffer.from(ref.sourceB64, 'base64').toString('utf-8');
  if (ref.kind === 'https') {
    const r = await fetch(ref.url);
    if (!r.ok) throw new Error(`codeRef https fetch ${r.status}`);
    return await r.text();
  }
  if (ref.kind === 'content') {
    // The coordinator inlines content for us; this branch only runs in
    // tests where the envelope was hand-built without resolution.
    throw new Error('content codeRef must be pre-resolved by the coordinator');
  }
  throw new Error(`unsupported codeRef in executor: ${(ref as { kind: string }).kind}`);
}

parentPort.on('message', async (env: InvocationEnvelope) => {
  const t0 = Date.now();
  try {
    const runtimeId = env.manifest.runtime as ExecutorRuntimeId;
    const runtime = RuntimeRegistry.get(runtimeId);
    if (!runtime || !runtime.enabled) {
      parentPort!.postMessage(<ExecutorResponse>{
        ok: false,
        status: 400,
        errorCode: runtime ? 'RUNTIME_DISABLED' : 'RUNTIME_UNKNOWN',
        errorMessage: `${runtimeId} unavailable`,
        durationMs: Date.now() - t0,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      });
      return;
    }
    const cacheKey = `${env.agentId}:${env.functionName}:${env.bundleSha256}`;
    let sandbox = SANDBOX_CACHE.get(cacheKey);
    if (!sandbox) {
      if (SANDBOX_CACHE.size >= MAX_SANDBOXES) {
        const oldest = SANDBOX_CACHE.keys().next().value as string | undefined;
        if (oldest) {
          const evicted = SANDBOX_CACHE.get(oldest);
          SANDBOX_CACHE.delete(oldest);
          await evicted?.dispose();
        }
      }
      const source = await resolveSource(env);
      sandbox = await runtime.prepare(source, env.manifest);
      SANDBOX_CACHE.set(cacheKey, sandbox);
    } else {
      SANDBOX_CACHE.delete(cacheKey);
      SANDBOX_CACHE.set(cacheKey, sandbox);
    }
    const result = await sandbox.invoke(env);
    parentPort!.postMessage(result);
  } catch (e) {
    parentPort!.postMessage(<ExecutorResponse>{
      ok: false,
      status: 500,
      errorCode: 'WORKER_ERROR',
      errorMessage: (e as Error).message,
      durationMs: Date.now() - t0,
      cpuMs: 0,
      ingressBytes: 0,
      egressBytes: 0,
    });
  }
});
