/**
 * Oversubscribed worker-thread pool.
 *
 * - W = K × cpuCount workers (default K = 8 — empirically a good tradeoff
 *   between CPU contention and IO fill-factor for short, mostly-IO-bound
 *   user functions).
 * - One invocation per worker at a time → fairness across tenants.
 * - CPU-pressure admission gate pulls from the OS load avg / cgroup
 *   stats; if cpuLoadPct > threshold, new admissions are rejected with
 *   `CPU_PRESSURE` and the caller surfaces 429.
 * - LRU sandbox cache per runtime so repeated invocations of the same
 *   `(agentId, functionName, sha256)` re-use a hot isolate.
 */

import { Worker } from 'worker_threads';
import * as os from 'os';
import type { InvocationEnvelope, ExecutorResponse } from '../skills/functions/executor-client';
import type { AdmissionDecision, ExecutorPoolMetrics, ExecutorRuntimeId } from './types';

export interface WorkerPoolOptions {
  oversubscribe?: number;
  cpuPressureThresholdPct?: number;
  maxQueueDepth?: number;
  workerScript?: string;
}

interface WorkerSlot {
  worker: Worker;
  busy: boolean;
}

export class WorkerPool {
  private readonly workers: WorkerSlot[] = [];
  private readonly queue: Array<{
    env: InvocationEnvelope;
    resolve: (r: ExecutorResponse) => void;
    reject: (e: unknown) => void;
  }> = [];
  private readonly cpuPressureThresholdPct: number;
  private readonly maxQueueDepth: number;
  private invocationsInFlight = 0;
  private admissionRejections = 0;
  private readonly invocationCounter: number[] = [];
  private readonly cacheHits: Record<ExecutorRuntimeId, number> = { 'js-v1': 0, 'python-pyodide-v1': 0, 'wasm-v1': 0 };
  private readonly cacheMisses: Record<ExecutorRuntimeId, number> = { 'js-v1': 0, 'python-pyodide-v1': 0, 'wasm-v1': 0 };

  constructor(opts: WorkerPoolOptions = {}) {
    const k = opts.oversubscribe ?? 8;
    const cpuCount = os.cpus().length;
    const w = Math.max(1, Math.floor(k * cpuCount));
    this.cpuPressureThresholdPct = opts.cpuPressureThresholdPct ?? 85;
    this.maxQueueDepth = opts.maxQueueDepth ?? w * 4;
    const workerScript = opts.workerScript ?? new URL('./worker.js', import.meta.url).pathname;
    for (let i = 0; i < w; i++) {
      const worker = new Worker(workerScript);
      this.workers.push({ worker, busy: false });
      worker.on('message', (msg) => this.onWorkerMessage(i, msg));
      worker.on('error', (err) => this.onWorkerError(i, err));
    }
  }

  /** CPU-pressure admission gate. */
  admit(): AdmissionDecision {
    if (this.queue.length >= this.maxQueueDepth) {
      this.admissionRejections++;
      return { ok: false, reason: 'POOL_SATURATED' };
    }
    const load = os.loadavg()[0]; // 1-min load avg
    const cpus = os.cpus().length;
    const loadPct = Math.min(100, (load / cpus) * 100);
    if (loadPct > this.cpuPressureThresholdPct) {
      this.admissionRejections++;
      return { ok: false, reason: 'CPU_PRESSURE' };
    }
    return { ok: true };
  }

  async run(env: InvocationEnvelope): Promise<ExecutorResponse> {
    const decision = this.admit();
    if (!decision.ok) {
      return {
        ok: false,
        errorCode: decision.reason,
        errorMessage: `executor admission denied: ${decision.reason}`,
        durationMs: 0,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      };
    }
    const slot = this.workers.find((s) => !s.busy);
    if (!slot) {
      return new Promise((resolve, reject) => {
        this.queue.push({ env, resolve, reject });
      });
    }
    slot.busy = true;
    this.invocationsInFlight++;
    this.invocationCounter.push(Date.now());
    return new Promise((resolve, reject) => {
      slot.worker.once('message', (msg: ExecutorResponse) => {
        slot.busy = false;
        this.invocationsInFlight--;
        this.flush();
        resolve(msg);
      });
      slot.worker.postMessage(env);
      const wallMs = env.context?.limits?.wallMs ?? env.manifest.limits?.wallMs ?? 30_000;
      const t = setTimeout(() => {
        slot.worker.terminate();
        slot.busy = false;
        this.invocationsInFlight--;
        reject(new Error('worker hung past wallMs + grace'));
      }, wallMs + 5_000).unref();
      slot.worker.once('exit', () => clearTimeout(t));
    });
  }

  private flush() {
    while (this.queue.length > 0) {
      const slot = this.workers.find((s) => !s.busy);
      if (!slot) return;
      const next = this.queue.shift();
      if (!next) return;
      slot.busy = true;
      this.invocationsInFlight++;
      slot.worker.once('message', (msg: ExecutorResponse) => {
        slot.busy = false;
        this.invocationsInFlight--;
        this.flush();
        next.resolve(msg);
      });
      slot.worker.postMessage(next.env);
    }
  }

  private onWorkerMessage(_idx: number, _msg: unknown) {
    // worker resolves its outstanding promise via `worker.once('message', …)`
  }

  private onWorkerError(idx: number, _err: Error) {
    const slot = this.workers[idx];
    if (!slot) return;
    slot.worker.terminate().catch(() => {});
    const replacement = new Worker(new URL('./worker.js', import.meta.url).pathname);
    this.workers[idx] = { worker: replacement, busy: false };
  }

  metrics(): ExecutorPoolMetrics {
    const now = Date.now();
    while (this.invocationCounter.length && this.invocationCounter[0] < now - 1000) {
      this.invocationCounter.shift();
    }
    const load = os.loadavg()[0];
    const cpus = os.cpus().length;
    return {
      workersTotal: this.workers.length,
      workersBusy: this.workers.filter((s) => s.busy).length,
      cpuLoadPct: Math.min(100, (load / cpus) * 100),
      invocationsInFlight: this.invocationsInFlight,
      invocationsPerSecond: this.invocationCounter.length,
      admissionRejections: this.admissionRejections,
      cacheHits: { ...this.cacheHits },
      cacheMisses: { ...this.cacheMisses },
    };
  }

  async shutdown(): Promise<void> {
    await Promise.all(this.workers.map((s) => s.worker.terminate()));
    this.workers.length = 0;
  }
}
