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
import { readFileSync } from 'fs';
import type { InvocationEnvelope, ExecutorResponse } from '../skills/functions/executor-client';
import type { AdmissionDecision, ExecutorPoolMetrics, ExecutorRuntimeId } from './types';

export interface WorkerPoolOptions {
  oversubscribe?: number;
  cpuPressureThresholdPct?: number;
  /**
   * CPU allotment (in cores) the pressure gate measures against.
   * Defaults to the container's cgroup quota when one is set
   * (`cpu.max` / cfs_quota), else the host CPU count. Override via
   * `EXECUTOR_CPU_BUDGET_CORES` when the cgroup isn't visible.
   */
  cpuBudgetCores?: number;
  maxQueueDepth?: number;
  workerScript?: string;
  /** Test seams — production code never passes these. */
  readCpuUsage?: () => NodeJS.CpuUsage;
  now?: () => number;
}

/**
 * Resolve the container's CPU quota in cores from the cgroup fs, or
 * null when no quota is visible (bare host / no limit set).
 * cgroup v2: `/sys/fs/cgroup/cpu.max` = `"<quota|max> <period>"`.
 * cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us (-1 = unlimited).
 */
export function detectCgroupCpuBudgetCores(): number | null {
  try {
    const [quotaRaw, periodRaw] = readFileSync('/sys/fs/cgroup/cpu.max', 'utf-8').trim().split(/\s+/);
    if (quotaRaw === 'max') return null;
    const quota = Number(quotaRaw);
    const period = Number(periodRaw || '100000');
    if (quota > 0 && period > 0) return quota / period;
    return null;
  } catch {
    /* not cgroup v2 — try v1 */
  }
  try {
    const quota = Number(readFileSync('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'utf-8').trim());
    const period = Number(readFileSync('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'utf-8').trim());
    if (quota > 0 && period > 0) return quota / period;
  } catch {
    /* no cgroup limits visible */
  }
  return null;
}

const CPU_SAMPLE_MIN_INTERVAL_MS = 500;

/**
 * Self-usage CPU gate: measures THIS PROCESS's cpu time (all threads,
 * workers included — `process.cpuUsage()` is process-wide) against its
 * own allotment, EWMA-smoothed across ≥500ms windows.
 *
 * This deliberately does NOT look at `os.loadavg()`: Linux never
 * namespaces loadavg, so inside a container it reports the NODE's run
 * queue. The previous loadavg-based gate rejected every invocation on
 * any busy Kubernetes node while the executor itself sat idle (observed
 * on GKE 2026-06-12: a noisy neighbor pushed node load past 85% and all
 * widget functions 500'd with CPU_PRESSURE at 1-2ms actual cpu per
 * call). Own-usage-vs-quota is correct in a container AND on bare
 * hosts; `/proc/pressure` (PSI) remains a future refinement for
 * detecting throttling-induced stalls.
 */
export class SelfCpuGate {
  private readonly budgetCores: number;
  private readonly readCpuUsage: () => NodeJS.CpuUsage;
  private readonly now: () => number;
  private lastSample: NodeJS.CpuUsage;
  private lastSampleAt: number;
  private ewmaPct: number | null = null;

  constructor(opts: { budgetCores?: number; readCpuUsage?: () => NodeJS.CpuUsage; now?: () => number } = {}) {
    this.budgetCores = opts.budgetCores ?? detectCgroupCpuBudgetCores() ?? os.cpus().length;
    this.readCpuUsage = opts.readCpuUsage ?? (() => process.cpuUsage());
    this.now = opts.now ?? (() => Date.now());
    this.lastSample = this.readCpuUsage();
    this.lastSampleAt = this.now();
  }

  /** Smoothed percent of the CPU allotment this process is consuming. */
  pct(): number {
    const now = this.now();
    const dtMs = now - this.lastSampleAt;
    if (dtMs >= CPU_SAMPLE_MIN_INTERVAL_MS) {
      const u = this.readCpuUsage();
      const usedMs = (u.user + u.system - this.lastSample.user - this.lastSample.system) / 1000;
      const instPct = Math.min(100, Math.max(0, (usedMs / (dtMs * this.budgetCores)) * 100));
      this.ewmaPct = this.ewmaPct === null ? instPct : this.ewmaPct * 0.5 + instPct * 0.5;
      this.lastSample = u;
      this.lastSampleAt = now;
    }
    return this.ewmaPct ?? 0;
  }
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
  private readonly cpuGate: SelfCpuGate;
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
    this.cpuGate = new SelfCpuGate({
      budgetCores: opts.cpuBudgetCores,
      readCpuUsage: opts.readCpuUsage,
      now: opts.now,
    });
    this.maxQueueDepth = opts.maxQueueDepth ?? w * 4;
    const workerScript = opts.workerScript ?? new URL('./worker.js', import.meta.url).pathname;
    for (let i = 0; i < w; i++) {
      const worker = new Worker(workerScript);
      this.workers.push({ worker, busy: false });
      worker.on('message', (msg) => this.onWorkerMessage(i, msg));
      worker.on('error', (err) => this.onWorkerError(i, err));
    }
  }

  /**
   * CPU-pressure admission gate.
   *
   * Measures THIS process's cpu usage against its own allotment (see
   * `SelfCpuGate`) — NOT `os.loadavg()`, which reads the host/node run
   * queue inside containers and rejected everything on busy Kubernetes
   * nodes while the executor sat idle. `cpuPressureThresholdPct <= 0`
   * disables the gate entirely — only the queue-depth `POOL_SATURATED`
   * gate remains.
   */
  admit(): AdmissionDecision {
    if (this.queue.length >= this.maxQueueDepth) {
      this.admissionRejections++;
      return { ok: false, reason: 'POOL_SATURATED' };
    }
    if (this.cpuPressureThresholdPct <= 0) {
      return { ok: true };
    }
    if (this.cpuGate.pct() > this.cpuPressureThresholdPct) {
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
      const wallMs = env.context?.limits?.wallMs ?? env.manifest.limits?.wallMs ?? 30_000;
      const t = setTimeout(() => {
        slot.worker.terminate();
        slot.busy = false;
        this.invocationsInFlight--;
        reject(new Error('worker hung past wallMs + grace'));
      }, wallMs + 5_000).unref();
      slot.worker.once('message', (msg: ExecutorResponse) => {
        // CRITICAL: clear the watchdog on success, otherwise it fires
        // ~wallMs+grace later and terminate()s the worker mid-next-invocation,
        // causing the *next* call on the same slot to reject with
        // "worker hung past wallMs + grace".
        clearTimeout(t);
        slot.busy = false;
        this.invocationsInFlight--;
        this.flush();
        resolve(msg);
      });
      slot.worker.once('exit', () => clearTimeout(t));
      slot.worker.postMessage(env);
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
      const wallMs =
        next.env.context?.limits?.wallMs ?? next.env.manifest.limits?.wallMs ?? 30_000;
      const t = setTimeout(() => {
        slot.worker.terminate();
        slot.busy = false;
        this.invocationsInFlight--;
        next.reject(new Error('worker hung past wallMs + grace'));
      }, wallMs + 5_000).unref();
      slot.worker.once('message', (msg: ExecutorResponse) => {
        clearTimeout(t);
        slot.busy = false;
        this.invocationsInFlight--;
        this.flush();
        next.resolve(msg);
      });
      slot.worker.once('exit', () => clearTimeout(t));
      slot.worker.postMessage(next.env);
    }
  }

  private onWorkerMessage(_idx: number, _msg: unknown) {
    // worker resolves its outstanding promise via `worker.once('message', …)`
  }

  private onWorkerError(idx: number, err: Error) {
    // A `worker.on('error')` event means the worker thread crashed before
    // it could post a structured `ExecutorResponse` back. The pending
    // promise registered in `run()` via `worker.once('message', …)` will
    // never resolve, so the request hangs until the wallMs+grace timeout.
    // Log the underlying error so post-mortem in the executor pod logs
    // explains the timeout instead of just showing "worker hung".
    console.error(
      `[executor] worker[${idx}] crashed: ${err?.message ?? String(err)}`,
      err?.stack,
    );
    const slot = this.workers[idx];
    if (!slot) return;
    slot.worker.terminate().catch(() => {});
    const replacement = new Worker(new URL('./worker.js', import.meta.url).pathname);
    replacement.on('message', (msg) => this.onWorkerMessage(idx, msg));
    replacement.on('error', (e) => this.onWorkerError(idx, e));
    this.workers[idx] = { worker: replacement, busy: false };
  }

  metrics(): ExecutorPoolMetrics {
    const now = Date.now();
    while (this.invocationCounter.length && this.invocationCounter[0] < now - 1000) {
      this.invocationCounter.shift();
    }
    return {
      workersTotal: this.workers.length,
      workersBusy: this.workers.filter((s) => s.busy).length,
      // Self-usage vs allotment — the same signal the admission gate
      // uses (loadavg is node-scoped inside containers; see SelfCpuGate).
      cpuLoadPct: this.cpuGate.pct(),
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
