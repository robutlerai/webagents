/**
 * Plain HTTP executor server.
 *
 * Endpoints:
 *   - POST /invoke    body: InvocationEnvelope    → ExecutorResponse
 *   - POST /validate  body: { runtime, source, manifest } → ExecutorValidationResult
 *   - GET  /healthz   liveness; pool metrics in body
 *   - GET  /metrics   Prometheus exposition format
 *
 * The server is intentionally minimal — it just unwraps requests and
 * delegates to the worker pool. Authentication on the portal→executor
 * leg is NetworkPolicy ingress (only portal + webagentsd may reach
 * :7070) — see ADR-0007 for the mTLS retirement rationale.
 */

import * as http from 'http';
import { WorkerPool } from './worker-pool';
import { RuntimeRegistry } from './runtime-registry';
import type {
  InvocationEnvelope,
  ExecutorResponse,
  ExecutorValidationResult,
} from '../skills/functions/executor-client';

export interface StartExecutorServerOptions {
  port: number;
  host?: string;
  pool?: WorkerPool;
  /** Custom worker.js path (override for testing or alt runtime sets). */
  workerScript?: string;
}

export async function startExecutorServer(opts: StartExecutorServerOptions): Promise<{
  server: http.Server;
  pool: WorkerPool;
  close: () => Promise<void>;
}> {
  const pool = opts.pool ?? new WorkerPool({ workerScript: opts.workerScript });

  const handler = async (req: http.IncomingMessage, res: http.ServerResponse) => {
    try {
      return await innerHandler(req, res);
    } catch (err) {
      // Last-resort guard so an uncaught exception inside the request path
      // (worker pool throws, JSON.stringify of a circular result, etc.)
      // becomes a structured 500 with the same {ok:false, errorCode,
      // errorMessage} shape user-function failures use. Without this the
      // socket would close with no body and the portal client would have
      // nothing useful to surface.
      const message = err instanceof Error ? err.message : String(err);
      console.error('[executor] handler exception:', message, err);
      const body: ExecutorResponse = {
        ok: false,
        errorCode: 'EXECUTOR_INTERNAL',
        errorMessage: message,
        durationMs: 0,
        cpuMs: 0,
        ingressBytes: 0,
        egressBytes: 0,
      };
      try {
        res.writeHead(500, { 'content-type': 'application/json' });
        res.end(JSON.stringify(body));
      } catch {
        // Headers already sent — nothing more we can do; the socket close
        // is the signal. The console.error above is the audit trail.
      }
    }
  };

  const innerHandler = async (req: http.IncomingMessage, res: http.ServerResponse) => {
    if (req.method === 'GET' && req.url === '/healthz') {
      const m = pool.metrics();
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', ...m }));
      return;
    }
    if (req.method === 'GET' && req.url === '/metrics') {
      const m = pool.metrics();
      const lines = [
        `# HELP executor_workers_total Total worker threads`,
        `# TYPE executor_workers_total gauge`,
        `executor_workers_total ${m.workersTotal}`,
        `# HELP executor_workers_busy Workers currently servicing an invocation`,
        `# TYPE executor_workers_busy gauge`,
        `executor_workers_busy ${m.workersBusy}`,
        `# HELP executor_cpu_load_pct CPU load percent (1-min loadavg / cpus)`,
        `# TYPE executor_cpu_load_pct gauge`,
        `executor_cpu_load_pct ${m.cpuLoadPct.toFixed(2)}`,
        `# HELP executor_invocations_in_flight Active invocations`,
        `# TYPE executor_invocations_in_flight gauge`,
        `executor_invocations_in_flight ${m.invocationsInFlight}`,
        `# HELP executor_invocations_per_second Rolling 1s rate`,
        `# TYPE executor_invocations_per_second gauge`,
        `executor_invocations_per_second ${m.invocationsPerSecond}`,
        `# HELP executor_admission_rejections_total Cumulative admission gate rejections`,
        `# TYPE executor_admission_rejections_total counter`,
        `executor_admission_rejections_total ${m.admissionRejections}`,
      ];
      for (const r of Object.keys(m.cacheHits) as Array<keyof typeof m.cacheHits>) {
        lines.push(`executor_sandbox_cache_hits_total{runtime="${r}"} ${m.cacheHits[r]}`);
        lines.push(`executor_sandbox_cache_misses_total{runtime="${r}"} ${m.cacheMisses[r]}`);
      }
      res.writeHead(200, { 'content-type': 'text/plain' });
      res.end(lines.join('\n') + '\n');
      return;
    }
    if (req.method !== 'POST') {
      res.writeHead(405);
      res.end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = Buffer.concat(chunks).toString('utf-8');
    let parsed: unknown;
    try {
      parsed = body.length > 0 ? JSON.parse(body) : {};
    } catch {
      res.writeHead(400, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'invalid JSON' }));
      return;
    }

    if (req.url === '/invoke') {
      const env = parsed as InvocationEnvelope;
      const r: ExecutorResponse = await pool.run(env);
      // Always 200 — the body's `ok` flag carries success/failure for
      // *user-function* outcomes. HTTP 500 is reserved for executor-server
      // internal panics (caught by the outer `try` in `handler`) so the
      // portal client can disambiguate "your function threw" from "the
      // executor itself is broken". This is the same protocol shape gRPC
      // uses (transport status vs. application status).
      const fnName = env.functionName ?? '<unknown>';
      const agentId = env.agentId ?? '<unknown>';
      const consumerId = env.context?.source?.consumerId ?? '<unknown>';
      if (r.ok) {
        console.log(
          `[executor] /invoke ok agent=${agentId} fn=${fnName} consumer=${consumerId} dur=${r.durationMs}ms cpu=${r.cpuMs}ms`,
        );
      } else {
        console.warn(
          `[executor] /invoke fail agent=${agentId} fn=${fnName} consumer=${consumerId} code=${r.errorCode} msg=${r.errorMessage}`,
        );
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(r));
      return;
    }
    if (req.url === '/validate') {
      const { runtime, source, manifest } = parsed as { runtime: string; source: string; manifest: import('../skills/functions/manifest').FunctionManifest };
      const rt = RuntimeRegistry.get(runtime as import('./types').ExecutorRuntimeId);
      if (!rt || !rt.enabled) {
        const result: ExecutorValidationResult = {
          ok: false,
          warnings: [],
          errors: [{ code: rt ? 'RUNTIME_DISABLED' : 'RUNTIME_UNKNOWN', message: `${runtime} not available` }],
        };
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(result));
        return;
      }
      const result = await rt.validate(source, manifest);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(result));
      return;
    }
    res.writeHead(404);
    res.end();
  };

  const server = http.createServer(handler);

  await new Promise<void>((resolve) => server.listen(opts.port, opts.host ?? '0.0.0.0', resolve));
  return {
    server,
    pool,
    close: async () => {
      await pool.shutdown();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}
