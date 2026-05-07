/**
 * HTTPS executor server (mTLS in production; plain HTTP for localhost).
 *
 * Endpoints:
 *   - POST /invoke    body: InvocationEnvelope    → ExecutorResponse
 *   - POST /validate  body: { runtime, source, manifest } → ExecutorValidationResult
 *   - GET  /healthz   liveness; pool metrics in body
 *   - GET  /metrics   Prometheus exposition format
 *
 * The server is intentionally minimal — it just unwraps requests and
 * delegates to the worker pool. mTLS verification, NetworkPolicy, and
 * pod security live at the infra layer (kustomize overlays).
 */

import * as http from 'http';
import * as https from 'https';
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
  mtls?: {
    key: Buffer | string;
    cert: Buffer | string;
    /** PEM-encoded CA bundle. */
    ca: Buffer | string;
    requestCert?: boolean;
    rejectUnauthorized?: boolean;
  };
  pool?: WorkerPool;
  /** Custom worker.js path (override for testing or alt runtime sets). */
  workerScript?: string;
}

export async function startExecutorServer(opts: StartExecutorServerOptions): Promise<{
  server: http.Server | https.Server;
  pool: WorkerPool;
  close: () => Promise<void>;
}> {
  const pool = opts.pool ?? new WorkerPool({ workerScript: opts.workerScript });

  const handler = async (req: http.IncomingMessage, res: http.ServerResponse) => {
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
      res.writeHead(r.ok ? 200 : 500, { 'content-type': 'application/json' });
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

  const server = opts.mtls
    ? https.createServer({
        key: opts.mtls.key,
        cert: opts.mtls.cert,
        ca: opts.mtls.ca,
        requestCert: opts.mtls.requestCert ?? true,
        rejectUnauthorized: opts.mtls.rejectUnauthorized ?? true,
      }, handler)
    : http.createServer(handler);

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
