/**
 * Executor entrypoint — same plain-HTTP protocol used both for localhost
 * development and the cloud (NetworkPolicy gates ingress; ADR-0007).
 * Shipped as the `webagents-executor` npm package and as the
 * function-executor container.
 *
 *   - HOST: defaults to `0.0.0.0` so the kubelet readiness probe can reach
 *     the pod IP. Local CLI defaults to `0.0.0.0` too — overridable via
 *     `--bind 127.0.0.1` or `EXECUTOR_BIND` for a loopback-only setup.
 *   - `file://` codeRefs are gated by `WEBAGENTS_EXECUTOR_DEV=1` (or `--dev`).
 */

import { startExecutorServer } from './server';
import { WorkerPool } from './worker-pool';
import * as path from 'path';

export interface LocalExecutorOptions {
  port?: number;
  bind?: string;
  socketPath?: string;
  oversubscribe?: number;
  cpuPressureThresholdPct?: number;
  /** Allow `file://` codeRefs and local folder bindings. Default true for local. */
  dev?: boolean;
}

export async function runLocalExecutor(opts: LocalExecutorOptions = {}) {
  const port = opts.port ?? Number(process.env.PORT ?? 7070);
  const host = opts.bind ?? process.env.EXECUTOR_BIND ?? '0.0.0.0';
  const dev = opts.dev ?? (process.env.WEBAGENTS_EXECUTOR_DEV === '1');
  if (dev) {
    process.env.WEBAGENTS_EXECUTOR_DEV = '1';
  }
  const pool = new WorkerPool({
    oversubscribe: opts.oversubscribe ?? Number(process.env.EXECUTOR_OVERSUBSCRIBE ?? 8),
    cpuPressureThresholdPct: opts.cpuPressureThresholdPct ?? Number(process.env.EXECUTOR_CPU_PRESSURE_PCT ?? 85),
  });
  const { server, close } = await startExecutorServer({
    port,
    host,
    pool,
  });
  console.log(
    `[webagents-executor] listening on http://${host}:${port}${dev ? ' (dev mode)' : ''}`,
  );
  process.on('SIGINT', async () => {
    await close();
    process.exit(0);
  });
  process.on('SIGTERM', async () => {
    await close();
    process.exit(0);
  });
  return { server, pool, close };
}

export async function cli(argv: string[] = process.argv.slice(2)) {
  const opts: LocalExecutorOptions = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--port' || a === '-p') opts.port = Number(argv[++i]);
    else if (a === '--bind') opts.bind = argv[++i];
    else if (a === '--socket') opts.socketPath = argv[++i];
    else if (a === '--oversubscribe') opts.oversubscribe = Number(argv[++i]);
    else if (a === '--dev') opts.dev = true;
    else if (a === '--help' || a === '-h') {
      console.log(`webagents-executor — function-execution daemon

Usage: webagents-executor [options]

  --port, -p <n>          Listen port (default 7070; PORT env)
  --bind <host>           Bind interface (default 0.0.0.0; EXECUTOR_BIND env)
  --oversubscribe <n>     Worker pool oversubscription factor (default 8)
  --dev                   Enable file:// codeRefs and local folder bindings
  --help, -h              Show this help`);
      process.exit(0);
    }
  }
  await runLocalExecutor(opts);
}

/**
 * Default unix-socket path used by `webagents dev` and the SDK fallback.
 */
export const DEFAULT_LOCAL_SOCKET = process.platform === 'win32'
  ? '\\\\.\\pipe\\webagents-executor'
  : path.join(process.env.XDG_RUNTIME_DIR || '/tmp', 'webagents-executor.sock');

/** Best-effort check that a local executor is reachable on the given port. */
export async function probeLocalExecutor(port: number): Promise<boolean> {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/healthz`, { signal: AbortSignal.timeout(500) });
    return r.ok;
  } catch {
    return false;
  }
}
