/**
 * A client for the local agent daemon.
 *
 * WHY THIS EXISTS RATHER THAN A SECOND IMPLEMENTATION (2026-09-23). The
 * TypeScript CLI was on course to reimplement agent lifecycle, sessions and
 * cron alongside the Python CLI's 13,840 lines, and then keep the two honest
 * forever. Every comparable framework runs ONE core with thin front-ends
 * instead, and webagents already has the core: `webagentsd` serves an HTTP API
 * that answers all three. So this file talks to it.
 *
 * THE CANONICAL SHAPE IS THE PYTHON DAEMON'S, because that is what
 * `webagentsd` runs (`python/webagents/cli/commands/daemon.py` builds the
 * server with `url_prefix="/agents"`). Routes:
 *
 *     GET    /health
 *     GET    /agents/                 list
 *     GET    /agents/{name}           one agent
 *     DELETE /agents/{name}           unregister
 *     POST   /agents/{name}/chat/completions
 *     GET    /agents/cron             list jobs
 *     POST   /agents/cron             add a job
 *
 * The TypeScript daemon in `src/daemon/` serves the same shape, so one client
 * reaches either. It had cron at `/cron` until this landed, which meant a
 * client could not be written against both without branching on which daemon
 * answered.
 *
 * Every method distinguishes "the daemon is not running" from "the daemon said
 * no", because the two have completely different next steps and a CLI that
 * conflates them sends people to the wrong place.
 */

import { cliCommand } from './config-store';

export class DaemonUnreachableError extends Error {
  constructor(public readonly baseUrl: string, cause?: unknown) {
    super(
      `No daemon answering at ${baseUrl}. Start one with \`${cliCommand('daemon')}\`, ` +
        `or point at another with \`${cliCommand('config set daemon.port <port>')}\`.`,
    );
    this.name = 'DaemonUnreachableError';
    this.cause = cause;
  }
}

export class DaemonRequestError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
    method: string,
    path: string,
  ) {
    super(`${method} ${path} failed: ${status} ${body.slice(0, 200)}`);
    this.name = 'DaemonRequestError';
  }
}

/** One agent as the daemon reports it. Fields beyond `name` are best-effort. */
export interface DaemonAgent {
  name: string;
  description?: string;
  namespace?: string;
  model?: string;
  skills?: string[];
  source?: string;
  [key: string]: unknown;
}

export interface DaemonCronJob {
  id?: string;
  agent?: string;
  schedule?: string;
  enabled?: boolean;
  next_run?: string;
  [key: string]: unknown;
}

export interface DaemonHealth {
  status?: string;
  version?: string;
  [key: string]: unknown;
}

export interface DaemonClientOptions {
  host?: string;
  port?: number;
  /** Milliseconds before an unanswered request is treated as unreachable. */
  timeoutMs?: number;
  token?: string;
}

export class DaemonClient {
  readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly token?: string;

  constructor(options: DaemonClientOptions = {}) {
    const host = options.host ?? '127.0.0.1';
    const port = options.port ?? 8765;
    this.baseUrl = `http://${host}:${port}`;
    this.timeoutMs = options.timeoutMs ?? 5000;
    this.token = options.token;
  }

  /**
   * Build a client from the CLI's own config, so `daemon.host` / `daemon.port`
   * mean the same thing here as everywhere else.
   */
  static async fromConfig(overrides: DaemonClientOptions = {}): Promise<DaemonClient> {
    const { ConfigStore } = await import('./config-store.js');
    const store = new ConfigStore();
    const port = Number(store.get('daemon.port') ?? 8765);
    return new DaemonClient({
      host: String(store.get('daemon.host') ?? '127.0.0.1'),
      port: Number.isFinite(port) ? port : 8765,
      ...overrides,
    });
  }

  /** Whether a daemon is answering. Never throws: this is the question `status` asks. */
  async isRunning(): Promise<boolean> {
    try {
      await this.health();
      return true;
    } catch {
      return false;
    }
  }

  async health(): Promise<DaemonHealth> {
    return this.request<DaemonHealth>('GET', '/health');
  }

  async listAgents(): Promise<DaemonAgent[]> {
    // The trailing slash is the registered path on the Python daemon; without
    // it FastAPI answers 307 and some clients drop the method on the redirect.
    const body = await this.request<DaemonAgent[] | { agents?: DaemonAgent[] }>('GET', '/agents/');
    if (Array.isArray(body)) return body;
    return Array.isArray(body?.agents) ? body.agents : [];
  }

  async getAgent(name: string): Promise<DaemonAgent> {
    return this.request<DaemonAgent>('GET', `/agents/${encodeURIComponent(name)}`);
  }

  async unregisterAgent(name: string): Promise<void> {
    await this.request('DELETE', `/agents/${encodeURIComponent(name)}`);
  }

  async listCronJobs(): Promise<DaemonCronJob[]> {
    const body = await this.request<{ jobs?: DaemonCronJob[] } | DaemonCronJob[]>(
      'GET',
      '/agents/cron',
    );
    if (Array.isArray(body)) return body;
    return Array.isArray(body?.jobs) ? body.jobs : [];
  }

  async addCronJob(agent: string, schedule: string): Promise<DaemonCronJob> {
    const query = `?agent=${encodeURIComponent(agent)}&schedule=${encodeURIComponent(schedule)}`;
    return this.request<DaemonCronJob>('POST', `/agents/cron${query}`);
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (this.token) headers.Authorization = `Bearer ${this.token}`;

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (error) {
      // A connection refused, a DNS failure and a timeout all mean the same
      // thing to the user: nothing is listening.
      throw new DaemonUnreachableError(this.baseUrl, error);
    }

    if (!response.ok) {
      throw new DaemonRequestError(response.status, await response.text(), method, path);
    }

    const text = await response.text();
    if (!text) return undefined as T;
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new DaemonRequestError(response.status, text, method, path);
    }
  }
}
