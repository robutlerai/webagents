/**
 * WebAgentsServer — Multi-Agent Server
 *
 * Full-featured server for hosting multiple agents with:
 * - Dynamic routing between agents
 * - Prometheus metrics
 * - Rate limiting
 * - WebSocket support (UAMP + Realtime)
 * - Scope-based access control
 * - Extension loading
 * - Storage backend configuration
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import type { Context as HonoContext } from 'hono';
import type { IAgent, Context, ISkill } from '../core/types';
import { ContextImpl } from '../core/context';
import type { ClientEvent, ServerEvent } from '../uamp/events';
import { serializeEvent } from '../uamp/events';
import type { Capabilities } from '../uamp/types';
import { AgentIdentity, type AgentIdentityConfig } from '../crypto/identity';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface WebAgentsServerConfig {
  port?: number;
  hostname?: string;
  cors?: boolean;
  logging?: boolean;
  basePath?: string;
  /** Prometheus metrics path (default: /metrics) */
  metricsPath?: string;
  /** Enable rate limiting */
  rateLimit?: RateLimitConfig;
  /** Default scopes required for access */
  defaultScopes?: string[];
  /** Extensions to load on all agents */
  extensions?: ExtensionLoader[];
  /** AOAuth identity config — enables /.well-known/jwks.json per agent */
  identity?: Omit<AgentIdentityConfig, 'agentId' | 'issuer'> & {
    /** Base public URL of this server (used as issuer). Required to enable AOAuth. */
    publicUrl: string;
  };
}

export interface RateLimitConfig {
  /** Max requests per window */
  maxRequests: number;
  /** Window size in ms (default: 60000 = 1 minute) */
  windowMs?: number;
  /** Key extractor (default: IP-based) */
  keyExtractor?: (c: HonoContext) => string;
}

export interface ExtensionLoader {
  name: string;
  load: (agent: IAgent) => Promise<ISkill[]>;
}

interface AgentEntry {
  agent: IAgent;
  scopes?: string[];
  rateLimit?: RateLimitConfig;
  mountPath: string;
}

interface RateLimitBucket {
  count: number;
  resetAt: number;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

class PrometheusMetrics {
  private counters = new Map<string, Map<string, number>>();
  private histograms = new Map<string, number[]>();

  inc(name: string, labels: Record<string, string> = {}): void {
    const key = this.labelKey(labels);
    const counter = this.counters.get(name) ?? new Map<string, number>();
    counter.set(key, (counter.get(key) ?? 0) + 1);
    this.counters.set(name, counter);
  }

  observe(name: string, value: number): void {
    const h = this.histograms.get(name) ?? [];
    h.push(value);
    if (h.length > 10_000) h.splice(0, h.length - 10_000);
    this.histograms.set(name, h);
  }

  render(): string {
    const lines: string[] = [];

    for (const [name, labelMap] of this.counters) {
      lines.push(`# TYPE ${name} counter`);
      for (const [labels, value] of labelMap) {
        lines.push(`${name}${labels} ${value}`);
      }
    }

    for (const [name, values] of this.histograms) {
      if (values.length === 0) continue;
      lines.push(`# TYPE ${name} summary`);
      const sorted = [...values].sort((a, b) => a - b);
      const sum = sorted.reduce((a, b) => a + b, 0);
      lines.push(`${name}_count ${sorted.length}`);
      lines.push(`${name}_sum ${sum.toFixed(3)}`);
      const p50 = sorted[Math.floor(sorted.length * 0.5)];
      const p99 = sorted[Math.floor(sorted.length * 0.99)];
      lines.push(`${name}{quantile="0.5"} ${p50.toFixed(3)}`);
      lines.push(`${name}{quantile="0.99"} ${p99.toFixed(3)}`);
    }

    return lines.join('\n') + '\n';
  }

  private labelKey(labels: Record<string, string>): string {
    const entries = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
    if (entries.length === 0) return '';
    return `{${entries.map(([k, v]) => `${k}="${v}"`).join(',')}}`;
  }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

export class WebAgentsServer {
  private agents = new Map<string, AgentEntry>();
  private agentIdentities = new Map<string, AgentIdentity>();
  private app: Hono;
  private metrics = new PrometheusMetrics();
  private rateLimitBuckets = new Map<string, RateLimitBucket>();
  private config: WebAgentsServerConfig;

  constructor(config: WebAgentsServerConfig = {}) {
    this.config = {
      port: 3000,
      hostname: '0.0.0.0',
      cors: true,
      logging: true,
      basePath: '',
      metricsPath: '/metrics',
      ...config,
    };
    this.app = this.createApp();
  }

  // ============================================================================
  // Agent Registration
  // ============================================================================

  async addAgent(
    name: string,
    agent: IAgent,
    options?: { scopes?: string[]; rateLimit?: RateLimitConfig; mountPath?: string },
  ): Promise<void> {
    const mountPath = options?.mountPath ?? `/agents/${name}`;

    // Load extensions
    if (this.config.extensions) {
      for (const ext of this.config.extensions) {
        const skills = await ext.load(agent);
        for (const skill of skills) {
          if (typeof agent.addSkill === 'function') {
            agent.addSkill(skill);
          }
        }
      }
    }

    this.agents.set(name, {
      agent,
      scopes: options?.scopes,
      rateLimit: options?.rateLimit,
      mountPath,
    });

    if (this.config.identity?.publicUrl) {
      const identity = new AgentIdentity({
        agentId: name,
        issuer: `${this.config.identity.publicUrl}${mountPath}`,
        kid: name,
        agentPath: mountPath,
        privateKey: this.config.identity.privateKey,
        publicKey: this.config.identity.publicKey,
      });
      await identity.initialize();
      this.agentIdentities.set(name, identity);
    }

    this.mountAgent(name, this.agents.get(name)!);
  }

  removeAgent(name: string): boolean {
    this.agentIdentities.delete(name);
    return this.agents.delete(name);
  }

  getAgent(name: string): IAgent | undefined {
    return this.agents.get(name)?.agent;
  }

  getIdentity(name: string): AgentIdentity | undefined {
    return this.agentIdentities.get(name);
  }

  listAgents(): Array<{ name: string; mountPath: string; capabilities: Capabilities }> {
    return [...this.agents.entries()].map(([name, entry]) => ({
      name,
      mountPath: entry.mountPath,
      capabilities: entry.agent.getCapabilities(),
    }));
  }

  // ============================================================================
  // App Construction
  // ============================================================================

  private createApp(): Hono {
    const app = new Hono();
    const bp = this.config.basePath!;

    if (this.config.cors) app.use('*', cors());
    if (this.config.logging) app.use('*', logger());

    // Global health
    app.get(`${bp}/health`, (c) => {
      const agentList = [...this.agents.entries()].map(([name]) => ({ name, healthy: true }));
      return c.json({ status: 'ok', agents: agentList });
    });

    // List all agents
    app.get(`${bp}/agents`, (c) => {
      return c.json({ agents: this.listAgents() });
    });

    // Metrics
    if (this.config.metricsPath) {
      app.get(this.config.metricsPath, (c) => {
        return c.text(this.metrics.render(), 200, { 'Content-Type': 'text/plain; version=0.0.4' });
      });
    }

    // Dynamic routing: forward to agent by name
    app.all(`${bp}/agents/:name/*`, async (c) => {
      const name = c.req.param('name');
      const entry = this.agents.get(name);
      if (!entry) return c.json({ error: 'Agent not found' }, 404);

      // Rate limiting
      if (this.config.rateLimit || entry.rateLimit) {
        const rlConfig = entry.rateLimit ?? this.config.rateLimit!;
        const key = rlConfig.keyExtractor?.(c) ?? (c.req.header('x-forwarded-for') ?? 'unknown');
        if (!this.checkRateLimit(key, rlConfig)) {
          this.metrics.inc('webagents_rate_limited_total', { agent: name });
          return c.json({ error: 'Rate limit exceeded' }, 429);
        }
      }

      // Scope check
      const requiredScopes = entry.scopes ?? this.config.defaultScopes ?? [];
      if (requiredScopes.length > 0) {
        const ctx = createContextFromHono(c);
        const userScopes = (ctx.auth?.scopes ?? []) as string[];
        const missing = requiredScopes.filter((s) => !userScopes.includes(s));
        if (missing.length > 0) {
          return c.json({ error: `Missing scopes: ${missing.join(', ')}` }, 403);
        }
      }

      this.metrics.inc('webagents_requests_total', { agent: name, method: c.req.method });
      const start = performance.now();

      try {
        const subPath = c.req.path.replace(`${bp}/agents/${name}`, '') || '/';
        return await this.routeToAgent(entry, subPath, c);
      } finally {
        this.metrics.observe('webagents_request_duration_seconds', (performance.now() - start) / 1000);
      }
    });

    return app;
  }

  private mountAgent(_name: string, _entry: AgentEntry): void {
    // Agents are routed dynamically via the catch-all route above
  }

  private async routeToAgent(entry: AgentEntry, subPath: string, c: HonoContext): Promise<Response> {
    const agent = entry.agent;

    // Consult httpRegistry first — transport skills register their endpoints here
    const httpHandler = agent.getHttpHandler?.(subPath, c.req.method);
    if (httpHandler) {
      const context = createContextFromHono(c);
      return httpHandler.handler(c.req.raw, context);
    }

    // Built-in routes (not covered by transport skills)
    if (subPath === '/health' || subPath === '/') {
      return c.json({ status: 'ok', agent: agent.name });
    }

    if (subPath === '/info') {
      return c.json({
        name: agent.name,
        description: agent.description,
        capabilities: agent.getCapabilities(),
        tools: agent.getToolDefinitions?.() ?? [],
      });
    }

    // Fallback UAMP HTTP POST (in case no UAMPTransportSkill is loaded)
    if (subPath === '/uamp' && c.req.method === 'POST') {
      const body = await c.req.json() as ClientEvent[];
      const events: ServerEvent[] = [];
      for await (const event of agent.processUAMP(body)) {
        events.push(event);
      }
      return c.json(events);
    }

    if (subPath === '/uamp/stream' && c.req.method === 'POST') {
      const body = await c.req.json() as ClientEvent[];
      return streamSSE(agent.processUAMP(body));
    }

    // Fallback chat/completions (in case no CompletionsTransportSkill is loaded)
    if ((subPath === '/chat/completions' || subPath === '/v1/chat/completions') && c.req.method === 'POST') {
      const body = await c.req.json() as {
        messages: Array<{ role: string; content: string }>;
        stream?: boolean;
      };
      const msgs = body.messages.map((m) => ({ role: m.role as 'user' | 'system' | 'assistant', content: m.content }));

      if (body.stream) {
        const response = agent.runStreaming(msgs);
        return streamCompletions(response);
      }

      const result = await agent.run(msgs);
      return c.json({
        choices: [{ message: { role: 'assistant', content: result.content }, finish_reason: 'stop' }],
        usage: result.usage,
      });
    }

    // .well-known/jwks.json — AOAuth public keys
    if (subPath === '/.well-known/jwks.json') {
      const agentName = [...this.agents.entries()]
        .find(([, e]) => e === entry)?.[0];
      const identity = agentName ? this.agentIdentities.get(agentName) : undefined;
      if (!identity) {
        return c.json({ error: 'AOAuth not configured for this agent' }, 404);
      }
      return c.json(identity.getJwks(), 200, {
        'Cache-Control': 'public, max-age=3600',
      });
    }

    // .well-known/openid-configuration — AOAuth discovery
    if (subPath === '/.well-known/openid-configuration') {
      const agentName = [...this.agents.entries()]
        .find(([, e]) => e === entry)?.[0];
      const identity = agentName ? this.agentIdentities.get(agentName) : undefined;
      if (!identity) {
        return c.json({ error: 'AOAuth not configured for this agent' }, 404);
      }
      return c.json(identity.getOpenIdConfiguration(), 200, {
        'Cache-Control': 'public, max-age=3600',
      });
    }

    return c.json({ error: 'Not found' }, 404);
  }

  // ============================================================================
  // Rate Limiting
  // ============================================================================

  private checkRateLimit(key: string, config: RateLimitConfig): boolean {
    const now = Date.now();
    const windowMs = config.windowMs ?? 60_000;
    const bucket = this.rateLimitBuckets.get(key);

    if (!bucket || now > bucket.resetAt) {
      this.rateLimitBuckets.set(key, { count: 1, resetAt: now + windowMs });
      return true;
    }

    bucket.count++;
    return bucket.count <= config.maxRequests;
  }

  // ============================================================================
  // Start / Stop
  // ============================================================================

  async start(): Promise<void> {
    const port = this.config.port!;
    const hostname = this.config.hostname!;

    console.log(`WebAgentsServer starting on http://${hostname}:${port}`);
    console.log(`Serving ${this.agents.size} agents`);

    if (typeof Bun !== 'undefined') {
      Bun.serve({ port, hostname, fetch: this.app.fetch });
    } else {
      try {
        const { serve } = await import(/* @vite-ignore */ '@hono/node-server');
        const server = serve({ fetch: this.app.fetch, port, hostname });

        // Wire WebSocket upgrades to agent wsRegistry handlers
        if (server && typeof server.on === 'function') {
          server.on('upgrade', (req: import('http').IncomingMessage, socket: import('stream').Duplex, head: Buffer) => {
            this.handleWebSocketUpgrade(req, socket, head);
          });
        }
      } catch {
        throw new Error('Install @hono/node-server for Node.js runtime');
      }
    }

    console.log('WebAgentsServer started');
  }

  /**
   * Handle a WebSocket upgrade by resolving the agent from the URL and
   * dispatching to its wsRegistry handler.
   */
  private handleWebSocketUpgrade(
    req: import('http').IncomingMessage,
    socket: import('stream').Duplex,
    head: Buffer,
  ): void {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
    const bp = this.config.basePath ?? '';
    const prefix = `${bp}/agents/`;
    if (!url.pathname.startsWith(prefix)) {
      socket.write('HTTP/1.1 404 Not Found\r\n\r\n');
      socket.destroy();
      return;
    }

    const rest = url.pathname.slice(prefix.length);
    const slashIdx = rest.indexOf('/');
    const agentName = slashIdx >= 0 ? rest.slice(0, slashIdx) : rest;
    const subPath = slashIdx >= 0 ? rest.slice(slashIdx) : '/';

    const entry = this.agents.get(agentName);
    if (!entry) {
      socket.write('HTTP/1.1 404 Agent Not Found\r\n\r\n');
      socket.destroy();
      return;
    }

    const wsEndpoint = entry.agent.getWebSocketHandler?.(subPath);
    if (!wsEndpoint) {
      socket.write('HTTP/1.1 404 No WebSocket Handler\r\n\r\n');
      socket.destroy();
      return;
    }

    // Lazy-init WebSocketServer
    if (!this._wss) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { WebSocketServer } = require('ws');
        this._wss = new WebSocketServer({ noServer: true });
      } catch {
        socket.write('HTTP/1.1 500 ws package not available\r\n\r\n');
        socket.destroy();
        return;
      }
    }

    const context = createContextFromIncomingMessage(req);
    this._wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
      wsEndpoint.handler(ws, context);
    });
  }

  private _wss: any = null;

  getApp(): Hono {
    return this.app;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createContextFromHono(c: HonoContext): Context {
  const context = new ContextImpl();
  const authHeader = c.req.header('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    context.setAuth({ authenticated: true });
  }
  const paymentToken = c.req.header('x-payment-token') ?? c.req.header('x-payment');
  if (paymentToken) {
    context.set('payment_token', paymentToken);
  }
  context.metadata = {
    userAgent: c.req.header('user-agent'),
    ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
    path: c.req.path,
    method: c.req.method,
  };
  return context;
}

function createContextFromIncomingMessage(req: import('http').IncomingMessage): Context {
  const context = new ContextImpl();
  const authHeader = req.headers['authorization'];
  if (typeof authHeader === 'string' && authHeader.startsWith('Bearer ')) {
    context.setAuth({ authenticated: true });
  }
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const paymentToken =
    (req.headers['x-payment-token'] as string) ??
    url.searchParams.get('payment_token') ??
    undefined;
  if (paymentToken) {
    context.set('payment_token', paymentToken);
  }
  context.metadata = {
    userAgent: req.headers['user-agent'],
    ip: (req.headers['x-forwarded-for'] as string) || (req.headers['x-real-ip'] as string),
    path: url.pathname,
    method: req.method,
  };
  return context;
}

function streamSSE(events: AsyncGenerator<ServerEvent, void, unknown>): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const event of events) {
          controller.enqueue(encoder.encode(`data: ${serializeEvent(event)}\n\n`));
        }
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });
  return new Response(stream, {
    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' },
  });
}

function streamCompletions(
  gen: AsyncGenerator<{ type: string; delta?: string; response?: unknown }, void, unknown>,
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of gen) {
          if (chunk.type === 'delta' && chunk.delta) {
            const data = { choices: [{ delta: { content: chunk.delta }, finish_reason: null }] };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
          }
        }
        const done = { choices: [{ delta: {}, finish_reason: 'stop' }] };
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(done)}\n\n`));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });
  return new Response(stream, {
    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' },
  });
}

declare const Bun: {
  serve(options: { port: number; hostname: string; fetch: (request: Request) => Response | Promise<Response> }): void;
} | undefined;
