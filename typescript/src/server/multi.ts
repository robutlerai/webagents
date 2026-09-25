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
import {
  agentVerifiesCredentials,
  defaultHostname,
  originPolicy,
  upgradeOriginAllowed,
  type CorsSetting,
  type OriginPolicy,
} from './origin-policy';
import { requestLog } from './request-log';
import type { Context as HonoContext } from 'hono';
import type { IAgent, Context, ISkill } from '../core/types';
import { ContextImpl } from '../core/context';
import type { ClientEvent, ServerEvent } from '../uamp/events';
import { serializeEvent } from '../uamp/events';
import type { Capabilities } from '../uamp/types';
import { AgentIdentity, type AgentIdentityConfig } from '../crypto/identity';
import { loadOrCreateAgentIdentity } from '../crypto/identity-store';
import { credentialFloor, webSocketUpgradeIsRefused } from './credential-floor';
import { DIRECTORY_WELL_KNOWN_PATH, keyDirectoryResponse } from './key-directory';
import { buildAgentCard } from './card';
import {
  admit,
  admitEndpoint,
  identificationContext,
  inboundRequest,
  inboundUpgrade,
  isOpen,
  needsCaller,
  refuseUpgrade,
} from './endpoint-gate';
import { replyText } from './error-reply';
import { createRequire } from 'node:module';

// A `require` that works in this ES module (2026-09-24): the bare `require('ws')`
// below threw in ESM and every WebSocket upgrade answered "500 ws package not
// available". Same defect and fix as `node.ts`. Not in the portal's bundle.
const nodeRequire = createRequire(import.meta.url);

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface WebAgentsServerConfig {
  port?: number;
  /** Unset: every interface when a public URL is configured or every agent has an AuthSkill, loopback otherwise (S-226). */
  hostname?: string;
  /** `true` any origin, a list those, `false` none. Unset: any origin when every agent has an AuthSkill, loopback origins otherwise. */
  cors?: CorsSetting;
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
  /**
   * Signing identity config. Enables, PER AGENT, the key set at
   * `{basePath}/agents/{name}/.well-known/jwks.json`, the self-naming card
   * beside it, and `getIdentity(name)` for `registerWithPlatform`.
   *
   * There is deliberately no key material here. Until 2026-09-18 this took
   * one server-wide `privateKey`/`publicKey` that every agent's identity was
   * built from (S-142): `kid` is the key's thumbprint, so agents `a` and `b`
   * published and signed with one key, and the platform's key-anchored
   * continuity, which moves the registration holding a presented thumbprint
   * to whatever URL now presents it, merged them into one registration that
   * flipped between the two on every request: `b` authenticated and was
   * billed as `a`, and `a`'s inbound routing pointed at `b`. Each agent now
   * gets its own persisted key from `keysDir` (or its own key material via
   * `addAgent`'s `identity` option), and `addAgent` refuses a key another
   * agent on this server already holds.
   */
  identity?: {
    /**
     * The base public URL of this server. Each agent's URL, the principal
     * the platform registers, is `publicUrl + basePath + /agents/ + name`:
     * exactly the path the router serves it at. Required to sign.
     */
    publicUrl: string;
    /**
     * Where each agent's Ed25519 key is persisted, one file per agent name,
     * as `serve()` does: `WEBAGENTS_KEYS_DIR`, then `~/.webagents/keys`,
     * when omitted. `null` is an explicitly ephemeral key per boot (tests):
     * registration pins the key set's thumbprints, so an ephemeral key
     * breaks on the first restart.
     */
    keysDir?: string | null;
  };
}

/** Per-agent options for `addAgent`. */
export interface AddAgentOptions {
  scopes?: string[];
  rateLimit?: RateLimitConfig;
  mountPath?: string;
  /**
   * This agent's OWN key material: the current pair and, while rotating,
   * the one previous pair (`AgentIdentityConfig.previousKeys`). Omitted, the
   * key is loaded or created under the server's `identity.keysDir`. Never
   * shared between agents: a second agent presenting a key this server
   * already holds is refused at `addAgent` (S-142).
   */
  identity?: Pick<AgentIdentityConfig, 'privateKey' | 'publicKey' | 'previousKeys'>;
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
    // The pre-2026-09-18 shape, refused loudly for a JavaScript caller the
    // type no longer stops: one key pair on the server IS the S-142 defect.
    const identity = config.identity as (Record<string, unknown> & { publicUrl?: string }) | undefined;
    if (identity && ('privateKey' in identity || 'publicKey' in identity || 'previousKeys' in identity)) {
      throw new Error(
        'WebAgentsServer: identity.privateKey / publicKey / previousKeys are no longer accepted: a key pair ' +
          'shared by every agent merges their platform registrations into one (S-142). Give each agent its ' +
          'own key material in addAgent(name, agent, { identity }) or let identity.keysDir persist one per agent',
      );
    }
    this.config = {
      port: 3000,
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

  async addAgent(name: string, agent: IAgent, options?: AddAgentOptions): Promise<void> {
    const mountPath = options?.mountPath ?? `/agents/${name}`;

    // The identity first, so a refused key leaves no half-added agent behind.
    let identity: AgentIdentity | undefined;
    if (this.config.identity?.publicUrl) {
      // The issuer is the URL the ROUTER serves the agent at (`createApp`
      // mounts every agent at `${basePath}/agents/:name`, whatever
      // `mountPath` says), because the card served there must name itself
      // and the key set must live under the URL every signature names
      // (W2 design sections 3.1 and 3.3, 2026-09-17). `kid` is the key's
      // thumbprint, no longer the agent name.
      const issuer = `${this.config.identity.publicUrl.replace(/\/+$/, '')}${this.routePrefix(name)}`;
      if (options?.identity) {
        identity = new AgentIdentity({ agentId: name, issuer, ...options.identity });
        await identity.initialize();
      } else {
        // One PERSISTED key per agent name, the same store `serve()` uses
        // (S-142, 2026-09-18): a fresh key per boot works exactly until the
        // first restart, and one key for every agent merges them all.
        identity = await loadOrCreateAgentIdentity(name, { issuer, keysDir: this.config.identity.keysDir });
      }
      // A key another agent on this server already holds is the S-142 shape
      // whatever route it arrived by (a copied key file, the same material
      // passed twice): refuse it here, at boot, with the two names, instead
      // of letting the platform merge the two registrations.
      const mine = new Set(identity.getJwks().keys.map((k) => k.kid));
      for (const [other, theirs] of this.agentIdentities) {
        if (other === name) continue;
        const shared = theirs.getJwks().keys.find((k) => mine.has(k.kid));
        if (shared) {
          throw new Error(
            `WebAgentsServer: agent "${name}" would hold key ${shared.kid}, which agent "${other}" already ` +
              'holds. Every agent needs its own key: the platform keys registrations by thumbprint and would ' +
              'merge these two into one identity (S-142)',
          );
        }
      }
    }

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
    if (identity) this.agentIdentities.set(name, identity);
    else this.agentIdentities.delete(name);
    // The same hand-over `serve()` makes (2026-09-23): the agent's skills sign
    // platform calls with the identity this server publishes for it. Left
    // alone when this server holds no identity for the agent, so an identity
    // the caller attached itself is not erased.
    if (identity) agent.identity = identity;

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

    // Decided per request, because agents are added after the app exists
    // (`addAgent`): the policy follows whoever is registered right now.
    if (this.config.cors !== false) {
      app.use('*', cors({ origin: (origin) => this.originPolicyNow()(origin) ?? undefined }));
    }
    if (this.config.logging) app.use('*', requestLog());

    // ========================================================================
    // THE CREDENTIAL FLOOR for this server class — registered before any route.
    //
    // `WebAgentsServer` is the DOCUMENTED multi-agent entry point
    // (docs/agent/overview.md, docs/api/typescript.md) and it had no floor on
    // ANY billable route: anonymous `POST /agents/:name/chat/completions`,
    // `/v1/chat/completions` and `/uamp` all answered 200 and really invoked
    // `agent.run` / `agent.processUAMP`. Adding four more per-route `if`s here
    // is what produced this class of bug three rounds running; this is one
    // check, upstream of `routeToAgent`, so it also stands in front of the
    // `getHttpHandler` dispatch that runs BEFORE any built-in branch — the door
    // a transport skill's `@http` handler comes through.
    //
    // Same predicate and same path set as every other server class, from
    // `credential-floor.ts`.
    // ========================================================================
    app.use('*', async (c, next) => {
      const refusal = credentialFloor(c.req.raw);
      if (refusal) return refusal;
      await next();
      return undefined;
    });

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

    // The signatures directory a `legacy-string` signer's bare origin
    // resolves to (key-directory.ts, 2026-09-19): at the ORIGIN, not under
    // `basePath`, and listing every hosted agent's keys, read per request so
    // an agent added or removed after boot is in or out of it at once.
    app.get(DIRECTORY_WELL_KNOWN_PATH, () => keyDirectoryResponse(this.agentIdentities.values()));

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

  /** The path `createApp` routes an agent's requests under: `${basePath}/agents/${name}`. */
  private routePrefix(name: string): string {
    return `${(this.config.basePath ?? '').replace(/\/+$/, '')}/agents/${name}`;
  }

  private async routeToAgent(entry: AgentEntry, subPath: string, c: HonoContext): Promise<Response> {
    const agent = entry.agent;
    const agentName = [...this.agents.entries()].find(([, e]) => e === entry)?.[0];
    const identity = agentName ? this.agentIdentities.get(agentName) : undefined;

    // .well-known/agent.json: the self-naming card (card.ts), the same one
    // `createFetchHandler` serves and with the same precedence over a
    // transport skill's `@http` handler, because it is what the platform
    // reads at registration and until 2026-09-17 this server served no card
    // at all (W2 design section 9.1). With an identity its `url` is the
    // identity's issuer; without one it is the route, a relative reference.
    if (subPath === '/.well-known/agent.json' && c.req.method === 'GET') {
      return c.json(
        buildAgentCard(agent, {
          principal: identity?.issuer ?? this.routePrefix(agentName ?? agent.name),
          signs: identity !== undefined,
        }),
      );
    }

    // Consult httpRegistry first — transport skills register their endpoints here
    const httpHandler = agent.getHttpHandler?.(subPath, c.req.method);
    if (httpHandler) {
      let context = createContextFromHono(c);
      // WHO MAY CALL IT (S-242, 2026-09-25): the one gate
      // (`endpoint-gate.ts`); an endpoint with no scopes is open. The agent
      // entry's own `scopes` above are a separate, server-wide check.
      if (needsCaller(httpHandler)) {
        const raw = new Uint8Array(await c.req.raw.clone().arrayBuffer());
        context = identificationContext(context, inboundRequest(c.req.raw, raw));
        const gate = await admitEndpoint(agent, httpHandler, context);
        if (gate.refusal) return c.json(gate.refusal.body, gate.refusal.status);
      }
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

    // .well-known/jwks.json: the key set every signature names, entries
    // `{ kty, crv, x, kid: thumbprint, use }` with no `alg`.
    if (subPath === '/.well-known/jwks.json') {
      if (!identity) {
        return c.json({ error: 'AOAuth not configured for this agent' }, 404);
      }
      return c.json(identity.getJwks(), 200, {
        'Cache-Control': 'public, max-age=3600',
      });
    }

    // .well-known/openid-configuration — AOAuth discovery
    if (subPath === '/.well-known/openid-configuration') {
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

  /** The origin policy for the agents registered right now (`origin-policy.ts`). */
  private originPolicyNow(): OriginPolicy {
    return originPolicy(this.config.cors, this.everyAgentVerifies());
  }

  /** True only when there are agents and each has an AuthSkill. */
  private everyAgentVerifies(): boolean {
    const entries = [...this.agents.values()];
    return entries.length > 0 && entries.every((entry) => agentVerifiesCredentials(entry.agent));
  }

  async start(): Promise<void> {
    const port = this.config.port!;
    const hostname =
      this.config.hostname ??
      defaultHostname({
        publicUrl: this.config.identity?.publicUrl,
        verifiesCredentials: this.everyAgentVerifies(),
      });

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

    // CORS never covers a WebSocket handshake; the origin rule is enforced here (S-226).
    if (!upgradeOriginAllowed(this.originPolicyNow(), req.headers.origin as string | undefined)) {
      socket.write('HTTP/1.1 403 Forbidden\r\n\r\n');
      socket.destroy();
      return;
    }

    // The floor, on the WebSocket door — see the note in node.ts's
    // `handleUpgrade`. `@websocket({ path: '/uamp' })` reaches the model.
    if (
      webSocketUpgradeIsRefused(
        subPath,
        { get: (name: string) => (req.headers[name.toLowerCase()] as string) ?? null },
        url.searchParams,
      )
    ) {
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
      socket.destroy();
      return;
    }

    // Lazy-init WebSocketServer
    if (!this._wss) {
      try {
        const { WebSocketServer } = nodeRequire('ws');
        this._wss = new WebSocketServer({ noServer: true });
      } catch {
        socket.write('HTTP/1.1 500 ws package not available\r\n\r\n');
        socket.destroy();
        return;
      }
    }

    const context = createContextFromIncomingMessage(req);
    const upgrade = (ctx: Context) => {
      this._wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
        wsEndpoint.handler(ws, ctx);
      });
    };
    if (isOpen(wsEndpoint.scopes)) {
      upgrade(context);
      return;
    }
    // WHO MAY OPEN IT (S-242, 2026-09-25): the one gate, before the handshake
    // completes, so a refusal is an HTTP status with the gate's body.
    admit(entry.agent, wsEndpoint.scopes, identificationContext(context, inboundUpgrade(req))).then(
      (gate) => (gate.refusal ? refuseUpgrade(socket, gate.refusal) : upgrade(gate.context)),
      (error) => {
        replyText(error, `${agentName} websocket ${subPath}`);
        socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\n');
        socket.destroy();
      },
    );
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
