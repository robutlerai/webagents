/**
 * Node.js Server
 * 
 * Hono-based HTTP/WebSocket server for running agents.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import type { Context as HonoContext } from 'hono';
import type { IAgent, Context } from '../core/types';
import { ContextImpl } from '../core/context';
import type { ClientEvent, ServerEvent } from '../uamp/events';
import { serializeEvent } from '../uamp/events';
import { createFetchHandler } from './handler';
import type { AgentIdentity } from '../crypto/identity';
import { loadOrCreateAgentIdentity } from '../crypto/identity-store';
import { startHeartbeat, type HeartbeatHandle } from './registration';
import { credentialFloor, webSocketUpgradeIsRefused } from './credential-floor';

/**
 * Server configuration
 */
export interface ServerConfig {
  /** Port to listen on */
  port?: number;
  /** Hostname to bind to */
  hostname?: string;
  /** Enable CORS */
  cors?: boolean;
  /** Enable request logging */
  logging?: boolean;
  /** Base path for routes */
  basePath?: string;
  /**
   * AgentIdentity whose public key goes on the agent card. `serve()` loads or
   * creates a PERSISTED one when this is omitted — registration pins the
   * card's key, so a fresh key per boot breaks on the first restart.
   */
  identity?: AgentIdentity;
  /** The URL this agent is reachable at (card `url`). Falls back to WEBAGENTS_PUBLIC_URL. */
  publicUrl?: string;
  /**
   * Where the agent's Ed25519 key is persisted. Defaults to
   * WEBAGENTS_KEYS_DIR, then `~/.webagents/keys`. `null` = ephemeral (tests).
   */
  keysDir?: string | null;
  /**
   * POST /api/agents/heartbeat every 60s when WEBAGENTS_AGENT_TOKEN and
   * ROBUTLER_API_URL are set (default true). Presence is a registration
   * requirement, not an extra.
   */
  heartbeat?: boolean;
}

/** What `serve()` hands back once the agent is actually listening. */
export interface ServeHandle {
  /** The fetch handler actually serving requests (usable in tests / other runtimes). */
  fetch: (request: Request) => Promise<Response>;
  /** The identity whose public key the agent card carries. */
  identity: AgentIdentity;
  /** The port that was really bound (meaningful when `port: 0` was requested). */
  port: number;
  /** Stop the server, the heartbeat and any portal bridge this agent opened. */
  close: () => Promise<void>;
}

/**
 * Result of createAgentApp: the Hono HTTP app plus a WebSocket upgrade handler.
 *
 * Breaking change from v1 which returned a bare Hono instance.
 * Callers that only need HTTP can use `result.app`.
 */
export interface AgentServer {
  /** Hono HTTP application */
  app: Hono;
  /**
   * Handle a WebSocket upgrade from a Node.js HTTP server.
   * Wire to `httpServer.on('upgrade', handleUpgrade)`.
   */
  handleUpgrade(req: import('http').IncomingMessage, socket: import('stream').Duplex, head: Buffer): void;
}

/**
 * Create a Hono app + WS upgrade handler for an agent.
 *
 * @returns AgentServer with `.app` (Hono) and `.handleUpgrade()` for WS
 */
export function createAgentApp(agent: IAgent, config: ServerConfig = {}): AgentServer {
  const app = new Hono();
  const basePath = config.basePath || '';
  
  // Middleware
  if (config.cors !== false) {
    app.use('*', cors());
  }
  
  if (config.logging !== false) {
    app.use('*', logger());
  }

  // ==========================================================================
  // THE CREDENTIAL FLOOR for this server class — registered before any route,
  // so it runs ahead of route dispatch rather than inside one branch.
  //
  // It has to be here and not only in `createFetchHandler`: this app registers
  // its OWN `/uamp` and `/uamp/stream` routes and mounts every skill `@http`
  // endpoint (including `CompletionsTransportSkill`'s
  // `@http({ path: '/v1/chat/completions' })`) as its own Hono route. All of
  // those SHADOW the `app.all('*')` fallback into the fetch handler, so a floor
  // that lived only down there guarded nothing that mattered — the TypeScript
  // twin of the Python per-skill static mount, door 3.
  //
  // One middleware covers: the two UAMP routes below, every mounted `@http`
  // handler, the fetch-handler fallback, and anything added later.
  // ==========================================================================
  app.use('*', async (c, next) => {
    const refusal = credentialFloor(c.req.raw);
    if (refusal) return refusal;
    await next();
    return undefined;
  });

  // Health check
  app.get(`${basePath}/health`, (c) => {
    return c.json({ status: 'ok', agent: agent.name });
  });
  
  // Agent info
  app.get(`${basePath}/info`, (c) => {
    return c.json({
      name: agent.name,
      description: agent.description,
      capabilities: agent.getCapabilities(),
    });
  });
  
  // UAMP endpoint (HTTP POST)
  app.post(`${basePath}/uamp`, async (c) => {
    try {
      const body = await c.req.json() as ClientEvent[];
      
      const events: ServerEvent[] = [];
      for await (const event of agent.processUAMP(body)) {
        events.push(event);
      }
      
      return c.json(events);
    } catch (error) {
      return c.json({
        error: {
          code: 'uamp_error',
          message: (error as Error).message,
        },
      }, 500);
    }
  });
  
  // UAMP streaming endpoint (SSE)
  app.post(`${basePath}/uamp/stream`, async (c) => {
    try {
      const body = await c.req.json() as ClientEvent[];
      
      return streamResponse(c, async function* () {
        for await (const event of agent.processUAMP(body)) {
          yield `data: ${serializeEvent(event)}\n\n`;
        }
        yield 'data: [DONE]\n\n';
      });
    } catch (error) {
      return c.json({
        error: {
          code: 'uamp_error',
          message: (error as Error).message,
        },
      }, 500);
    }
  });
  
  // Mount HTTP endpoints from agent skills (httpRegistry)
  for (const [key, endpoint] of (agent as { httpRegistry?: Map<string, { path: string; method: string; handler: (req: Request, ctx: Context) => Promise<Response> }> }).httpRegistry || new Map()) {
    const [method, path] = key.split(':');
    const fullPath = `${basePath}${path}`;
    
    const handler = async (c: HonoContext) => {
      const context = createContextFromHono(c);
      try {
        const response = await endpoint.handler(c.req.raw, context);
        return response;
      } catch (error) {
        return c.json({
          error: {
            code: 'handler_error',
            message: (error as Error).message,
          },
        }, 500);
      }
    };
    
    switch (method.toLowerCase()) {
      case 'get':
        app.get(fullPath, handler);
        break;
      case 'post':
        app.post(fullPath, handler);
        break;
      case 'put':
        app.put(fullPath, handler);
        break;
      case 'patch':
        app.patch(fullPath, handler);
        break;
      case 'delete':
        app.delete(fullPath, handler);
        break;
    }
  }

  // Fallback: delegate everything else to the universal fetch handler, which
  // serves the routes this app never registered on its own — most
  // importantly POST {basePath}/chat/completions (the endpoint the platform
  // dials and the documented quickstart curls) plus /.well-known/agent.json
  // (at the origin AND under the prefix, with metadata.publicKey) and
  // /.well-known/jwks.json. Before this delegation, `serve()` registered no
  // chat-completions route at all, so the documented TS quickstart could
  // not work.
  const fetchHandler = createFetchHandler(agent, {
    basePath,
    identity: config.identity,
    // The card's `url` is the address the agent PUBLISHES for itself.
    // Without this it was derived from the REQUEST HOST, so a configured
    // `publicUrl` / WEBAGENTS_PUBLIC_URL was documented, accepted, used as
    // the identity issuer — and silently absent from the card.
    //
    // (For the record, since two earlier comments here overstated this in
    // opposite directions: the platform does not store `card.url`, and it is
    // NOT true that `AgentMetadata` "declares only" name/description/avatar/
    // capabilities/publicKey — `lib/auth/agent-auth.ts:82` carries an index
    // signature `[key: string]: unknown`, so `url` does come through the
    // interface. What is true is that nothing dereferences it: the only
    // `metadata.` reads in that file are `capabilities` (272, 344) and
    // `publicKey` (485, 509), and the callable address a registration is keyed
    // on is `composeAgentRegistrationUrl(iss, agent_path, sub)` (186), from the
    // agent's own signed token. `card.url` is what every OTHER A2A consumer
    // reads, which is why it still has to be right, and why the last-resort
    // fallback is a relative reference rather than a Host-header guess.)
    publicUrl: config.publicUrl,
  });
  app.all('*', (c) => fetchHandler(c.req.raw));

  // WebSocket upgrade handler using wsRegistry
  const handleUpgrade = (
    req: import('http').IncomingMessage,
    socket: import('stream').Duplex,
    head: Buffer,
  ) => {
    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
    let subPath = url.pathname;
    if (basePath && subPath.startsWith(basePath)) {
      subPath = subPath.slice(basePath.length) || '/';
    }

    const wsEndpoint = agent.getWebSocketHandler?.(subPath);
    if (!wsEndpoint) {
      socket.write('HTTP/1.1 404 Not Found\r\n\r\n');
      socket.destroy();
      return;
    }

    // The floor, on the WebSocket door. `UAMPTransportSkill` registers
    // `@websocket({ path: '/uamp' })`, which runs the model on the owner's
    // credit exactly like `POST /uamp` does — and nothing here checked
    // anything at all. Same predicate, same path set, from
    // `credential-floor.ts`; a handshake carries its credential in a header or
    // in `?token=`, because a browser cannot set headers on an upgrade.
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

    // Lazy-init a noServer WebSocketServer
    if (!(handleUpgrade as any)._wss) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const { WebSocketServer } = require('ws');
        (handleUpgrade as any)._wss = new WebSocketServer({ noServer: true });
      } catch {
        socket.write('HTTP/1.1 500 ws package not available\r\n\r\n');
        socket.destroy();
        return;
      }
    }
    const wss = (handleUpgrade as any)._wss;

    const context = createContextFromIncomingMessage(req);
    wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
      wsEndpoint.handler(ws, context);
    });
  };
  
  return { app, handleUpgrade };
}

/**
 * Create context from Hono request
 */
function createContextFromHono(c: HonoContext): Context {
  const context = new ContextImpl();
  
  const authHeader = c.req.header('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    context.setAuth({
      authenticated: true,
    });
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

/**
 * Create context from Node.js IncomingMessage (for WS upgrades)
 */
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

/**
 * Create an SSE streaming response
 */
function streamResponse(
  _c: HonoContext,
  generator: () => AsyncGenerator<string, void, unknown>
): Response {
  const encoder = new TextEncoder();
  
  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of generator()) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });
  
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

/**
 * The `agent_path` claim for a server mounted at `basePath`.
 *
 * `/agents/mini` serving `mini` is `/agents`: the platform composes the
 * registration URL as `iss + agent_path + '/' + sub` and supplies `sub`
 * itself, so returning the whole `basePath` would key the registration on
 * `/agents/mini/mini`. A `basePath` that does not end in the agent's name is
 * not a hosting prefix and yields no claim rather than a guess.
 */
export function agentPathFromBasePath(
  basePath: string | undefined,
  agentName: string,
): string | undefined {
  const base = (basePath ?? '').replace(/\/+$/, '');
  if (!base) return undefined;
  const suffix = `/${agentName}`;
  if (!base.endsWith(suffix)) return undefined;
  return base.slice(0, base.length - suffix.length) || undefined;
}

/**
 * Serve an agent: HTTP + WebSocket, the platform registration surface, and
 * any reverse bridge the agent's skills declare.
 *
 * This is the ONE documented way to put a TypeScript agent on the platform.
 * It absorbed everything the deleted `host()` wrapper used to add, because
 * none of it was a convenience:
 *
 *  * a PERSISTED Ed25519 identity, so `/.well-known/agent.json` carries
 *    `metadata.publicKey` (SPKI PEM) and the key survives a restart —
 *    registration pins that key and verifies every later AOAuth token
 *    against it;
 *  * the agent card at the ORIGIN as well as under `basePath` (the platform
 *    resolves the card origin-relative and DISCARDS the agent path);
 *  * a 60s presence heartbeat;
 *  * `agent.initialize()`, which starts an attached `PortalConnectSkill`.
 *
 * That last one is the lifecycle fix: a bridged agent cannot run until it is
 * connected, and skills initialise lazily on first run. `connect()` used to
 * paper over that deadlock for one caller; starting the skill here fixes it
 * for every caller.
 */
export async function serve(agent: IAgent, config: ServerConfig = {}): Promise<ServeHandle> {
  const port = config.port ?? 3000;
  const hostname = config.hostname || '0.0.0.0';
  const publicUrl =
    config.publicUrl ??
    (typeof process !== 'undefined' ? process.env?.WEBAGENTS_PUBLIC_URL : undefined) ??
    `http://localhost:${port}`;

  const identity =
    config.identity ??
    (await loadOrCreateAgentIdentity(agent.name, {
      issuer: publicUrl,
      keysDir: config.keysDir,
      // `basePath` is the prefix PLUS the agent name (`/agents/mini`); the
      // `agent_path` claim is the prefix alone, because the platform appends
      // `sub` itself. Deriving it here is what makes the composed
      // registration URL match the path the card is actually served at.
      agentPath: agentPathFromBasePath(config.basePath, agent.name),
    }));

  // Initialise the agent BEFORE binding: this is what starts an attached
  // PortalConnectSkill, and a bridged agent that only initialises on its
  // first request waits for a request that is supposed to arrive over the
  // socket it never opened.
  const initFn = (agent as { initialize?: () => Promise<void> }).initialize;
  if (typeof initFn === 'function') await initFn.call(agent);

  // `POST {basePath}/chat/completions` refuses a request with no credential,
  // but only an AuthSkill actually VERIFIES the one that is presented. Say so
  // out loud rather than letting a bearer-shaped string look like security.
  const skills = (agent as { skills?: Array<{ constructor?: { name?: string } }> }).skills ?? [];
  if (!skills.some((s) => s?.constructor?.name === 'AuthSkill')) {
    console.warn(
      `[webagents] ${agent.name} has no AuthSkill: /chat/completions requires an ` +
        'Authorization header but cannot verify it. Add AuthSkill to validate api keys, ' +
        'owner assertions and platform service tokens.',
    );
  }

  const { app, handleUpgrade } = createAgentApp(agent, { ...config, identity });
  const fetchHandler = (request: Request) => Promise.resolve(app.fetch(request));

  let heartbeatHandle: HeartbeatHandle | null = null;
  if (config.heartbeat !== false) {
    heartbeatHandle = startHeartbeat(agent.name);
  }

  const stopAgent = async () => {
    heartbeatHandle?.stop();
    const cleanupFn = (agent as { cleanup?: () => Promise<void> }).cleanup;
    if (typeof cleanupFn === 'function') await cleanupFn.call(agent);
  };

  if (typeof Bun !== 'undefined') {
    console.log(`[webagents] ${agent.name} on http://${hostname}:${port} (Bun)`);
    Bun.serve({ port, hostname, fetch: app.fetch });
    return { fetch: fetchHandler, identity, port, close: stopAgent };
  }

  let nodeServe: (
    options: { fetch: unknown; port: number; hostname: string },
    onListening?: (info: { port: number }) => void,
  ) => { on?: (evt: string, cb: unknown) => void; close?: (cb?: () => void) => void };
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ({ serve: nodeServe } = await import('@hono/node-server' as any));
  } catch {
    console.error('Failed to start server. Install @hono/node-server for Node.js support.');
    throw new Error('No compatible server runtime found');
  }

  let resolveListening: (port: number) => void;
  const listening = new Promise<number>((resolve) => {
    resolveListening = resolve;
  });
  const server = nodeServe({ fetch: app.fetch, port, hostname }, (info) =>
    resolveListening(info.port),
  );
  // Wire WebSocket upgrades to the transport skill handlers
  server.on?.('upgrade', handleUpgrade);
  const boundPort = await listening;
  console.log(`[webagents] ${agent.name} on http://${hostname}:${boundPort}`);

  return {
    fetch: fetchHandler,
    identity,
    port: boundPort,
    close: async () => {
      await stopAgent();
      await new Promise<void>((resolve) => {
        if (server?.close) server.close(() => resolve());
        else resolve();
      });
    },
  };
}

// Bun type declaration
declare const Bun: {
  serve(options: { port: number; hostname: string; fetch: (request: Request) => Response | Promise<Response> }): void;
} | undefined;
