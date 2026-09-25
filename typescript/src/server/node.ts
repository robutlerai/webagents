/**
 * Node.js Server
 * 
 * Hono-based HTTP/WebSocket server for running agents.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { requestLog } from './request-log';
import type { Context as HonoContext } from 'hono';
import type { IAgent, Context } from '../core/types';
import { ContextImpl } from '../core/context';
import type { ClientEvent, ServerEvent } from '../uamp/events';
import { serializeEvent } from '../uamp/events';
import { createFetchHandler } from './handler';
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
import type { AgentIdentity } from '../crypto/identity';
import { loadOrCreateAgentIdentity } from '../crypto/identity-store';
import { resolveAgentCredential } from './agent-credential';
import { resolvePortalApiUrl, startHeartbeat, stopHeartbeat, type HeartbeatHandle } from './registration';
import { createRequire } from 'node:module';

// A `require` that works in this ES module (2026-09-24). The WebSocket upgrade
// below called the bare `require('ws')`, which does not exist in ESM, so it
// threw `ReferenceError`, the `catch` answered "500 ws package not available",
// and every UAMP WebSocket to a `serve()`d agent failed although `ws` is a
// declared dependency. This file is Node-only and not in the portal's bundle.
const nodeRequire = createRequire(import.meta.url);
import { credentialFloor, webSocketUpgradeIsRefused } from './credential-floor';
import { replyText } from './error-reply';
import {
  agentVerifiesCredentials,
  defaultHostname,
  originPolicy,
  upgradeOriginAllowed,
  type CorsSetting,
} from './origin-policy';

/**
 * Server configuration
 */
export interface ServerConfig {
  /** Port to listen on */
  port?: number;
  /**
   * Hostname to bind to. Unset: every interface when a public URL is
   * configured or the agent has an AuthSkill, loopback otherwise
   * (`origin-policy.ts`, S-226).
   */
  hostname?: string;
  /**
   * Which browser origins may call the agent. `true` any, a list those, `false`
   * none. Unset: any origin when the agent has an AuthSkill (it verifies what
   * it is sent), loopback origins only when it does not (`origin-policy.ts`).
   */
  cors?: CorsSetting;
  /** Enable request logging */
  logging?: boolean;
  /** Base path for routes */
  basePath?: string;
  /**
   * The identity that signs for this agent and whose key set the card names.
   * `serve()` loads or creates a PERSISTED one when this is omitted, with
   * `publicUrl + basePath` as its issuer: registration pins the key set's
   * thumbprints, so a fresh key per boot breaks on the first restart. An
   * identity passed in must carry that same issuer, or the card cannot name
   * itself.
   */
  identity?: AgentIdentity;
  /**
   * The base URL this agent is reachable at. With `basePath` it composes the
   * agent URL, `publicUrl + basePath`: the card's `url`, the identity's
   * issuer and the principal the platform registers. Falls back to
   * WEBAGENTS_PUBLIC_URL, then to `http://localhost:<port>`, which serves but
   * cannot sign (the platform refuses loopback by name).
   */
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
  /** The identity that signs for this agent: what `registerWithPlatform` takes. */
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

  // One origin decision for every door this app has: the CORS middleware, the
  // fetch-handler fallback (which stamped `*` on its own responses) and the
  // WebSocket upgrade, which CORS never covered (S-226).
  const policy = originPolicy(config.cors, agentVerifiesCredentials(agent));

  // Middleware
  if (config.cors !== false) {
    app.use('*', cors({ origin: (origin) => policy(origin) ?? undefined }));
  }
  
  if (config.logging !== false) {
    app.use('*', requestLog());
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
      // A fixed sentence and a logged reference, not the error's own message
      // (S-228, `error-reply.ts`).
      return c.json({
        error: {
          code: 'uamp_error',
          message: replyText(error, `${agent.name} uamp`),
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
          message: replyText(error, `${agent.name} uamp/stream`),
        },
      }, 500);
    }
  });
  
  // Mount HTTP endpoints from agent skills (httpRegistry)
  for (const [key, endpoint] of (agent as { httpRegistry?: Map<string, { path: string; method: string; scopes?: string[]; auth?: string; handler: (req: Request, ctx: Context) => Promise<Response> }> }).httpRegistry || new Map()) {
    const [method, path] = key.split(':');
    const fullPath = `${basePath}${path}`;
    
    const handler = async (c: HonoContext) => {
      let context = createContextFromHono(c);
      try {
        // WHO MAY CALL IT (S-242, 2026-09-25): the one gate
        // (`endpoint-gate.ts`); an endpoint with no scopes is open.
        if (needsCaller(endpoint)) {
          const raw = new Uint8Array(await c.req.raw.clone().arrayBuffer());
          context = identificationContext(context, inboundRequest(c.req.raw, raw));
          const gate = await admitEndpoint(agent, endpoint, context);
          if (gate.refusal) return c.json(gate.refusal.body, gate.refusal.status);
        }
        const response = await endpoint.handler(c.req.raw, context);
        return response;
      } catch (error) {
        return c.json({
          error: {
            code: 'handler_error',
            message: replyText(error, `${agent.name} ${key}`),
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
  // serves the routes this app never registered on its own: most
  // importantly POST {basePath}/chat/completions (the endpoint the platform
  // dials and the documented quickstart curls) plus the self-naming
  // /.well-known/agent.json under the prefix and /.well-known/jwks.json.
  // Before this delegation, `serve()` registered no chat-completions route at
  // all, so the documented TS quickstart could not work.
  const fetchHandler = createFetchHandler(agent, {
    basePath,
    identity: config.identity,
    // The card's `url` is the address the agent PUBLISHES for itself: the
    // identity's issuer when there is one (`serve()` composes it from
    // `publicUrl + basePath`), else this plus `basePath`. Since 2026-09-17
    // the platform READS it: the card must name the URL it was fetched from
    // (`client_id`) and the principal (`url`), or registration refuses it as
    // `card_not_self_naming`. A Host-header guess was never acceptable here
    // and is now a refusal.
    publicUrl: config.publicUrl,
    originPolicy: policy,
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

    // A page from another origin, refused before anything else is read. CORS
    // does not apply to WebSocket handshakes, so without this any website the
    // developer visited could open `/uamp?token=<anything>` (S-226).
    if (!upgradeOriginAllowed(policy, req.headers.origin as string | undefined)) {
      socket.write('HTTP/1.1 403 Forbidden\r\n\r\n');
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
        const { WebSocketServer } = nodeRequire('ws');
        (handleUpgrade as any)._wss = new WebSocketServer({ noServer: true });
      } catch {
        socket.write('HTTP/1.1 500 ws package not available\r\n\r\n');
        socket.destroy();
        return;
      }
    }
    const wss = (handleUpgrade as any)._wss;

    const context = createContextFromIncomingMessage(req);
    const upgrade = (ctx: Context) => {
      wss.handleUpgrade(req, socket, head, (ws: WebSocket) => {
        wsEndpoint.handler(ws, ctx);
      });
    };
    if (isOpen(wsEndpoint.scopes)) {
      upgrade(context);
      return;
    }
    // WHO MAY OPEN IT (S-242, 2026-09-25): the one gate, before the
    // handshake completes, so a refusal is an HTTP status with the gate's body.
    admit(agent, wsEndpoint.scopes, identificationContext(context, inboundUpgrade(req))).then(
      (gate) => (gate.refusal ? refuseUpgrade(socket, gate.refusal) : upgrade(gate.context)),
      (error) => {
        replyText(error, `${agent.name} websocket ${subPath}`);
        socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\n');
        socket.destroy();
      },
    );
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
 * Serve an agent: HTTP + WebSocket, the platform registration surface, and
 * any reverse bridge the agent's skills declare.
 *
 * This is the ONE documented way to put a TypeScript agent on the platform.
 * It absorbed everything the deleted `host()` wrapper used to add, because
 * none of it was a convenience:
 *
 *  * a PERSISTED Ed25519 identity whose issuer is the agent URL,
 *    `publicUrl + basePath`, so `{agentUrl}/.well-known/jwks.json` lists a
 *    key that survives a restart: registration pins the key set's
 *    thumbprints and verifies every later signed request against them
 *    (ADR 0038 step 5, 2026-09-17);
 *  * the self-naming agent card under `basePath` (`client_id`, `url` and
 *    `jwks_uri` all derived from that same agent URL, W2 design section 3.3);
 *  * a 60s presence heartbeat;
 *  * `agent.initialize()`, which starts an attached `PortalConnectSkill`.
 *
 * That last one is the lifecycle fix: a bridged agent cannot run until it is
 * connected, and skills initialise lazily on first run. `connect()` used to
 * paper over that deadlock for one caller; starting the skill here fixes it
 * for every caller.
 *
 * THE LOOPBACK FALLBACK STAYS FOR SERVING AND CANNOT SIGN. With no
 * `publicUrl` and no WEBAGENTS_PUBLIC_URL the agent URL is
 * `http://localhost:<port>` plus `basePath`: fine for a local curl, and the
 * signer refuses it with the sentence `registerWithPlatform` answers, because
 * the platform refuses `localhost` by name before it resolves anything and a
 * bare 401 would send the reader to look at their keys.
 */
export async function serve(agent: IAgent, config: ServerConfig = {}): Promise<ServeHandle> {
  const port = config.port ?? 3000;
  const configuredPublicUrl =
    config.publicUrl ??
    (typeof process !== 'undefined' ? process.env?.WEBAGENTS_PUBLIC_URL : undefined);
  // Loopback unless the agent is meant to be reached or verifies its callers
  // (S-226). It was every interface, always, including for the unauthenticated
  // local agent `webagents serve` starts, whose model key anyone on the same
  // network could then spend.
  const verifiesCredentials = agentVerifiesCredentials(agent);
  const hostname =
    config.hostname || defaultHostname({ publicUrl: configuredPublicUrl, verifiesCredentials });
  if (!config.hostname && hostname === '127.0.0.1') {
    console.info(
      `[webagents] ${agent.name}: listening on 127.0.0.1 only, because it has no public URL and no ` +
        'AuthSkill. Pass `hostname` (`--host 0.0.0.0` on the CLI) to accept other machines.',
    );
  }
  const publicUrl = (configuredPublicUrl ?? `http://localhost:${port}`).replace(/\/+$/, '');
  // `basePath` is the whole mount, prefix PLUS agent name (`/agents/mini`),
  // and the agent URL is `publicUrl + basePath`: the principal the platform
  // registers, where the card and the key set are served, and what every
  // signature names. (The bearer era split this into an `iss` and an
  // `agent_path` claim the platform recomposed; there is one URL now.)
  const agentUrl = `${publicUrl}${(config.basePath ?? '').replace(/\/+$/, '')}`;

  // A PURELY LOCAL RUN SAYS SO ONCE (2026-09-24). With no public URL, no agent
  // key and no platform URL, the public-URL and heartbeat diagnostics were two
  // paragraphs about a platform the person had not tried to use yet, and on
  // `webagents serve` they were a first-time developer's first screen. The
  // moment ANY of the three is set, each diagnostic prints as before, because
  // that is when a missing half is a mistake worth naming.
  // The agent's own key, found rather than configured (`agent-credential.ts`).
  const credential = await resolveAgentCredential(agent.name);
  const localOnly = !configuredPublicUrl && !credential && !resolvePortalApiUrl();
  if (localOnly) {
    console.info(
      `[webagents] ${agent.name}: serving locally, not registered with the platform. ` +
        'Registering needs WEBAGENTS_PUBLIC_URL, WEBAGENTS_AGENT_TOKEN and ROBUTLER_API_URL.',
    );
  } else if (!configuredPublicUrl) {
    console.info(
      `[webagents] ${agent.name}: no public URL (WEBAGENTS_PUBLIC_URL), serving as ${agentUrl}. ` +
        'This identity cannot sign a platform request from a loopback address; set ' +
        'WEBAGENTS_PUBLIC_URL to the https address the agent is reachable at before registering.',
    );
  }

  const identity =
    config.identity ??
    (await loadOrCreateAgentIdentity(agent.name, {
      issuer: agentUrl,
      keysDir: config.keysDir,
    }));

  // Hand the agent its identity (2026-09-23), and do it BEFORE `initialize()`
  // so a skill that publishes on boot can sign. A skill that calls the
  // platform for the agent (PortalDiscoverySkill) reads `agent.identity` and
  // signs with the key this very server publishes, which is the credential
  // the platform takes first; until then it asked the developer for a
  // platform key the platform never needed from a signing agent. The served
  // identity is the one whose key set is reachable, so it wins over anything
  // set on the agent beforehand.
  agent.identity = identity;

  // Initialise the agent BEFORE binding: this is what starts an attached
  // PortalConnectSkill, and a bridged agent that only initialises on its
  // first request waits for a request that is supposed to arrive over the
  // socket it never opened.
  const initFn = (agent as { initialize?: () => Promise<void> }).initialize;
  if (typeof initFn === 'function') await initFn.call(agent);

  // `POST {basePath}/chat/completions` refuses a request with no credential,
  // but only an AuthSkill actually VERIFIES the one that is presented. Say so
  // out loud rather than letting a bearer-shaped string look like security.
  // An agent file's `access:` block (ADR-0045) verifies signed callers
  // itself, so the sentence would be false for it.
  const hasAccessBlock = ((agent as { skills?: Array<{ constructor?: { name?: string } }> }).skills ?? []).some(
    (skill) => skill?.constructor?.name === 'AccessSkill',
  );
  if (!verifiesCredentials && !hasAccessBlock) {
    console.warn(
      `[webagents] ${agent.name} has no AuthSkill: /chat/completions requires an ` +
        'Authorization header but cannot verify it. Add AuthSkill to validate api keys, ' +
        'owner assertions and platform service tokens.',
    );
  }

  const { app, handleUpgrade } = createAgentApp(agent, { ...config, identity });
  const fetchHandler = (request: Request) => Promise.resolve(app.fetch(request));

  let heartbeatHandle: HeartbeatHandle | null = null;
  // Local-only: there is nothing to beat with, and the line above said so.
  if (config.heartbeat !== false && !localOnly) {
    heartbeatHandle = startHeartbeat(agent.name, { token: credential?.token, key: identity.issuer });
  }

  const stopAgent = async () => {
    heartbeatHandle?.stop();
    // And the one `registerWithPlatform` started for this identity with the
    // bearer it minted: a closed agent must stop claiming to be online.
    stopHeartbeat(identity.issuer);
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
