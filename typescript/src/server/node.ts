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
 * Serve an agent on HTTP + WebSocket.
 * 
 * Note: This uses Bun or Node.js built-in serve if available.
 * For production, use Hono's adapter for your platform.
 */
export async function serve(agent: IAgent, config: ServerConfig = {}): Promise<void> {
  const { app, handleUpgrade } = createAgentApp(agent, config);
  const port = config.port || 3000;
  const hostname = config.hostname || '0.0.0.0';
  
  if (typeof Bun !== 'undefined') {
    console.log(`Starting server on http://${hostname}:${port} (Bun)`);
    Bun.serve({
      port,
      hostname,
      fetch: app.fetch,
    });
  } else {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { serve: nodeServe } = await import('@hono/node-server' as any);
      console.log(`Starting server on http://${hostname}:${port} (Node.js)`);
      const server = nodeServe({
        fetch: app.fetch,
        port,
        hostname,
      });

      // Wire WebSocket upgrades to the transport skill handlers
      if (server && typeof server.on === 'function') {
        server.on('upgrade', handleUpgrade);
      }
    } catch {
      console.error('Failed to start server. Install @hono/node-server for Node.js support.');
      throw new Error('No compatible server runtime found');
    }
  }
}

// Bun type declaration
declare const Bun: {
  serve(options: { port: number; hostname: string; fetch: (request: Request) => Response | Promise<Response> }): void;
} | undefined;
