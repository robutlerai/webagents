/**
 * Universal Fetch Handler
 * 
 * A fetch handler that works in any environment (Node.js, Bun, Cloudflare Workers, etc.)
 */

import type { IAgent, Context } from '../core/types.js';
import { ContextImpl } from '../core/context.js';
import type { ClientEvent, ServerEvent } from '../uamp/events.js';
import { serializeEvent } from '../uamp/events.js';
import type { AgentIdentity } from '../crypto/identity.js';

/**
 * Handler options
 */
export interface HandlerOptions {
  /** Base path for routes */
  basePath?: string;
  /** CORS origin */
  corsOrigin?: string;
  /** AgentIdentity for AOAuth JWKS/OpenID serving */
  identity?: AgentIdentity;
}

/**
 * Create a fetch handler for an agent
 * 
 * @example
 * ```typescript
 * const agent = new BaseAgent({ ... });
 * const handler = createFetchHandler(agent);
 * 
 * // Cloudflare Workers
 * export default { fetch: handler };
 * 
 * // Bun
 * Bun.serve({ fetch: handler });
 * ```
 */
export function createFetchHandler(
  agent: IAgent,
  options: HandlerOptions = {}
): (request: Request) => Promise<Response> {
  const basePath = options.basePath || '';
  
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;
    
    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, {
        headers: getCorsHeaders(options.corsOrigin),
      });
    }
    
    // Health check
    if (path === `${basePath}/health` && method === 'GET') {
      return jsonResponse({ status: 'healthy', agent: agent.name }, options.corsOrigin);
    }
    
    // Agent info
    if (path === `${basePath}/info` && method === 'GET') {
      return jsonResponse({
        name: agent.name,
        description: agent.description,
        capabilities: agent.getCapabilities(),
        tools: agent.getToolDefinitions?.() ?? [],
      }, options.corsOrigin);
    }
    
    // .well-known/agent.json — A2A agent card
    if (path === `${basePath}/.well-known/agent.json` && method === 'GET') {
      const baseUrl = `${url.protocol}//${url.host}${basePath}`;
      return jsonResponse({
        name: agent.name,
        description: agent.description,
        url: baseUrl,
        capabilities: { streaming: true, pushNotifications: false },
        authentication: { schemes: ['Bearer'] },
        skills: (agent.getToolDefinitions?.() ?? []).map((t: { function: { name: string; description?: string } }) => ({
          id: t.function.name,
          name: t.function.name,
          description: t.function.description,
        })),
      }, options.corsOrigin);
    }

    // .well-known/jwks.json — AOAuth public keys
    if (path === `${basePath}/.well-known/jwks.json` && method === 'GET') {
      if (!options.identity) {
        return jsonResponse({ error: 'AOAuth not configured' }, options.corsOrigin, 404);
      }
      return new Response(JSON.stringify(options.identity.getJwks()), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600',
          ...getCorsHeaders(options.corsOrigin),
        },
      });
    }

    // .well-known/openid-configuration — AOAuth discovery
    if (path === `${basePath}/.well-known/openid-configuration` && method === 'GET') {
      if (!options.identity) {
        return jsonResponse({ error: 'AOAuth not configured' }, options.corsOrigin, 404);
      }
      return new Response(JSON.stringify(options.identity.getOpenIdConfiguration()), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600',
          ...getCorsHeaders(options.corsOrigin),
        },
      });
    }

    // OpenAI-compatible chat completions
    if ((path === `${basePath}/chat/completions` || path === `${basePath}/v1/chat/completions`) && method === 'POST') {
      try {
        const body = await request.json() as {
          messages: Array<{ role: string; content: string }>;
          stream?: boolean;
        };
        const msgs = body.messages.map((m) => ({ role: m.role as 'user' | 'system' | 'assistant', content: m.content }));

        if (body.stream && typeof agent.runStreaming === 'function') {
          const gen = agent.runStreaming(msgs);
          return streamCompletionsResponse(gen, options.corsOrigin);
        }

        const result = await agent.run(msgs);
        return jsonResponse({
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          choices: [{ index: 0, message: { role: 'assistant', content: result.content }, finish_reason: 'stop' }],
          usage: result.usage,
        }, options.corsOrigin);
      } catch (error) {
        return jsonResponse({
          error: { code: 'completions_error', message: (error as Error).message },
        }, options.corsOrigin, 500);
      }
    }

    // UAMP endpoint
    if (path === `${basePath}/uamp` && method === 'POST') {
      try {
        const body = await request.json() as ClientEvent[];
        
        const events: ServerEvent[] = [];
        for await (const event of agent.processUAMP(body)) {
          events.push(event);
        }
        
        return jsonResponse(events, options.corsOrigin);
      } catch (error) {
        return jsonResponse({
          error: { code: 'uamp_error', message: (error as Error).message },
        }, options.corsOrigin, 500);
      }
    }
    
    // UAMP streaming
    if (path === `${basePath}/uamp/stream` && method === 'POST') {
      try {
        const body = await request.json() as ClientEvent[];
        
        return streamResponse(agent.processUAMP(body), options.corsOrigin);
      } catch (error) {
        return jsonResponse({
          error: { code: 'uamp_error', message: (error as Error).message },
        }, options.corsOrigin, 500);
      }
    }
    
    // Check agent HTTP endpoints
    const httpRegistry = (agent as { httpRegistry?: Map<string, { handler: (req: Request, ctx: Context) => Promise<Response> }> }).httpRegistry;
    if (httpRegistry) {
      const key = `${method}:${path.replace(basePath, '')}`;
      const endpoint = httpRegistry.get(key);
      if (endpoint) {
        const context = createContextFromRequest(request);
        try {
          const response = await endpoint.handler(request, context);
          // Add CORS headers
          const headers = new Headers(response.headers);
          for (const [key, value] of Object.entries(getCorsHeaders(options.corsOrigin))) {
            headers.set(key, value);
          }
          return new Response(response.body, {
            status: response.status,
            headers,
          });
        } catch (error) {
          return jsonResponse({
            error: { code: 'handler_error', message: (error as Error).message },
          }, options.corsOrigin, 500);
        }
      }
    }
    
    // Not found
    return jsonResponse({ error: { code: 'not_found', message: 'Not found' } }, options.corsOrigin, 404);
  };
}

/**
 * Create context from request
 */
function createContextFromRequest(request: Request): Context {
  const context = new ContextImpl();
  
  const authHeader = request.headers.get('authorization');
  if (authHeader?.startsWith('Bearer ')) {
    context.setAuth({ authenticated: true });
  }
  
  context.metadata = {
    userAgent: request.headers.get('user-agent'),
    method: request.method,
  };
  
  return context;
}

/**
 * Create JSON response
 */
function jsonResponse(data: unknown, corsOrigin?: string, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...getCorsHeaders(corsOrigin),
    },
  });
}

/**
 * Create SSE streaming response
 */
function streamResponse(
  events: AsyncGenerator<ServerEvent, void, unknown>,
  corsOrigin?: string
): Response {
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
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      ...getCorsHeaders(corsOrigin),
    },
  });
}

/**
 * Create SSE streaming response for OpenAI-compatible completions
 */
function streamCompletionsResponse(
  gen: AsyncGenerator<{ type: string; delta?: string; response?: unknown }, void, unknown>,
  corsOrigin?: string,
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
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      ...getCorsHeaders(corsOrigin),
    },
  });
}

/**
 * Get CORS headers
 */
function getCorsHeaders(origin?: string): Record<string, string> {
  return {
    'Access-Control-Allow-Origin': origin || '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}
