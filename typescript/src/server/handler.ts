/**
 * Universal Fetch Handler
 * 
 * A fetch handler that works in any environment (Node.js, Bun, Cloudflare Workers, etc.)
 */

import type { IAgent, Context } from '../core/types';
import { ContextImpl } from '../core/context';
import type { ClientEvent, ServerEvent } from '../uamp/events';
import { serializeEvent } from '../uamp/events';
import type { AgentIdentity } from '../crypto/identity';
import { CREDENTIAL_HEADERS, credentialFloor } from './credential-floor';

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
  /**
   * The URL this agent is actually reachable at, published as the agent
   * card's `url`. Falls back to `WEBAGENTS_PUBLIC_URL`, then to `basePath` as
   * a RELATIVE reference.
   *
   * `serve()` documented `publicUrl` as "the card `url`" but never threaded it
   * here, so an agent started with WEBAGENTS_PUBLIC_URL=https://agent.example.com
   * still published `http://127.0.0.1:<ephemeral>/agents/mini` — an address
   * nothing outside the process could dial.
   *
   * Python resolves this identically
   * (`server/core/registration.py: resolve_public_base_url`), including the
   * last tier — see the note on `resolveCardUrl`.
   */
  publicUrl?: string;
}

/**
 * The card's `url`: explicit config, then the environment, then `basePath` as
 * a RELATIVE reference. Byte-for-byte the same decision as
 * `resolve_public_base_url` in the Python SDK, whitespace handling included.
 *
 * WHY RELATIVE AND NOT THE REQUEST ORIGIN (the two SDKs used to disagree here,
 * TS answering `http://127.0.0.1:8816/agents/og` where Python answered
 * `/mini`): the request origin is a guess derived from the Host header, and
 * behind a proxy, a tunnel or a container it is the WRONG guess stated as
 * fact. A relative reference is resolved by any consumer against the document
 * it just fetched — which IS, by construction, an origin the agent is
 * reachable at.
 *
 * The platform settles it from the other side: NO CONSUMER READS `card.url`.
 * Precisely, because an earlier version of this note said `AgentMetadata`
 * "declares only" a handful of fields and would not survive someone opening
 * the file: the interface (portal `lib/auth/agent-auth.ts:71`) carries an index
 * signature `[key: string]: unknown` at line 82, so `url` IS carried through
 * it. What holds is that nothing DEREFERENCES it — the only `metadata.` reads
 * in that file are `capabilities` (lines 272, 344) and `publicKey` (lines 485,
 * 509), and the callable address a registration is keyed on is
 * `composeAgentRegistrationUrl(iss, agent_path, sub)` (line 186), built from
 * the agent's OWN signed token. So the last tier cannot break registration
 * either way, and the honest value wins.
 */
function resolveCardUrl(configured: string | undefined, basePath: string): string {
  const env =
    typeof process !== 'undefined' ? process.env?.WEBAGENTS_PUBLIC_URL : undefined;
  const base = (configured || env || '').trim().replace(/\/+$/, '');
  return base || basePath || '/';
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

    // ========================================================================
    // THE CREDENTIAL FLOOR — one check, above every branch below AND above the
    // `httpRegistry` dispatch at the bottom, so a route added to this handler
    // is behind the floor the moment it is written.
    //
    // It used to live inside the `/chat/completions` branch, and that is
    // precisely how `POST {basePath}/uamp` and `/uamp/stream` — the same
    // `agent.processUAMP` call, the same owner's credit — stayed anonymous
    // 200s: nobody added a second copy of the `if`. Four doors to the same
    // billable endpoint were found that way across two SDKs. There is one
    // decision now, and it lives in `credential-floor.ts`.
    //
    // It runs before `await request.json()`, which is a property in its own
    // right: an anonymous caller cannot make the server parse arbitrary bytes,
    // and anonymous-plus-malformed observes 401 here and 401 on the Python
    // side rather than a parser error on one of them.
    // ========================================================================
    const refusal = credentialFloor(request, getCorsHeaders(options.corsOrigin));
    if (refusal) return refusal;

    
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
    
    // .well-known/agent.json — A2A agent card.
    // Served at the ORIGIN as well as under the agent prefix: the platform
    // fetches `new URL('/.well-known/agent.json', agentUrl)`, which is
    // origin-relative and discards the agent path — a card served only under
    // the prefix is invisible to registration.
    if (
      (path === `${basePath}/.well-known/agent.json` || path === '/.well-known/agent.json') &&
      method === 'GET'
    ) {
      const baseUrl = resolveCardUrl(options.publicUrl, basePath);
      let publicKey: string | undefined;
      try {
        publicKey = options.identity?.getPublicKeySpki();
      } catch {
        publicKey = undefined;
      }
      return jsonResponse({
        name: agent.name,
        description: agent.description,
        url: baseUrl,
        capabilities: { streaming: true, pushNotifications: false },
        authentication: { schemes: ['Bearer'] },
        // metadata.publicKey (SPKI PEM) is hard-required by platform
        // registration (verifyExternalAOAuthToken -> importSPKI).
        ...(publicKey ? { metadata: { publicKey } } : {}),
        skills: (agent.getToolDefinitions?.() ?? [])
          .filter(t => t.type === 'function' && 'function' in t)
          .map((t) => {
            const ft = t as { type: 'function'; function: { name: string; description?: string } };
            return { id: ft.function.name, name: ft.function.name, description: ft.function.description };
          }),
      }, options.corsOrigin);
    }

    // .well-known/jwks.json — AOAuth public keys (origin and prefix)
    if (
      (path === `${basePath}/.well-known/jwks.json` || path === '/.well-known/jwks.json') &&
      method === 'GET'
    ) {
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
    //
    // AUTH: this endpoint runs the agent's model on the owner's credit, so it
    // goes through the SAME path `/uamp` does — the run's `on_connection`
    // hooks, where `AuthSkill.authenticateConnection` validates the api key /
    // owner assertion / platform service token and THROWS on failure. Two
    // things make that work from here:
    //
    //  * the request's credential headers are seeded onto the RUN's context
    //    metadata (`buildRequestMetadata`), which is where `AuthSkill` reads
    //    them from — without this the hook sees no token and refuses
    //    everything;
    //  * the credential floor at the top of this handler refuses a request
    //    that carries no credential at all, so an agent assembled WITHOUT an
    //    AuthSkill is not an open model-billing endpoint for anyone who can
    //    reach the port.
    //
    // Streaming is primed before the Response is constructed: an
    // AuthenticationError raised inside the generator would otherwise surface
    // after a 200 header had already been written.
    if ((path === `${basePath}/chat/completions` || path === `${basePath}/v1/chat/completions`) && method === 'POST') {
      try {
        const body = await request.json() as {
          messages: Array<{ role: string; content: string }>;
          stream?: boolean;
          metadata?: Record<string, unknown>;
        };

        const msgs = body.messages.map((m) => ({ role: m.role as 'user' | 'system' | 'assistant', content: m.content }));

        // The platform sends `metadata: {chat_id, chat_type, platform, sender}`
        // on every routed turn. `sender` is what attributes the call to a
        // person (AuthSkill._serviceAuthInfo reads `metadata.sender.id` to
        // decide USER vs OWNER scope), so it must reach the context — a
        // service token alone names the ROUTER, not the caller.
        const requestMetadata = buildRequestMetadata(request, body.metadata);
        const chatId =
          typeof body.metadata?.chat_id === 'string' ? (body.metadata.chat_id as string) : undefined;
        const runOptions = { metadata: requestMetadata, ...(chatId ? { chatId } : {}) };

        if (body.stream && typeof agent.runStreaming === 'function') {
          const gen = agent.runStreaming(msgs, runOptions);
          // Pull the first chunk here so an auth refusal is a 401, not a
          // 200 whose body happens to contain an error.
          const first = await gen.next();
          return streamCompletionsResponse(gen, options.corsOrigin, first.done ? undefined : first.value);
        }

        const result = await agent.run(msgs, runOptions);
        return jsonResponse({
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          choices: [{ index: 0, message: { role: 'assistant', content: result.content }, finish_reason: 'stop' }],
          usage: result.usage,
        }, options.corsOrigin);
      } catch (error) {
        if (isAuthError(error)) {
          return unauthorizedResponse((error as Error).message, options.corsOrigin);
        }
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
 * Request metadata for a run: the credential headers `AuthSkill` reads, plus
 * the platform's own request-body `metadata` (chat_id, chat_type, platform,
 * sender). Body keys are merged UNDER the headers so a request body can never
 * overwrite the header a credential was actually presented in.
 */
function buildRequestMetadata(
  request: Request,
  bodyMetadata?: Record<string, unknown>,
): Record<string, unknown> {
  const metadata: Record<string, unknown> = { ...(bodyMetadata ?? {}) };
  metadata.userAgent = request.headers.get('user-agent');
  metadata.method = request.method;
  for (const name of CREDENTIAL_HEADERS) {
    const value = request.headers.get(name);
    if (value) metadata[name] = value;
  }
  const paymentToken = request.headers.get('x-payment-token');
  if (paymentToken) metadata['x-payment-token'] = paymentToken;
  return metadata;
}

/**
 * `AuthenticationError` / `AuthorizationError` are matched by NAME so this
 * module keeps no import edge into `skills/auth` (which pulls the whole JWKS
 * stack into every fetch-handler bundle).
 */
function isAuthError(error: unknown): boolean {
  const name = (error as { name?: string } | null)?.name;
  return name === 'AuthenticationError' || name === 'AuthorizationError';
}

function unauthorizedResponse(message: string, corsOrigin?: string): Response {
  return jsonResponse(
    { error: { code: 'unauthorized', message } },
    corsOrigin,
    401,
  );
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
  /**
   * The first chunk, already pulled by the caller so that an auth refusal
   * raised on the generator's first step becomes a 401 instead of a 200 with
   * an error buried in the SSE body.
   */
  firstChunk?: { type: string; delta?: string; response?: unknown },
): Response {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const emit = (chunk: { type: string; delta?: string; response?: unknown }) => {
          if (chunk.type === 'delta' && chunk.delta) {
            const data = { choices: [{ delta: { content: chunk.delta }, finish_reason: null }] };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
          }
        };
        if (firstChunk) emit(firstChunk);
        for await (const chunk of gen) {
          emit(chunk);
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
