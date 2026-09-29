/**
 * Universal Fetch Handler
 * 
 * A fetch handler that works in any environment (Node.js, Bun, Cloudflare Workers, etc.)
 */

import type { IAgent, Context, HttpEndpoint, RunOptions, RunResponse } from '../core/types';
import { ContextImpl } from '../core/context';
import { isAgentFinish } from '../core/tool-budget';
import type { ClientEvent, ServerEvent } from '../uamp/events';
import { serializeEvent } from '../uamp/events';
import type { AgentIdentity } from '../crypto/identity';
import { CREDENTIAL_HEADERS, challengeHeaders, credentialFloor, refusalHeaders } from './credential-floor';
import type { OriginPolicy } from './origin-policy';
import { buildAgentCard } from './card';
import { isKeyDirectoryRequest, keyDirectoryResponse } from './key-directory';
import { isMeantToBeShown, replyText, shownResponseError } from './error-reply';
import {
  admitEndpoint,
  identificationContext,
  inboundRequest,
  isAuthError,
  needsCaller,
  refusalResponse,
} from './endpoint-gate';
import { serveThroughPaywall } from '../skills/payments/paywall';
import { X402_CORS_ALLOW_HEADERS, X402_CORS_EXPOSE_HEADERS } from '../skills/payments/x402-wire';

// Moved to `endpoint-gate.ts` with the S-242 gate; still importable from here.
export { inboundRequest, refusalResponse };

/**
 * Handler options
 */
export interface HandlerOptions {
  /** Base path for routes */
  basePath?: string;
  /** CORS origin, answered to every request. Ignored when `originPolicy` is given. */
  corsOrigin?: string;
  /**
   * Decides the `Access-Control-Allow-Origin` value per request from its
   * `Origin`, or `null` for no CORS headers at all (see `origin-policy.ts`).
   * `serve()` passes one; a bare handler keeps the old `corsOrigin` behaviour.
   */
  originPolicy?: OriginPolicy;
  /**
   * The identity that signs for this agent. Its key set is served at
   * `{basePath}/.well-known/jwks.json` and its `issuer` is the agent URL the
   * card names (see `resolvePrincipal`).
   */
  identity?: AgentIdentity;
  /**
   * The base URL this agent is reachable at. With `basePath` it composes the
   * agent URL, `publicUrl + basePath`, which is the card's `url` and the
   * principal the platform registers. Falls back to `WEBAGENTS_PUBLIC_URL`,
   * then to `basePath` as a RELATIVE reference. Ignored for the card when an
   * `identity` is given, because the identity's issuer IS the agent URL.
   *
   * `serve()` documented `publicUrl` as "the card `url`" but never threaded it
   * here, so an agent started with WEBAGENTS_PUBLIC_URL=https://agent.example.com
   * still published `http://127.0.0.1:<ephemeral>/agents/mini`, an address
   * nothing outside the process could dial.
   */
  publicUrl?: string;
}

/**
 * The principal the card names as `url` (and derives `client_id` and
 * `jwks_uri` from, W2 design section 3.3, 2026-09-17).
 *
 * ONE SOURCE OF TRUTH: when the agent has an identity, the principal is that
 * identity's issuer, because that is the URL every signature names in
 * `Signature-Agent` and the platform requires the card's `url` to equal it.
 * `serve()` composes the issuer as `publicUrl + basePath`; a caller who
 * builds both by hand and lets them disagree would fail the self-naming
 * check at registration, and the card following the SIGNER is what makes the
 * failure show up as `card_not_self_naming` rather than as a silent
 * registration under the wrong URL.
 *
 * Without an identity: explicit config or the environment, plus `basePath`,
 * else `basePath` alone as a RELATIVE reference (the same last tier as
 * `resolve_public_base_url` in the Python SDK). WHY RELATIVE AND NOT THE
 * REQUEST ORIGIN: the request origin is a guess derived from the Host header,
 * and behind a proxy, a tunnel or a container it is the WRONG guess stated as
 * fact. A relative reference is resolved by any consumer against the document
 * it just fetched, which IS, by construction, an origin the agent is
 * reachable at. Such an agent cannot register anyway (a signature needs an
 * absolute https agent URL), so nothing is lost by being honest.
 */
function resolvePrincipal(options: HandlerOptions, basePath: string): string {
  if (options.identity) return options.identity.issuer;
  const env =
    typeof process !== 'undefined' ? process.env?.WEBAGENTS_PUBLIC_URL : undefined;
  const base = (options.publicUrl || env || '').trim().replace(/\/+$/, '');
  return base ? `${base}${basePath}` : basePath || '/';
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
    // One CORS decision per request, used by every response below. `null`
    // means no CORS headers at all (S-226): the fallback used to stamp `*` on
    // everything, whatever the server in front of it had decided.
    const corsOrigin: string | null | undefined = options.originPolicy
      ? options.originPolicy(request.headers.get('origin'))
      : options.corsOrigin;
    
    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, {
        headers: getCorsHeaders(corsOrigin),
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
    const refusal = credentialFloor(request, getCorsHeaders(corsOrigin));
    if (refusal) return refusal;

    
    // Health check
    if (path === `${basePath}/health` && method === 'GET') {
      return jsonResponse({ status: 'healthy', agent: agent.name }, corsOrigin);
    }
    
    // Agent info
    if (path === `${basePath}/info` && method === 'GET') {
      return jsonResponse({
        name: agent.name,
        description: agent.description,
        capabilities: agent.getCapabilities(),
        tools: agent.getToolDefinitions?.() ?? [],
      }, corsOrigin);
    }
    
    // .well-known/agent.json: the self-naming agent card (card.ts), served
    // under `basePath` ONLY. It used to be served at the origin as well, for
    // the origin-level fallback the platform deleted with S-015; an origin
    // copy cannot name itself for a prefixed agent (`client_id` would say the
    // origin while the card says `/agents/mini`), so it went with the
    // fallback (W2 design section 9.1, 2026-09-17). With an empty `basePath`
    // this path IS the origin path.
    if (path === `${basePath}/.well-known/agent.json` && method === 'GET') {
      return jsonResponse(
        buildAgentCard(agent, {
          principal: resolvePrincipal(options, basePath),
          signs: options.identity !== undefined,
        }),
        corsOrigin,
      );
    }

    // .well-known/jwks.json: the key set every signature names (origin and
    // prefix). The entries are `{ kty, crv, x, kid: thumbprint, use }`, no
    // `alg`; `Cache-Control: max-age` is what the platform's refresh clamp
    // reads (W2 design section 4.4).
    if (
      (path === `${basePath}/.well-known/jwks.json` || path === '/.well-known/jwks.json') &&
      method === 'GET'
    ) {
      if (!options.identity) {
        return jsonResponse({ error: 'AOAuth not configured' }, corsOrigin, 404);
      }
      return new Response(JSON.stringify(options.identity.getJwks()), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600',
          ...getCorsHeaders(corsOrigin),
        },
      });
    }

    // .well-known/http-message-signatures-directory: what a `legacy-string`
    // signer's bare-origin `Signature-Agent` resolves to, at the ORIGIN
    // whatever `basePath` is, with the media type a verifier requires
    // (key-directory.ts holds the reasoning; 2026-09-19).
    if (isKeyDirectoryRequest(method, path)) {
      return keyDirectoryResponse([options.identity], getCorsHeaders(corsOrigin));
    }

    // .well-known/openid-configuration — AOAuth discovery
    if (path === `${basePath}/.well-known/openid-configuration` && method === 'GET') {
      if (!options.identity) {
        return jsonResponse({ error: 'AOAuth not configured' }, corsOrigin, 404);
      }
      return new Response(JSON.stringify(options.identity.getOpenIdConfiguration()), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600',
          ...getCorsHeaders(corsOrigin),
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
        // The bytes first, then the JSON: the access skill checks a signed
        // request's Content-Digest against exactly what arrived (ADR-0045).
        const raw = new Uint8Array(await request.arrayBuffer());
        const body = JSON.parse(new TextDecoder().decode(raw)) as {
          messages: Array<{ role: string; content: string }>;
          stream?: boolean;
          metadata?: Record<string, unknown>;
        };

        const msgs = body.messages.map((m) => ({ role: m.role as 'user' | 'system' | 'assistant', content: m.content }));

        // The platform sends `metadata: {chat_id, chat_type, platform, sender}`
        // on every routed turn. `sender` is what attributes the call to a
        // person (AuthSkill._serviceAuthInfo reads `metadata.sender.id` to
        // decide USER vs OWNER scope), so it must reach the context — a
        // service token alone names the ROUTER, not the caller. The request
        // itself goes in SESSION data, which only this server writes; the
        // body's `metadata` cannot reach it (ADR-0045). One builder for every
        // served route (`servedRunOptions`, S-345).
        const runOptions = servedRunOptions(request, raw, body.metadata);

        const requested = (body as { model?: unknown }).model;
        const model = typeof requested === 'string' && requested ? requested : agentModelName(agent);
        if (body.stream && typeof agent.runStreaming === 'function') {
          const gen = agent.runStreaming(msgs, runOptions);
          // Pull the first chunk here so an auth refusal is a 401, not a
          // 200 whose body happens to contain an error.
          const first = await gen.next();
          // A refusal written for the caller that arrives as the stream's
          // FIRST chunk (the proxy skill's "send a payment token", S-327) is
          // its status, before any stream, as a thrown one is.
          const firstError = !first.done && first.value.type === 'error'
            ? shownResponseError((first.value as { error?: unknown }).error)
            : null;
          if (firstError) {
            return jsonResponse({ error: { code: firstError.code, message: firstError.message } }, corsOrigin, firstError.status);
          }
          return streamCompletionsResponse(gen, corsOrigin, first.done ? undefined : first.value, completionMeta(model), `${agent.name} chat/completions`);
        }

        const result = await agent.run(msgs, runOptions);
        return jsonResponse(completionBody(result, model), corsOrigin);
      } catch (error) {
        if (isAuthError(error)) {
          return unauthorizedResponse((error as Error).message, corsOrigin, error, request);
        }
        // A refusal the run carried as a `response.error` written for the
        // caller (`details.shown`, S-327): its status and words.
        const carried = shownResponseError(error);
        if (carried) return jsonResponse({ error: { code: carried.code, message: carried.message } }, corsOrigin, carried.status);
        // A REFUSAL IS ITS STATUS (the ptypass-fixes lane, 2026-09-27; the
        // Python server's `_refusal`, S-236): a payment refusal in a
        // non-streaming call, or before a stream's first chunk, came back as
        // 500 `completions_error`. An error thrown to be shown keeps its own
        // status and message (`shownRefusal`).
        const refusal = shownRefusal(error);
        if (refusal) return jsonResponse(refusal.body, corsOrigin, refusal.status);
        // Not the error's own message (S-228): that is a provider's body or a
        // tool's paths as often as not. A fixed sentence and a reference the
        // server's log carries too (`error-reply.ts`).
        return jsonResponse({
          error: { code: 'completions_error', message: replyText(error, `${agent.name} chat/completions`) },
        }, corsOrigin, 500);
      }
    }

    // UAMP endpoint
    //
    // THE RUN GETS THE REQUEST (S-345, 2026-09-29), as `chat/completions`
    // above hands it over: this called `agent.processUAMP(body)` with
    // nothing else, so the turn ran on a context that carried no credential
    // headers and no request, an auth skill on the agent could neither
    // verify nor refuse the token the floor had let through, and a refusal
    // that did surface was a 500 `uamp_error`. The bytes are read first, for
    // the access skill's Content-Digest check (ADR-0045), the turn is bound
    // to its own context (`processUAMP` with options), and an auth
    // refusal is its status with the bearer challenge (`refusalResponse`).
    if (path === `${basePath}/uamp` && method === 'POST') {
      try {
        const raw = new Uint8Array(await request.arrayBuffer());
        const body = JSON.parse(new TextDecoder().decode(raw)) as ClientEvent[];

        const events: ServerEvent[] = [];
        for await (const event of agent.processUAMP(body, servedRunOptions(request, raw))) {
          events.push(event);
        }

        return jsonResponse(events, corsOrigin);
      } catch (error) {
        const refusal = refusalResponse(error, request);
        if (refusal) return jsonResponse(refusal.body, corsOrigin, refusal.status, refusal.headers);
        return jsonResponse({
          error: { code: 'uamp_error', message: replyText(error, `${agent.name} uamp`) },
        }, corsOrigin, 500);
      }
    }

    // UAMP streaming
    if (path === `${basePath}/uamp/stream` && method === 'POST') {
      try {
        const raw = new Uint8Array(await request.arrayBuffer());
        const body = JSON.parse(new TextDecoder().decode(raw)) as ClientEvent[];

        // The first event is pulled before the Response exists, so a refusal
        // raised in the run's connection hooks is a 401 and not a 200 whose
        // body tears (S-345).
        const events = agent.processUAMP(body, servedRunOptions(request, raw));
        const first = await events.next();
        return streamResponse(withFirst(first, events), corsOrigin);
      } catch (error) {
        const refusal = refusalResponse(error, request);
        if (refusal) return jsonResponse(refusal.body, corsOrigin, refusal.status, refusal.headers);
        return jsonResponse({
          error: { code: 'uamp_error', message: replyText(error, `${agent.name} uamp/stream`) },
        }, corsOrigin, 500);
      }
    }
    
    // Check agent HTTP endpoints
    const httpRegistry = (agent as { httpRegistry?: Map<string, HttpEndpoint> }).httpRegistry;
    if (httpRegistry) {
      const subPath = basePath && path.startsWith(basePath) ? path.slice(basePath.length) : path;
      const key = `${method}:${subPath}`;
      // `getHttpHandler` also matches `{param}` patterns (the A2A task
      // routes, 2026-09-26); the bare map is the fallback for an agent that
      // only carries the registry.
      const endpoint = agent.getHttpHandler?.(subPath, method) ?? httpRegistry.get(key);
      if (endpoint) {
        let context = createContextFromRequest(request);
        try {
          // WHO MAY CALL IT (S-242, 2026-09-25): the one gate
          // (`endpoint-gate.ts`). An endpoint with no scopes is open, as it
          // always was; a scoped one gets the caller the agent verified. The
          // body is read from a copy, so the handler still has its own.
          if (needsCaller(endpoint)) {
            const raw = new Uint8Array(await request.clone().arrayBuffer());
            context = identificationContext(context, inboundRequest(request, raw));
            const gate = await admitEndpoint(agent, endpoint, context);
            if (gate.refusal) return jsonResponse(gate.refusal.body, corsOrigin, gate.refusal.status, gate.refusal.headers);
          }
          // WHO PAYS FOR IT (2026-09-26): a priced endpoint goes through the
          // payment skill's paywall, which answers a standard x402 402 to an
          // unpaid request, verifies a payment before the handler runs and
          // settles after it answered (skills/payments/paywall.ts). A free
          // endpoint is served as it always was.
          const response = await serveThroughPaywall(agent, endpoint, request, () => endpoint.handler(request, context));
          // Add CORS headers
          const headers = new Headers(response.headers);
          for (const [key, value] of Object.entries(getCorsHeaders(corsOrigin))) {
            headers.set(key, value);
          }
          return new Response(response.body, {
            status: response.status,
            headers,
          });
        } catch (error) {
          return jsonResponse({
            error: { code: 'handler_error', message: replyText(error, `${agent.name} ${key}`) },
          }, corsOrigin, 500);
        }
      }
    }
    
    // Not found
    return jsonResponse({ error: { code: 'not_found', message: 'Not found' } }, corsOrigin, 404);
  };
}

/**
 * An OpenAI chat completion for a finished run: the shape and key order the
 * OpenAI API returns, and the Python server's answer byte for byte
 * (`server/core/app.py`, `openai_completion_body`, 2026-09-25). It carried
 * this SDK's own usage names (`input_tokens`, `output_tokens`) and no `model`,
 * so an OpenAI client read no usage from it.
 */
export function completionBody(result: RunResponse, model: string): Record<string, unknown> {
  const usage = result.usage;
  const meta = completionMeta(model);
  // The OpenAI key order, and the Python server's: id, object, created, model, choices, usage.
  return {
    id: meta.id,
    object: 'chat.completion',
    created: meta.created,
    model: meta.model,
    choices: [{ index: 0, message: { role: 'assistant', content: result.content }, finish_reason: 'stop' }],
    ...(usage
      ? { usage: { prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens, total_tokens: usage.total_tokens } }
      : {}),
    // A turn the agent ended by its tool budget (2026-09-28,
    // `core/tool-budget.ts`) says so beside OpenAI's own fields, as the
    // Python server's completion does; `finish_reason` keeps OpenAI's values.
    ...(result.finish && isAgentFinish(result.finish.reason) ? { webagents_finish: agentFinishBody(result.finish) } : {}),
  };
}

/** `webagents_finish` as both servers send it: the Python agent's shape. */
export function agentFinishBody(finish: NonNullable<RunResponse['finish']>): Record<string, unknown> {
  return {
    reason: finish.reason,
    blocked: false,
    retried: false,
    ...(finish.rounds !== undefined ? { rounds: finish.rounds } : {}),
    ...(finish.tool ? { tool: finish.tool } : {}),
  };
}

/** What one completion and every chunk of a streamed one carry: its id, the timestamp and the model (the Python server's twin). */
export interface CompletionMeta {
  id: string;
  created: number;
  model: string;
}

export function completionMeta(model: string): CompletionMeta {
  const now = Date.now();
  return { id: `chatcmpl-${now}`, created: Math.floor(now / 1000), model };
}

/** The model the agent's LLM skill runs, without its `provider/` prefix, for a completion's `model`. */
export function agentModelName(agent: IAgent): string {
  for (const skill of (agent as { skills?: Array<{ getCapabilities?: () => { id?: string; provider?: string } }> }).skills ?? []) {
    const capabilities = typeof skill.getCapabilities === 'function' ? skill.getCapabilities() : undefined;
    if (capabilities?.provider && typeof capabilities.id === 'string' && capabilities.id) {
      const slash = capabilities.id.indexOf('/');
      return slash === -1 ? capabilities.id : capabilities.id.slice(slash + 1);
    }
  }
  return agent.name;
}

/**
 * Request metadata for a run: the credential headers `AuthSkill` reads, plus
 * the platform's own request-body `metadata` (chat_id, chat_type, platform,
 * sender). Body keys are merged UNDER the headers so a request body can never
 * overwrite the header a credential was actually presented in.
 *
 * A credential named in the BODY is dropped (S-345, 2026-09-29): the merge
 * only replaced a body key when the request carried that header, so a body
 * could plant `x-api-key` or `authorization` for a header the request never
 * sent, and the auth skill reads `metadata` first. A credential is a header,
 * as the Python auth skill has it (it reads `context.request.headers` and
 * nothing else). No privilege was to be had, since whatever a body can plant
 * a caller can send as the header, but the run's metadata should say only
 * what was actually presented.
 *
 * THE ONE BUILDER for every served route (`servedRunOptions`): exported so
 * `WebAgentsServer` and the daemon build the same metadata rather than a
 * second one that drifts.
 */
export function buildRequestMetadata(
  request: Request,
  bodyMetadata?: Record<string, unknown>,
): Record<string, unknown> {
  const metadata: Record<string, unknown> = { ...(bodyMetadata ?? {}) };
  for (const name of CREDENTIAL_HEADERS) delete metadata[name];
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

/** A plain object, as a request body's `metadata` must be to be read at all. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * The run options every served route hands the agent (S-345, 2026-09-29):
 * the request metadata `buildRequestMetadata` builds (the credential headers
 * the auth skill reads, the platform's body `metadata` under them), the
 * platform's `chat_id` as the run's chat binding, and the request itself in
 * SESSION data, which only the server writes (ADR-0045), where the access
 * skill verifies a signature.
 *
 * `chat/completions` in this handler built exactly this and nothing else
 * did: `WebAgentsServer`'s built-in `chat/completions`, `uamp` and
 * `uamp/stream`, the `uamp` routes here and in `serve()`, and the daemon's
 * `chat/completions` (session data only, no metadata) ran the agent with no
 * credential on its metadata, so an auth skill on those doors saw nothing
 * and could refuse nothing behind the floor's presence check. One function,
 * so the next served route cannot leave the credential behind.
 */
export function servedRunOptions(request: Request, raw: Uint8Array, bodyMetadata?: unknown): RunOptions {
  const metadata = isRecord(bodyMetadata) ? bodyMetadata : undefined;
  const chatId = typeof metadata?.chat_id === 'string' && metadata.chat_id ? metadata.chat_id : undefined;
  return {
    metadata: buildRequestMetadata(request, metadata),
    ...(chatId ? { chatId } : {}),
    sessionData: { _inboundRequest: inboundRequest(request, raw) },
  };
}

/**
 * `rest`, preceded by its `first` step, which the route already pulled so a
 * refusal raised on the generator's first step is a status rather than a
 * torn 200 (the streamed `uamp` routes, S-345).
 */
export async function* withFirst<T>(
  first: IteratorResult<T, void>,
  rest: AsyncGenerator<T, void, unknown>,
): AsyncGenerator<T, void, unknown> {
  if (first.done) return;
  yield first.value;
  yield* rest;
}

/**
 * A refusal's answer: 401 `unauthorized`, unless the error carries its own
 * status and code (the access skill's 403 `forbidden`, or a signature
 * refusal's code, ADR-0045).
 */
/**
 * An error thrown to be shown (`isMeantToBeShown`: this SDK's refusals,
 * `PaymentRequiredError` among them) that carries an HTTP status, as that
 * status and a JSON body in this handler's shape; null for anything else. A
 * payment refusal keeps what the caller needs to pay (`accepts`).
 */
function shownRefusal(error: unknown): { status: number; body: Record<string, unknown> } | null {
  if (!isMeantToBeShown(error)) return null;
  const { status_code: snake, statusCode: camel, accepts } = error as { status_code?: unknown; statusCode?: unknown; accepts?: unknown };
  const status = typeof snake === 'number' ? snake : typeof camel === 'number' ? camel : undefined;
  if (status === undefined || status < 400 || status > 599) return null;
  return {
    status,
    body: {
      error: {
        code: status === 402 ? 'payment_required' : 'refused',
        message: (error as Error).message,
        ...(Array.isArray(accepts) ? { accepts } : {}),
      },
    },
  };
}

function unauthorizedResponse(message: string, corsOrigin?: string | null, error?: unknown, request?: Request): Response {
  const { statusCode, code } = (error ?? {}) as { statusCode?: unknown; code?: unknown };
  const status = statusCode === 403 ? 403 : 401;
  // The bearer challenge on a 401 (2026-09-29): `invalid_token` when the
  // request carried a credential, which behind the floor it did.
  return jsonResponse(
    { error: { code: typeof code === 'string' && code ? code : 'unauthorized', message } },
    corsOrigin,
    status,
    request ? refusalHeaders(status, request) : status === 401 ? challengeHeaders(true) : {},
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
/**
 * A JSON reply. A 401 always carries the bearer challenge (2026-09-29, RFC
 * 7235): the plain one unless `headers` names the variant, so a 401 that
 * reaches here from an error whose cause is not the caller's credential (a
 * provider's refusal of the owner's own key, carried as its status) still
 * says the scheme without calling the caller's token invalid.
 */
function jsonResponse(data: unknown, corsOrigin?: string | null, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...getCorsHeaders(corsOrigin),
      ...(status === 401 ? challengeHeaders() : {}),
      ...headers,
    },
  });
}

/**
 * Create SSE streaming response
 */
function streamResponse(
  events: AsyncGenerator<ServerEvent, void, unknown>,
  corsOrigin?: string | null
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
  corsOrigin?: string | null,
  /**
   * The first chunk, already pulled by the caller so that an auth refusal
   * raised on the generator's first step becomes a 401 instead of a 200 with
   * an error buried in the SSE body.
   */
  firstChunk?: { type: string; delta?: string; response?: unknown },
  /**
   * The completion's id, timestamp and model, stamped on EVERY chunk with
   * `object: chat.completion.chunk` (2026-09-26, the e2e run): the chunks
   * carried `choices` alone, so an OpenAI client could not tell which
   * completion, or which model, it was reading.
   */
  meta: CompletionMeta = completionMeta('unknown'),
  /** Where a failure is logged from (`replyText`'s `where`). */
  where = 'chat/completions',
): Response {
  const encoder = new TextEncoder();
  // The OpenAI key order, as the non-streamed answer has it.
  const stamp = (choices: unknown[]) => ({ id: meta.id, object: 'chat.completion.chunk', created: meta.created, model: meta.model, choices });

  const stream = new ReadableStream({
    async start(controller) {
      try {
        // AN ERROR IN THE STREAM IS SENT, NOT DROPPED (B1, 2026-09-28). A run
        // that failed after the first chunk (a provider's 402 or 429, a
        // refused platform credential) yields an `error` chunk, and this
        // passed `delta`s only: the caller got a `stop` and 200 with nothing
        // in it, and the server logged nothing. It is now OpenAI's in-stream
        // error, logged under a reference like any failed run (S-228: the
        // caller reads the reference, or the sentence of a refusal written
        // for it), and the stream ends there, as the Python server's does.
        let agentFinish: Record<string, unknown> | undefined;
        const emit = (chunk: { type: string; delta?: string; response?: unknown; error?: unknown }): boolean => {
          if (chunk.type === 'done') {
            // The turn's finish, when the agent's tool budget ended it (2026-09-28).
            const finish = (chunk.response as RunResponse | undefined)?.finish;
            if (finish && isAgentFinish(finish.reason)) agentFinish = agentFinishBody(finish);
          }
          if (chunk.type === 'delta' && chunk.delta) {
            const data = stamp([{ index: 0, delta: { content: chunk.delta }, finish_reason: null }]);
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
          } else if (chunk.type === 'error') {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(streamErrorBody(chunk.error, where))}\n\n`));
            return false;
          }
          return true;
        };
        let going = firstChunk ? emit(firstChunk) : true;
        if (going) {
          for await (const chunk of gen) {
            if (!emit(chunk)) {
              going = false;
              break;
            }
          }
        }
        if (going) {
          const done = { ...stamp([{ index: 0, delta: {}, finish_reason: 'stop' }]), ...(agentFinish ? { webagents_finish: agentFinish } : {}) };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(done)}\n\n`));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        }
        controller.close();
      } catch (error) {
        // Thrown after the stream began: the same error event, not a torn body.
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(streamErrorBody(error, where))}\n\n`));
          controller.close();
        } catch {
          controller.error(error);
        }
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
 * OpenAI's in-stream error, `{error: {message, type, code}}` (B1,
 * 2026-09-28): a refusal written for the caller keeps its words and code;
 * anything else is the fixed sentence and a reference, logged with the whole
 * error (`replyText`). The Python server sends the same shape (fixture
 * `cli/final_sdk_serve_model.json`, `stream_error`).
 */
export function streamErrorBody(error: unknown, where: string): { error: { message: string; type: string; code: string } } {
  const shown = shownResponseError(error);
  if (shown) return { error: { message: shown.message, type: 'invalid_request_error', code: shown.code } };
  if (isMeantToBeShown(error)) {
    const { code } = (error ?? {}) as { code?: unknown };
    return { error: { message: (error as Error).message, type: 'invalid_request_error', code: typeof code === 'string' && code ? code : 'refused' } };
  }
  return { error: { message: replyText(error, where), type: 'server_error', code: 'completions_error' } };
}

/**
 * Get CORS headers.
 *
 * The x402 and MPP payment headers are always in the allow and expose lists
 * (2026-09-26): a browser client cannot read a `PAYMENT-REQUIRED` it is not
 * allowed to see, nor send a `PAYMENT-SIGNATURE` the preflight refused, and
 * the preflight is answered above before any route is matched, so the lists
 * cannot depend on the endpoint. Listing them for a free endpoint costs
 * nothing.
 */
function getCorsHeaders(origin?: string | null): Record<string, string> {
  if (origin === null) return {};
  return {
    ...(origin && origin !== '*' ? { Vary: 'Origin' } : {}),
    'Access-Control-Allow-Origin': origin || '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': X402_CORS_ALLOW_HEADERS.join(', '),
    'Access-Control-Expose-Headers': X402_CORS_EXPOSE_HEADERS.join(', '),
  };
}
