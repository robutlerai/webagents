/**
 * Who may call a scoped `@http` or `@websocket` endpoint (S-242, 2026-09-25).
 *
 * The decorators store `scopes`, and the docs said `scopes: ['owner']`
 * restricts callers, but no server looked at them: `createFetchHandler`, the
 * Hono app `serve()` builds and `WebAgentsServer` all called the handler for
 * anyone who reached it. The Python servers had the same hole on their static
 * mount (S-243).
 *
 * ONE GATE, the same in both SDKs (`python/webagents/server/core/endpoint_gate.py`,
 * answers pinned by `python/tests/fixtures/endpoint_gate/refusals.json`):
 *
 *   - An endpoint with no scopes, or `all`, is OPEN, exactly as before:
 *     nothing is asked of the caller and nothing extra runs.
 *   - Otherwise the agent identifies the caller the way a chat turn does
 *     (`BaseAgent.identifyCaller`: its auth skills and its access block, and
 *     nothing else, so no payment is taken for an endpoint call), and the one
 *     scope rule (`core/scopes.ts`) decides.
 *   - A refusal raised while identifying keeps its own status and body: a
 *     signature that does not verify is 401, a caller the access block keeps
 *     out is 403, a credential the auth skill cannot verify is 401.
 *   - Then: a caller the agent could not verify gets 401 `unauthorized`, and a
 *     verified caller the scopes do not include gets 403 `forbidden`.
 *
 * A scoped endpoint's handler gets the identified context. The servers' own
 * context builders count any `Authorization: Bearer` header as
 * `authenticated`, which is why a scoped endpoint never starts from one of
 * them as it is (`identificationContext` resets that); an open endpoint keeps
 * the context it always had.
 *
 * AN `@http` ENDPOINT'S `auth` MODE TOO (S-247, 2026-09-25). `HttpAuthMode`
 * says what the host must do before the handler runs, and when the SDK serves
 * the agent the SDK is the host, but none of its servers read it: an
 * `auth: 'session'` endpoint ("owner-only admin pages") answered anyone.
 * `authModeScope` turns the modes into this gate's terms: `session` needs the
 * owner, `portal_token` a caller the agent verified (the auth skill checks
 * platform tokens), both on top of the endpoint's own scopes.
 * `visitor_session` refuses no one for being anonymous: a caller that sends a
 * credential is identified (and refused if it does not verify), and one that
 * sends none reaches the handler as `{ authenticated: false }`. `public` and
 * `signature` ask nothing (a signature endpoint checks its provider itself).
 */

import { createDefaultAuthInfo } from '../core/context';
import { callerScopes, scopeAllows, type RequiredScope } from '../core/scopes';
import type { AuthInfo, Context, IAgent } from '../core/types';
import { CREDENTIAL_HEADERS } from './credential-floor';

export const NEEDS_CALLER = 'This endpoint needs a caller this agent can verify, and the request carries none.';
export const NOT_OPEN = 'This endpoint is not open to this caller.';

/** A refusal's status and JSON body. */
export interface GateRefusal {
  status: 401 | 403;
  body: { error: { code: string; message: string } };
}

/** The request as the access skill verifies it: method, path and query, headers, body bytes. */
export interface InboundRequestShape {
  method: string;
  target: string;
  headers: Record<string, string>;
  body: Uint8Array;
}

/**
 * The session-data key the access skill reads the request from
 * (`INBOUND_REQUEST_KEY` in `skills/access/skill.ts`, not imported so the
 * fetch handler keeps no import edge into the access skill's crypto).
 */
const INBOUND_REQUEST_KEY = '_inboundRequest';

/** Whether an endpoint declared `scopes` is open to an anonymous caller. */
export function isOpen(scopes: RequiredScope): boolean {
  return scopeAllows(scopes, []);
}

/**
 * `AuthenticationError` / `AuthorizationError` are matched by NAME so this
 * module keeps no import edge into `skills/auth` (which pulls the whole JWKS
 * stack into every fetch-handler bundle).
 */
export function isAuthError(error: unknown): boolean {
  const name = (error as { name?: string } | null)?.name;
  return name === 'AuthenticationError' || name === 'AuthorizationError';
}

/**
 * An auth or access refusal's status and body, or null for any other error.
 * Shared by `createFetchHandler`, the daemon's route and this gate, which all
 * answer the same way.
 */
export function refusalResponse(error: unknown): GateRefusal | null {
  if (!isAuthError(error)) return null;
  const { statusCode, code } = error as { statusCode?: unknown; code?: unknown };
  return {
    status: statusCode === 403 ? 403 : 401,
    body: { error: { code: typeof code === 'string' && code ? code : 'unauthorized', message: (error as Error).message } },
  };
}

/** What the access skill verifies: method, the path and query, headers, and the body bytes. */
export function inboundRequest(request: Request, body: Uint8Array): InboundRequestShape {
  const url = new URL(request.url);
  const headers: Record<string, string> = {};
  request.headers.forEach((value, key) => {
    headers[key.toLowerCase()] = value;
  });
  return { method: request.method, target: `${url.pathname}${url.search}`, headers, body };
}

/**
 * A websocket upgrade as the identity skills read it. A browser cannot set
 * headers on an upgrade, so a `?token` is read as the bearer credential when
 * no Authorization header came with it.
 */
export function inboundUpgrade(req: import('http').IncomingMessage): InboundRequestShape {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined) continue;
    headers[key.toLowerCase()] = Array.isArray(value) ? value.join(', ') : value;
  }
  const token = url.searchParams.get('token');
  if (token && !headers.authorization) headers.authorization = `Bearer ${token}`;
  return { method: req.method ?? 'GET', target: `${url.pathname}${url.search}`, headers, body: new Uint8Array() };
}

/**
 * `context`, made ready for the identity skills: anonymous until one of them
 * says otherwise, the credential headers on `metadata` (where the auth skill
 * reads them) and the request in session data (where the access skill does).
 */
export function identificationContext(context: Context, inbound: InboundRequestShape): Context {
  const anonymous = createDefaultAuthInfo();
  const setter = (context as { setAuth?: (auth: AuthInfo) => void }).setAuth;
  if (typeof setter === 'function') setter.call(context, anonymous);
  else (context as { auth: AuthInfo }).auth = anonymous;
  const metadata: Record<string, unknown> = { ...(context.metadata ?? {}) };
  for (const name of CREDENTIAL_HEADERS) {
    const value = inbound.headers[name];
    if (value) metadata[name] = value;
  }
  context.metadata = metadata;
  context.set(INBOUND_REQUEST_KEY, inbound);
  return context;
}

/**
 * Whether the caller of `context` may use an endpoint declared `scopes`.
 * `refusal` is set when not; otherwise `context` names the caller. For a
 * scoped endpoint, build `context` with `identificationContext` first.
 */
export async function admit(
  agent: IAgent,
  scopes: RequiredScope,
  context: Context,
): Promise<{ context: Context; refusal?: GateRefusal }> {
  return admitEndpoint(agent, { scopes }, context);
}

/** What an `@http` endpoint's `auth` mode asks of its caller in this gate's terms (see the file comment). */
export function authModeScope(auth: string | undefined): string | undefined {
  if (auth === 'session') return 'owner';
  if (auth === 'portal_token') return 'user';
  return undefined;
}

/** An endpoint as the gate reads it: its scopes and, for `@http`, its `auth` mode. */
export interface GatedEndpoint {
  scopes?: RequiredScope;
  auth?: string;
}

/** Whether serving `endpoint` needs to know who is calling. */
export function needsCaller(endpoint: GatedEndpoint): boolean {
  return !isOpen(endpoint.scopes) || authModeScope(endpoint.auth) !== undefined || endpoint.auth === 'visitor_session';
}

/**
 * `admit` for an endpoint: its scopes and its `auth` mode both (see the file
 * comment). Build `context` with `identificationContext` first.
 */
export async function admitEndpoint(
  agent: IAgent,
  endpoint: GatedEndpoint,
  context: Context,
): Promise<{ context: Context; refusal?: GateRefusal }> {
  if (!needsCaller(endpoint)) return { context };
  const modeScope = authModeScope(endpoint.auth);
  if (endpoint.auth === 'visitor_session' && isOpen(endpoint.scopes)) {
    // Anonymous is an answer here, not a refusal: identify only a caller that
    // sent something to identify.
    const sent = CREDENTIAL_HEADERS.some((name) => Boolean(context.metadata?.[name]));
    if (!sent) return { context };
  }
  const required: RequiredScope[] = [endpoint.scopes, ...(modeScope ? [modeScope] : [])];
  try {
    await agent.identifyCaller?.(context);
  } catch (error) {
    const refusal = refusalResponse(error);
    if (refusal) return { context, refusal };
    throw error;
  }
  const held = callerScopes(context.auth);
  if (required.every((scopes) => scopeAllows(scopes, held))) return { context };
  if (!context.auth?.authenticated) {
    return { context, refusal: { status: 401, body: { error: { code: 'unauthorized', message: NEEDS_CALLER } } } };
  }
  return { context, refusal: { status: 403, body: { error: { code: 'forbidden', message: NOT_OPEN } } } };
}

/** Refuse a websocket upgrade with the gate's status and JSON body. */
export function refuseUpgrade(socket: import('stream').Duplex, refusal: GateRefusal): void {
  const body = JSON.stringify(refusal.body);
  const reason = refusal.status === 403 ? 'Forbidden' : 'Unauthorized';
  socket.write(
    `HTTP/1.1 ${refusal.status} ${reason}\r\n` +
      'Content-Type: application/json\r\n' +
      `Content-Length: ${new TextEncoder().encode(body).length}\r\n` +
      'Connection: close\r\n\r\n' +
      body,
  );
  socket.destroy();
}
