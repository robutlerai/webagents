/**
 * The credential floor — ONE predicate, ONE path set, ONE guard, for every
 * TypeScript server class.
 *
 * WHY THIS MODULE EXISTS AT ALL. The floor used to be a per-route `if` in the
 * one branch someone remembered. Four separate doors to the same billable
 * endpoint were found that way, in three rounds, across two SDKs:
 *
 *   1. `POST {basePath}/chat/completions` in `createFetchHandler` — guarded.
 *   2. the Python dynamic-agent catch-all — found later, guarded later.
 *   3. the Python per-skill `@http` static mount — found later still.
 *   4. `POST {basePath}/uamp` and `/uamp/stream` in `createFetchHandler`, and
 *      EVERY billable route on `WebAgentsServer` — anonymous 200s, proven with
 *      invocation counters on `agent.run` / `agent.processUAMP`.
 *
 * The recurrence is the finding. A per-route check means every new route and
 * every new server class silently reintroduces the hole, and the tests only
 * ever cover the doors somebody thought of. So the decision — "is this request
 * allowed to reach the model?" — lives here, is made from the request line
 * alone (method + path, no body), and is applied at exactly ONE point per
 * server class:
 *
 *   * `createFetchHandler` (handler.ts): the top of the returned handler,
 *     above every branch AND above the `httpRegistry` dispatch.
 *   * `createAgentApp` / `serve` (node.ts): a `app.use('*')` Hono middleware
 *     registered before any route.
 *   * `WebAgentsServer` (multi.ts): the same middleware in `createApp`.
 *
 * A new billable route added to any of those three is covered the moment it is
 * added, because the guard is upstream of route dispatch rather than inside a
 * branch. What still has to be maintained by hand is `BILLABLE_PATHS`: nothing
 * can infer that a NEW path reaches the model. That is what the enumerating
 * test in `tests/unit/server/billable-routes.test.ts` is for — it walks the
 * real route tables and fails on any mounted route that is neither in this set
 * nor in an explicit public allow-list, so a new route cannot be added without
 * someone classifying it.
 */

/**
 * Header names a caller can present a credential in.
 *
 * Kept byte-identical to `CREDENTIAL_HEADERS` in
 * `python/webagents/server/core/credential_floor.py`; the two SDKs serve the
 * same endpoint and must not disagree about what counts as "authenticated
 * enough to reach the model". `tests/unit/server/floor-parity.test.ts` reads
 * the Python file and asserts the two lists are equal.
 */
// `signature-input` since 2026-09-25 (ADR-0045): an agent that only SIGNS its
// request (Web Bot Auth) carries no bearer, and the access skill verifies the
// signature behind this floor. Presence is all the floor checks, as for the others.
export const CREDENTIAL_HEADERS = ['authorization', 'x-api-key', 'x-owner-assertion', 'signature-input'] as const;

/**
 * Sub-paths that ARE a billable model endpoint, whichever door they are
 * reached through: the built-in branch, a Hono route, or a transport skill's
 * `@http` handler mounted at the same subpath.
 *
 * `uamp` and `uamp/stream` are here for the same reason `chat/completions` is:
 * both call `agent.processUAMP`, which runs the model on the owner's credit —
 * the same provider, the same money, a different wire format. `uamp/completions`
 * is the Python `CompletionsTransportSkill` subpath; it is listed here too so
 * the two SDKs cannot drift apart on the set, even though no TypeScript skill
 * currently mounts it.
 *
 * `a2a`, `tasks` and `acp` were found by the enumerating test once it started
 * walking the handler REGISTRIES instead of the paths the floor already knew:
 *
 *   - `a2a` — `A2ATransportSkill` declares `@http({ path: '/a2a', method:
 *     'POST' })` in `src/skills/transport/a2a/skill.ts`, the A2A JSON-RPC
 *     endpoint, whose sends run the agent (`agent.run` since v1.0; the pre-v1
 *     `tasks/send` ran `processUAMP`). It answered an anonymous 200 with the
 *     model reached, because the skill was not in the enumerating test's
 *     fixture and its route was therefore never classified. The two
 *     `a2a/message:*` entries are the same run over the HTTP+JSON binding.
 *   - `tasks` WAS here: the pre-v1 Python `A2ATransportSkill` mounted
 *     `@http("/tasks", method="post")`, its 0.2.1 REST binding, with public
 *     `GET /tasks/{task_id}` reads beside it. Both SDKs serve A2A v1.0 at
 *     `a2a` now (2026-09-26) and the 0.2.1 routes are gone, so the entry
 *     went with them.
 *   - `acp` WAS here (2026-09-26): the Python `ACPTransportSkill` mounted
 *     `@http("/acp", method="post")`, whose `session/prompt` reached
 *     `process_uamp`. ACP is served over stdio now (`webagents acp`, plan
 *     item 1.6) and no SDK mounts an ACP route, so the entry is gone with it.
 *
 * Kept identical to `BILLABLE_PATHS` in
 * `python/webagents/server/core/credential_floor.py` — asserted by
 * `tests/unit/server/floor-parity.test.ts`.
 */
export const BILLABLE_PATHS = [
  'chat/completions',
  'v1/chat/completions',
  'uamp',
  'uamp/stream',
  'uamp/completions',
  'a2a',
  // The A2A v1.0 HTTP+JSON sends (2026-09-26): the same run as `POST /a2a`.
  // The pre-v1 `tasks` route is gone with the 0.2.1 implementation.
  'a2a/message:send',
  'a2a/message:stream',
] as const;

/**
 * The floor applies to POST only.
 *
 * Every billable path is POST-served, and restricting the method is what keeps
 * the SUFFIX match below from swallowing an unrelated route: `GET /uamp` is the
 * agent-info page of an agent that happens to be named `uamp`, not a model
 * call. An OPTIONS preflight is likewise never billable and must not be
 * refused, or a browser can never reach the endpoint at all.
 */
export const BILLABLE_METHODS = ['POST'] as const;

/**
 * WebSocket sub-paths that reach the model. `UAMPTransportSkill` registers
 * `@websocket({ path: '/uamp' })`, and the upgrade handler dispatches to it
 * with no credential check of any kind — the same open billable endpoint as
 * `POST /uamp`, over a different protocol.
 *
 * `realtime` and `acp/stream` were the two Python sockets the previous round
 * left open, and they were left open for a structural reason worth writing
 * down: both enumerating tests expanded the billable set THROUGH the WebSocket
 * catch-all instead of walking the agent's WebSocket registry, so a socket was
 * only ever probed if it was already in this array. An array that can only
 * confirm itself is a memo, not a test. Both tests now walk the registry
 * (`agent.listWebSocketEndpoints()` here, `agent.get_all_websocket_handlers()`
 * in Python), so a new `@websocket` handler fails classification on the day it
 * is written.
 *
 * `acp/stream` is gone with the HTTP ACP endpoint (2026-09-26, plan item 1.6):
 * ACP is stdio, and the socket no longer exists in either SDK. `realtime` is
 * served by both: the Python `@websocket("/realtime")`, and since the same day
 * the TypeScript `RealtimeTransportSkill`'s `@websocket({ path: '/realtime' })`
 * (plan item 1.5), so `serve()` answers a voice session behind this floor.
 */
export const BILLABLE_WS_PATHS = ['uamp', 'realtime'] as const;

/**
 * THE OTHER HALF OF THE CLASSIFICATION. Agent-surface sub-paths that are
 * deliberately anonymous.
 *
 * A path set on its own cannot make a new route fail by default — it can only
 * confirm the routes already in it. What makes the enumerating test a tripwire
 * is that every handler discovered in the agent's registries must land in
 * EXACTLY ONE of two declared sets: `BILLABLE_PATHS` or this one. A handler in
 * neither is a test failure, so the author of the next transport skill has to
 * say which it is before the suite goes green.
 *
 * That is why this list lives in the shipped module next to the billable set
 * rather than in the test file: it is a security declaration, it must not drift
 * between the two SDKs, and the parity tests assert it does not. Adding a line
 * here is a decision that a route is safe to serve to anyone who can reach the
 * port. It should feel like one.
 *
 * The reasons, in order:
 *
 *   - the empty string — the agent info page at the mount root; static
 *     metadata.
 *   - capabilities, models, v1/models, info — static descriptions of the agent.
 *     They are also how a client discovers the billable endpoints, so gating
 *     them would break discovery without protecting anything.
 *   - health, metrics — liveness and counters, deliberately reachable by a load
 *     balancer that has no credential.
 *   - the four .well-known documents — the registration card, the A2A v1.0
 *     card beside it (`agent-card.json`, a peer must read it before it can
 *     authenticate), JWKS and OIDC discovery, all of which are useless
 *     unless they are public.
 *   - the well-known signatures directory — the same public keys as the JWKS,
 *     under the path and media type a `legacy-string` signer's bare origin
 *     resolves to (`key-directory.ts`). Served at the ORIGIN only.
 *   - NOT command or command/-path, the Python slash-command surface (S-235,
 *     2026-09-25): not billable, but not harmless either (owner commands
 *     restore checkpoints and install plugins), so it needs a credential like
 *     any agent route. This SDK serves no command route.
 *   - NOT the A2A task routes (2026-09-26): `a2a/tasks` and its `{id}`
 *     reads, cancel and subscribe never call the model, but a task belongs
 *     to the caller that made it, so they sit in `CREDENTIALED_SUBPATHS`.
 *     The pre-v1 public `tasks/{task_id}` reads went with the 0.2.1
 *     implementation.
 *
 * Framework chrome that is not agent surface — FastAPI docs, openapi.json, the
 * server-level readiness probes, the Hono agents listing — is allow-listed in
 * the test files instead, because it differs per framework and is not part of
 * the cross-SDK contract this array encodes.
 *
 * Kept identical to `PUBLIC_SUBPATHS` in
 * `python/webagents/server/core/credential_floor.py`.
 */
export const PUBLIC_SUBPATHS = [
  '',
  'capabilities',
  'models',
  'v1/models',
  'info',
  'health',
  'metrics',
  '.well-known/agent.json',
  // The A2A v1.0 card beside the registration card (2026-09-26): public
  // for the same reason, a peer must read it before it can authenticate.
  '.well-known/agent-card.json',
  '.well-known/jwks.json',
  '.well-known/openid-configuration',
  '.well-known/http-message-signatures-directory',
] as const;

/**
 * Agent routes that need a credential although they cannot reach the model:
 * the Python slash-command surface (S-235, 2026-09-25), whose commands restore
 * checkpoints, install plugins and act as the owner. The floor refuses them
 * without a credential, for every method. This SDK serves no such route today;
 * the list is shared so a future one is guarded the same way.
 *
 * Kept identical to `CREDENTIALED_SUBPATHS` in
 * `python/webagents/server/core/credential_floor.py`.
 */
export const CREDENTIALED_SUBPATHS = [
  'command',
  'command/{path:path}',
  // The A2A v1.0 task routes (2026-09-26): they read, cancel and replay
  // tasks the billable sends created, and a task belongs to the caller that
  // made it, so an anonymous request has nothing to read. The floor refuses
  // them for every method (the reads are GETs), and the skill then names the
  // caller and answers only the tasks of that caller. The pre-v1 public
  // `tasks/{task_id}` reads are gone with the 0.2.1 implementation. (No
  // semicolon and no apostrophe inside this block: the Python parity test
  // cuts it at the first semicolon and pairs quotes with a regex.)
  'a2a/tasks',
  'a2a/tasks/{id}',
  'a2a/tasks/{id}:cancel',
  'a2a/tasks/{id}:subscribe',
] as const;

/**
 * The WebSocket half of the same declaration, and it is EMPTY on purpose.
 *
 * Every `@websocket` handler either of these SDKs ships today reaches the
 * model, so every one of them is in `BILLABLE_WS_PATHS`. This array is where a
 * socket that genuinely does not — a presence feed, a log tail — would be
 * declared. It exists so that the classification has somewhere to put such a
 * handler other than silence, which is what `/realtime` and `/acp/stream` got
 * for two rounds.
 *
 * It is deliberately SEPARATE from `PUBLIC_SUBPATHS`: an HTTP path being safe
 * says nothing about a socket at the same name. The probe that proved the old
 * test blind was a `@websocket` handler at `/live`, and `live` is exactly the
 * kind of name that is an innocuous HTTP probe.
 *
 * Kept identical to `PUBLIC_WS_SUBPATHS` in
 * `python/webagents/server/core/credential_floor.py`.
 */
export const PUBLIC_WS_SUBPATHS = [] as const;

/**
 * Query parameters a WebSocket client can present a credential in. A browser
 * cannot set headers on a WebSocket handshake, which is why the existing
 * upgrade path already reads `?token=`; the floor accepts the same thing
 * rather than locking out the callers the server documents.
 */
export const WS_CREDENTIAL_QUERY_PARAMS = ['token', 'access_token', 'api_key'] as const;

export const UNAUTHORIZED_MESSAGE =
  'Authentication required: send the platform service token or an api key ' +
  'in the Authorization header.';

/** The minimum a header source has to do for the floor to read it. */
export interface HeaderSource {
  get(name: string): string | null | undefined;
}

/**
 * True when the request carries SOMETHING that can be authenticated.
 *
 * This is a FLOOR, not the authentication itself: `AuthSkill` (when the agent
 * has one) verifies the credential inside the run's `on_connection` hook and
 * throws when it does not check out. The floor exists so that an agent with no
 * AuthSkill — the quickstart shape — is not an anonymous, BILLABLE model
 * endpoint for anyone who can reach the port.
 *
 * A bare `Bearer` with nothing after it is not a credential.
 */
export function hasCredential(request: { headers: HeaderSource }): boolean {
  for (const name of CREDENTIAL_HEADERS) {
    const value = request.headers.get(name);
    if (value && value.trim() && value.trim().toLowerCase() !== 'bearer') return true;
  }
  return false;
}

/** Strip leading/trailing slashes so a path can be compared segment-wise. */
function normalizePath(pathname: string): string {
  return pathname.replace(/^\/+/, '').replace(/\/+$/, '');
}

/**
 * True when `pathname` ends in one of `BILLABLE_PATHS`.
 *
 * Suffix matching rather than equality because the same subpath is mounted
 * under every prefix the SDK supports and the floor must not care which:
 * `/chat/completions` (bare `createFetchHandler`), `/agents/og/chat/completions`
 * (`serve()` with a basePath), `/api/agents/m1/v1/chat/completions`
 * (`WebAgentsServer` with a basePath). Matching the tail is what makes ONE
 * check cover all of them.
 */
export function isBillablePath(pathname: string): boolean {
  const path = normalizePath(pathname);
  return BILLABLE_PATHS.some((billable) => path === billable || path.endsWith(`/${billable}`));
}

/** True when `pathname` ends in a WebSocket sub-path that reaches the model. */
export function isBillableWebSocketPath(pathname: string): boolean {
  const path = normalizePath(pathname);
  return BILLABLE_WS_PATHS.some((billable) => path === billable || path.endsWith(`/${billable}`));
}

/** True when `pathname` is under one of `CREDENTIALED_SUBPATHS`, whatever prefix it is mounted under. */
export function isCredentialedPath(pathname: string): boolean {
  const path = `/${normalizePath(pathname)}/`;
  return CREDENTIALED_SUBPATHS.some((candidate) => path.includes(`/${candidate.split('/{')[0]}/`));
}

/** The whole floor decision, from the request line alone. No body is read. */
export function isBillableRequest(method: string, pathname: string): boolean {
  return (
    (BILLABLE_METHODS as readonly string[]).includes(method.toUpperCase()) &&
    isBillablePath(pathname)
  );
}

/**
 * THE GUARD. Returns a 401 `Response` when this request must be refused, and
 * `null` when it may proceed.
 *
 * Deliberately takes only the `Request`: it decides from method and path, so it
 * can run above route dispatch and above `await request.json()`. Running it
 * before the body is read is a property in its own right — an anonymous caller
 * must not be able to make the server parse arbitrary bytes, and an
 * anonymous-plus-malformed request must observe 401 on both SDKs rather than
 * 401 on one and a parser error on the other.
 */
export function credentialFloor(
  request: Request,
  extraHeaders: Record<string, string> = {},
): Response | null {
  let pathname: string;
  try {
    pathname = new URL(request.url).pathname;
  } catch {
    return null;
  }
  if (!isBillableRequest(request.method, pathname) && !isCredentialedPath(pathname)) return null;
  if (hasCredential(request)) return null;
  return unauthorizedResponse(extraHeaders);
}

/** The realm of the RFC 6750 bearer challenge (`bearerChallenge`), the Python floor's `BEARER_REALM`. */
export const BEARER_REALM = 'webagents';

/**
 * The `WWW-Authenticate` value of a 401 (RFC 6750 section 3, 2026-09-29):
 * `Bearer realm="webagents"` when the request carried no credential, with
 * `error="invalid_token"` when it carried one that was refused. `mcp serve
 * --http` answered 401 with no challenge at all, in both SDKs, so an MCP
 * client had nothing to read the scheme from; the Python twin is
 * `bearer_challenge` in `credential_floor.py`, and
 * `python/tests/fixtures/credential_floor/www_authenticate.json` pins both
 * values, and the doors.
 *
 * EVERY 401 CARRIES IT (2026-09-29, the same day, once the MCP route had
 * one). RFC 7235 section 3.1 makes the header a MUST on every 401, and the
 * MCP route was the only one of these servers' 401s that sent it: the
 * floor's own refusal (`credentialFloor`, so `createFetchHandler`, `serve()`,
 * `WebAgentsServer` and the daemon), the gate's `NEEDS_CALLER`, an auth
 * skill's refusal on `chat/completions`, the raw `HTTP/1.1 401` on a billable
 * WebSocket upgrade and the daemon's registry routes all answered bare. So
 * the header is no longer something a route remembers to add:
 * `unauthorizedResponse` carries the plain challenge by default, and a
 * refusal answered for a request goes through `refusalHeaders`, which reads
 * the ONE rule for the variant, `refused` when the request carried a
 * credential (`hasCredential`, the floor's own predicate): a token was
 * presented and not accepted, whoever refused it, and RFC 6750 calls that
 * `invalid_token`.
 */
export function bearerChallenge(refused = false): string {
  const challenge = `Bearer realm="${BEARER_REALM}"`;
  return refused ? `${challenge}, error="invalid_token"` : challenge;
}

/** The header a 401 carries, as a header record for a response. */
export function challengeHeaders(refused = false): Record<string, string> {
  return { 'WWW-Authenticate': bearerChallenge(refused) };
}

/**
 * The challenge for a 401 answered to `request` (anything `hasCredential`
 * reads): `invalid_token` when it carried a credential, plain otherwise.
 */
export function challengeFor(request: { headers: HeaderSource }): string {
  return bearerChallenge(hasCredential(request));
}

/**
 * The headers a refusal of `request` with `status` carries: the challenge
 * for a 401 (`challengeFor`), nothing for any other status. The one call
 * every route answering a gate or hook refusal makes, so a 401 cannot be
 * answered bare by a route that forgot.
 */
export function refusalHeaders(status: number, request: { headers: HeaderSource }): Record<string, string> {
  return status === 401 ? { 'WWW-Authenticate': challengeFor(request) } : {};
}

/**
 * The one 401 body both SDKs answer with. The `WWW-Authenticate` header
 * carries the plain challenge unless `extraHeaders` names one: this is the
 * floor's answer to a request that carried nothing, so there is no token to
 * call invalid.
 */
export function unauthorizedResponse(extraHeaders: Record<string, string> = {}): Response {
  return new Response(
    JSON.stringify({ error: { code: 'unauthorized', message: UNAUTHORIZED_MESSAGE } }),
    {
      status: 401,
      headers: { 'Content-Type': 'application/json', ...challengeHeaders(), ...extraHeaders },
    },
  );
}

/**
 * The raw reply a billable WebSocket upgrade with no credential gets
 * (`serve()` and `WebAgentsServer` write it to the socket before the
 * handshake): a 401 with the plain challenge, as any other 401.
 */
export function unauthorizedUpgradeReply(): string {
  return `HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: ${bearerChallenge()}\r\n\r\n`;
}

/**
 * The WebSocket half of the same decision, for an upgrade request.
 *
 * `headers` is the handshake's header bag; `url` is the request URL, whose
 * query string is also consulted because a browser cannot set headers on a
 * WebSocket handshake.
 *
 * Returns true when the handshake must be refused.
 */
export function webSocketUpgradeIsRefused(
  pathname: string,
  headers: HeaderSource,
  searchParams?: { get(name: string): string | null },
): boolean {
  if (!isBillableWebSocketPath(pathname)) return false;
  if (hasCredential({ headers })) return false;
  if (searchParams) {
    for (const param of WS_CREDENTIAL_QUERY_PARAMS) {
      const value = searchParams.get(param);
      if (value && value.trim()) return false;
    }
  }
  return true;
}
