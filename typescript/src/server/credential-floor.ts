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
export const CREDENTIAL_HEADERS = ['authorization', 'x-api-key', 'x-owner-assertion'] as const;

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
 *     'POST' })` in `src/skills/transport/a2a/skill.ts`, a JSON-RPC envelope
 *     whose `tasks/send` runs `this.agent.processUAMP`. It answered an
 *     anonymous 200 with the model reached, because the skill was not in the
 *     enumerating test's fixture and its route was therefore never classified.
 *   - `tasks` — the Python `A2ATransportSkill` mounts `@http("/tasks",
 *     method="post")`, which calls `agent.process_uamp` directly. Only the POST
 *     is billable; the `GET /tasks/{task_id}` status reads are in
 *     `PUBLIC_SUBPATHS`, which is part of why the floor is POST-only.
 *   - `acp` — the Python `ACPTransportSkill` mounts `@http("/acp",
 *     method="post")`, whose `session/prompt` reaches `process_uamp`. Neither
 *     is a TypeScript route today; both are listed here for the same reason
 *     `uamp/completions` is.
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
  'tasks',
  'acp',
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
 * `realtime` and `acp/stream` are the two Python sockets the previous round
 * left open, and they were left open for a structural reason worth writing
 * down: both enumerating tests expanded the billable set THROUGH the WebSocket
 * catch-all instead of walking the agent's WebSocket registry, so a socket was
 * only ever probed if it was already in this array. An array that can only
 * confirm itself is a memo, not a test. Both tests now walk the registry
 * (`agent.listWebSocketEndpoints()` here, `agent.get_all_websocket_handlers()`
 * in Python), so a new `@websocket` handler fails classification on the day it
 * is written.
 *
 * No TypeScript skill registers either path today. They are listed for the same
 * reason `uamp/completions` is: the two SDKs serve one platform, and the parity
 * tests assert the sets are equal.
 */
export const BILLABLE_WS_PATHS = ['uamp', 'realtime', 'acp/stream'] as const;

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
 *   - the three .well-known documents — agent card, JWKS and OIDC discovery,
 *     all of which are useless unless they are public.
 *   - command and command/-path — the Python slash-command surface. It
 *     dispatches through agent.execute_command, and no shipped command handler
 *     reaches execute_handoff, process_uamp or run. A future one that does is
 *     billable and belongs in BILLABLE_PATHS, not here.
 *   - tasks/-task_id and tasks/-task_id/artifacts — A2A task status reads and a
 *     cancel. They serve results already stored by the billable POST /tasks and
 *     never call the model themselves.
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
  '.well-known/jwks.json',
  '.well-known/openid-configuration',
  'command',
  'command/{path:path}',
  'tasks/{task_id}',
  'tasks/{task_id}/artifacts',
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
  if (!isBillableRequest(request.method, pathname)) return null;
  if (hasCredential(request)) return null;
  return unauthorizedResponse(extraHeaders);
}

/** The one 401 body both SDKs answer with. */
export function unauthorizedResponse(extraHeaders: Record<string, string> = {}): Response {
  return new Response(
    JSON.stringify({ error: { code: 'unauthorized', message: UNAUTHORIZED_MESSAGE } }),
    {
      status: 401,
      headers: { 'Content-Type': 'application/json', ...extraHeaders },
    },
  );
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
