/**
 * Who a served agent answers from another origin, and where it listens by
 * default (2026-09-24, S-226 in the portal's security log).
 *
 * THE HOLE THIS CLOSES. The credential floor in front of the model routes
 * checks that a credential is PRESENT; verifying it is `AuthSkill`'s job. That
 * is fine for callers the network already keeps out, and it was not fine in
 * combination with two defaults: every served agent answered ANY origin's
 * preflight (`cors()` defaults, plus the fetch handler stamping
 * `Access-Control-Allow-Origin: *` on its own responses), and `serve()`
 * listened on every interface. So while a developer ran `webagents serve`
 * with their model key exported, any web page they visited, and anyone on the
 * same network, could run the agent's model and read the answers by sending
 * `Authorization: Bearer <anything>`. Measured, not inferred.
 *
 * THE RULE. A server that cannot tell one caller from another must not invite
 * callers it cannot see:
 *   - an agent with an `AuthSkill` verifies what it is sent, so it keeps the
 *     permissive policy it always had (hosted agents called from browsers);
 *   - an agent without one answers loopback origins only (a local UI on
 *     another port still works), and binds loopback unless it has a public URL
 *     or is told otherwise;
 *   - an explicit `cors` setting always wins: `true` for any origin, a list for
 *     exactly those, `false` for none.
 * WebSocket handshakes are held to the same rule by the server itself, because
 * CORS does not apply to them: a browser opens a socket to any origin and only
 * reports where the page came from in `Origin`.
 */

/** `true` any origin, a list for those origins, `false` none, unset for the rule above. */
export type CorsSetting = boolean | string[] | undefined;

/** The `Access-Control-Allow-Origin` value for a request's `Origin`, or `null` for none. */
export type OriginPolicy = (origin: string | null | undefined) => string | null;

const LOOPBACK_ORIGIN = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/i;

/** A page served from this machine: `http(s)://localhost`, `127.0.0.1` or `[::1]`, any port. */
export function isLoopbackOrigin(origin: string): boolean {
  return LOOPBACK_ORIGIN.test(origin);
}

/** Whether the agent verifies the credentials it is sent, rather than only requiring one. */
export function agentVerifiesCredentials(agent: unknown): boolean {
  const skills = (agent as { skills?: Array<{ constructor?: { name?: string } }> })?.skills ?? [];
  return skills.some((skill) => skill?.constructor?.name === 'AuthSkill');
}

/** The policy for one server, from its `cors` setting and whether its agent verifies callers. */
export function originPolicy(setting: CorsSetting, verifiesCredentials: boolean): OriginPolicy {
  if (setting === false) return () => null;
  if (setting === true) return () => '*';
  if (Array.isArray(setting)) {
    const allowed = new Set(setting);
    return (origin) => (origin && allowed.has(origin) ? origin : null);
  }
  if (verifiesCredentials) return () => '*';
  return (origin) => (origin && isLoopbackOrigin(origin) ? origin : null);
}

/**
 * Whether a WebSocket handshake may proceed. A request with no `Origin` did
 * not come from a browser page (the CLI, the platform, a script), so the
 * policy has nothing to say about it; the credential floor still applies.
 */
export function upgradeOriginAllowed(policy: OriginPolicy, origin: string | null | undefined): boolean {
  if (!origin) return true;
  return policy(origin) !== null;
}

/**
 * Where `serve()` listens when no hostname is given: every interface for an
 * agent that verifies its callers (an AuthSkill), loopback otherwise.
 *
 * A PUBLIC URL ALONE NO LONGER WIDENS THE BIND (S-327, 2026-09-28). It did,
 * even for a loopback URL and with no AuthSkill, so anyone on the network
 * could run the agent's model with any bearer string. The public URL names
 * where the agent is reached; a tunnel or proxy on this machine forwards to
 * the loopback port, and `hostname` (`--host`) still binds anywhere on
 * purpose. `publicUrl` is kept in the options for the callers that pass it,
 * and chooses only the sentence (`loopbackBindLine`).
 */
export function defaultHostname(options: { publicUrl?: string; verifiesCredentials: boolean }): string {
  return options.verifiesCredentials ? '0.0.0.0' : '127.0.0.1';
}

/**
 * Why a served agent listens on loopback, said once at startup. The Python
 * `serve` prints the same (`cli/serve.py` `loopback_bind_line`; fixture
 * `cli/final_sdk_serve_model.json`, `bind`).
 */
export function loopbackBindLine(name: string, publicUrl?: string): string {
  if (publicUrl) {
    return (
      `[webagents] ${name}: listening on 127.0.0.1 only, because it has no AuthSkill to verify its ` +
      'callers; WEBAGENTS_PUBLIC_URL alone does not open it to other machines. A tunnel or proxy on ' +
      'this machine can forward to it, or pass `hostname` (`--host 0.0.0.0` on the CLI) to accept ' +
      'other machines.'
    );
  }
  return (
    `[webagents] ${name}: listening on 127.0.0.1 only, because it has no AuthSkill to verify its ` +
    'callers. Pass `hostname` (`--host 0.0.0.0` on the CLI) to accept other machines.'
  );
}
