/**
 * What the MCP client says when a remote server refuses it (2026-09-29, the
 * skills and MCP e2e). The Python twin is
 * `webagents/agents/skills/local/mcp/connect_errors.py`; the words are
 * `python/tests/fixtures/mcp_tool/connect_errors.json`, read by both suites.
 *
 * Pointed at a Streamable HTTP server that answers 401, `doctor` printed the
 * transport's own line, `Streamable HTTP error: Error POSTing to endpoint:
 * {..."Unauthorized"...}`, readable but with no word about a credential, and
 * its fix line said "Fix the server's entry in the agent file". The Python
 * client printed `unhandled errors in a TaskGroup (1 sub-exception)`, the
 * `str()` of the exception group its transport raised through.
 *
 *  - `rootCause` walks an `AggregateError` (the group shape Node has) down
 *    to the leaf that matters, an HTTP status error first; a plain error is
 *    its own root cause.
 *  - `httpStatusOf` reads the status the MCP SDK's transports put on their
 *    errors (`StreamableHTTPError` and `SseError` carry it as `code`, their
 *    messages prefixed `Streamable HTTP error:` and `SSE error:`), or a
 *    `status`, `statusCode` or `response.status` field.
 *  - A 401 or 403 (`CREDENTIAL_STATUSES`) is said as `NEEDS_CREDENTIAL`:
 *    the server needs a credential, a bearer token goes in the entry's
 *    `headers` as `Authorization: Bearer ${secret:NAME}`, and OAuth sign-in
 *    to MCP servers is not supported yet (neither SDK does the OAuth flow).
 *    NAME is `suggestedSecretName(server, 'token')`, `<SERVER>_TOKEN`. The
 *    report row carries `needsCredential: true`, and `doctor`'s fix line is
 *    then the recipe with the `webagents secrets set` command
 *    (`cli/doctor.ts`, `MCP_CHECK_WORDS.fixCredential`), never "fix the
 *    server's entry".
 */

import { suggestedSecretName } from '../secrets/references';

/** The HTTP statuses that mean the server wants a credential. */
export const CREDENTIAL_STATUSES: readonly number[] = [401, 403];

/** The row's sentence for those statuses; `{status}` and `{name}` are filled. */
export const NEEDS_CREDENTIAL =
  "needs a credential (HTTP {status}): put a bearer token in the entry's headers as " +
  'Authorization: Bearer ${secret:{name}}; OAuth sign-in to MCP servers is not supported yet';

/** The leaf of an error group (`AggregateError.errors`), an HTTP status error first; a plain error as it is. */
export function rootCause(error: unknown): unknown {
  const members = (error as { errors?: unknown } | null | undefined)?.errors;
  if (!Array.isArray(members) || !members.length) return error;
  const leaves = members.map(rootCause);
  return leaves.find((leaf) => httpStatusOf(leaf) !== undefined) ?? leaves[0];
}

function statusIn(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 100 && value <= 599 ? value : undefined;
}

/**
 * The HTTP status an error carries: `status`, `statusCode`, `response.status`,
 * or the MCP SDK transport errors' `code` (a JSON-RPC `McpError` has a numeric
 * `code` too, so that one counts only under the transports' own prefixes).
 */
export function httpStatusOf(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined;
  const e = error as { status?: unknown; statusCode?: unknown; response?: { status?: unknown }; code?: unknown; message?: unknown };
  const direct = statusIn(e.status) ?? statusIn(e.statusCode) ?? statusIn(e.response?.status);
  if (direct !== undefined) return direct;
  const message = typeof e.message === 'string' ? e.message : '';
  if (/^(Streamable HTTP error|SSE error):/.test(message)) return statusIn(e.code);
  return undefined;
}

/** The `${secret:NAME}` the sentence suggests for `server`: `<SERVER>_TOKEN`. */
export function credentialSecretName(server: string): string {
  return suggestedSecretName(server, 'token');
}

/** The row's sentence for a server that answered `status`. */
export function needsCredentialSentence(server: string, status: number): string {
  return NEEDS_CREDENTIAL.replace('{status}', String(status)).replace('{name}', credentialSecretName(server));
}

/**
 * The sentence a failed connection to `server` is recorded with (values not
 * yet masked: the caller does that), and whether the server wants a credential.
 */
export function describeConnectError(server: string, error: unknown): { message: string; needsCredential: boolean } {
  const leaf = rootCause(error);
  const status = httpStatusOf(leaf);
  if (status !== undefined && CREDENTIAL_STATUSES.includes(status)) {
    return { message: needsCredentialSentence(server, status), needsCredential: true };
  }
  const message = (leaf as { message?: unknown } | null | undefined)?.message;
  return { message: typeof message === 'string' && message ? message : String(leaf), needsCredential: false };
}
