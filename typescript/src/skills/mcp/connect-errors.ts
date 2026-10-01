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
 *
 * A SERVER THAT STOPS BEFORE IT ANSWERS (2026-09-29, the owner's yoyo agent).
 * `uvx mcp-server-sqlite` fetched the server's last release with `mcp` 2.2.0,
 * the server died at start (`@server.list_resources()` is gone from `mcp` 2),
 * and `/mcp` said `not connected: MCP error -32000: Connection closed`, with
 * nothing about why. The reason was in the server's own stderr, which goes to
 * `<profile folder>/logs/mcp-<name>.log` (B8), and nothing pointed there. Now
 * a stdio server whose connection closes before the handshake is said as
 * `serverStoppedSentence`: the last error line it wrote during this attempt
 * (`lastErrorLine`: stack frames, `Node.js vN` and brace lines skipped, the
 * last unindented line preferred) and where its whole output is. Words and
 * cases: the same fixture, `server_stopped`.
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

/** The JSON-RPC code both MCP SDKs give a request whose connection closed. */
export const CONNECTION_CLOSED_CODE = -32000;

/** The row's sentence for a stdio server that stopped before the handshake; `{line}` and `{log}` are filled. */
export const SERVER_STOPPED = 'the server stopped before it answered: {line} (its output: {log})';
/** The same when it wrote nothing to its error output this time. */
export const SERVER_STOPPED_SILENT = 'the server stopped before it answered and wrote nothing to its error output';
/** The same when its error output could not be kept (no log folder). */
export const SERVER_STOPPED_UNLOGGED = 'the server stopped before it answered';
/** The longest error line quoted; longer ones end in "…". */
export const LINE_MAX = 200;

/** Lines that are never the error: blank, a stack frame, Node's version footer, a caret or tilde marker, a lone brace or bracket. */
const NOISE: RegExp[] = [/^\s*$/, /^\s+at\s/, /^Node\.js v\d/, /^\s*[\^~]+\s*$/, /^\s*[{}[\]]\s*$/];
/** What a quoted line starts with that is decoration, not words (uv's "×" and "╰─▶"). */
const LEAD = /^[\s×✗✘╰╭│─▶►→]+/;

/** Whether the connection closed under the client: the SDK's `Connection closed` (code -32000). */
export function connectionClosed(error: unknown): boolean {
  const leaf = rootCause(error) as { code?: unknown; message?: unknown } | null | undefined;
  if (leaf && typeof leaf === 'object' && leaf.code === CONNECTION_CLOSED_CODE) return true;
  const message = typeof leaf?.message === 'string' ? leaf.message : String(leaf ?? '');
  return /(^|: )Connection closed$/i.test(message.trim());
}

/** The line of a server's error output that says what went wrong, trimmed to `LINE_MAX`; undefined when there is none. */
export function lastErrorLine(text: string): string | undefined {
  const lines = (text ?? '').split(/\r?\n/).filter((line) => !NOISE.some((noise) => noise.test(line)));
  if (!lines.length) return undefined;
  const unindented = lines.filter((line) => !/^\s/.test(line));
  const pool = unindented.length ? unindented : lines;
  const line = pool[pool.length - 1].replace(LEAD, '').trimEnd();
  if (!line) return undefined;
  return line.length <= LINE_MAX ? line : `${line.slice(0, LINE_MAX - 1)}…`;
}

/** `path` with the home folder written `~`. */
export function displayPath(path: string, home: string): string {
  if (home && (path === home || path.startsWith(`${home}/`))) return `~${path.slice(home.length)}`;
  return path;
}

/**
 * The row's sentence for a stdio server that stopped before it answered:
 * `stderr` is what it wrote to its error output during this attempt,
 * undefined when that could not be kept; `log` is where its output is.
 */
export function serverStoppedSentence(stderr: string | undefined, log: string | undefined, home = ''): string {
  if (stderr === undefined || !log) return SERVER_STOPPED_UNLOGGED;
  const line = lastErrorLine(stderr);
  if (line === undefined) return SERVER_STOPPED_SILENT;
  return SERVER_STOPPED.replace('{line}', line).replace('{log}', displayPath(log, home));
}

/**
 * The row's sentence when a stdio server's command is not there (2026-09-29):
 * it was `spawn uvx ENOENT`, with nothing about what to install. `{command}`
 * and `{hint}` are filled.
 */
export const COMMAND_MISSING = '{command} is not installed or not on PATH: {hint}';
/** What to install, by the command's name; `default` for any other. */
export const COMMAND_HINTS: Readonly<Record<string, string>> = {
  uvx: 'it comes with uv (https://docs.astral.sh/uv/, or `pip install uv`)',
  uv: 'install uv (https://docs.astral.sh/uv/, or `pip install uv`)',
  npx: 'it comes with Node.js (https://nodejs.org)',
  npm: 'it comes with Node.js (https://nodejs.org)',
  node: 'install Node.js (https://nodejs.org)',
  bunx: 'it comes with Bun (https://bun.sh)',
  bun: 'install Bun (https://bun.sh)',
  deno: 'install Deno (https://deno.com)',
  docker: 'install Docker (https://docs.docker.com/get-docker/)',
  python: "install Python 3, or name the interpreter's full path",
  python3: "install Python 3, or name the interpreter's full path",
  default: "install it, or give its full path as the entry's command",
};

/** What to install for `command` (its base name, `.exe` and friends dropped). */
export function commandHint(command: string): string {
  const base = (command.split(/[\\/]/).pop() ?? command).toLowerCase().replace(/\.(exe|cmd|bat)$/, '');
  return COMMAND_HINTS[base] ?? COMMAND_HINTS.default;
}

/** The row's sentence for a command that is not there. */
export function commandMissingSentence(command: string): string {
  return COMMAND_MISSING.replace('{command}', command).replace('{hint}', commandHint(command));
}
