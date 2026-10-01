"""
What the MCP client says when a remote server refuses it (2026-09-29, the
skills and MCP e2e). The TypeScript twin is `skills/mcp/connect-errors.ts`;
the words are `tests/fixtures/mcp_tool/connect_errors.json`, read by both
suites.

Pointed at a Streamable HTTP server that answers 401, `doctor` printed
`✗ mcp robutler: unhandled errors in a TaskGroup (1 sub-exception)`, and its
fix line said "Fix the server's entry in the agent file". The SDK's client
runs its transport in an anyio task group, the HTTP status error is raised
inside it, and `str()` of the exception group that leaves is that sentence.
The TypeScript client printed the transport's own line instead
(`Streamable HTTP error: Error POSTing to endpoint: {..."Unauthorized"...}`),
readable but with no word about a credential.

  - `root_cause` walks an exception group down to the leaf that matters: an
    HTTP status error first when the group holds one, else the first leaf.
    A plain exception is its own root cause. Every sentence the skill records
    about a failed connection goes through it (`skill._connect_server`).
  - `http_status_of` reads the status an httpx status error carries (the
    SDK's transports raise one for every non-2xx answer), or a `status_code`
    or `status` attribute.
  - A 401 or 403 (`CREDENTIAL_STATUSES`) is said as `needs_credential`: the
    server needs a credential, a bearer token goes in the entry's `headers`
    as `Authorization: Bearer ${secret:NAME}`, and OAuth sign-in to MCP
    servers is not supported yet (neither SDK does the OAuth flow). NAME is
    `suggested_secret_name(server, "token")`, `<SERVER>_TOKEN`. The report
    row carries `needs_credential: True`, and `doctor`'s fix line is then the
    recipe with the `webagents secrets set` command (`cli/doctor.py`,
    `MCP_CHECK_WORDS["fixCredential"]`), never "fix the server's entry".

A SERVER THAT STOPS BEFORE IT ANSWERS (2026-09-29, the owner's yoyo agent).
`uvx mcp-server-sqlite` fetched the server's last release with `mcp` 2.2.0,
the server died at start on `@server.list_resources()` (the decorator is gone
from `mcp` 2), and the chat said `not connected: Connection closed`: the
client's word for the pipe closing, with nothing about why. The reason was in
the server's own stderr, which goes to `<profile folder>/logs/mcp-<name>.log`
(B8), and nothing pointed there. Now a stdio server whose connection closes
before the handshake is said as `server_stopped_sentence`: the last error line
it wrote during this attempt (`last_error_line`: stack frames, `Node.js vN`
and brace lines skipped, the last unindented line preferred, which is the
exception line of a Python traceback and the `Error...` line of Node's) and
where its whole output is. Words and cases: the same fixture, `server_stopped`.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from ..secrets.references import suggested_secret_name

#: The HTTP statuses that mean the server wants a credential.
CREDENTIAL_STATUSES = (401, 403)

#: The row's sentence for those statuses; `{status}` and `{name}` are filled.
NEEDS_CREDENTIAL = (
    "needs a credential (HTTP {status}): put a bearer token in the entry's headers as "
    "Authorization: Bearer ${secret:{name}}; OAuth sign-in to MCP servers is not supported yet"
)


def root_cause(error: BaseException) -> BaseException:
    """The leaf of an exception group (anyio's `ExceptionGroup`, or Python's
    own `BaseExceptionGroup`), an HTTP status error first; a plain exception
    as it is. Duck-typed on `.exceptions`, so it needs no 3.11 name."""
    members = getattr(error, "exceptions", None)
    if not isinstance(members, (list, tuple)) or not members:
        return error
    leaves = [root_cause(member) for member in members if isinstance(member, BaseException)]
    if not leaves:
        return error
    for leaf in leaves:
        if http_status_of(leaf) is not None:
            return leaf
    return leaves[0]


def http_status_of(error: BaseException) -> Optional[int]:
    """The HTTP status an error carries: an httpx `HTTPStatusError`'s
    response status, or a `status_code` or `status` attribute; None otherwise."""
    response = getattr(error, "response", None)
    candidates = (getattr(response, "status_code", None), getattr(error, "status_code", None), getattr(error, "status", None))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def credential_secret_name(server: str) -> str:
    """The `${secret:NAME}` the sentence suggests for `server`: `<SERVER>_TOKEN`."""
    return suggested_secret_name(server, "token")


def needs_credential_sentence(server: str, status: int) -> str:
    """The row's sentence for a server that answered `status`."""
    return NEEDS_CREDENTIAL.replace("{status}", str(status)).replace("{name}", credential_secret_name(server))


def describe_connect_error(server: str, error: BaseException) -> Tuple[str, bool]:
    """The sentence a failed connection to `server` is recorded with (values
    not yet masked: the caller does that), and whether the server wants a
    credential. An HTTP error other than 401 or 403 keeps its first line:
    httpx appends a "For more information check: <url>" line nobody needs."""
    leaf = root_cause(error)
    status = http_status_of(leaf)
    if status in CREDENTIAL_STATUSES:
        return needs_credential_sentence(server, status), True
    text = str(leaf) or leaf.__class__.__name__
    if status is not None:
        text = text.splitlines()[0] if text.strip() else text
    return text, False


#: The JSON-RPC code both MCP SDKs give a request whose connection closed.
CONNECTION_CLOSED_CODE = -32000

#: The row's sentence for a stdio server that stopped before the handshake;
#: `{line}` and `{log}` are filled (module docstring).
SERVER_STOPPED = "the server stopped before it answered: {line} (its output: {log})"
#: The same when it wrote nothing to its error output this time.
SERVER_STOPPED_SILENT = "the server stopped before it answered and wrote nothing to its error output"
#: The same when its error output could not be kept (no log folder).
SERVER_STOPPED_UNLOGGED = "the server stopped before it answered"
#: The longest error line quoted; longer ones end in "…".
LINE_MAX = 200

#: Lines that are never the error: blank, a JavaScript stack frame, Node's
#: version footer, a caret or tilde marker, a lone brace or bracket.
_NOISE = (
    re.compile(r"^\s*$"),
    re.compile(r"^\s+at\s"),
    re.compile(r"^Node\.js v\d"),
    re.compile(r"^\s*[\^~]+\s*$"),
    re.compile(r"^\s*[{}\[\]]\s*$"),
)
#: What a quoted line starts with that is decoration, not words (uv's "×" and "╰─▶").
_LEAD = re.compile(r"^[\s×✗✘╰╭│─▶►→]+")


def connection_closed(error: BaseException) -> bool:
    """Whether the connection closed under the client: the SDK's
    "Connection closed" (code -32000), or anyio's end of stream."""
    leaf = root_cause(error)
    if getattr(getattr(leaf, "error", None), "code", None) == CONNECTION_CLOSED_CODE:
        return True
    if type(leaf).__name__ in ("EndOfStream", "ClosedResourceError", "BrokenResourceError"):
        return True
    return str(leaf).strip().lower() == "connection closed"


def last_error_line(text: str) -> Optional[str]:
    """The line of a server's error output that says what went wrong
    (module docstring), trimmed to `LINE_MAX`; None when there is none."""
    lines = [line for line in (text or "").splitlines() if not any(noise.match(line) for noise in _NOISE)]
    if not lines:
        return None
    unindented = [line for line in lines if not line[:1].isspace()]
    line = _LEAD.sub("", (unindented or lines)[-1]).rstrip()
    if not line:
        return None
    return line if len(line) <= LINE_MAX else line[: LINE_MAX - 1] + "…"


def display_path(path: str) -> str:
    """`path` with the home folder written `~`."""
    home = os.path.expanduser("~")
    if home and home != "~" and (path == home or path.startswith(home + os.sep)):
        return "~" + path[len(home):]
    return path


def server_stopped_sentence(stderr_text: Optional[str], log: Optional[str]) -> str:
    """The row's sentence for a stdio server that stopped before it answered:
    `stderr_text` is what it wrote to its error output during this attempt,
    None when that could not be kept; `log` is where its output is."""
    if stderr_text is None or not log:
        return SERVER_STOPPED_UNLOGGED
    line = last_error_line(stderr_text)
    if line is None:
        return SERVER_STOPPED_SILENT
    return SERVER_STOPPED.replace("{line}", line).replace("{log}", display_path(log))


#: The row's sentence when a stdio server's command is not there (2026-09-29):
#: it was `[Errno 2] No such file or directory: 'uvx'`, the operating system's
#: words, with nothing about what to install. `{command}` and `{hint}` are filled.
COMMAND_MISSING = "{command} is not installed or not on PATH: {hint}"
#: What to install, by the command's name; `default` for any other.
COMMAND_HINTS = {
    "uvx": "it comes with uv (https://docs.astral.sh/uv/, or `pip install uv`)",
    "uv": "install uv (https://docs.astral.sh/uv/, or `pip install uv`)",
    "npx": "it comes with Node.js (https://nodejs.org)",
    "npm": "it comes with Node.js (https://nodejs.org)",
    "node": "install Node.js (https://nodejs.org)",
    "bunx": "it comes with Bun (https://bun.sh)",
    "bun": "install Bun (https://bun.sh)",
    "deno": "install Deno (https://deno.com)",
    "docker": "install Docker (https://docs.docker.com/get-docker/)",
    "python": "install Python 3, or name the interpreter's full path",
    "python3": "install Python 3, or name the interpreter's full path",
    "default": "install it, or give its full path as the entry's command",
}


def command_hint(command: str) -> str:
    """What to install for `command` (its base name, `.exe` and friends dropped)."""
    base = os.path.basename(command).lower()
    for suffix in (".exe", ".cmd", ".bat"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return COMMAND_HINTS.get(base, COMMAND_HINTS["default"])


def command_missing_sentence(command: str) -> str:
    """The row's sentence for a command that is not there."""
    return COMMAND_MISSING.replace("{command}", command).replace("{hint}", command_hint(command))
