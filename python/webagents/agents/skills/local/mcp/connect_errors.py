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
"""

from __future__ import annotations

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
