"""
Who may call a scoped `@http` or websocket endpoint (S-242/S-243, 2026-09-25).

An endpoint can be declared with a scope (`@http("/setup", scope="owner")`).
Until this module, the scope was checked on the dynamic route only, through an
import of a module that does not exist, so there every scoped endpoint answered
403 to everyone but a localhost `?token` equal to the agent's own key; and the
static mount never checked it at all, so the OpenAI skill's owner-only setup
form saved credentials for anyone who could reach the server (S-243). The
TypeScript servers never checked it either (S-242).

ONE GATE, the same in both SDKs (`typescript/src/server/endpoint-gate.ts`):

  - An endpoint with no scope, or `all`, is OPEN, exactly as before: nothing
    is asked of the caller and nothing extra runs.
  - Otherwise the agent identifies the caller the way a chat turn does
    (`BaseAgent.identify_caller`: its auth skills and its access block, and
    nothing else, so no payment is taken for an endpoint call), and the one
    scope rule (`agents.core.scopes.scope_allows`) decides.
  - A refusal raised while identifying keeps its own status and body: a
    signature that does not verify is 401, a caller the access block keeps
    out is 403, a credential the Robutler auth skill cannot verify is 401.
  - Then: a caller the agent could not verify gets 401 `unauthorized`, and a
    verified caller the scope does not include gets 403 `forbidden`.

NO CREDENTIAL IN A URL. The dynamic route used to make a localhost `?token`
equal to the agent's own platform credential its owner, which is what the
OpenAI skill's setup link relied on, and that link handed the credential to
every caller (S-246). The link is a one-time code now, and the rule is gone
(2026-09-25): a URL names a caller only through the auth skills, like any
other credential.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from webagents.agents.core.scopes import scope_allows

from .error_reply import is_meant_to_be_shown

NEEDS_CALLER = "This endpoint needs a caller this agent can verify, and the request carries none."
NOT_OPEN = "This endpoint is not open to this caller."

Refusal = Tuple[int, Dict[str, Any]]


def is_open(scope: Any) -> bool:
    """Whether an endpoint declared `scope` is open to an anonymous caller."""
    return scope_allows(scope, ())


def refusal_of(error: BaseException) -> Optional[Refusal]:
    """`(status, body)` for an auth or access refusal meant for the caller;
    None for anything else, which stays an error."""
    status = getattr(error, "status_code", None)
    if not (isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 500):
        return None
    if not is_meant_to_be_shown(error):
        return None
    if hasattr(error, "to_dict"):
        return status, error.to_dict()
    code = getattr(error, "error_code", None) or ("unauthorized" if status == 401 else "forbidden")
    return status, {"error": {"code": code, "message": str(error)}}


async def identify(agent: Any, scope: Any, context: Any) -> Tuple[Any, Optional[Refusal]]:
    """Identify the caller of `context` when `scope` is not open.

    Returns the context (now naming the caller) and None, or the context and
    the refusal identification raised. An open scope identifies no one.
    """
    if is_open(scope) or context is None or not hasattr(agent, "identify_caller"):
        return context, None
    try:
        return await agent.identify_caller(context), None
    except Exception as error:  # noqa: BLE001 - only a refusal is answered here
        refusal = refusal_of(error)
        if refusal is None:
            raise
        return context, refusal


async def admit(agent: Any, scope: Any, context: Any) -> Tuple[Any, Optional[Refusal]]:
    """Whether the caller of `context` may use an endpoint declared `scope`.

    `(context, None)` lets the handler run, with `context` naming the caller;
    `(context, (status, body))` refuses.
    """
    context, refusal = await identify(agent, scope, context)
    if refusal is not None or is_open(scope):
        return context, refusal
    held = set(getattr(context, "auth_scopes", None) or ())
    if scope_allows(scope, held):
        return context, None
    auth = getattr(context, "auth", None)
    if not getattr(auth, "authenticated", False):
        return context, (401, {"error": {"code": "unauthorized", "message": NEEDS_CALLER}})
    return context, (403, {"error": {"code": "forbidden", "message": NOT_OPEN}})
