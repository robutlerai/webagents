"""
Who a turn is for, as the scope checks read it (ADR-0045).

`scope` is the tier (`owner`, `admin`, `user`, or `all` for anonymous); `groups`
the access groups the caller was placed in (`group:<name>` scopes, see
`agents.core.scopes`); `principals` the verified identities the placement was
made from (`user:...`, `agent:...`, `key:...`), for logs and for the model's
prompt, never for authorization after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional


@dataclass
class CallerAuth:
    scope: str
    groups: List[str] = field(default_factory=list)
    principals: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    authenticated: bool = True
    provider: str = "local"


#: The person at the terminal, in the local chat and `webagents -p`: the owner
#: of the agent they run (2026-09-25). Every turn ran anonymous, so an
#: owner-scoped tool or prompt (the REST tool is one) never reached the one
#: person the local chat serves. The TypeScript chat passes the same
#: (`LOCAL_OWNER` in `typescript/src/cli/app.ts`).
LOCAL_OWNER = CallerAuth(scope="owner", provider="local")


def run_as_local_owner(agent: Any) -> Any:
    """Set the current context to a fresh one whose caller is the local owner,
    for the turn about to run (`run_streaming` reuses the current context)."""
    from webagents.server.context.context_vars import create_context, set_context

    context = create_context(messages=[], stream=True, agent=agent)
    # A copy per turn: a skill that sets something on `context.auth` must not
    # carry it into the next turn through a shared object.
    context.auth = replace(LOCAL_OWNER, groups=[], principals=[])
    set_context(context)
    return context
