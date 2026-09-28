"""
Who a turn is for, as the scope checks read it (ADR-0045).

`scope` is the tier (`owner`, `admin`, `user`, or `all` for anonymous); `groups`
the access groups the caller was placed in (`group:<name>` scopes, see
`agents.core.scopes`); `principals` the verified identities the placement was
made from (`user:...`, `agent:...`, `key:...`), for logs and for the model's
prompt, never for authorization after the fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass
class CallerAuth:
    scope: str
    groups: List[str] = field(default_factory=list)
    principals: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    authenticated: bool = True
    provider: str = "local"
    #: The platform handle behind `user_id`, when the credential carried one;
    #: the access skill turns it into a `user:@<handle>` principal.
    username: Optional[str] = None
    #: The channel the caller wrote from, when the platform relayed the turn
    #: from a connected channel (the channel relay, plan item 2.2):
    #: ``{"type": ..., "sender_id": ...}``, which the access skill turns into
    #: the ``channel:<type>:<sender id>`` principal, listed first, so it is
    #: also the caller's memory namespace. Set only from the platform's own
    #: caller assertion (`portal_caller_auth`), never from a request.
    channel: Optional[Dict[str, str]] = None


#: A channel type is a slug, the same rule as a group name.
_CHANNEL_TYPE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


def channel_identity_of(raw: Any) -> Optional[Dict[str, str]]:
    """The channel a relayed sender wrote from, as the platform asserts it
    (``caller.channel = {type, sender_id}`` on the relayed frame). The type is
    lower-cased; the sender id is one token with no whitespace, never the
    wildcard. Anything else is not a channel and reads as no channel, so the
    caller stays a plain user. The vocabulary is pinned by
    ``tests/fixtures/access/channel_caller.json``; the TypeScript twin is
    ``channelIdentityOf`` in ``access/caller.ts``."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    sender_id = raw.get("sender_id")
    if not isinstance(kind, str) or not isinstance(sender_id, str):
        return None
    lowered = kind.lower()
    if not _CHANNEL_TYPE.match(lowered):
        return None
    if not sender_id or len(sender_id) > 200 or sender_id == "*" or any(c.isspace() for c in sender_id):
        return None
    return {"type": lowered, "sender_id": sender_id}


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
