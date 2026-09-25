"""
The access skill: who is calling, and which group they are in (ADR-0045).

An agent file with an `access:` block gets this skill. On every turn, before the
model runs (`on_connection`, after an auth skill and before payments), it:

  1. takes the caller's TIER as already established by something this process
     trusts: the local chat's owner, or an auth skill that verified a platform
     credential. Nothing in the request can set it;
  2. collects the caller's PRINCIPALS from verified credentials only:
       - `user:<id>` from a platform credential an auth skill verified (an API
         key or an owner assertion), or from the `sender` the platform signs
         into a service token whose audience is this agent's own URL (S-240);
         the request body's sender names no one;
       - `agent:<URL>` and `key:<thumbprint>` from a Web Bot Auth signature,
         verified here (`crypto.web_bot_auth_verify`). A signature that is
         present and does not verify refuses the request (401); it is never
         read as "anonymous". A bearer the agent has no way to verify names no
         one, as it always has;
  3. decides with the block (`access.policy.decide`): refused (403), or let in
     with its groups, which become `group:<name>` scopes on the context.

It also adds each group's instructions file as a prompt scoped to that group,
and one short prompt telling the model who this turn is from.

The TypeScript twin is `typescript/src/skills/access/skill.ts`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from ...base import Skill
from webagents.access.caller import CallerAuth
from webagents.access.policy import AccessPolicy, decide
from webagents.agents.tools.decorators import hook, prompt
from webagents.crypto.web_bot_auth_verify import (
    InboundRequest,
    KeySetFetcher,
    MemoryNonceStore,
    normalize_authority,
    verify_web_bot_auth,
)

NO_ADDRESS = (
    "This agent has no public address, so it cannot check a signature made for one. "
    "Serve it with WEBAGENTS_PUBLIC_URL set to the address callers sign for."
)
FORBIDDEN = "This agent does not accept requests from this caller."


class AccessRefused(Exception):
    """A request the access block refuses: 401 for a signature that does not
    verify, 403 for a caller the block keeps out."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = code

    def to_dict(self) -> Dict[str, Any]:
        return {"error": {"code": self.error_code, "message": str(self)}}


def _tier(auth: Any) -> Optional[str]:
    """`owner` or `admin` when something trusted already said so; None otherwise."""
    if auth is None:
        return None
    scope = getattr(auth, "scope", None)
    value = getattr(scope, "value", scope)
    authenticated = getattr(auth, "authenticated", True)
    if authenticated and value in ("owner", "admin"):
        return value
    return None


def _user_principals(auth: Any) -> List[str]:
    """`user:` principals from a platform credential an auth skill verified."""
    if auth is None or not getattr(auth, "authenticated", False):
        return []
    if getattr(auth, "provider", None) == "local":
        return []
    claims = getattr(auth, "assertion", None)
    if isinstance(claims, dict) and str(claims.get("sub", "")).startswith("service:"):
        # A service token names the router. The person is the platform's
        # signed `sender` claim, and only on a token whose audience is this
        # agent's own URL: any other could have been minted for another agent
        # and replayed here (S-240). The body's sender names no one.
        sender = claims.get("sender") if getattr(auth, "audience_verified", False) else None
        user_id = sender.get("id") if isinstance(sender, dict) else None
        username = sender.get("username") if isinstance(sender, dict) else None
    else:
        user_id = getattr(auth, "user_id", None)
        username = getattr(auth, "username", None)
    out = []
    if isinstance(user_id, str) and user_id:
        out.append(f"user:{user_id}")
    if isinstance(username, str) and username:
        out.append(f"user:@{username}")
    return out


async def _inbound(context: Any) -> Optional[InboundRequest]:
    """The HTTP request this turn came from, when it came from one."""
    request = getattr(context, "request", None)
    if request is None or not hasattr(request, "headers") or not hasattr(request, "body"):
        return None
    scope = getattr(request, "scope", {}) or {}
    raw_path = scope.get("raw_path") or str(getattr(getattr(request, "url", None), "path", "/")).encode()
    query = scope.get("query_string") or b""
    target = raw_path.decode("latin-1") + (("?" + query.decode("latin-1")) if query else "")
    try:
        body = await request.body()
    except Exception:  # noqa: BLE001 - a stream already consumed reads as no body
        body = b""
    return InboundRequest(method=request.method, target=target, headers=request.headers, body=body)


def _own_address(public_url: Optional[str]) -> Optional[Tuple[str, str]]:
    """`(authority, scheme)` of the address callers sign for, or None."""
    url = public_url if public_url is not None else os.environ.get("WEBAGENTS_PUBLIC_URL")
    if not url:
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    authority = normalize_authority(parts.netloc.rpartition("@")[2])
    return (authority, parts.scheme) if authority else None


def who_is_calling(auth: Any) -> str:
    """One sentence for the model about this turn's caller."""
    tier = _tier(auth)
    if tier == "owner":
        return "This turn is from this agent's owner."
    if tier == "admin":
        return "This turn is from an admin of this agent."
    principals = list(getattr(auth, "principals", None) or [])
    groups = list(getattr(auth, "groups", None) or [])
    listed = ", ".join(groups)
    agent = next((p[len("agent:"):] for p in principals if p.startswith("agent:")), None)
    user = next((p[len("user:"):] for p in principals if p.startswith("user:")), None)
    if agent:
        who = f"the agent {agent}, which proved it with a Web Bot Auth signature"
    elif user:
        who = f"the Robutler user {user}"
    else:
        who = "a caller who did not identify itself"
    return f"This turn is from {who}. Groups: {listed or 'none'}."


class AccessSkill(Skill):
    """The access block, enforced. Added by the agent file loader, never by hand."""

    #: Establishes who is calling (`BaseAgent.identify_caller`): a scoped
    #: `@http` or websocket endpoint runs this skill's `on_connection` hook.
    identifies_caller = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        settings = config or {}
        super().__init__(settings, scope="all")
        self.policy: AccessPolicy = settings["policy"]
        self.instruction_texts: Dict[str, str] = dict(settings.get("instruction_texts") or {})
        allow_private = os.environ.get("ROBUTLER_AGENT_URL_ALLOW_PRIVATE") == "1"
        self.key_sets = settings.get("key_sets") or KeySetFetcher(allow_private=allow_private)
        self.nonces = settings.get("nonces") or MemoryNonceStore()
        self.public_url: Optional[str] = settings.get("public_url")
        #: The agent file's skill names, for `access.tools` (set by the loader).
        self.skill_names: List[str] = list(settings.get("skill_names") or [])

    async def initialize(self, agent) -> None:
        await super().initialize(agent)
        # A tool a skill registered only now, when it started, gets its scopes too.
        from webagents.access.install import apply_access_tools

        apply_access_tools(agent, self.policy, self.skill_names, strict=False)
        for group, text in self.instruction_texts.items():

            def make(body: str):
                def group_instructions() -> str:
                    return body

                return group_instructions

            fn = make(text)
            fn.__name__ = f"access_instructions_{group}"
            agent.register_prompt(fn, priority=5, source="access", scope=f"group:{group}")

    @prompt(priority=4, scope="all")
    def caller_prompt(self, context) -> str:
        return "## Caller\n" + who_is_calling(getattr(context, "auth", None))

    @hook("on_connection", priority=1, scope="all")
    async def admit(self, context):
        auth = getattr(context, "auth", None)
        tier = _tier(auth)
        principals = _user_principals(auth)

        inbound = await _inbound(context)
        header = (lambda name: inbound.headers.get(name)) if inbound is not None else (lambda name: None)
        if inbound is not None and header("signature-input") is not None:
            own = _own_address(self.public_url)
            if own is None:
                raise AccessRefused(401, "signature_authority_mismatch", NO_ADDRESS)
            authority, scheme = own
            outcome = await verify_web_bot_auth(
                inbound,
                authorities=[authority],
                scheme=scheme,
                key_sets=self.key_sets,
                nonces=self.nonces,
                allow_http=os.environ.get("ROBUTLER_AGENT_URL_ALLOW_PRIVATE") == "1",
            )
            if not outcome.ok:
                raise AccessRefused(401, outcome.refusal.code, outcome.refusal.description)
            principals.append(f"agent:{outcome.agent.principal}")
            principals.extend(f"key:{t}" for t in outcome.agent.thumbprints)

        decision = decide(self.policy, principals, tier)
        if not decision.allow:
            raise AccessRefused(403, "forbidden", FORBIDDEN)

        if auth is None or getattr(auth, "provider", None) == "local" or not hasattr(auth, "__dict__"):
            context.auth = CallerAuth(
                scope=tier or ("user" if principals else "all"),
                groups=list(decision.groups),
                principals=principals,
                user_id=getattr(auth, "user_id", None) if auth is not None else None,
                authenticated=bool(tier or principals),
                provider=getattr(auth, "provider", None) or ("signature" if principals else "anonymous"),
            )
        else:
            # Keep the auth skill's own context (other skills read its fields);
            # add what the block decided.
            auth.groups = list(decision.groups)
            auth.principals = principals
        return context
