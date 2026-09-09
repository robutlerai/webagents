"""
Platform registration surface for a served agent.

These are not conveniences and they are not optional extras — they are what
the platform's registration path READS. They used to live in a `host()`
wrapper, so the documented server (`create_server` + `uvicorn.run`) served an
agent the platform could never finish registering. They now belong to the
server itself.

Two requirements, both discovered the hard way:

1. **The agent card must be reachable at the ORIGIN.** The platform fetches
   ``new URL('/.well-known/agent.json', agentUrl)``, which is origin-relative
   and DISCARDS the agent path. A card served only under ``/{agent}/`` is
   invisible to it.
2. **The card must carry the SPKI PEM as top-level ``publicKey`` AND as
   ``metadata.publicKey``.** ``verifyExternalAOAuthToken`` feeds that value to
   ``importSPKI`` and verifies the presented token against it: proof of key
   possession is the whole guard on auto-registration. No key, no
   registration, ever. It is published twice on purpose (build plan 1M-00,
   ADR-0038 step 1): the platform's verifier reads the card's TOP-LEVEL
   ``publicKey`` (portal ``lib/auth/agent-auth.ts``: ``metadata?.publicKey``,
   where ``metadata`` is the whole fetched card), while this SDK and the
   TypeScript one wrote only the nested ``metadata.publicKey``. The two never
   met, so no SDK-served agent could auto-register at all. Both shapes stay
   until every verifier reads both.

Plus presence: ``POST /api/agents/heartbeat`` every 60 seconds, identity
derived from the bearer. Without it the platform shows the agent as unknown.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("webagents.server.registration")

HEARTBEAT_INTERVAL_S = 60


def build_agent_card(
    agent: Any,
    public_base_url: str,
    public_key_pem: Optional[str] = None,
) -> Dict[str, Any]:
    """The A2A agent card in the shape platform registration reads."""
    card: Dict[str, Any] = {
        "name": agent.name,
        "description": (
            getattr(agent, "instructions", "") or getattr(agent, "description", "") or ""
        ),
        "url": public_base_url,
        "capabilities": {"streaming": True, "pushNotifications": False},
        "authentication": {"schemes": ["Bearer"]},
    }
    if public_key_pem:
        # Both placements, see item 2 in the module docstring.
        card["publicKey"] = public_key_pem
        card["metadata"] = {"publicKey": public_key_pem}
    return card


def resolve_public_base_url(configured: Optional[str], agent_name: str) -> str:
    """The URL that goes on the card: explicit config, then
    ``WEBAGENTS_PUBLIC_URL``, then the agent-prefixed path as a RELATIVE last
    resort.

    Relative is deliberate, and the TypeScript half now does the same
    (``resolveCardUrl`` in ``typescript/src/server/handler.ts``, which used to
    fall back to the request origin instead — the two SDKs answered ``/mini``
    and ``http://127.0.0.1:8816/agents/og`` to the same question). The request
    origin is a guess derived from the Host header; behind a proxy, a tunnel or
    a container it is the wrong guess stated as fact. A relative reference is
    resolved by any consumer against the document it just fetched, which is by
    construction an origin the agent is reachable at.

    The platform settles it from the other side: NO CONSUMER READS
    ``card.url``. Being precise about that, because an earlier version of this
    note said ``AgentMetadata`` "declares only" a handful of fields and would
    not survive someone opening the file: the interface (portal
    ``lib/auth/agent-auth.ts:71``) carries an index signature
    ``[key: string]: unknown`` at line 82, so ``url`` IS carried through it.
    What holds is that nothing DEREFERENCES it — the only ``metadata.`` reads
    in that file are ``capabilities`` (lines 272 and 344) and ``publicKey``
    (lines 485 and 509), and the callable address a registration is keyed on is
    ``composeAgentRegistrationUrl(iss, agent_path, sub)`` (line 186), built from
    the agent's OWN signed token rather than from the card.

    ``.strip()`` before ``.rstrip("/")`` so a whitespace-only configured value
    is treated as unset rather than published verbatim — again matching the
    TypeScript ``.trim()``, including the detail that a whitespace-only
    ``configured`` suppresses the environment variable rather than falling
    through to it.
    """
    base = (configured or os.getenv("WEBAGENTS_PUBLIC_URL") or "").strip().rstrip("/")
    return base or f"/{agent_name}"


def resolve_portal_api_url(configured: Optional[str] = None) -> Optional[str]:
    """The platform's HTTP API base, for the heartbeat."""
    return (
        configured
        or os.getenv("ROBUTLER_INTERNAL_API_URL")
        or os.getenv("ROBUTLER_API_URL")
        or None
    )


def resolve_agent_token(agent: Any = None) -> Optional[str]:
    """The per-agent platform key. ``WEBAGENTS_AGENT_TOKEN`` is the documented
    variable; an ``api_key`` set on the agent object is honoured as a fallback
    so a programmatically configured agent does not need the environment."""
    return os.getenv("WEBAGENTS_AGENT_TOKEN") or (
        getattr(agent, "api_key", None) if agent is not None else None
    )


async def run_heartbeat_loop(
    portal_api_url: str,
    token: str,
    agent_name: str,
    interval_s: float = HEARTBEAT_INTERVAL_S,
) -> None:
    """POST ``/api/agents/heartbeat`` forever with the per-agent token.

    A failed beat is a warning, never a crash: an agent that cannot reach the
    platform must keep serving the callers that can reach IT. The route takes
    no body and derives identity from the bearer.
    """
    import httpx

    url = f"{portal_api_url.rstrip('/')}/api/agents/heartbeat"
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(
                        "heartbeat for %s: HTTP %s from %s",
                        agent_name,
                        resp.status_code,
                        url,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - presence must not kill the server
            logger.warning("heartbeat for %s failed: %s", agent_name, e)
        await asyncio.sleep(interval_s)


# ---------------------------------------------------------------------------
# Dynamic registration
#
# Serving the card is only half of it. The platform registers an agent on the
# FIRST REQUEST THAT VERIFIES: it reads ``iss`` off the unverified payload of a
# bearer, fetches the card at that URL, imports the ``publicKey`` it finds and
# checks the signature. Until something presents such a token the agent has a
# card nobody has read and no row anywhere.
#
# Nothing in either SDK presented one. The key was persisted, the card was
# correct, ``JWKSManager.mint_aoauth_token`` existed and had no callers, and
# its own docstring said the wiring was not built. The one fact that wiring
# has to get right is the audience: ``aud`` is the PLATFORM's base URL, never
# the agent's own URL and never the path of the endpoint being called. A token
# addressed to the agent's URL fails with ``unexpected "aud" claim value``,
# which reads like a signature problem and is not one.
# ---------------------------------------------------------------------------


def resolve_platform_base_url(configured: Optional[str] = None) -> Optional[str]:
    """The platform base URL an AOAuth token is addressed to (``aud``)."""
    return (
        configured
        or os.getenv("ROBUTLER_API_URL")
        or os.getenv("ROBUTLER_INTERNAL_API_URL")
        or None
    )


#: Default name the platform bearer is filed under in a secret store.
PLATFORM_TOKEN_SECRET = "platform_token"


def _seconds_until_expiry(token: str) -> Optional[int]:
    """Seconds until a JWT's ``exp``, read WITHOUT verifying the signature.

    Unverified is correct here and only here: this is our own stored copy of a
    token we are deciding whether to reuse, so the question is "has this
    lapsed" rather than "is this genuine". The platform verifies it properly
    on every call. Returns ``None`` when there is no readable ``exp``, which
    is treated as unusable rather than as eternal.
    """
    import base64
    import json
    import time

    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        exp = json.loads(base64.urlsafe_b64decode(padded)).get("exp")
    except Exception:  # noqa: BLE001 - anything unreadable means "do not reuse"
        return None
    if not isinstance(exp, (int, float)):
        return None
    return int(exp - time.time())


async def register_with_platform(
    agent_name: str,
    public_url: Optional[str] = None,
    platform_url: Optional[str] = None,
    keys_dir: Optional[str] = None,
    scopes: str = "read write",
    ttl_seconds: int = 300,
    agent_path: Optional[str] = None,
    secrets: Any = None,
    token_name: str = PLATFORM_TOKEN_SECRET,
    refresh: bool = False,
    expiry_skew_seconds: int = 300,
) -> Dict[str, Any]:
    """Register this agent with the platform by making one authenticated call.

    There is no registration endpoint to post to. ``POST /api/auth/agent/register``
    exists and answers **410 by design** — it used to mint an agent from an
    unauthenticated body with no proof the caller held the key it was storing.
    Registration is now implicit in verification, so the way to register is to
    call an authenticated route and let the verifier do it.

    ``POST /api/auth/cli/token`` is the call this uses, because it is the one
    that answers with the identity that was just minted (``user_id``,
    ``username``) plus a platform bearer the agent can keep. Any
    AOAuth-accepting route registers the agent equally well; this one lets the
    caller SEE that it happened.

    Requirements the caller has to meet, all of them about reachability rather
    than about crypto:

      * ``public_url`` (the card's URL, and the token's ``iss``) must be an
        address the PLATFORM can fetch. It resolves the card over the public
        internet through an SSRF guard that refuses loopback, RFC 1918,
        link-local and 100.64.0.0/10 — which includes Tailscale addresses, so
        a funnel host that serves the platform itself is still not a place an
        agent card can live.
      * the key on the card must be the key signing the token, and it must
        survive restarts: ``keys_dir`` has to be the same directory the server
        served the card from, because the platform stores what it read at
        registration and verifies every later token against that stored copy.

    ``secrets`` is where to keep the platform bearer this returns, so the next
    start reads it instead of registering again. Anything with ``get(name)``,
    ``set(name, value)`` and ``delete(name)`` will do;
    ``webagents.agents.skills.local.secrets.open_secret_store()`` builds one.

    It is OPTIONAL, and it must stay optional: an agent on a box with no
    keystore and no writable home still has to register. Absent, this behaves
    exactly as it did before the store existed.

    It is worth passing when you consider what is being stored. The bearer is
    good for seven days, carries ``agents:own``, and the signer sets no
    ``jti``, so there is no revocation lever: rotating the key published on
    the agent card does not invalidate an already-minted one (platform
    security log S-037, which amplifies S-034). Re-registering on every boot
    mints another week-long unrevocable credential each time; reading one back
    mints none. Pass ``refresh=True`` to ignore a stored bearer, which is the
    lever for "the stored one stopped working" (the local check reads ``exp``
    and cannot know the platform suspended the principal).

    Returns a dict with ``ok`` and ``status``, plus ``username``, ``user_id``
    and ``access_token`` on success or ``error`` on failure. A reused bearer
    comes back with ``reused: True`` and ``status: 0``, because nothing was
    called.
    """
    import httpx

    from ...crypto.jwks import JWKSManager

    # Read back before minting. A stored bearer that has not lapsed is the
    # same credential a fresh registration would hand back, so calling again
    # buys nothing and costs another seven-day unrevocable token.
    if secrets is not None and not refresh:
        try:
            stored = secrets.get(token_name)
            if stored:
                remaining = _seconds_until_expiry(stored)
                if remaining is not None and remaining > expiry_skew_seconds:
                    return {
                        "ok": True,
                        "status": 0,
                        "access_token": stored,
                        "reused": True,
                        "stored": "saved",
                    }
                # Lapsed or unreadable. Drop it rather than leave a dead
                # credential sitting in the keystore looking live.
                try:
                    secrets.delete(token_name)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001 - a store that cannot be read is
            # a reason to register normally, never a reason to fail. Say so,
            # because silence here looks like a cache miss.
            logger.warning(
                'could not read "%s" from the secret store: %s. Registering instead.',
                token_name,
                e,
            )

    platform = resolve_platform_base_url(platform_url)
    if not platform:
        return {
            "ok": False,
            "status": 0,
            "error": (
                "no platform URL: pass platform_url or set ROBUTLER_API_URL to "
                "the platform's base URL (this is also the token's `aud`)"
            ),
        }
    audience = platform.rstrip("/")

    issuer = resolve_public_base_url(public_url, agent_name)
    if not issuer.startswith("http"):
        return {
            "ok": False,
            "status": 0,
            "error": (
                "no public URL: pass public_url or set WEBAGENTS_PUBLIC_URL to "
                "the absolute address the PLATFORM can fetch this agent's card at"
            ),
        }

    jwks_config: Dict[str, Any] = {}
    if keys_dir:
        jwks_config["keys_dir"] = keys_dir
    manager = JWKSManager(jwks_config)
    manager.ensure_keys(agent_name)
    token = manager.mint_aoauth_token(
        agent_name,
        issuer=issuer,
        audience=audience,
        scopes=scopes,
        ttl_seconds=ttl_seconds,
        agent_path=agent_path,
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{audience}/api/auth/cli/token",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={},
            )
    except Exception as e:  # noqa: BLE001 - the reason is the whole value here
        return {"ok": False, "status": 0, "error": str(e)}

    if resp.status_code != 200:
        # 401 here is almost never a bad signature. In order of how often it
        # is actually the cause: the platform could not FETCH the card
        # (private or unroutable public_url), the card carries no top-level
        # `publicKey`, or the audience is not the platform base URL.
        return {
            "ok": False,
            "status": resp.status_code,
            "error": (
                f"{resp.status_code} from {audience}/api/auth/cli/token: "
                f"{resp.text[:300]}"
            ),
        }

    body = resp.json()
    access_token = body.get("access_token")

    stored_state = "not-requested"
    if secrets is not None and access_token:
        try:
            secrets.set(token_name, access_token)
            stored_state = "saved"
        except Exception as e:  # noqa: BLE001 - registration SUCCEEDED. The
            # caller holds a working bearer; only the persistence failed, so
            # report it and hand the token over rather than throwing away a
            # live credential over a storage problem.
            stored_state = f"not stored: {e}"
            logger.warning(
                'registered, but could not persist "%s": %s', token_name, e
            )

    return {
        "ok": True,
        "status": resp.status_code,
        "username": body.get("username"),
        "user_id": body.get("user_id"),
        "access_token": access_token,
        "reused": False,
        "stored": stored_state,
    }


# The set of scheduled registrations, held so the garbage collector cannot
# take a task mid-flight. `asyncio.create_task` returns the only strong
# reference there is: drop it and CPython is free to collect the task before
# it finishes, which produces a registration that silently never happened.
# Documented behaviour, not a CPython quirk, and the exact failure this whole
# helper exists to remove.
_PENDING_REGISTRATIONS: "set[asyncio.Task[Any]]" = set()


def register_after_startup(
    server_or_app: Any,
    agent_name: str,
    *,
    on_result: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Register with the platform once this server is actually serving.

    Use this instead of awaiting :func:`register_with_platform` inside your own
    startup handler. THE ORDERING IS A TRAP and it cost a live debugging run to
    find (2026-09-08 registration pass):

    Registration is a call the platform ANSWERS BY CALLING BACK. It fetches
    this agent's card from this very server while the registering request is
    still in flight. uvicorn serves no request at all until every
    ``on_event("startup")`` handler has returned, so awaiting the registration
    inside one deadlocks the callback against the handler waiting for it.

    What that looks like from outside is a 502 on the card fetch and a bare 401
    on the registering call, with nothing in either message pointing at the
    ordering. Both are plausible enough to send you looking at keys, at the
    audience, at the SSRF guard: everywhere except the one place that is wrong.

    This schedules the registration as a task and returns immediately, so the
    handler completes, uvicorn starts serving, and the platform's fetch of the
    card is answered by a server that is up. Same call, same arguments, correct
    order.

    ``server_or_app`` is what :func:`create_server` returned, or its ``.app``.
    ``on_result`` is called with the result dict when the registration settles;
    the default logs it. Everything else is passed straight through to
    :func:`register_with_platform`.

    Nothing here raises into startup: an agent that cannot register still has
    to serve the callers that can reach it, exactly as with the heartbeat.
    """
    app = getattr(server_or_app, "app", server_or_app)

    # Default `agent_path` to the prefix the agents are actually mounted
    # behind. `create_server(url_prefix=...)` mounts each agent at
    # `{url_prefix}/{name}`, and the platform composes the registration URL as
    # `issuer + agent_path + "/" + sub` where `sub` is the agent name. So the
    # prefix is what belongs here, and the agent name must NOT be appended.
    #
    # Passing it matters even when the prefix is empty. `agent_registrations
    # .agent_url` is unique, and until 2026-09-08 the platform read
    # `agent_path` with a truthiness check, so `""` was indistinguishable from
    # absent and every agent on one origin registered as the bare issuer: a
    # host serving N agents could register exactly ONE, and the others
    # verified against the first one's key and were refused. The portal now
    # tests `!== undefined` (`composeAgentRegistrationUrl`,
    # lib/auth/agent-auth.ts), so an explicit empty prefix means "mounted at
    # the root" and the agents get distinct URLs. A caller that passes
    # `agent_path` itself still wins.
    if "agent_path" not in kwargs:
        prefix = getattr(server_or_app, "url_prefix", None)
        if prefix is not None:
            kwargs["agent_path"] = str(prefix).rstrip("/")

    def _default_on_result(result: Dict[str, Any]) -> None:
        if result.get("ok"):
            if result.get("reused"):
                logger.info(
                    "%s: reusing the stored platform bearer, nothing registered", agent_name
                )
            else:
                logger.info(
                    "%s: registered as %s (%s)",
                    agent_name,
                    result.get("username"),
                    result.get("user_id"),
                )
        else:
            logger.warning("%s: not registered: %s", agent_name, result.get("error"))

    report = on_result or _default_on_result

    async def _run() -> None:
        try:
            result = await register_with_platform(agent_name, **kwargs)
        except Exception as e:  # noqa: BLE001 - a failed registration is a
            # warning, never a crash. See the docstring's last paragraph.
            logger.warning("%s: registration raised: %s", agent_name, e)
            return
        try:
            report(result)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: registration callback raised: %s", agent_name, e)

    @app.on_event("startup")
    async def _schedule_registration() -> None:  # pragma: no cover - trivial
        task = asyncio.create_task(_run())
        # See _PENDING_REGISTRATIONS: without this the task can be collected
        # before it runs, and the failure is silence.
        _PENDING_REGISTRATIONS.add(task)
        task.add_done_callback(_PENDING_REGISTRATIONS.discard)
