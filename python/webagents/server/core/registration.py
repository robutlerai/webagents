"""
Platform registration surface for a served agent.

These are not conveniences and they are not optional extras: they are what
the platform's registration path READS. They used to live in a `host()`
wrapper, so the documented server (`create_server` + `uvicorn.run`) served an
agent the platform could never finish registering. They belong to the server
itself.

What the platform reads (ADR 0038 step 5, W2 design sections 2, 3.1 and 3.3,
2026-09-17):

1. **The key set at ``{agent_url}/.well-known/jwks.json``.** Every request
   the agent makes to the platform is signed with its Ed25519 key (RFC 9421
   HTTP Message Signatures under the Web Bot Auth profile,
   ``webagents.crypto.http_signature``). The ``Signature-Agent`` header names
   this URL; the platform fetches it, selects the key by the ``keyid``
   thumbprint, and derives the principal, the agent URL, by stripping the
   well-known suffix. The agent URL IS the registration
   (``agent_registrations.agent_url``): ``{public_url}{url_prefix}/{name}``,
   the address the agent is mounted at, with no trailing slash
   (``compose_principal``).
2. **The self-naming card at ``{agent_url}/.well-known/agent.json``.** Read
   once, on first verified use, for the name, description and capabilities
   the registration stores, and checked by simple string comparison:
   ``client_id`` equals the card's own URL, ``url`` equals the agent URL and
   ``jwks_uri`` equals the key-set URL the signature was resolved through.
   A card failing any of these is refused (``card_not_self_naming``,
   ``card_key_set_mismatch``). The card carries no key material: the key
   comes from the key set, never from the card, and an origin-level card
   cannot self-name for an agent mounted under a path, so none is served.

Plus presence: ``POST /api/agents/heartbeat`` every 60 seconds, identity
derived from the bearer. Without it the platform shows the agent as unknown.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from ...crypto.http_signature import (
    CARD_SUFFIX,
    DEFAULT_SIGNATURE_AGENT_FORM,
    KEY_SET_SUFFIX,
    SIGNATURE_LIFETIME_S,
    SigningError,
    WebBotAuth,
    assert_signable_agent_url,
    canonical_agent_url,
)

logger = logging.getLogger("webagents.server.registration")

HEARTBEAT_INTERVAL_S = 60


def compose_principal(
    public_base_url: str,
    agent_name: str,
    agent_path: Optional[str] = None,
) -> str:
    """The agent URL: the principal the platform registers (W2 design sections
    3.1 and 3.2), which is also where the card and the key set are served.

    ``public_base_url`` is what ``resolve_public_base_url`` answered. Absolute
    (the configured or ``WEBAGENTS_PUBLIC_URL`` tier), the agent is mounted
    at ``{base}{agent_path}/{agent_name}``, ``agent_path`` being the server's
    ``url_prefix`` (``create_server(url_prefix=...)`` mounts every agent at
    ``{url_prefix}/{name}``). Relative (the last-resort tier, which is
    already ``/{agent_name}``), the prefix goes in front so the card resolves
    against the origin it was fetched from to the real mount path. Never a
    trailing slash, and in the ONE canonical spelling (``canonical_agent_url``:
    host lowercased, default port dropped): the platform derives the principal
    from ``Signature-Agent`` through a WHATWG parse and compares the card's
    ``url`` to it by string equality, so ``https://Agents.Example.com:443``
    on the card was ``card_not_self_naming`` against a signature that verified
    (2026-09-18, W2 review). The signer reads the same function, so the two
    cannot disagree.
    """
    prefix = (agent_path or "").strip().rstrip("/")
    base = (public_base_url or "").strip().rstrip("/")
    if base.startswith("http://") or base.startswith("https://"):
        return canonical_agent_url(f"{base}{prefix}/{agent_name}")
    return f"{prefix}{base}" if base else f"{prefix}/{agent_name}"


def build_agent_card(agent: Any, principal: str) -> Dict[str, Any]:
    """The self-naming agent card (W2 design section 3.3) for the agent served
    at ``principal``: ``client_id`` is the card's own URL, ``url`` the
    principal, ``jwks_uri`` the key set, and there is no key material on it."""
    principal = principal.rstrip("/")
    return {
        "name": agent.name,
        "description": (
            getattr(agent, "instructions", "") or getattr(agent, "description", "") or ""
        ),
        "client_id": principal + CARD_SUFFIX,
        "url": principal,
        "jwks_uri": principal + KEY_SET_SUFFIX,
        "capabilities": {"streaming": True, "pushNotifications": False},
        "authentication": {"schemes": ["HTTPSig"]},
    }


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
    """The per-agent platform key, found rather than configured (2026-09-24).

    An ``api_key`` set on the agent in code, then ``WEBAGENTS_AGENT_TOKEN``,
    then the key ``webagents deploy`` stored for the agent this directory is
    linked to (``webagents.utils.agent_credential``). Agent-bound sources only:
    the older ``WEBAGENTS_API_KEY`` is often an owner's key, which the
    heartbeat route refuses.
    """
    from ...utils.agent_credential import resolve_agent_credential

    explicit = getattr(agent, "_api_key_explicit", None) if agent is not None else None
    found = resolve_agent_credential(
        getattr(agent, "name", None) if agent is not None else None,
        explicit=explicit,
        include_legacy=False,
    )
    return found[0] if found else None


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
# Serving the card and the key set is only half of it. The platform registers
# an agent on the FIRST REQUEST THAT VERIFIES: it reads the key-set URL off
# the request's ``Signature-Agent`` header, fetches the key set, selects the
# key the ``keyid`` names, checks the signature over the request, and then
# reads the card at the principal it derived. Until the agent makes such a
# request it has a card nobody has read and no row anywhere.
#
# ``register_with_platform`` makes that request. The one fact it has to get
# right is the AUTHORITY: the signature covers ``@authority``, and the
# platform verifies a request only against its own host, so the call goes to
# the platform's public base URL and to nothing that proxies it under another
# name (W2 design section 3.4).
# ---------------------------------------------------------------------------


def resolve_platform_base_url(configured: Optional[str] = None) -> Optional[str]:
    """The platform base URL: where the signed registration call is made, and
    the authority the signature is bound to."""
    return (
        configured
        or os.getenv("ROBUTLER_API_URL")
        or os.getenv("ROBUTLER_INTERNAL_API_URL")
        or None
    )


#: Default name the platform bearer is filed under in a secret store.
PLATFORM_TOKEN_SECRET = "platform_token"

#: The redirect statuses of the Fetch standard, which the TypeScript SDK's
#: ``redirect: 'error'`` refuses (S-190). A 300 or a 304 is an answer, not a
#: redirect, and fails registration as any other non-200 does.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

#: The header that carries the OPERATOR's platform API key on the registering
#: request. Always among the signature's covered components when it is sent
#: (S-184, 2026-09-19).
OWNER_KEY_HEADER = "X-Robutler-Owner-Key"


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


#: How long a claim link lives unless the caller says otherwise: ten minutes,
#: the TypeScript `claimUrl` default. The platform refuses a claim token that
#: lives longer than an hour whatever is asked for here.
CLAIM_URL_TTL_SECONDS = 600


def claim_url(
    identity: Any,
    agent_name: str,
    agent_user_id: str,
    *,
    platform_url: Optional[str] = None,
    ttl_seconds: Optional[int] = CLAIM_URL_TTL_SECONDS,
    agent_url: Optional[str] = None,
) -> Optional[str]:
    """A URL a human can open to take ownership of this agent:
    ``{platform}/claim/{agent_user_id}#{token}``. The twin of the TypeScript
    SDK's ``claimUrl`` (``src/server/registration.ts``), same shape and same
    lifetime.

    THE PROOF IS THE KEY, so this works where DNS cannot. The agent mints a
    short-lived, single-use claim token (``JWKSManager.mint_claim_token``, the
    one JWT left in this SDK: it is not a request from the agent, so it cannot
    be a signed request) with a key the platform pinned at registration; the
    person opens the link while signed in; the platform selects the key by the
    token's ``kid`` from the registration's key set, verifies it and records
    them as the owner. The token proves the agent, the session proves the
    human, and redemption binds them. That matters most on day one: an agent
    behind a tunnel or on a bare IP cannot be claimed by DNS at all.

    THE TOKEN IS IN THE FRAGMENT, NOT THE QUERY, and that is the reason this
    helper exists. A claim token is a bearer until it is spent. In ``?token=``
    it reaches the platform's access logs and any ``Referer`` a redirect
    leaks; after ``#`` the browser keeps it client-side. Until 2026-09-19
    Python had only ``mint_claim_token``, which left building the link, and so
    the fragment, to each operator. Print THIS, not the bare token. Ten
    minutes, one use, and never in a log is the whole security story.

    Args:
        identity: whatever holds the agent's Ed25519 key and mints the token:
            a ``JWKSManager`` on which ``ensure_ed25519_key(agent_name)`` has
            run over the SAME keys directory the server publishes its key set
            from (the platform selects the key by thumbprint from that set).
            Anything with ``mint_claim_token(agent_id, platform_url,
            ttl_seconds, issuer=...)`` will do.
        agent_name: the token's ``sub``, as ``AgentIdentity.agentId`` is in
            TypeScript.
        agent_user_id: the platform's id for the agent, ``user_id`` in the
            result of ``register_with_platform``. It names the claim page.
        platform_url: the platform's base URL, else ``ROBUTLER_API_URL`` /
            ``ROBUTLER_INTERNAL_API_URL`` (``resolve_platform_base_url``). The
            token's audience is derived from it, never from the agent's URL.
        ttl_seconds: the token's lifetime; ``None`` means the default.
        agent_url: written as the token's ``iss`` when given (``agent_url``
            in the result of ``register_with_platform``). The claim route
            does not read it.

    Returns ``None`` when no platform URL can be resolved, as ``claimUrl``
    does. Prefer being owned from birth: pass ``owner_api_key`` (or set
    ``ROBUTLER_API_KEY``) to ``register_with_platform`` and no claim is needed.
    """
    platform = resolve_platform_base_url(platform_url)
    if not platform:
        return None
    base = platform.rstrip("/")
    token = identity.mint_claim_token(
        agent_name,
        base,
        CLAIM_URL_TTL_SECONDS if ttl_seconds is None else ttl_seconds,
        issuer=agent_url,
    )
    return f"{base}/claim/{agent_user_id}#{token}"


async def register_with_platform(
    agent_name: str,
    public_url: Optional[str] = None,
    platform_url: Optional[str] = None,
    keys_dir: Optional[str] = None,
    agent_path: Optional[str] = None,
    secrets: Any = None,
    token_name: str = PLATFORM_TOKEN_SECRET,
    refresh: bool = False,
    expiry_skew_seconds: int = 300,
    signature_agent_form: str = DEFAULT_SIGNATURE_AGENT_FORM,
    signature_lifetime_seconds: int = SIGNATURE_LIFETIME_S,
    allow_http: Optional[bool] = None,
    owner_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Register this agent with the platform by making one signed call.

    There is no registration endpoint to post to. ``POST /api/auth/agent/register``
    exists and answers **410 by design**: it used to mint an agent from an
    unauthenticated body with no proof the caller held the key it was storing.
    Registration is implicit in verification, so the way to register is to
    call an authenticated route and let the verifier do it.

    ``POST /api/auth/cli/token`` is the call this uses, because it is the one
    that answers with the identity that was just minted (``user_id``,
    ``username``) plus a platform bearer the agent can keep. Any route that
    accepts a signed request registers the agent equally well; this one lets
    the caller SEE that it happened. The request is signed by
    ``WebBotAuth`` with every Ed25519 key the agent holds (W2 design section
    2.5); ``signature_agent_form`` is the ``Signature-Agent`` form of design
    section 2.4 (leave the default) and ``signature_lifetime_seconds`` the
    signing window (60 seconds; the platform refuses more than 3600).

    Requirements the caller has to meet, all of them about reachability rather
    than about crypto:

      * ``public_url`` must be an https address the PLATFORM can fetch
        (operator decision 2 of the W2 design: plain http is refused here
        because a plaintext key set lets an on-path attacker substitute
        keys), except under ``allow_http`` or
        ``ROBUTLER_AGENT_URL_ALLOW_PRIVATE=1`` in this process's environment,
        the switch the platform's local overlay and the TypeScript SDK read
        (2026-09-18: the two SDKs disagreed, and a Python agent could not
        register on the local cluster). Loopback by name is refused always.
        The platform resolves the key set and the card over the public
        internet through an SSRF guard that refuses loopback, RFC 1918,
        link-local and 100.64.0.0/10, which includes Tailscale addresses, so
        a funnel host that serves the platform itself is still not a place an
        agent's key set can live.
      * ``agent_path`` is the server's ``url_prefix``: the principal is
        ``{public_url}{agent_path}/{agent_name}`` (``compose_principal``),
        and it must be the URL the card and the key set are actually served
        under. ``register_after_startup`` fills it in from the server.
      * the key in the key set must be the key signing the request, and it
        must survive restarts: ``keys_dir`` has to be the same directory the
        server serves the key set from, because the platform selects the
        key by thumbprint from what it fetched at that URL.

    ``owner_api_key`` is the OPERATOR's own platform API key, falling back to
    ``ROBUTLER_API_KEY``: TWO CREDENTIALS, TWO PRINCIPALS. The signature proves
    the agent is itself; ``X-Robutler-Owner-Key`` says who owns it, so the
    agent is owned from birth and needs no claim flow. This is your key, not
    the agent's. Without it the agent registers OWNERLESS, which works but
    leaves it with no payer (the platform will not fund inference for it),
    and on a tunnel or bare-IP host it cannot be claimed afterwards either,
    because a DNS proof needs a domain you control. Optional by design, and a
    key the platform will not accept is ignored rather than fatal: the
    registration still succeeds, ownerless, and ``owned`` in the result says
    so. The TypeScript SDK has sent this since the claim flow existed; this
    function had no parameter for it, no environment fallback, no ``owned``
    in its result and no warning until 2026-09-19, so every Python agent
    registered ownerless.

    THE OWNER KEY IS INSIDE THE SIGNATURE (S-184, 2026-09-19): whenever the
    header is sent it is one of the covered components, so whoever can rewrite
    headers between this process and the platform's TLS edge cannot swap in a
    key of their own and register the agent under their account. The platform
    honours the header on a signed request only when it is covered.

    NEVER FOLLOW A REDIRECT WITH IT (S-190, 2026-09-19). Coverage stops
    tampering, not disclosure: httpx strips only ``Authorization`` when a
    redirect crosses origins, so a followed redirect carries the operator's
    platform key to whatever the ``Location`` names. The TypeScript SDK sets
    ``redirect: 'error'``; this function used to rely on httpx's DEFAULT not
    to follow, which is a property of how the client was built and not of
    this request. The request now says ``follow_redirects=False`` itself, so
    it holds for any client, and a redirect is a failed registration whose
    ``error`` says so (``status`` is the 3xx the platform answered).

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

    Returns a dict with ``ok`` and ``status``, plus ``username``, ``user_id``,
    ``access_token`` and ``owned`` on success or ``error`` on failure.
    ``owned`` is whether the agent has an accountable human behind it: ``False``
    is not a failure of the call (the agent is registered and will serve), it
    says the agent has no payer; ``None`` when the platform did not report it.
    A reused bearer comes back with ``reused: True`` and ``status: 0``, because
    nothing was called.
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
                "the platform's base URL (the signature is bound to its host)"
            ),
        }
    platform = platform.rstrip("/")

    issuer = resolve_public_base_url(public_url, agent_name)
    if not issuer.startswith("http"):
        return {
            "ok": False,
            "status": 0,
            "error": (
                "no public URL: pass public_url or set WEBAGENTS_PUBLIC_URL to "
                "the absolute https address the PLATFORM can fetch this agent's "
                "key set and card at"
            ),
        }
    agent_url = compose_principal(issuer, agent_name, agent_path)
    # The same rule the signer applies (`assert_signable_agent_url`), checked
    # here first so the answer is this dict and its sentence rather than an
    # exception out of httpx's auth flow.
    try:
        assert_signable_agent_url(agent_url, allow_http=allow_http)
    except SigningError as e:
        return {"ok": False, "status": 0, "error": str(e)}

    jwks_config: Dict[str, Any] = {}
    if keys_dir:
        jwks_config["keys_dir"] = keys_dir
    manager = JWKSManager(jwks_config)
    manager.ensure_ed25519_key(agent_name)
    # The operator's key (docstring): covered by the signature whenever sent.
    owner_key = (owner_api_key if owner_api_key is not None else os.getenv("ROBUTLER_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if owner_key:
        headers[OWNER_KEY_HEADER] = owner_key
    try:
        auth = WebBotAuth(
            manager.held_ed25519_keys(),
            agent_url,
            form=signature_agent_form,
            lifetime=signature_lifetime_seconds,
            allow_http=allow_http,
            covered_headers=[OWNER_KEY_HEADER] if owner_key else None,
        )
    except SigningError as e:
        return {"ok": False, "status": 0, "error": str(e)}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{platform}/api/auth/cli/token",
                headers=headers,
                json={},
                auth=auth,
                # S-190 (docstring): said HERE, on the request that carries
                # the operator's key, never left to the client's default.
                follow_redirects=False,
            )
    except Exception as e:  # noqa: BLE001 - the reason is the whole value here
        return {"ok": False, "status": 0, "error": str(e)}

    if resp.status_code in _REDIRECT_STATUSES:
        location = (resp.headers.get("location") or "").strip()
        return {
            "ok": False,
            "status": resp.status_code,
            "error": (
                f"{platform}/api/auth/cli/token answered with a redirect ({resp.status_code})"
                f"{' to ' + location[:300] if location else ''}; a redirect is never followed on the registering "
                "request, because it carries the operator's platform key. Set platform_url (or ROBUTLER_API_URL) "
                "to the platform's own base URL, the host that answers without redirecting"
            ),
        }

    if resp.status_code != 200:
        # 401 here is almost never a bad signature. In order of how often it
        # is actually the cause: the platform could not FETCH the key set or
        # the card at the agent URL (private, unroutable or plain-http
        # public_url, or an agent_path that is not the real mount prefix), the
        # card does not self-name (`client_id`, `url`, `jwks_uri` compared by
        # string equality), the key set does not carry the `keyid` (a keys_dir
        # other than the one the server serves from), the clock is more than
        # 60 seconds off, or the platform was reached under a host that is not
        # its own. The body's `error_code` says which.
        return {
            "ok": False,
            "status": resp.status_code,
            "error": (
                f"{resp.status_code} from {platform}/api/auth/cli/token: "
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

    # Say it out loud. An ownerless agent is the state a developer is most
    # likely to be in without knowing, and the one that quietly costs them
    # the most: no payer, and no way to claim it later from a tunnel or IP
    # host. The TypeScript SDK warns in the same words.
    owned = body.get("owned") if isinstance(body.get("owned"), bool) else None
    if owned is False:
        logger.warning(
            "registered as %s but it is UNCLAIMED: it has no owner, so the platform will not fund "
            "inference for it. Set ROBUTLER_API_KEY (your own platform key) and restart to be owned "
            "from birth.",
            body.get("username") or "this agent",
        )

    return {
        "ok": True,
        "status": resp.status_code,
        "username": body.get("username"),
        "user_id": body.get("user_id"),
        "access_token": access_token,
        "owned": owned,
        "agent_url": agent_url,
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


def _start_heartbeat_with_bearer(
    server: Any, agent_name: str, result: Dict[str, Any], platform_url: Optional[str]
) -> None:
    """Hand the bearer registration returned to the presence heartbeat (2026-09-24).

    Registration is the one place that holds it. The docs used to tell the
    developer to export it as WEBAGENTS_AGENT_TOKEN and restart, and the
    bridge then read that same variable and refused it (it carries no
    `agent_id`). Needs the server object (not only `.app`) to know which
    agents already beat; with just the app, nothing is started.
    """
    token = result.get("access_token") if result.get("ok") else None
    running = getattr(server, "_heartbeat_agents", None)
    tasks = getattr(server, "_heartbeat_tasks", None)
    if not token or running is None or tasks is None or agent_name in running:
        return
    portal_api_url = resolve_portal_api_url() or resolve_platform_base_url(platform_url)
    if not portal_api_url:
        return
    tasks.append(
        asyncio.create_task(run_heartbeat_loop(portal_api_url, token, agent_name, HEARTBEAT_INTERVAL_S))
    )
    running.add(agent_name)
    logger.info("%s: heartbeat started with the bearer registration returned", agent_name)


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
    # `{url_prefix}/{name}`, and the principal the platform registers is
    # `{public_url}{agent_path}/{name}` (`compose_principal`): the URL the
    # card and the key set are served under. So the prefix is what belongs
    # here, and the agent name must NOT be appended. A caller that passes
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
        _start_heartbeat_with_bearer(server_or_app, agent_name, result, kwargs.get("platform_url"))
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
