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
2. **The card must carry ``metadata.publicKey`` as an SPKI PEM.**
   ``verifyExternalAOAuthToken`` feeds that value to ``importSPKI`` and
   verifies the presented token against it — proof of key possession is the
   whole guard on auto-registration. No key, no registration, ever.

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
