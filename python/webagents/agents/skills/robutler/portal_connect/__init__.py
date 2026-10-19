"""Portal Connect Skill - UAMP WebSocket connection to the platform.

Attach it to an agent and serve that agent normally; the skill reads
``WEBAGENTS_PORTAL_URL`` / ``WEBAGENTS_AGENT_TOKEN`` itself and opens the
reverse bridge from the agent/server lifecycle.
"""

from .skill import (
    PORTAL_TURN_CONTEXT_KEY,
    PortalConnectConfigError,
    PortalConnectSkill,
    PortalCredentialError,
    check_agent_token,
    check_portal_credential,
    portal_turn_of,
    resolve_portal_ws_url,
    sanitize_portal_messages,
)

__all__ = [
    "PORTAL_TURN_CONTEXT_KEY",
    "portal_turn_of",
    "PortalConnectSkill",
    "PortalConnectConfigError",
    "PortalCredentialError",
    "check_agent_token",
    "check_portal_credential",
    "resolve_portal_ws_url",
    "sanitize_portal_messages",
]
