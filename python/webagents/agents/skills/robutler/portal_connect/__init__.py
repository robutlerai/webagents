"""Portal Connect Skill - UAMP WebSocket connection to the platform.

Attach it to an agent and serve that agent normally; the skill reads
``WEBAGENTS_PORTAL_URL`` / ``WEBAGENTS_AGENT_TOKEN`` itself and opens the
reverse bridge from the agent/server lifecycle.
"""

from .skill import (
    PortalConnectConfigError,
    PortalConnectSkill,
    PortalCredentialError,
    check_agent_token,
    resolve_portal_ws_url,
    sanitize_portal_messages,
)

__all__ = [
    "PortalConnectSkill",
    "PortalConnectConfigError",
    "PortalCredentialError",
    "check_agent_token",
    "resolve_portal_ws_url",
    "sanitize_portal_messages",
]
