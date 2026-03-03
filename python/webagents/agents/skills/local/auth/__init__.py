"""
AOAuth: Agent OAuth Protocol Implementation

OAuth 2.0 extension for agent-to-agent authentication.
Supports both Portal-delegated mode and self-issued mode.
"""

from .skill import AuthSkill, AuthContext, AuthError, normalize_agent_url
from .config import AOAuthConfig, AuthMode
from .jwks import JWKSManager, CacheEntry

__all__ = [
    "AuthSkill",
    "AuthContext",
    "AuthError",
    "normalize_agent_url",
    "AOAuthConfig",
    "AuthMode",
    "JWKSManager",
    "CacheEntry",
]
