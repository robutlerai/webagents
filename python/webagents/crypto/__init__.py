"""
Crypto utilities for WebAgents: the key store and published key set (JWKS),
and Web Bot Auth request signing (RFC 9421 under the
draft-ietf-webbotauth-httpsig-protocol-00 profile, 2026-09-17).
"""

from .jwks import JWKSManager, CacheEntry
from .http_signature import (
    SIGNATURE_AGENT_FORMS,
    SigningError,
    SigningKey,
    WebBotAuth,
    sign_request,
)

__all__ = [
    "JWKSManager",
    "CacheEntry",
    "SIGNATURE_AGENT_FORMS",
    "SigningError",
    "SigningKey",
    "WebBotAuth",
    "sign_request",
]
