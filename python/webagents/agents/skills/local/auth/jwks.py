"""
Backward-compatible re-export of JWKSManager from webagents.crypto.jwks.
"""

from webagents.crypto.jwks import JWKSManager, CacheEntry

__all__ = ["JWKSManager", "CacheEntry"]
