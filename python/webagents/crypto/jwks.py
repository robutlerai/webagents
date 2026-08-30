"""
JWKS Manager with Smart Caching

Manages RSA key pairs for JWT signing and provides JWKS caching
with ETag support and automatic refresh on key miss (rotation handling).
Shared by auth and payment skills.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import os
import time
import base64
import hashlib
import logging
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import httpx
import jwt


@dataclass
class CacheEntry:
    """JWKS cache entry with TTL and ETag support."""
    keys: List[Dict[str, Any]]
    expires_at: float
    etag: Optional[str] = None
    last_fetch: float = 0


class JWKSManager:
    """JWKS management with smart caching.
    
    Features:
    - RSA key pair generation and persistence
    - JWKS caching with configurable TTL
    - ETag support for efficient cache validation
    - Automatic refresh on key miss (handles key rotation)
    - Rate limiting to prevent cache stampede
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize JWKS manager.
        
        Args:
            config: Configuration dictionary with optional keys:
                - keys_dir: Directory for storing RSA keys
                - jwks_cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.config = config or {}
        self._private_key = None
        self._public_key = None
        self._kid: Optional[str] = None
        self._jwks_cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = self.config.get("jwks_cache_ttl", 3600)
        self._min_refetch_interval = 60  # Prevent spam
        
        # Determine keys directory. WEBAGENTS_KEYS_DIR is the documented
        # override (the TS SDK reads the same variable); the key MUST persist
        # across restarts because platform registration pins the public key
        # published on the agent card.
        keys_dir = self.config.get("keys_dir") or os.getenv("WEBAGENTS_KEYS_DIR")
        if keys_dir:
            self._keys_dir = Path(keys_dir)
        else:
            self._keys_dir = Path.home() / ".webagents" / "keys"
        
        self.logger = logging.getLogger(__name__)
    
    def ensure_keys(self, agent_id: str) -> str:
        """Generate or load RSA key pair for the agent.
        
        Keys are persisted to disk for consistency across restarts.
        The key ID (kid) is derived from the public key hash.
        
        Args:
            agent_id: Agent identifier for key file naming
            
        Returns:
            Key ID (kid) for the generated/loaded key
        """
        # Sanitize agent_id for filesystem
        safe_id = agent_id.replace("/", "_").replace("@", "").replace(":", "_")
        key_file = self._keys_dir / f"{safe_id}.pem"
        
        if key_file.exists():
            # Load existing key
            self._private_key = serialization.load_pem_private_key(
                key_file.read_bytes(),
                password=None,
                backend=default_backend()
            )
            self.logger.debug(f"Loaded existing RSA key for {agent_id}")
        else:
            # Generate new key pair
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Persist to disk. This is an UNENCRYPTED PKCS8 private key — the
            # identity the platform pins at registration — so it is written
            # owner-only, and the directory is owner-only too. Default mkdir
            # (0755) and default write (0644) left it world-readable on any
            # shared machine. The mode= on mkdir only applies when the
            # directory is created, so chmod unconditionally to repair a
            # directory an earlier version already made.
            self._keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self._keys_dir.chmod(0o700)
            except OSError:  # e.g. a dir we do not own; the file mode still holds
                pass
            # Create with 0600 rather than write-then-chmod: the latter leaves a
            # window where the key exists world-readable.
            fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(
                    self._private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    )
                )
            os.chmod(key_file, 0o600)
            self.logger.info(f"Generated new RSA key for {agent_id}")
        
        # Extract public key
        self._public_key = self._private_key.public_key()
        
        # Generate key ID from public key hash
        pub_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self._kid = hashlib.sha256(pub_bytes).hexdigest()[:16]
        
        return self._kid
    
    def get_signing_key(self) -> Any:
        """Get the private key for signing JWTs.
        
        Returns:
            RSA private key object
            
        Raises:
            RuntimeError: If keys haven't been initialized
        """
        if not self._private_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        return self._private_key
    
    def get_kid(self) -> Optional[str]:
        """Get the current key ID."""
        return self._kid
    
    def get_public_jwk(self) -> Dict[str, Any]:
        """Get public key in JWK format for JWKS endpoint.
        
        Returns:
            JWK dictionary with RSA public key
            
        Raises:
            RuntimeError: If keys haven't been initialized
        """
        if not self._public_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        
        numbers = self._public_key.public_numbers()
        
        def int_to_base64(n: int, length: int) -> str:
            """Convert integer to base64url encoding without padding."""
            return base64.urlsafe_b64encode(
                n.to_bytes(length, 'big')
            ).decode().rstrip('=')
        
        return {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": self._kid,
            "n": int_to_base64(numbers.n, 256),
            "e": int_to_base64(numbers.e, 3),
        }
    
    def get_jwks(self) -> Dict[str, Any]:
        """Get full JWKS response for /.well-known/jwks.json endpoint.

        Returns:
            JWKS dictionary with keys array
        """
        keys = []
        if self._public_key:
            keys.append(self.get_public_jwk())

        return {"keys": keys}

    def get_public_key_spki_pem(self) -> str:
        """Public key as an SPKI PEM string.

        This is the exact shape the platform's agent registration requires:
        `verifyExternalAOAuthToken` reads `metadata.publicKey` from the agent
        card and imports it with `importSPKI`. The JWK forms above cannot be
        fed to it, and until this method existed the Python SDK had no way to
        produce a card the platform would accept.

        Raises:
            RuntimeError: If keys haven't been initialized
        """
        if not self._public_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    def mint_aoauth_token(
        self,
        agent_id: str,
        issuer: str,
        audience: str,
        scopes: str = "read write",
        ttl_seconds: int = 300,
    ) -> str:
        """Mint an RS256 AOAuth JWT proving possession of this agent's key.

        The platform verifies it against the SPKI public key published in the
        agent card (`metadata.publicKey`), which is what auto-registration
        keys on (`verifyExternalAOAuthToken` -> `importSPKI` ->
        `jwtVerify`).

        EXPERIMENTAL — nothing in this SDK calls it yet. `create_server`
        serves the card and the JWKS (`WebAgentsServer._create_registration_endpoints`)
        but does not present an AOAuth token of its own, so auto-registration
        still requires the developer to send this token on a request to the
        platform's registration endpoints themselves:

            mgr = JWKSManager({"keys_dir": ...})
            mgr.ensure_keys(agent_id)
            token = mgr.mint_aoauth_token(agent_id, issuer=public_url,
                                          audience=platform_url)
            httpx.get(f"{platform_url}/api/agents",
                      headers={"Authorization": f"Bearer {token}"})

        The shape is covered by tests/test_aoauth_mint.py; the wiring into
        `WebAgentsServer` startup is not built.

        Args:
            agent_id: `sub` / `client_id` claim.
            issuer: the agent's public URL (must match the card's origin).
            audience: the platform issuer (its public base URL).
            scopes: space-separated scope string.
            ttl_seconds: token TTL.
        """
        if not self._private_key:
            raise RuntimeError("Keys not initialized. Call ensure_keys() first.")
        import uuid
        now = int(time.time())
        payload = {
            "sub": agent_id,
            "iss": issuer.rstrip("/"),
            "aud": audience,
            "iat": now,
            "nbf": now,
            "exp": now + ttl_seconds,
            "jti": str(uuid.uuid4()),
            "scope": scopes,
            "client_id": agent_id,
            "token_type": "Bearer",
        }
        return jwt.encode(
            payload,
            self._private_key,
            algorithm="RS256",
            headers={"kid": self._kid} if self._kid else None,
        )
    
    async def fetch_jwks(
        self,
        jwks_uri: str,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch JWKS from remote endpoint with smart caching.
        
        Features:
        - Respects Cache-Control max-age
        - Uses ETag for conditional requests (304 Not Modified)
        - Rate limits refetches to prevent cache stampede
        
        Args:
            jwks_uri: URL to fetch JWKS from
            force_refresh: Force fetch even if cache is valid
            
        Returns:
            List of JWK dictionaries
        """
        now = time.time()
        cached = self._jwks_cache.get(jwks_uri)
        
        # Return cached if valid and not forcing refresh
        if cached and not force_refresh and cached.expires_at > now:
            return cached.keys
        
        # Rate limit refetches to prevent spam
        if cached and (now - cached.last_fetch) < self._min_refetch_interval:
            self.logger.debug(f"Rate limiting JWKS fetch for {jwks_uri}")
            return cached.keys
        
        try:
            headers = {}
            if cached and cached.etag:
                headers["If-None-Match"] = cached.etag
            
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    jwks_uri,
                    timeout=10,
                    headers=headers,
                    follow_redirects=True
                )
                
                # Handle 304 Not Modified
                if resp.status_code == 304 and cached:
                    self._jwks_cache[jwks_uri] = CacheEntry(
                        keys=cached.keys,
                        expires_at=now + self._cache_ttl,
                        etag=cached.etag,
                        last_fetch=now
                    )
                    self.logger.debug(f"JWKS cache validated (304) for {jwks_uri}")
                    return cached.keys
                
                resp.raise_for_status()
                jwks = resp.json()
                
                # Parse Cache-Control for TTL
                ttl = self._cache_ttl
                cache_control = resp.headers.get("Cache-Control", "")
                if "max-age=" in cache_control:
                    try:
                        max_age_str = cache_control.split("max-age=")[1].split(",")[0].strip()
                        ttl = int(max_age_str)
                    except (ValueError, IndexError):
                        pass
                
                # Update cache
                self._jwks_cache[jwks_uri] = CacheEntry(
                    keys=jwks.get("keys", []),
                    expires_at=now + ttl,
                    etag=resp.headers.get("ETag"),
                    last_fetch=now
                )
                
                self.logger.debug(f"JWKS fetched from {jwks_uri}, TTL={ttl}s")
                return jwks.get("keys", [])
                
        except httpx.HTTPStatusError as e:
            self.logger.warning(f"JWKS fetch failed for {jwks_uri}: HTTP {e.response.status_code}")
            if cached:
                return cached.keys
            return []
        except Exception as e:
            self.logger.warning(f"JWKS fetch failed for {jwks_uri}: {e}")
            if cached:
                return cached.keys
            return []
    
    async def get_public_key_from_jwks(
        self,
        jwks_uri: str,
        kid: str
    ) -> Optional[Any]:
        """Get public key by kid from JWKS, with auto-refresh on miss.
        
        This handles key rotation gracefully:
        1. First tries to find key in cache
        2. If not found, refreshes JWKS and tries again
        
        Args:
            jwks_uri: URL to fetch JWKS from
            kid: Key ID to look for
            
        Returns:
            Public key object or None if not found
        """
        # First try with cached JWKS
        keys = await self.fetch_jwks(jwks_uri)
        
        for key in keys:
            if key.get("kid") == kid:
                try:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)
                except Exception as e:
                    self.logger.warning(f"Failed to parse JWK {kid}: {e}")
                    return None
        
        # Key not found - try refreshing JWKS (handles key rotation)
        self.logger.info(f"Key {kid} not found in cache, refreshing JWKS from {jwks_uri}")
        keys = await self.fetch_jwks(jwks_uri, force_refresh=True)
        
        for key in keys:
            if key.get("kid") == kid:
                try:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)
                except Exception as e:
                    self.logger.warning(f"Failed to parse JWK {kid} after refresh: {e}")
                    return None
        
        self.logger.warning(f"Key {kid} not found at {jwks_uri} even after refresh")
        return None
    
    def invalidate_cache(self, jwks_uri: Optional[str] = None) -> None:
        """Invalidate JWKS cache.
        
        Args:
            jwks_uri: Specific URI to invalidate, or None to clear all
        """
        if jwks_uri:
            self._jwks_cache.pop(jwks_uri, None)
            self.logger.debug(f"Invalidated JWKS cache for {jwks_uri}")
        else:
            self._jwks_cache.clear()
            self.logger.debug("Cleared all JWKS cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging.
        
        Returns:
            Dictionary with cache statistics
        """
        now = time.time()
        stats = {
            "total_entries": len(self._jwks_cache),
            "entries": {}
        }
        
        for uri, entry in self._jwks_cache.items():
            stats["entries"][uri] = {
                "keys_count": len(entry.keys),
                "expires_in": max(0, int(entry.expires_at - now)),
                "has_etag": entry.etag is not None,
                "last_fetch_ago": int(now - entry.last_fetch),
            }
        
        return stats
