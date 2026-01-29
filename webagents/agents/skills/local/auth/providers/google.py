"""
Google OAuth Provider

Implements Google OAuth 2.0 / OpenID Connect for user authentication.
"""

from typing import Dict, Any, Optional
from urllib.parse import urlencode
import logging

import httpx
import jwt

from .base import BaseProvider


# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


class GoogleProvider(BaseProvider):
    """Google OAuth / OpenID Connect provider.
    
    Supports:
    - Authorization code flow
    - ID token validation via JWKS
    - User info retrieval
    
    Configuration:
        client_id: Google OAuth client ID
        client_secret: Google OAuth client secret
        hosted_domain: Optional G Suite domain restriction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Google provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.hosted_domain = config.get("hosted_domain")
        self.logger = logging.getLogger(__name__)
        
        # JWKS cache
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_expires_at: float = 0
    
    def get_authorization_url(
        self,
        redirect_uri: str,
        scope: str,
        state: str,
        **kwargs
    ) -> str:
        """Generate Google authorization URL.
        
        Args:
            redirect_uri: Callback URL
            scope: Requested scopes
            state: Anti-CSRF state
            **kwargs: Additional parameters (nonce, login_hint, prompt)
            
        Returns:
            Google authorization URL
        """
        # Ensure OpenID scopes are included
        scopes = scope.split() if scope else []
        if "openid" not in scopes:
            scopes.insert(0, "openid")
        if "email" not in scopes:
            scopes.append("email")
        if "profile" not in scopes:
            scopes.append("profile")
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": state,
            "access_type": "offline",
            "include_granted_scopes": "true",
        }
        
        # Add optional parameters
        if kwargs.get("nonce"):
            params["nonce"] = kwargs["nonce"]
        if kwargs.get("login_hint"):
            params["login_hint"] = kwargs["login_hint"]
        if kwargs.get("prompt"):
            params["prompt"] = kwargs["prompt"]
        
        # Restrict to hosted domain if configured
        if self.hosted_domain:
            params["hd"] = self.hosted_domain
        
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    
    async def exchange_code(
        self,
        code: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens.
        
        Args:
            code: Authorization code
            redirect_uri: Callback URL
            
        Returns:
            Token response with access_token, id_token, refresh_token
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            
            response.raise_for_status()
            return response.json()
    
    async def validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """Validate Google ID token.
        
        Validates:
        - Signature against Google JWKS
        - Issuer is Google
        - Audience matches client_id
        - Token is not expired
        
        Args:
            id_token: Google ID token
            
        Returns:
            Decoded token claims
            
        Raises:
            ValueError: If token is invalid
        """
        # Get key ID from header
        header = jwt.get_unverified_header(id_token)
        kid = header.get("kid")
        
        if not kid:
            raise ValueError("ID token missing kid header")
        
        # Fetch JWKS and find matching key
        public_key = await self._get_public_key(kid)
        
        if not public_key:
            raise ValueError(f"Key {kid} not found in Google JWKS")
        
        # Validate and decode
        claims = jwt.decode(
            id_token,
            public_key,
            algorithms=["RS256"],
            audience=self.client_id,
            issuer=["https://accounts.google.com", "accounts.google.com"],
        )
        
        # Validate hosted domain if configured
        if self.hosted_domain:
            token_hd = claims.get("hd")
            if token_hd != self.hosted_domain:
                raise ValueError(f"Token hosted domain {token_hd} doesn't match {self.hosted_domain}")
        
        return claims
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user info from Google.
        
        Args:
            access_token: Google access token
            
        Returns:
            User info dictionary
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.warning(f"Failed to fetch Google user info: {e}")
            return None
    
    async def _get_public_key(self, kid: str) -> Optional[Any]:
        """Get public key from Google JWKS.
        
        Args:
            kid: Key ID
            
        Returns:
            Public key or None
        """
        import time
        
        now = time.time()
        
        # Fetch JWKS if cache expired
        if not self._jwks_cache or now > self._jwks_expires_at:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(GOOGLE_JWKS_URL, timeout=10)
                    response.raise_for_status()
                    self._jwks_cache = response.json()
                    
                    # Parse cache-control header
                    cache_control = response.headers.get("Cache-Control", "")
                    ttl = 3600  # Default 1 hour
                    if "max-age=" in cache_control:
                        try:
                            ttl = int(cache_control.split("max-age=")[1].split(",")[0])
                        except ValueError:
                            pass
                    
                    self._jwks_expires_at = now + ttl
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch Google JWKS: {e}")
                if not self._jwks_cache:
                    return None
        
        # Find matching key
        for key in self._jwks_cache.get("keys", []):
            if key.get("kid") == kid:
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)
        
        return None
