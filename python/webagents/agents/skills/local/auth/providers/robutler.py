"""
Robutler Portal OAuth Provider

Implements OAuth for Robutler Portal, supporting both
user authorization and agent-to-agent token requests.
"""

from typing import Dict, Any, Optional, List
from urllib.parse import urlencode
import logging

import httpx
import jwt

from .base import BaseProvider


# Default Portal URL
DEFAULT_PORTAL_URL = "https://robutler.ai"


class RobutlerProvider(BaseProvider):
    """Robutler Portal OAuth provider.
    
    Supports:
    - Authorization code flow for user authentication
    - Client credentials flow for agent-to-agent auth
    - Token validation via Portal JWKS
    - Namespace scope assignment
    
    Configuration:
        client_id: Agent identifier
        client_secret: Agent secret (optional for some flows)
        authority: Portal URL (default: https://robutler.ai)
        jwks_manager: Shared JWKSManager instance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Robutler provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        
        self.authority = config.get("authority", DEFAULT_PORTAL_URL)
        self.jwks_manager = config.get("jwks_manager")
        self.logger = logging.getLogger(__name__)
        
        # Portal endpoints
        self.jwks_uri = f"{self.authority}/api/auth/jwks"
        self.token_uri = f"{self.authority}/api/auth/token"
        self.auth_uri = f"{self.authority}/auth/authorize"
        self.userinfo_uri = f"{self.authority}/api/auth/userinfo"
    
    def get_authorization_url(
        self,
        redirect_uri: str,
        scope: str,
        state: str,
        **kwargs
    ) -> str:
        """Generate Portal authorization URL.
        
        Args:
            redirect_uri: Callback URL
            scope: Requested scopes
            state: Anti-CSRF state
            **kwargs: Additional parameters:
                - on_behalf_of: Agent acting on behalf of user
                - prompt: "none", "consent", or "login"
            
        Returns:
            Portal authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
        }
        
        # Add optional parameters
        if kwargs.get("on_behalf_of"):
            params["on_behalf_of"] = kwargs["on_behalf_of"]
        if kwargs.get("prompt"):
            params["prompt"] = kwargs["prompt"]
        if kwargs.get("login_hint"):
            params["login_hint"] = kwargs["login_hint"]
        
        return f"{self.auth_uri}?{urlencode(params)}"
    
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
            Token response with access_token, id_token
        """
        async with httpx.AsyncClient() as client:
            data = {
                "client_id": self.client_id,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
            
            if self.client_secret:
                data["client_secret"] = self.client_secret
            
            response = await client.post(
                self.token_uri,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            
            response.raise_for_status()
            return response.json()
    
    async def validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """Validate Portal ID token.
        
        Args:
            id_token: Portal ID token
            
        Returns:
            Decoded token claims
            
        Raises:
            ValueError: If token is invalid
        """
        if not self.jwks_manager:
            raise ValueError("JWKS manager not configured")
        
        # Get key ID from header
        header = jwt.get_unverified_header(id_token)
        kid = header.get("kid")
        
        if not kid:
            raise ValueError("ID token missing kid header")
        
        # Fetch public key from Portal JWKS
        public_key = await self.jwks_manager.get_public_key_from_jwks(
            self.jwks_uri, kid
        )
        
        if not public_key:
            raise ValueError(f"Key {kid} not found in Portal JWKS")
        
        # Validate and decode
        return jwt.decode(
            id_token,
            public_key,
            algorithms=["RS256"],
            audience=self.client_id,
        )
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user info from Portal.
        
        Args:
            access_token: Portal access token
            
        Returns:
            User info dictionary
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.userinfo_uri,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.warning(f"Failed to fetch Portal user info: {e}")
            return None
    
    def request_token(
        self,
        target: str,
        scopes: List[str],
        **kwargs
    ) -> str:
        """Request token from Portal for agent-to-agent auth.
        
        This is a synchronous call used for client_credentials flow.
        Portal will issue a token with namespace scopes.
        
        Args:
            target: Target agent URL or @name
            scopes: Requested scopes
            **kwargs: Additional claims
            
        Returns:
            JWT access token
            
        Raises:
            httpx.HTTPStatusError: On Portal error
        """
        response = httpx.post(
            self.token_uri,
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "target": target,
                "scope": " ".join(scopes),
            },
            timeout=10,
        )
        
        response.raise_for_status()
        return response.json()["access_token"]
    
    async def request_token_async(
        self,
        target: str,
        scopes: List[str],
        **kwargs
    ) -> str:
        """Async version of request_token.
        
        Args:
            target: Target agent
            scopes: Requested scopes
            
        Returns:
            JWT access token
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_uri,
                json={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "target": target,
                    "scope": " ".join(scopes),
                },
                timeout=10,
            )
            
            response.raise_for_status()
            return response.json()["access_token"]
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an access token.
        
        Args:
            refresh_token: Refresh token from original token response
            
        Returns:
            New token response
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_uri,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            
            response.raise_for_status()
            return response.json()
