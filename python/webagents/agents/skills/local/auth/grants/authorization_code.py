"""
Authorization Code Grant Handler

Implements OAuth 2.0 Authorization Code grant for user authorization
flows involving browser redirects.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import secrets
import time
import logging

if TYPE_CHECKING:
    from ..skill import AuthSkill


class AuthorizationCodeGrant:
    """OAuth 2.0 Authorization Code Grant Handler.
    
    Used for user authorization flows where:
    - User is redirected to authorization endpoint
    - User authorizes the request
    - Callback receives authorization code
    - Code is exchanged for tokens
    
    Supports external OAuth providers (Google, Portal) for
    the actual user authentication.
    """
    
    def __init__(self, auth_skill: 'AuthSkill', providers: Dict[str, Any]):
        """Initialize grant handler.
        
        Args:
            auth_skill: Parent AuthSkill instance
            providers: Dictionary of OAuth providers
        """
        self.auth_skill = auth_skill
        self.providers = providers
        self.logger = logging.getLogger(__name__)
        
        # In-memory authorization code store
        # In production, use Redis or database
        self._pending_codes: Dict[str, Dict[str, Any]] = {}
        self._code_ttl = 600  # 10 minutes
    
    async def handle(
        self,
        client_id: str = None,
        client_secret: str = None,
        scope: str = None,
        code: str = None,
        redirect_uri: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle authorization code grant request.
        
        Args:
            client_id: Client identifier
            client_secret: Client secret (optional)
            scope: Requested scopes (for code generation phase)
            code: Authorization code (for token exchange phase)
            redirect_uri: Redirect URI for callback
            **kwargs: Additional parameters
            
        Returns:
            Token response or error
        """
        if not code:
            return {
                "error": "invalid_request",
                "error_description": "Missing authorization code"
            }
        
        if not redirect_uri:
            return {
                "error": "invalid_request",
                "error_description": "Missing redirect_uri"
            }
        
        # Look up pending authorization
        pending = self._pending_codes.get(code)
        
        if not pending:
            return {
                "error": "invalid_grant",
                "error_description": "Invalid or expired authorization code"
            }
        
        # Check expiration
        if time.time() > pending.get("expires_at", 0):
            del self._pending_codes[code]
            return {
                "error": "invalid_grant",
                "error_description": "Authorization code expired"
            }
        
        # Validate redirect_uri matches
        if pending.get("redirect_uri") != redirect_uri:
            return {
                "error": "invalid_grant",
                "error_description": "redirect_uri mismatch"
            }
        
        # Validate client_id matches
        if pending.get("client_id") != client_id:
            return {
                "error": "invalid_grant",
                "error_description": "client_id mismatch"
            }
        
        # Remove used code (one-time use)
        del self._pending_codes[code]
        
        # Get the user/agent info from pending authorization
        user_info = pending.get("user_info", {})
        granted_scopes = pending.get("scopes", [])
        
        # Generate access token
        token = self.auth_skill._generate_self_issued_token(
            target=pending.get("target") or self.auth_skill._base_url,
            scopes=granted_scopes,
            extra_claims={
                "user_id": user_info.get("sub") or user_info.get("email"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "grant_type": "authorization_code",
            }
        )
        
        response = {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": self.auth_skill.aoauth_config.token_ttl,
            "scope": " ".join(granted_scopes),
        }
        
        # Include ID token if openid scope was requested
        if "openid" in granted_scopes:
            response["id_token"] = self._generate_id_token(
                user_info, granted_scopes, client_id
            )
        
        return response
    
    def create_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        scopes: list,
        user_info: Dict[str, Any],
        target: str = None,
    ) -> str:
        """Create authorization code for callback.
        
        Called after successful user authentication to generate
        the authorization code that will be returned to the client.
        
        Args:
            client_id: Client that initiated the request
            redirect_uri: Where to redirect with the code
            scopes: Granted scopes
            user_info: Authenticated user information
            target: Target for the eventual token
            
        Returns:
            Authorization code string
        """
        code = secrets.token_urlsafe(32)
        
        self._pending_codes[code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scopes": scopes,
            "user_info": user_info,
            "target": target,
            "created_at": time.time(),
            "expires_at": time.time() + self._code_ttl,
        }
        
        # Clean up expired codes
        self._cleanup_expired_codes()
        
        return code
    
    def _cleanup_expired_codes(self) -> None:
        """Remove expired authorization codes."""
        now = time.time()
        expired = [
            code for code, data in self._pending_codes.items()
            if now > data.get("expires_at", 0)
        ]
        for code in expired:
            del self._pending_codes[code]
    
    def _generate_id_token(
        self,
        user_info: Dict[str, Any],
        scopes: list,
        client_id: str
    ) -> str:
        """Generate OpenID Connect ID token.
        
        Args:
            user_info: User information
            scopes: Granted scopes
            client_id: Audience for the token
            
        Returns:
            Signed ID token JWT
        """
        import uuid
        from datetime import datetime, timedelta
        import jwt
        
        now = datetime.utcnow()
        
        claims = {
            "iss": self.auth_skill._issuer,
            "sub": user_info.get("sub") or user_info.get("email"),
            "aud": client_id,
            "iat": now,
            "exp": now + timedelta(seconds=self.auth_skill.aoauth_config.token_ttl),
            "auth_time": int(time.time()),
            "nonce": user_info.get("nonce"),
        }
        
        # Add profile claims if scope allows
        if "profile" in scopes:
            claims["name"] = user_info.get("name")
            claims["picture"] = user_info.get("picture")
        
        if "email" in scopes:
            claims["email"] = user_info.get("email")
            claims["email_verified"] = user_info.get("email_verified", False)
        
        return jwt.encode(
            claims,
            self.auth_skill.jwks.get_signing_key(),
            algorithm="RS256",
            headers={"kid": self.auth_skill._kid}
        )
    
    async def initiate_external_auth(
        self,
        provider_name: str,
        redirect_uri: str,
        scope: str,
        state: str,
        **kwargs
    ) -> str:
        """Initiate external OAuth flow.
        
        Generates authorization URL for external provider (Google, Portal).
        
        Args:
            provider_name: Name of provider ("google", "robutler")
            redirect_uri: Callback URL
            scope: Requested scopes
            state: Anti-CSRF state parameter
            **kwargs: Provider-specific parameters
            
        Returns:
            Authorization URL
            
        Raises:
            ValueError: If provider not configured
        """
        provider = self.providers.get(provider_name)
        
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not configured")
        
        return provider.get_authorization_url(
            redirect_uri=redirect_uri,
            scope=scope,
            state=state,
            **kwargs
        )
    
    async def complete_external_auth(
        self,
        provider_name: str,
        code: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """Complete external OAuth flow.
        
        Exchanges code for tokens with external provider and
        validates the ID token.
        
        Args:
            provider_name: Name of provider
            code: Authorization code from callback
            redirect_uri: Callback URL used in authorization
            
        Returns:
            Dictionary with tokens and user_info
            
        Raises:
            ValueError: If provider not configured or validation fails
        """
        provider = self.providers.get(provider_name)
        
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not configured")
        
        # Exchange code for tokens
        token_response = await provider.exchange_code(code, redirect_uri)
        
        # Validate ID token if present
        user_info = {}
        if token_response.get("id_token"):
            user_info = await provider.validate_id_token(token_response["id_token"])
        
        # Optionally fetch additional user info
        if token_response.get("access_token"):
            additional_info = await provider.get_user_info(token_response["access_token"])
            if additional_info:
                user_info.update(additional_info)
        
        return {
            "tokens": token_response,
            "user_info": user_info,
            "provider": provider_name,
        }
