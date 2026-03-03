"""
Client Credentials Grant Handler

Implements OAuth 2.0 Client Credentials grant for agent-to-agent
authentication without user involvement.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..skill import AuthSkill


class ClientCredentialsGrant:
    """OAuth 2.0 Client Credentials Grant Handler.
    
    Used for agent-to-agent authentication where:
    - The calling agent authenticates with client_id/client_secret
    - No user authorization is required
    - Token is issued directly to the agent
    
    In Portal mode:
    - Credentials are validated against Portal
    - Portal assigns namespace scopes
    
    In Self-issued mode:
    - Agent generates and signs own token
    - Scopes are limited to allowed_scopes
    """
    
    def __init__(self, auth_skill: 'AuthSkill'):
        """Initialize grant handler.
        
        Args:
            auth_skill: Parent AuthSkill instance
        """
        self.auth_skill = auth_skill
        self.logger = logging.getLogger(__name__)
    
    async def handle(
        self,
        client_id: str = None,
        client_secret: str = None,
        scope: str = None,
        target: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle client credentials grant request.
        
        Args:
            client_id: Client identifier (agent ID)
            client_secret: Client secret for authentication
            scope: Requested scopes (space-separated)
            target: Target agent for the token (AOAuth extension)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Token response with access_token, token_type, expires_in
        """
        # Validate required parameters
        if not client_id:
            return {
                "error": "invalid_request",
                "error_description": "Missing client_id parameter"
            }
        
        # Parse requested scopes
        requested_scopes = scope.split() if scope else []
        
        # In self-issued mode, validate against allowed scopes
        # In portal mode, Portal handles scope validation
        from ..config import AuthMode
        
        if self.auth_skill.aoauth_config.mode == AuthMode.SELF_ISSUED:
            # For self-issued mode, we authenticate the client ourselves
            # In a real implementation, you'd validate client_secret
            # For now, we trust the request if it comes from an allowed source
            
            if not self._validate_client(client_id, client_secret):
                return {
                    "error": "invalid_client",
                    "error_description": "Client authentication failed"
                }
            
            # Filter scopes to allowed
            granted_scopes = [
                s for s in requested_scopes 
                if self.auth_skill._scope_allowed(s)
            ]
            
            # Use default scopes if none requested
            if not granted_scopes:
                granted_scopes = list(self.auth_skill.aoauth_config.allowed_scopes[:2])
            
            # Determine target (default to self if not specified)
            token_target = target or self.auth_skill._base_url
            
            # Generate self-issued token
            token = self.auth_skill._generate_self_issued_token(
                target=token_target,
                scopes=granted_scopes,
                extra_claims={
                    "client_id": client_id,
                    "grant_type": "client_credentials",
                }
            )
            
            return {
                "access_token": token,
                "token_type": "Bearer",
                "expires_in": self.auth_skill.aoauth_config.token_ttl,
                "scope": " ".join(granted_scopes),
            }
        
        else:
            # Portal mode - delegate to Portal
            try:
                robutler = self.auth_skill._providers.get("robutler")
                if not robutler:
                    return {
                        "error": "server_error",
                        "error_description": "Portal provider not configured"
                    }
                
                token = await robutler.request_token_async(
                    target=target or self.auth_skill._base_url,
                    scopes=requested_scopes or ["read"],
                )
                
                return {
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": self.auth_skill.aoauth_config.token_ttl,
                    "scope": scope or "read",
                }
                
            except Exception as e:
                self.logger.error(f"Portal token request failed: {e}")
                return {
                    "error": "server_error",
                    "error_description": f"Failed to obtain token from Portal: {e}"
                }
    
    def _validate_client(self, client_id: str, client_secret: str = None) -> bool:
        """Validate client credentials.
        
        In self-issued mode, this is a simple check.
        Override for more sophisticated validation.
        
        Args:
            client_id: Client identifier
            client_secret: Client secret (optional)
            
        Returns:
            True if client is valid
        """
        # For self-issued mode, we check the allow list
        # The actual authentication happens via JWT signature validation
        # on subsequent requests
        
        # Check if client is in allow list (or allow list is empty)
        return self.auth_skill._is_allowed(client_id)
