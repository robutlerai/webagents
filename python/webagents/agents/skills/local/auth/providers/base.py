"""
Base OAuth Provider Interface

Defines the interface for OAuth providers that can be used
with the AOAuth skill for user authentication flows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseProvider(ABC):
    """Base class for OAuth providers.
    
    OAuth providers handle:
    - Authorization URL generation
    - Token exchange (code -> tokens)
    - ID token validation
    - User info retrieval
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider.
        
        Args:
            config: Provider configuration dictionary
        """
        self.config = config
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
    
    @abstractmethod
    def get_authorization_url(
        self,
        redirect_uri: str,
        scope: str,
        state: str,
        **kwargs
    ) -> str:
        """Generate authorization URL for OAuth flow.
        
        Args:
            redirect_uri: Callback URL after authorization
            scope: Requested scopes (space-separated)
            state: Anti-CSRF state parameter
            **kwargs: Provider-specific parameters
            
        Returns:
            Authorization URL to redirect user to
        """
        pass
    
    @abstractmethod
    async def exchange_code(
        self,
        code: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from callback
            redirect_uri: Callback URL used in authorization
            
        Returns:
            Token response with access_token, id_token, etc.
        """
        pass
    
    @abstractmethod
    async def validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """Validate and decode ID token.
        
        Args:
            id_token: JWT ID token from token response
            
        Returns:
            Decoded token claims
            
        Raises:
            ValueError: If token is invalid
        """
        pass
    
    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user info from provider.
        
        Optional method - not all providers support this.
        
        Args:
            access_token: Access token from token response
            
        Returns:
            User info dictionary or None
        """
        return None
    
    def get_provider_name(self) -> str:
        """Get provider name for identification.
        
        Returns:
            Provider name (e.g., "google", "robutler")
        """
        return self.__class__.__name__.lower().replace("provider", "")
