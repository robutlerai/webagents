"""
Robutler Payment Handler - UCP Payment Handler for Robutler Tokens

Integrates with the existing Robutler payment token infrastructure
to enable UCP transactions using Robutler credits.

Namespace: ai.robutler.token
"""

import logging
from typing import Optional, Dict, Any, List

from .base import PaymentHandler, PaymentInstrument, PaymentResult, PaymentStatus
from ..exceptions import UCPPaymentError, UCPHandlerError

logger = logging.getLogger("webagents.skills.ucp.handlers.robutler")


class RobutlerHandler(PaymentHandler):
    """
    Robutler Token Payment Handler.
    
    Uses Robutler payment tokens (from the WebAgents platform) to pay
    for UCP transactions. Integrates with the existing PaymentSkill
    infrastructure.
    
    Configuration:
        token: Pre-configured payment token
        webagents_api_url: WebAgents API URL for token validation
        
    Credentials:
        token: Payment token in format "id:secret"
    """
    
    namespace = "ai.robutler.token"
    display_name = "Robutler Credits"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Pre-configured token (optional)
        self.token = self.config.get("token")
        
        # WebAgents API configuration
        self.webagents_api_url = self.config.get(
            "webagents_api_url",
            "https://webagents.ai"
        )
    
    async def initialize(self) -> None:
        """Initialize handler"""
        # Validate pre-configured token if provided
        if self.token:
            logger.debug(f"RobutlerHandler initialized with pre-configured token")
        else:
            logger.debug(f"RobutlerHandler initialized (token from context)")
        
        self._initialized = True
    
    async def create_instrument(
        self,
        credentials: Dict[str, Any],
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """
        Create payment instrument from Robutler token.
        
        Args:
            credentials: {"token": "token_id:token_secret"}
            handler_config: Handler config from merchant profile
            
        Returns:
            PaymentInstrument with token data
        """
        # Get token from credentials or use pre-configured
        token = credentials.get("token") or self.token
        
        if not token:
            raise UCPPaymentError(
                message="No Robutler payment token provided",
                handler_name=self.namespace,
                payment_status="missing_credentials"
            )
        
        # Validate token format
        if not self._validate_token_format(token):
            raise UCPPaymentError(
                message="Invalid token format - expected 'id:secret'",
                handler_name=self.namespace,
                payment_status="invalid_credentials"
            )
        
        # Get handler ID from config
        handler_id = handler_config.get("id", "robutler_token")
        
        # Create instrument
        # The token is sent to the merchant who will verify and redeem it
        instrument = PaymentInstrument(
            handler_id=handler_id,
            type="token",
            data={
                "scheme": "token",
                "network": "robutler",
                "token": token,
                # Include metadata for verification
                "api_url": self.webagents_api_url,
            }
        )
        
        logger.info(f"Created Robutler payment instrument for handler {handler_id}")
        
        return instrument
    
    def _validate_token_format(self, token: str) -> bool:
        """Validate token format is 'id:secret'"""
        if not isinstance(token, str):
            return False
        
        parts = token.split(":", 1)
        if len(parts) != 2:
            return False
        
        # Both parts should be non-empty
        return len(parts[0]) > 0 and len(parts[1]) > 0
    
    def can_handle(self, handler_config: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the merchant's handler config.
        
        Matches on:
        - name == "ai.robutler.token"
        - name contains "robutler" and config has network == "robutler"
        """
        name = handler_config.get("name", "")
        config = handler_config.get("config", {})
        
        # Direct namespace match
        if name == self.namespace:
            return True
        
        # Also match "dev.ucp.mock_payment" with robutler token support
        # (for testing against UCP playground)
        if "mock" in name.lower():
            supported_tokens = config.get("supported_tokens", [])
            # Mock handler that supports tokens
            return len(supported_tokens) > 0
        
        return False
    
    def get_required_credentials(self) -> List[Dict[str, Any]]:
        """Get required credential fields"""
        return [
            {
                "name": "token",
                "type": "string",
                "required": not bool(self.token),  # Required if no pre-configured token
                "description": "Robutler payment token (format: id:secret)",
                "sensitive": True
            }
        ]
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information"""
        info = super().get_handler_info()
        info.update({
            "has_preconfigured_token": bool(self.token),
            "webagents_api_url": self.webagents_api_url,
        })
        return info
