"""
Google Pay Payment Handler - UCP Payment Handler for Google Pay

Integrates with Google Pay to create payment instruments for UCP transactions.
Handles Google Pay payment data tokens.

Namespace: google.pay
"""

import logging
import json
import base64
from typing import Optional, Dict, Any, List

from .base import PaymentHandler, PaymentInstrument, PaymentResult, PaymentStatus
from ..exceptions import UCPPaymentError, UCPHandlerError

logger = logging.getLogger("webagents.skills.ucp.handlers.google_pay")


class GooglePayHandler(PaymentHandler):
    """
    Google Pay Payment Handler.
    
    Creates payment instruments from Google Pay payment data.
    The payment data token is obtained from the Google Pay Web SDK
    or mobile SDKs on the client side.
    
    Configuration:
        merchant_id: Google Pay merchant ID
        merchant_name: Display name for the merchant
        environment: "TEST" or "PRODUCTION"
        
    Credentials:
        payment_data: Google Pay payment data token (from client SDK)
        payment_token: Extracted payment token (alternative)
    """
    
    namespace = "google.pay"
    display_name = "Google Pay"
    
    # Supported card networks
    ALLOWED_CARD_NETWORKS = ["VISA", "MASTERCARD", "AMEX", "DISCOVER"]
    
    # Supported auth methods
    ALLOWED_AUTH_METHODS = ["PAN_ONLY", "CRYPTOGRAM_3DS"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Google Pay configuration
        self.merchant_id = self.config.get("merchant_id", "TEST")
        self.merchant_name = self.config.get("merchant_name", "WebAgents")
        self.environment = self.config.get("environment", "TEST")
    
    async def initialize(self) -> None:
        """Initialize Google Pay handler"""
        logger.debug(
            f"GooglePayHandler initialized: merchant_id={self.merchant_id}, "
            f"environment={self.environment}"
        )
        self._initialized = True
    
    async def create_instrument(
        self,
        credentials: Dict[str, Any],
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """
        Create payment instrument from Google Pay data.
        
        Args:
            credentials: {"payment_data": {...}} or {"payment_token": "..."}
            handler_config: Handler config from merchant profile
            
        Returns:
            PaymentInstrument with Google Pay token
        """
        handler_id = handler_config.get("id", "google_pay")
        
        # Get payment data
        payment_data = credentials.get("payment_data")
        payment_token = credentials.get("payment_token")
        
        if payment_data:
            return self._create_instrument_from_payment_data(
                payment_data,
                handler_id,
                handler_config
            )
        
        if payment_token:
            return self._create_instrument_from_token(
                payment_token,
                handler_id,
                handler_config
            )
        
        raise UCPPaymentError(
            message="No Google Pay payment data provided",
            handler_name=self.namespace,
            payment_status="missing_credentials",
            details={
                "accepted_credentials": ["payment_data", "payment_token"]
            }
        )
    
    def _create_instrument_from_payment_data(
        self,
        payment_data: Dict[str, Any],
        handler_id: str,
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """Create instrument from Google Pay payment data object"""
        try:
            # Extract payment method data
            payment_method_data = payment_data.get("paymentMethodData", {})
            
            # Get tokenization data
            tokenization_data = payment_method_data.get("tokenizationData", {})
            token_type = tokenization_data.get("type", "PAYMENT_GATEWAY")
            token = tokenization_data.get("token", "")
            
            # Parse token if it's JSON string
            if isinstance(token, str) and token.startswith("{"):
                try:
                    token_data = json.loads(token)
                except json.JSONDecodeError:
                    token_data = {"raw": token}
            else:
                token_data = {"raw": token}
            
            # Get card info for display
            card_info = payment_method_data.get("info", {})
            card_network = card_info.get("cardNetwork", "UNKNOWN")
            card_details = card_info.get("cardDetails", "****")
            
            # Build instrument data
            instrument_data = {
                "scheme": "exact",
                "network": "google_pay",
                "token_type": token_type,
                "payment_token": token,
                "card_network": card_network,
                "card_last4": card_details,
            }
            
            # Include signature data if present (for CRYPTOGRAM_3DS)
            if "signature" in token_data:
                instrument_data["signature"] = token_data["signature"]
            if "signedMessage" in token_data:
                instrument_data["signed_message"] = token_data["signedMessage"]
            
            logger.info(
                f"Created Google Pay instrument: {card_network} ****{card_details}"
            )
            
            return PaymentInstrument(
                handler_id=handler_id,
                type="wallet",
                data=instrument_data
            )
            
        except Exception as e:
            raise UCPPaymentError(
                message=f"Failed to parse Google Pay payment data: {str(e)}",
                handler_name=self.namespace,
                payment_status="parse_error"
            )
    
    def _create_instrument_from_token(
        self,
        payment_token: str,
        handler_id: str,
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """Create instrument from pre-extracted payment token"""
        # Token is already extracted, just wrap it
        return PaymentInstrument(
            handler_id=handler_id,
            type="wallet",
            data={
                "scheme": "exact",
                "network": "google_pay",
                "payment_token": payment_token,
            }
        )
    
    def can_handle(self, handler_config: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the merchant's handler config.
        """
        name = handler_config.get("name", "")
        
        # Match Google Pay namespaces
        return any(ns in name.lower() for ns in ["google.pay", "googlepay", "google_pay"])
    
    def get_required_credentials(self) -> List[Dict[str, Any]]:
        """Get required credential fields"""
        return [
            {
                "name": "payment_data",
                "type": "object",
                "required": False,
                "description": "Google Pay payment data object from SDK",
                "sensitive": True
            },
            {
                "name": "payment_token",
                "type": "string",
                "required": False,
                "description": "Google Pay payment token (pre-extracted)",
                "sensitive": True
            },
        ]
    
    def get_gpay_config(self, handler_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get Google Pay client configuration for web SDK.
        
        Returns configuration that can be passed to the Google Pay JS API.
        """
        config = handler_config.get("config", {})
        
        # Merge merchant config with defaults
        return {
            "apiVersion": config.get("api_version", 2),
            "apiVersionMinor": config.get("api_version_minor", 0),
            "merchantInfo": {
                "merchantId": config.get("merchant_info", {}).get("merchant_id") or self.merchant_id,
                "merchantName": config.get("merchant_info", {}).get("merchant_name") or self.merchant_name,
            },
            "allowedPaymentMethods": config.get("allowed_payment_methods", [
                {
                    "type": "CARD",
                    "parameters": {
                        "allowedAuthMethods": self.ALLOWED_AUTH_METHODS,
                        "allowedCardNetworks": self.ALLOWED_CARD_NETWORKS,
                    },
                }
            ]),
            "environment": self.environment,
        }
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information"""
        info = super().get_handler_info()
        info.update({
            "merchant_id": self.merchant_id,
            "merchant_name": self.merchant_name,
            "environment": self.environment,
            "supported_networks": self.ALLOWED_CARD_NETWORKS,
            "supported_auth_methods": self.ALLOWED_AUTH_METHODS,
        })
        return info
