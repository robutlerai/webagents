"""
Stripe Payment Handler - UCP Payment Handler for Stripe

Integrates with Stripe to create payment instruments for UCP transactions.
Supports card tokenization and payment method creation.

Namespace: com.stripe.payments.card
"""

import logging
from typing import Optional, Dict, Any, List

from .base import PaymentHandler, PaymentInstrument, PaymentResult, PaymentStatus
from ..exceptions import UCPPaymentError, UCPHandlerError

logger = logging.getLogger("webagents.skills.ucp.handlers.stripe")

# Optional Stripe import
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None


class StripeHandler(PaymentHandler):
    """
    Stripe Payment Handler.
    
    Creates payment instruments for Stripe-based payments.
    Supports:
    - Pre-tokenized payment methods (payment_method_id)
    - Card tokenization via Stripe.js (client-side)
    - Direct card data (requires PCI compliance)
    
    Configuration:
        api_key: Stripe secret API key
        publishable_key: Stripe publishable key (for client-side)
        
    Credentials (one of):
        payment_method_id: Pre-created Stripe PaymentMethod ID
        token: Stripe token from Stripe.js
        card: Card details (number, exp_month, exp_year, cvc) - requires PCI
    """
    
    namespace = "com.stripe.payments.card"
    display_name = "Credit/Debit Card (Stripe)"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Stripe API keys
        self.api_key = self.config.get("api_key")
        self.publishable_key = self.config.get("publishable_key")
        
        # Stripe client configured
        self._stripe_configured = False
    
    async def initialize(self) -> None:
        """Initialize Stripe handler"""
        if not STRIPE_AVAILABLE:
            logger.warning("Stripe package not installed. Install with: pip install stripe")
            self._initialized = True
            return
        
        if self.api_key:
            stripe.api_key = self.api_key
            self._stripe_configured = True
            logger.debug("Stripe handler initialized with API key")
        else:
            logger.warning("Stripe handler initialized without API key - limited functionality")
        
        self._initialized = True
    
    async def create_instrument(
        self,
        credentials: Dict[str, Any],
        handler_config: Dict[str, Any]
    ) -> PaymentInstrument:
        """
        Create payment instrument from Stripe credentials.
        
        Args:
            credentials: Payment credentials (see class docstring)
            handler_config: Handler config from merchant profile
            
        Returns:
            PaymentInstrument with Stripe payment data
        """
        handler_id = handler_config.get("id", "stripe_card")
        
        # Check for pre-existing payment method
        if "payment_method_id" in credentials:
            return self._create_instrument_from_payment_method(
                credentials["payment_method_id"],
                handler_id
            )
        
        # Check for Stripe token (from Stripe.js/Elements)
        if "token" in credentials:
            return await self._create_instrument_from_token(
                credentials["token"],
                handler_id
            )
        
        # Check for card data (requires PCI compliance)
        if "card" in credentials:
            return await self._create_instrument_from_card(
                credentials["card"],
                handler_id
            )
        
        raise UCPPaymentError(
            message="No valid Stripe credentials provided",
            handler_name=self.namespace,
            payment_status="missing_credentials",
            details={
                "accepted_credentials": [
                    "payment_method_id",
                    "token",
                    "card"
                ]
            }
        )
    
    def _create_instrument_from_payment_method(
        self,
        payment_method_id: str,
        handler_id: str
    ) -> PaymentInstrument:
        """Create instrument from existing PaymentMethod"""
        return PaymentInstrument(
            handler_id=handler_id,
            type="card",
            data={
                "scheme": "exact",
                "network": "stripe",
                "payment_method_id": payment_method_id,
            }
        )
    
    async def _create_instrument_from_token(
        self,
        token: str,
        handler_id: str
    ) -> PaymentInstrument:
        """Create instrument from Stripe token"""
        if not STRIPE_AVAILABLE or not self._stripe_configured:
            # Just pass through the token
            return PaymentInstrument(
                handler_id=handler_id,
                type="card",
                data={
                    "scheme": "exact",
                    "network": "stripe",
                    "token": token,
                }
            )
        
        try:
            # Convert token to PaymentMethod
            payment_method = stripe.PaymentMethod.create(
                type="card",
                card={"token": token}
            )
            
            return PaymentInstrument(
                handler_id=handler_id,
                type="card",
                data={
                    "scheme": "exact",
                    "network": "stripe",
                    "payment_method_id": payment_method.id,
                }
            )
            
        except Exception as e:
            raise UCPPaymentError(
                message=f"Failed to create payment method from token: {str(e)}",
                handler_name=self.namespace,
                payment_status="tokenization_failed"
            )
    
    async def _create_instrument_from_card(
        self,
        card: Dict[str, Any],
        handler_id: str
    ) -> PaymentInstrument:
        """
        Create instrument from raw card data.
        
        WARNING: Handling raw card data requires PCI DSS compliance.
        Prefer using Stripe.js for client-side tokenization.
        """
        if not STRIPE_AVAILABLE or not self._stripe_configured:
            raise UCPPaymentError(
                message="Stripe not configured - cannot process raw card data",
                handler_name=self.namespace,
                payment_status="not_configured"
            )
        
        required = ["number", "exp_month", "exp_year", "cvc"]
        missing = [f for f in required if f not in card]
        if missing:
            raise UCPPaymentError(
                message=f"Missing card fields: {missing}",
                handler_name=self.namespace,
                payment_status="invalid_credentials"
            )
        
        try:
            # Create PaymentMethod from card data
            payment_method = stripe.PaymentMethod.create(
                type="card",
                card={
                    "number": card["number"],
                    "exp_month": card["exp_month"],
                    "exp_year": card["exp_year"],
                    "cvc": card["cvc"],
                }
            )
            
            return PaymentInstrument(
                handler_id=handler_id,
                type="card",
                data={
                    "scheme": "exact",
                    "network": "stripe",
                    "payment_method_id": payment_method.id,
                    "card_brand": payment_method.card.brand if payment_method.card else None,
                    "last4": payment_method.card.last4 if payment_method.card else None,
                }
            )
            
        except stripe.error.CardError as e:
            raise UCPPaymentError(
                message=f"Card error: {e.user_message}",
                handler_name=self.namespace,
                payment_status="card_declined",
                details={"decline_code": e.code}
            )
        except Exception as e:
            raise UCPPaymentError(
                message=f"Failed to create payment method: {str(e)}",
                handler_name=self.namespace,
                payment_status="tokenization_failed"
            )
    
    def can_handle(self, handler_config: Dict[str, Any]) -> bool:
        """
        Check if this handler can process the merchant's handler config.
        """
        name = handler_config.get("name", "")
        
        # Match Stripe namespaces
        stripe_namespaces = [
            "com.stripe.payments.card",
            "com.stripe.payments",
            "stripe",
        ]
        
        return any(ns in name.lower() for ns in ["stripe"])
    
    def get_required_credentials(self) -> List[Dict[str, Any]]:
        """Get required credential fields"""
        return [
            {
                "name": "payment_method_id",
                "type": "string",
                "required": False,
                "description": "Stripe PaymentMethod ID (pm_...)",
                "sensitive": True
            },
            {
                "name": "token",
                "type": "string", 
                "required": False,
                "description": "Stripe token from Stripe.js (tok_...)",
                "sensitive": True
            },
        ]
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information"""
        info = super().get_handler_info()
        info.update({
            "stripe_available": STRIPE_AVAILABLE,
            "stripe_configured": self._stripe_configured,
            "publishable_key": self.publishable_key,  # Safe to expose
        })
        return info
    
    async def cleanup(self) -> None:
        """Cleanup handler"""
        self._stripe_configured = False
        await super().cleanup()
