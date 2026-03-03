"""
UCP Skill - Universal Commerce Protocol Integration

Enables WebAgents to participate in the UCP ecosystem:
- Discover and transact with UCP-compliant merchants (Client Mode)
- Expose commerce capabilities as a UCP server (Server/Merchant Mode)
- Support multiple payment handlers (Stripe, Google Pay, Robutler)

Modes:
- "client": Agent can discover and purchase from UCP merchants
- "server": Agent acts as a UCP merchant, selling services
- "both": Agent can both purchase and sell via UCP

UCP Spec: https://ucp.dev/specification/overview/
"""

from .skill import UCPSkill
from .client import UCPClient
from .server import UCPServer, ServiceOffering
from .discovery import UCPDiscovery
from .exceptions import (
    UCPError,
    UCPDiscoveryError,
    UCPCheckoutError,
    UCPPaymentError,
    UCPHandlerError,
    UCPCapabilityNotSupported,
    UCPEscalationRequired,
)
from .schemas import (
    CheckoutSession,
    CheckoutStatus,
    MerchantProfile,
    LineItem,
    Buyer,
    PaymentHandlerConfig,
    PaymentInstrumentData,
)

__all__ = [
    # Main skill
    "UCPSkill",
    
    # Client mode
    "UCPClient",
    "UCPDiscovery",
    
    # Server mode
    "UCPServer",
    "ServiceOffering",
    
    # Exceptions
    "UCPError",
    "UCPDiscoveryError", 
    "UCPCheckoutError",
    "UCPPaymentError",
    "UCPHandlerError",
    "UCPCapabilityNotSupported",
    "UCPEscalationRequired",
    
    # Schemas
    "CheckoutSession",
    "CheckoutStatus",
    "MerchantProfile",
    "LineItem",
    "Buyer",
    "PaymentHandlerConfig",
    "PaymentInstrumentData",
]
