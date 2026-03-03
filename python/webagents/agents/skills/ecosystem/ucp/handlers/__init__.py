"""
UCP Payment Handlers

Modular payment handler implementations for different payment providers.
Each handler implements the PaymentHandler interface.
"""

from .base import PaymentHandler, PaymentInstrument, PaymentResult, PaymentStatus
from .robutler import RobutlerHandler
from .stripe import StripeHandler
from .google_pay import GooglePayHandler
from .elaisium import ElaisiumHandler

__all__ = [
    "PaymentHandler",
    "PaymentInstrument", 
    "PaymentResult",
    "PaymentStatus",
    "RobutlerHandler",
    "StripeHandler",
    "GooglePayHandler",
    "ElaisiumHandler",
]

# Handler registry by namespace
HANDLER_REGISTRY = {
    "ai.robutler.token": RobutlerHandler,
    "com.stripe.payments.card": StripeHandler,
    "google.pay": GooglePayHandler,
    "world.elaisium.vibe": ElaisiumHandler,
}

def get_handler(namespace: str) -> type:
    """Get handler class by namespace"""
    return HANDLER_REGISTRY.get(namespace)
