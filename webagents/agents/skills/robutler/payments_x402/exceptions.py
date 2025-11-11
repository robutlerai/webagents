"""
x402-specific exceptions
"""

from typing import Dict, Any


class X402Error(Exception):
    """Base exception for x402-related errors"""
    
    def __init__(self, message: str, status_code: int = 500, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class PaymentRequired402(X402Error):
    """
    HTTP 402 Payment Required exception
    
    Raised when an HTTP endpoint requires payment.
    Contains x402 payment requirements in details.
    """
    
    def __init__(self, payment_requirements: Dict[str, Any]):
        super().__init__(
            "Payment required",
            status_code=402,
            details=payment_requirements
        )
        self.payment_requirements = payment_requirements


class X402UnsupportedScheme(X402Error):
    """Raised when no compatible payment scheme is available"""
    
    def __init__(self, message: str = "No compatible payment method available"):
        super().__init__(message, status_code=400)


class X402VerificationFailed(X402Error):
    """Raised when payment verification fails"""
    
    def __init__(self, reason: str):
        super().__init__(f"Payment verification failed: {reason}", status_code=402)


class X402SettlementFailed(X402Error):
    """Raised when payment settlement fails"""
    
    def __init__(self, reason: str):
        super().__init__(f"Payment settlement failed: {reason}", status_code=402)


class X402ExchangeFailed(X402Error):
    """Raised when crypto-to-credits exchange fails"""
    
    def __init__(self, reason: str):
        super().__init__(f"Exchange failed: {reason}", status_code=400)

